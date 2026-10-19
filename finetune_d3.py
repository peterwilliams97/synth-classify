"""finetune_d3.py fine-tunes `vllm-sr/d3-flash` with LoRA on a jsonl file of {text, label} rows and
saves the merged model to `--out`, for score.py `--d3 DIR`. It follows the Liquid4All cookbook's
finetuning/notebooks/sft_for_decision_models_with_trl.ipynb (LoRA rank, alpha, dropout, learning
rate and schedule, effective batch, gradient checkpointing, new option orders each epoch, merge and
save with the model's code) with one difference the model forces: d3-flash has no language-model
head. Its answer is a 255-way readout over the answer codes at the last prompt position
(`readout.safetensors`), computed over a prompt whose full-attention layers see the whole prompt
(d3_runtime.py's `PromptEncoder`). So the loss is the cross-entropy of that readout, over the
question's codes, on the correct option's code, through the runtime's own forward pass; there is
no completion to generate and no TRL trainer.

Each row is asked jev.py's `jev_question` for `fewshot`, the question `--d3 --prompt fewshot` asks,
with its options in a random order per row and epoch, rendered by the runtime's own `D3.text`. The
`fewshot` examples come from `dev_and_shots` in score.py, which reads Banking77; this file reads only
`--train`.

Fixed, none tuned on a test split: LoRA rank 16, alpha 32, dropout 0.05 on the language model's
attention q/k/v/o, Gated DeltaNet in_proj_qkv/z/a/b and out_proj, and MLP gate/up/down projections;
the vision tower, the readout and the `conv1d` kernels (the forward reads `conv1d.weight` directly,
so a LoRA wrapper on it is never applied) stay frozen. AdamW at 2e-4, weight decay 0, cosine
schedule with 5% warmup, gradient norm clipped at 1.0, one prompt per forward pass and 16 per
optimizer step. It runs on MPS only."""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time

from classes import CLASS_SETS
from d3 import REVISIONS, load
from jev import jev_question
from score import dev_and_shots
from train import read_jsonl, write_minutes

BASE = "vllm-sr/d3-flash"
EFFECTIVE_BATCH = 16
LORA_TARGETS = (
    r"language_model\.layers\.\d+\.("
    r"self_attn\.(q|k|v|o)_proj"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"
    r"|mlp\.(gate|up|down)_proj)"
)
# FINETUNE is the file `main` writes into `--out`; d3.py's `revision` reads it so a cached response
# from an earlier model saved to the same directory is not reused.
FINETUNE = "finetune.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", choices=list(CLASS_SETS), default="confusable")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    classes = CLASS_SETS[args.set]
    names = list(classes)
    texts, labels = read_jsonl(args.train, names)
    _, _, examples, _ = dev_and_shots(names)

    start = time.perf_counter()
    torch.manual_seed(args.seed)
    rt = load(BASE, REVISIONS[BASE]).runtime
    rng = random.Random(args.seed)
    # (token ids, index of the correct option); every epoch's rows get new option orders and the
    # concatenation is shuffled, as the notebook's trainer shuffles its concatenated epochs.
    rows = []
    for _ in range(args.epochs):
        for text, label in zip(texts, labels):
            order = rng.sample(names, len(names))
            question = jev_question({name: classes[name] for name in order}, "fewshot", examples)
            ids = rt.tokenizer(rt.text(text, question), add_special_tokens=False)["input_ids"]
            rows.append((ids, order.index(names[label])))
    rng.shuffle(rows)

    model = get_peft_model(
        rt.backbone,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=LORA_TARGETS),
    )
    model.print_trainable_parameters()
    rt.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    steps = math.ceil(len(rows) / EFFECTIVE_BATCH)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(0.05 * steps), steps)

    losses = []
    for step in range(steps):
        batch = rows[step * EFFECTIVE_BATCH : (step + 1) * EFFECTIVE_BATCH]
        total = 0.0
        for ids, target in batch:
            loss = readout_loss(rt, ids, target, len(names)) / len(batch)
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(total)
        print(f"step {step + 1}/{steps} loss {total:.4f}", file=sys.stderr, flush=True)

    rt.backbone.gradient_checkpointing_disable()
    merged = model.merge_and_unload()
    save(merged, args.out)
    with open(os.path.join(args.out, FINETUNE), "w") as f:
        record = {
            "base": BASE,
            "revision": REVISIONS[BASE],
            "train": args.train,
            "rows": len(texts),
            "epochs": args.epochs,
            "seed": args.seed,
            "step_losses": losses,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        json.dump(record, f, indent=1)
    write_minutes(args.set, args.out, args.train, len(texts), time.perf_counter() - start)


def readout_loss(rt, ids, target, options):
    """readout_loss is the cross-entropy, on option `target`, of the readout over the first `options`
    answer codes at the last position of prompt `ids`: d3_runtime.py's `D3.logits` for one unpadded
    prompt, with gradients."""
    import torch

    x = torch.tensor([ids], device=rt.device)
    hidden = rt.encoder(rt.backbone.get_input_embeddings()(x), torch.ones_like(x))[:, -1]
    logits = (hidden.float() @ rt.readout.T)[:, :options] / rt.temperature
    return torch.nn.functional.cross_entropy(logits, torch.tensor([target], device=rt.device))


def save(backbone, out):
    """save writes the merged backbone's weights to `out` and copies in every other file of the base
    snapshot (the model's code, config, decision_config.json, readout, tokenizer, processor configs,
    LICENSE and MODEL_MANIFEST.json), which is what `--d3 DIR` loads. The manifest's hashes are the base
    model's, so d3.py loads a local directory with the check skipped."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(BASE, revision=REVISIONS[BASE], local_files_only=True)
    os.makedirs(out, exist_ok=True)
    backbone.save_pretrained(out, max_shard_size="5GB")
    for name in os.listdir(snapshot):
        path = os.path.join(snapshot, name)
        if os.path.isfile(path) and not name.startswith(("model-", "model.safetensors")):
            shutil.copyfile(path, os.path.join(out, name))


if __name__ == "__main__":
    main()
