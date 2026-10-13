"""finetune.py trains a LoRA adapter for an mlx-lm model on a jsonl file of {text, label} rows and
saves it to `--out`, for score.py `--mlx MODEL --adapter DIR`. Each row becomes a two-turn chat:
the user turn is llm.py's `user_turn` of the `MLX_PROMPT` prompt and the row's text, which is what
`classify_mlx` sends at scoring time, and the assistant turn is the row's class name.

The hyperparameters are fixed and none was tuned on a test split: LoRA rank 8, scale 20, dropout 0
on every linear layer of the last 16 transformer blocks; Adam at learning rate 1e-4; batch size 4;
2 passes over the rows; seed 0; loss on the assistant turn only. Like generate.py and llm.py it
does not import `datasets` or read Banking77 — `--train` is whatever jsonl it is given."""

import argparse
import time
import types

from classes import CLASS_SETS
from llm import MLX_PROMPT, PROMPTS, user_turn
from train import read_jsonl, write_minutes

# 4-bit quantisation of google/gemma-4-E4B-it, the model Ollama serves as `gemma4:e4b`.
BASE = "mlx-community/gemma-4-e4b-it-4bit"
EPOCHS = 2
BATCH_SIZE = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", choices=list(CLASS_SETS), default="distinct")
    ap.add_argument("--model", default=BASE, help="mlx-lm model path or Hugging Face repo")
    args = ap.parse_args()

    import numpy as np
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import ChatDataset

    classes = CLASS_SETS[args.set]
    names = list(classes)
    texts, labels = read_jsonl(args.train, names)
    instructions = PROMPTS[MLX_PROMPT](classes, None)
    chats = [
        {
            "messages": [
                {"role": "user", "content": user_turn(instructions, text)},
                {"role": "assistant", "content": names[label]},
            ]
        }
        for text, label in zip(texts, labels)
    ]

    # `train_model` reads its settings from the namespace mlx-lm's own command line builds, and
    # writes them to adapter_config.json in `--out`, which is what loading the adapter reads.
    iters = EPOCHS * (len(chats) // BATCH_SIZE)
    config = types.SimpleNamespace(
        **{
            **CONFIG_DEFAULTS,
            "model": args.model,
            "data": args.train,
            "adapter_path": args.out,
            "fine_tune_type": "lora",
            "num_layers": 16,
            "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "batch_size": BATCH_SIZE,
            "iters": iters,
            "mask_prompt": True,
            "seed": 0,
            "save_every": iters,
        }
    )
    np.random.seed(config.seed)  # the batch order; `train_model` seeds mlx itself

    start = time.perf_counter()
    model, tok = load(args.model)
    train_set = ChatDataset(chats, tok, mask_prompt=True)
    train_model(config, model, train_set, [])
    write_minutes(args.set, args.out, args.train, len(texts), time.perf_counter() - start)


if __name__ == "__main__":
    main()
