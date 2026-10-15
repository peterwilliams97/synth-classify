"""score.py scores a classifier on Banking77 restricted to the `CLASS_SETS` entry named by `--set`,
and writes the report it prints to results/<set>/<split>/<classifier>.txt. The classifier is a model
saved by train.py (`--model`: ModernBERT, TF-IDF or frozen embeddings), an Ollama model with one
of llm.py's `PROMPTS` (`--llm`, `--prompt`), or an mlx-lm model with llm.py's `MLX_PROMPT` and
optionally a LoRA adapter trained by finetune.py (`--mlx`, `--adapter`). `--split test` is the Banking77 test split; `--split dev` is
`dev_and_shots`' sample of the train split, which is what prompts are chosen on.

It is the only file that reads Banking77, loaded from the `mteb/banking77` parquet copy (the
original `PolyAI/banking77` is a dataset script, which `datasets` no longer runs). `--write-train`
exports the restricted train split as jsonl so train.py can fit the real-data comparison model."""

import argparse
import json
import os
import random
import statistics
import sys
import time

from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from classes import CLASS_SETS
from llm import MLX_PROMPT, PROMPTS, classify

DATASET = "mteb/banking77"
DEV_PER_CLASS = 20
SHOTS_PER_CLASS = 3


def banking77(split, names):
    ds = load_dataset(DATASET, split=split)
    rows = [(r["text"], r["label_text"]) for r in ds if r["label_text"] in names]
    missing = set(names) - {label for _, label in rows}
    if missing:
        sys.exit(f"{DATASET} {split}: no rows for {sorted(missing)}")
    return [t for t, _ in rows], [names.index(label) for _, label in rows]


def dev_and_shots(names):
    """dev_and_shots draws, per class and with a fixed seed, `DEV_PER_CLASS` train rows as the dev
    split and the next `SHOTS_PER_CLASS` as that class's `fewshot` examples, so no example is a
    dev row. Returns dev texts, dev labels and {class name: [example text]}."""
    texts, labels = banking77("train", names)
    rng = random.Random(0)
    dev, shots = [], {}
    for i, name in enumerate(names):
        rows = [j for j, label in enumerate(labels) if label == i]
        rng.shuffle(rows)
        dev += rows[:DEV_PER_CLASS]
        shots[name] = [texts[j] for j in rows[DEV_PER_CLASS : DEV_PER_CLASS + SHOTS_PER_CLASS]]
    return [texts[j] for j in dev], [labels[j] for j in dev], shots


def predict(model_dir, texts, names):
    """predict returns a label index and the seconds taken for each of `texts`. Each text is
    classified in a call of its own, so the seconds are what one message costs and compare with
    `classify` in llm.py, which also sends one message at a time."""
    preds, seconds = [], []
    tfidf = os.path.join(model_dir, "tfidf.joblib")
    if os.path.exists(tfidf):
        import joblib

        clf = joblib.load(tfidf)
        for text in texts:
            start = time.perf_counter()
            preds.append(clf.predict([text])[0])
            seconds.append(time.perf_counter() - start)
        return preds, seconds

    frozen = os.path.join(model_dir, "embed.joblib")
    if os.path.exists(frozen):
        import joblib

        from train import EMBED_PROMPT, NO_PROMPT, embed, load_embedder

        clf = joblib.load(frozen)
        prompt = "" if os.path.exists(os.path.join(model_dir, NO_PROMPT)) else EMBED_PROMPT
        tok, model, device = load_embedder()
        for text in texts:
            start = time.perf_counter()
            preds.append(clf.predict(embed([text], prompt, tok, model, device))[0])
            seconds.append(time.perf_counter() - start)
        return preds, seconds

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    # The saved config carries train.py's label order; map through names, not positions.
    to_ours = {i: names.index(name) for i, name in model.config.id2label.items()}
    with torch.no_grad():
        for text in texts:
            start = time.perf_counter()
            batch = tok(text, truncation=True, max_length=128, return_tensors="pt").to(device)
            preds.append(to_ours[model(**batch).logits.argmax(-1).item()])
            seconds.append(time.perf_counter() - start)
    return preds, seconds


def report(classifier, split, gold, preds, seconds, names, invalid=None):
    """report formats the scores. `invalid` is the number of `preds` that are -1, an mlx-lm reply
    that is no class name, or None for a classifier that cannot give one; the line is printed only
    for the former."""
    ids = range(len(names))
    width = max(len(name) for name in names)
    recalls = recall_score(gold, preds, labels=ids, average=None)
    lines = [
        f"model     {classifier}",
        f"{split + ' rows':<9} {len(gold)} ({DATASET}, {len(names)} classes)",
        f"accuracy  {accuracy_score(gold, preds):.4f}",
        f"macro F1  {f1_score(gold, preds, labels=ids, average='macro'):.4f}",
        f"seconds   {statistics.median(seconds):.3g} (median per message, one message per call)",
    ]
    if invalid is not None:
        lines.append(f"invalid   {invalid} (replies that are not a class name; counted as wrong)")
    lines.append("recall")
    lines += [f"  {name:<{width}} {r:.4f}" for name, r in zip(names, recalls)]
    lines.append("confusion (row: true class, column: predicted class, numbered as the rows are)")
    if invalid:
        lines.append("  invalid replies are in no column, so a row can sum to less than its class")
    lines.append(f"  {'':<{width + 3}}" + "".join(f"{i:>4}" for i in ids))
    for i, (name, row) in enumerate(zip(names, confusion_matrix(gold, preds, labels=ids))):
        lines.append(f"  {i:>2} {name:<{width}}" + "".join(f"{n:>4}" for n in row))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(CLASS_SETS), default="distinct")
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--model", help="directory written by train.py")
    ap.add_argument("--llm", metavar="MODEL", help="Ollama model to score in place of --model")
    ap.add_argument("--prompt", choices=list(PROMPTS), help="required with --llm")
    ap.add_argument("--mlx", metavar="MODEL", help=f"mlx-lm model to score, {MLX_PROMPT} prompt")
    ap.add_argument("--adapter", metavar="DIR", help="LoRA adapter from finetune.py, with --mlx")
    ap.add_argument("--write-train", metavar="JSONL", help="export the restricted train split")
    args = ap.parse_args()
    names = list(CLASS_SETS[args.set])

    if args.write_train:
        texts, labels = banking77("train", names)
        os.makedirs(os.path.dirname(args.write_train) or ".", exist_ok=True)
        with open(args.write_train, "w") as f:
            for text, label in zip(texts, labels):
                f.write(json.dumps({"text": text, "label": names[label]}) + "\n")
        print(f"wrote {len(texts)} rows to {args.write_train}", file=sys.stderr)
    if bool(args.model) + bool(args.llm) + bool(args.mlx) != 1:
        if args.write_train and not args.model and not args.llm and not args.mlx:
            return
        ap.error("give exactly one of --model, --llm and --mlx, or --write-train alone")
    if bool(args.llm) != bool(args.prompt):
        ap.error("--llm and --prompt go together")
    if args.adapter and not args.mlx:
        ap.error("--adapter needs --mlx")

    shots = None
    if args.split == "dev":
        texts, gold, shots = dev_and_shots(names)
    else:
        texts, gold = banking77("test", names)
        if args.prompt == "fewshot":
            _, _, shots = dev_and_shots(names)

    invalid = None
    if args.mlx:
        classifier = f"{args.mlx} prompt {MLX_PROMPT}"
        stem = f"{os.path.basename(os.path.normpath(args.mlx))}-{MLX_PROMPT}"
        if args.adapter:
            classifier += f" adapter {args.adapter}"
            stem = os.path.basename(os.path.normpath(args.adapter))
        answers = classify(texts, args.set, args.mlx, MLX_PROMPT, mlx=True, adapter=args.adapter)
        preds = [-1 if label is None else names.index(label) for label, _ in answers]
        seconds = [s for _, s in answers]
        invalid = preds.count(-1)
    elif args.llm:
        classifier = f"{args.llm} prompt {args.prompt}"
        stem = f"{args.llm.replace(':', '-')}-{args.prompt}"
        answers = classify(texts, args.set, args.llm, args.prompt, shots)
        preds = [names.index(label) for label, _ in answers]
        seconds = [s for _, s in answers]
    else:
        classifier = args.model
        stem = os.path.basename(os.path.normpath(args.model))
        preds, seconds = predict(args.model, texts, names)

    text = report(classifier, args.split, gold, preds, seconds, names, invalid)
    sys.stdout.write(text)
    out = os.path.join("results", args.set, args.split, stem + ".txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
