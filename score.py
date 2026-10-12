"""score.py scores a model saved by train.py on the Banking77 test split, restricted to the
`CLASS_SETS` entry named by `--set`, and writes the report it prints to results/<set>/<model>.txt.
It is the only file that reads Banking77, loaded from the `mteb/banking77` parquet copy (the
original `PolyAI/banking77` is a dataset script, which `datasets` no longer runs). `--write-train`
exports the restricted train split as jsonl so train.py can fit the real-data comparison model."""

import argparse
import json
import os
import sys

from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from classes import CLASS_SETS

DATASET = "mteb/banking77"


def banking77(split, names):
    ds = load_dataset(DATASET, split=split)
    rows = [(r["text"], r["label_text"]) for r in ds if r["label_text"] in names]
    missing = set(names) - {label for _, label in rows}
    if missing:
        sys.exit(f"{DATASET} {split}: no rows for {sorted(missing)}")
    return [t for t, _ in rows], [names.index(label) for _, label in rows]


def predict(model_dir, texts, names):
    tfidf = os.path.join(model_dir, "tfidf.joblib")
    if os.path.exists(tfidf):
        import joblib

        return list(joblib.load(tfidf).predict(texts))

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    # The saved config carries train.py's label order; map through names, not positions.
    to_ours = {i: names.index(name) for i, name in model.config.id2label.items()}
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = tok(
                texts[i : i + 32],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            preds += [to_ours[p] for p in model(**batch).logits.argmax(-1).tolist()]
    return preds


def report(model_dir, texts, gold, preds, names):
    ids = range(len(names))
    width = max(len(name) for name in names)
    recalls = recall_score(gold, preds, labels=ids, average=None)
    lines = [
        f"model     {model_dir}",
        f"test rows {len(texts)} ({DATASET}, {len(names)} classes)",
        f"accuracy  {accuracy_score(gold, preds):.4f}",
        f"macro F1  {f1_score(gold, preds, labels=ids, average='macro'):.4f}",
        "recall",
    ]
    lines += [f"  {name:<{width}} {r:.4f}" for name, r in zip(names, recalls)]
    lines.append("confusion (row: true class, column: predicted class, numbered as the rows are)")
    lines.append(f"  {'':<{width + 3}}" + "".join(f"{i:>4}" for i in ids))
    for i, (name, row) in enumerate(zip(names, confusion_matrix(gold, preds, labels=ids))):
        lines.append(f"  {i:>2} {name:<{width}}" + "".join(f"{n:>4}" for n in row))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(CLASS_SETS), default="distinct")
    ap.add_argument("--model", help="directory written by train.py")
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
    if not args.model:
        if not args.write_train:
            ap.error("give --model, --write-train, or both")
        return

    texts, gold = banking77("test", names)
    text = report(args.model, texts, gold, predict(args.model, texts, names), names)
    sys.stdout.write(text)
    out = os.path.join("results", args.set, os.path.basename(os.path.normpath(args.model)) + ".txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
