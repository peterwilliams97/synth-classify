"""score.py scores a model saved by train.py on the Banking77 test split, restricted to `CLASSES`.
It is the only file that reads Banking77, loaded from the `mteb/banking77` parquet copy (the
original `PolyAI/banking77` is a dataset script, which `datasets` no longer runs). `--write-train`
exports the restricted train split as jsonl so train.py can fit the real-data comparison model."""

import argparse
import json
import os
import sys

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score

from classes import CLASSES

DATASET = "mteb/banking77"
LABELS = list(CLASSES)


def banking77(split):
    ds = load_dataset(DATASET, split=split)
    rows = [(r["text"], r["label_text"]) for r in ds if r["label_text"] in CLASSES]
    missing = set(LABELS) - {label for _, label in rows}
    if missing:
        sys.exit(f"{DATASET} {split}: no rows for {sorted(missing)}")
    return [t for t, _ in rows], [LABELS.index(label) for _, label in rows]


def predict(model_dir, texts):
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
    to_ours = {i: LABELS.index(name) for i, name in model.config.id2label.items()}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="directory written by train.py")
    ap.add_argument("--write-train", metavar="JSONL", help="export the restricted train split")
    args = ap.parse_args()

    if args.write_train:
        texts, labels = banking77("train")
        with open(args.write_train, "w") as f:
            for text, label in zip(texts, labels):
                f.write(json.dumps({"text": text, "label": LABELS[label]}) + "\n")
        print(f"wrote {len(texts)} rows to {args.write_train}", file=sys.stderr)
    if not args.model:
        if not args.write_train:
            ap.error("give --model, --write-train, or both")
        return

    texts, gold = banking77("test")
    preds = predict(args.model, texts)
    recalls = recall_score(gold, preds, labels=range(len(LABELS)), average=None)
    print(f"model     {args.model}")
    print(f"test rows {len(texts)} ({DATASET}, {len(LABELS)} classes)")
    print(f"accuracy  {accuracy_score(gold, preds):.4f}")
    print(f"macro F1  {f1_score(gold, preds, labels=range(len(LABELS)), average='macro'):.4f}")
    print("recall")
    for name, r in zip(LABELS, recalls):
        print(f"  {name:<24} {r:.4f}")


if __name__ == "__main__":
    main()
