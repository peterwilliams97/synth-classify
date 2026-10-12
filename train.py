"""train.py fits a classifier over the `CLASS_SETS` entry named by `--set` on a jsonl file of
{text, label} rows and saves it to `--out`. The default is a fine-tune of
answerdotai/ModernBERT-base; `--tfidf` fits the TF-IDF + logistic regression baseline instead.
Neither looks at a test set: epochs and learning rate are fixed, so score.py's numbers are not tuned
on what they measure."""

import argparse
import json
import os
import random
import sys

from classes import CLASS_SETS

BASE = "answerdotai/ModernBERT-base"


def read_jsonl(path, names):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        sys.exit(f"{path}: no rows")
    unknown = sorted({r["label"] for r in rows} - set(names))
    if unknown:
        sys.exit(f"{path}: labels not in the class set: {unknown}")
    return [r["text"] for r in rows], [names.index(r["label"]) for r in rows]


def train_tfidf(texts, labels, out):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(C=10, max_iter=1000),
    )
    model.fit(texts, labels)
    joblib.dump(model, os.path.join(out, "tfidf.joblib"))


def train_modernbert(texts, labels, names, out, epochs, lr, batch_size):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup

    # float32 throughout: MPS has no float64.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={name: i for i, name in enumerate(names)},
    ).to(device)

    steps = epochs * -(-len(texts) // batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    model.train()
    order = list(range(len(texts)))
    for epoch in range(epochs):
        random.shuffle(order)
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            batch = tok(
                [texts[j] for j in idx],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            target = torch.tensor([labels[j] for j in idx], device=device)
            loss = model(**batch, labels=target).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{epochs} loss {total / len(order):.4f}", file=sys.stderr)

    model.save_pretrained(out)
    tok.save_pretrained(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", choices=list(CLASS_SETS), default="distinct")
    ap.add_argument("--tfidf", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = list(CLASS_SETS[args.set])
    texts, labels = read_jsonl(args.train, names)
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    if args.tfidf:
        train_tfidf(texts, labels, args.out)
    else:
        import torch

        torch.manual_seed(args.seed)
        train_modernbert(texts, labels, names, args.out, args.epochs, args.lr, args.batch_size)
    print(f"trained on {len(texts)} rows, saved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
