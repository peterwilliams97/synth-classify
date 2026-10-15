"""train.py fits a classifier over the `CLASS_SETS` entry named by `--set` on a jsonl file of
{text, label} rows and saves it to `--out`. The default is a fine-tune of
answerdotai/ModernBERT-base; `--tfidf` fits the TF-IDF + logistic regression baseline instead, and
`--embed` fits logistic regression on frozen `EMBED_BASE` embeddings. None looks at a test set:
epochs, learning rate and regularisation are fixed, so score.py's numbers are not tuned on what they
measure."""

import argparse
import json
import os
import random
import sys
import time

from classes import CLASS_SETS

BASE = "answerdotai/ModernBERT-base"

# The embedding model is pinned: its modeling code is fetched from the Hub and run
# (trust_remote_code), so a revision that moves would change what runs as well as the weights.
EMBED_BASE = "dmis-lab/UltraBERT-Embed-150M"
EMBED_REVISION = "367d5343b5a0164d57b810cedf5de4b8983d5984"
# The model card wants a task instruction on the query side; this is MTEB's Banking77Classification
# instruction, which names the domain and the task and no class.
EMBED_PROMPT = "Instruct: Given a online banking query, find the corresponding intents\nQuery: "


def read_jsonl(path, names):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        sys.exit(f"{path}: no rows")
    unknown = sorted({r["label"] for r in rows} - set(names))
    if unknown:
        sys.exit(f"{path}: labels not in the class set: {unknown}")
    return [r["text"] for r in rows], [names.index(r["label"]) for r in rows]


def write_minutes(class_set, out, train, rows, seconds):
    """write_minutes records how long a training run took, base model load included, in
    results/<class_set>/train/<name>.txt, `name` being the last component of `out` as in
    score.py's reports. finetune.py calls it too, so both kinds of training are timed alike. A run
    under a minute is recorded in seconds: a TF-IDF fit is 0.0 minutes to one decimal."""
    took = f"minutes   {seconds / 60:.1f}" if seconds >= 60 else f"seconds   {seconds:.2f}"
    text = (
        f"model     {out}\n"
        f"train     {train} ({rows} rows)\n"
        f"{took} (base model load to save)\n"
    )
    sys.stderr.write(text)
    path = os.path.join("results", class_set, "train", os.path.basename(os.path.normpath(out)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".txt", "w") as f:
        f.write(text)


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


def load_embedder():
    """load_embedder returns the tokenizer, model and device `embed` takes. SDPA attention in
    float32: FlashAttention does not run on MPS, and MPS has no float64."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(EMBED_BASE, revision=EMBED_REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMBED_BASE,
        revision=EMBED_REVISION,
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=torch.float32,
    )
    return tok, model.to(device).eval(), device


def embed(texts, tok, model, device):
    """embed returns one row per text: the final hidden state at position 0, `EMBED_PROMPT`
    prepended, unnormalised. That is the model card's Sentence Transformers pipeline (CLS pooling,
    prompt included, no Normalize module) without the sentence-transformers dependency. score.py's
    `predict` calls it with one text at a time; train.py with a batch."""
    import torch

    batch = tok(
        [EMBED_PROMPT + t for t in texts],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        return model(**batch).last_hidden_state[:, 0].float().cpu().numpy()


def train_embed(texts, labels, out, batch_size):
    """train_embed fits logistic regression on frozen embeddings; the encoder is not updated. C is
    sklearn's default, as in MTEB's classification evaluator, and is not tuned."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    tok, model, device = load_embedder()
    starts = range(0, len(texts), batch_size)
    x = np.concatenate([embed(texts[i : i + batch_size], tok, model, device) for i in starts])
    clf = LogisticRegression(max_iter=1000).fit(x, labels)
    joblib.dump(clf, os.path.join(out, "embed.joblib"))


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
    ap.add_argument("--embed", action="store_true", help=f"frozen {EMBED_BASE} + logistic regression")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.tfidf and args.embed:
        ap.error("--tfidf and --embed are two different models; give one")

    names = list(CLASS_SETS[args.set])
    texts, labels = read_jsonl(args.train, names)
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    start = time.perf_counter()
    if args.tfidf:
        train_tfidf(texts, labels, args.out)
    elif args.embed:
        train_embed(texts, labels, args.out, args.batch_size)
    else:
        import torch

        torch.manual_seed(args.seed)
        train_modernbert(texts, labels, names, args.out, args.epochs, args.lr, args.batch_size)
    write_minutes(args.set, args.out, args.train, len(texts), time.perf_counter() - start)


if __name__ == "__main__":
    main()
