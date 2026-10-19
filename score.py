"""score.py scores a classifier on Banking77 restricted to the `CLASS_SETS` entry named by `--set`,
and writes the report it prints to results/<set>/<split>/<classifier>.txt. The classifier is a model
saved by train.py (`--model`: ModernBERT, TF-IDF or frozen embeddings), an Ollama model with one
of llm.py's `PROMPTS` (`--llm`, `--prompt`), or an mlx-lm model with llm.py's `MLX_PROMPT` and
optionally a LoRA adapter trained by finetune.py (`--mlx`, `--adapter`; with `--reply logits`, no
adapter and the class read off the letter logits of one forward pass, with `--prompt` `definitions`, the
default, or `fewshot`), or the Laya decision model
zero-shot with one of llm.py's `LAYA_PROMPTS` (`--laya`, `--prompt`), or TypeSafe's System One API
asked jev.py's question (`--jev`, `--prompt`, default `definitions`), or a Decision 3.0 model such
as `vllm-sr/d3-flash` run locally and asked the same question (`--d3`, `--prompt`, default
`definitions`; d3.py, in `.venv-d3`), or a directory finetune_d3.py wrote. `--split test` is the
Banking77 test split; `--split dev` is `dev_and_shots`' sample of the train split, which is what
prompts are chosen on; `--split rest` is the rows of data/<set>/banking77_train_nodev.jsonl that
are not in `--train-file`, for a `--model` trained on a subset of that file, or for `--jev`, `--d3` or
`--mlx --reply logits`, which were trained on none of it; with `--prompt fewshot` the rows that are
`fewshot` examples are left out. `--split file` is the rows of `--val-file`, scored under the
same rules as `--split rest`, with the report written to results/<set>/<val file name>/ so each
fold of `--write-folds` has its own. `--dump` also writes each message's probability over the class set, for
confidence.py; it refuses `--llm` and `--mlx` without `--reply logits`, which give a class and no
probabilities. A `--jev` report also gives the input tokens per message and their price.

It is the only file that reads Banking77, loaded from the `mteb/banking77` parquet copy (the
original `PolyAI/banking77` is a dataset script, which `datasets` no longer runs). `--write-train`
exports the restricted train split as jsonl so train.py can fit the real-data comparison model;
with `--no-dev` it leaves out the dev rows, so a model trained on it can be scored on dev.
`--write-folds K` deals the same rows into K folds and writes each fold's train and val files."""

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
from d3 import D3_PROMPTS, classify_d3
from jev import DOLLARS_PER_MILLION_TOKENS, JEV_PROMPTS, classify_jev
from llm import (
    LAYA_MODEL,
    LAYA_PROMPTS,
    MLX_PROMPT,
    PROMPTS,
    classify,
    classify_laya,
    classify_mlx_logits,
)

DATASET = "mteb/banking77"
DEV_PER_CLASS = 20
SHOTS_PER_CLASS = 3
# The top probability above which a Laya or `--reply logits` answer counts as high-confidence on
# the report's `p > 0.9` line.
HIGH_CONFIDENCE = 0.9


def banking77(split, names):
    """banking77 returns the texts, label indices in `names` and ids of the `split` rows whose class
    is in `names`. An id is "<split>:<row index in the unrestricted split>"; Banking77 has none of
    its own."""
    ds = load_dataset(DATASET, split=split)
    rows = [(i, r["text"], r["label_text"]) for i, r in enumerate(ds) if r["label_text"] in names]
    missing = set(names) - {label for _, _, label in rows}
    if missing:
        sys.exit(f"{DATASET} {split}: no rows for {sorted(missing)}")
    return (
        [t for _, t, _ in rows],
        [names.index(label) for _, _, label in rows],
        [f"{split}:{i}" for i, _, _ in rows],
    )


def dev_and_shots(names):
    """dev_and_shots draws, per class and with a fixed seed, `DEV_PER_CLASS` train rows as the dev
    split and the next `SHOTS_PER_CLASS` as that class's `fewshot` examples, so no example is a
    dev row. Returns dev texts, dev labels, {class name: [example text]} and dev ids."""
    texts, labels, ids = banking77("train", names)
    rng = random.Random(0)
    dev, shots = [], {}
    for i, name in enumerate(names):
        rows = [j for j, label in enumerate(labels) if label == i]
        rng.shuffle(rows)
        dev += rows[:DEV_PER_CLASS]
        shots[name] = [texts[j] for j in rows[DEV_PER_CLASS : DEV_PER_CLASS + SHOTS_PER_CLASS]]
    return [texts[j] for j in dev], [labels[j] for j in dev], shots, [ids[j] for j in dev]


def rest(set_name, train_file, names):
    """rest returns the texts, label indices and ids of the rows of
    data/<set>/banking77_train_nodev.jsonl that are not in `train_file`, so a model trained on a
    subset of that file has its routing threshold set on rows it did not see. Rows match on text and
    label; exits if a `train_file` row is not in the file. An id is the row's `banking77` train id,
    which `classify_jev` caches under."""
    def read(path):
        with open(path) as f:
            return [(r["text"], r["label"]) for r in map(json.loads, f)]
    every = read(os.path.join("data", set_name, "banking77_train_nodev.jsonl"))
    train = set(read(train_file))
    if missing := train - set(every):
        sys.exit(f"{train_file}: {len(missing)} rows not in banking77_train_nodev.jsonl")
    kept = [(text, label) for text, label in every if (text, label) not in train]
    texts, labels, ids = banking77("train", names)
    by_row = {(text, names[label]): mid for text, label, mid in zip(texts, labels, ids)}
    return (
        [text for text, _ in kept],
        [names.index(label) for _, label in kept],
        [by_row[row] for row in kept],
    )


def from_file(path, names):
    """from_file returns the texts, label indices and `banking77` train ids of the rows of the jsonl
    file at `path`, matched on text and label; exits if a row is not in the train split."""
    with open(path) as f:
        rows = [(r["text"], r["label"]) for r in map(json.loads, f)]
    texts, labels, ids = banking77("train", names)
    by_row = {(text, names[label]): mid for text, label, mid in zip(texts, labels, ids)}
    if missing := [row for row in rows if row not in by_row]:
        sys.exit(f"{path}: {len(missing)} rows not in the {DATASET} train split")
    return [text for text, _ in rows], [names.index(label) for _, label in rows], [by_row[r] for r in rows]


def deal_folds(labels, names, k, seed):
    """deal_folds assigns each of the rows with `labels` a fold in 0..`k`-1: each class's rows, in
    `names` order, are shuffled by one random.Random(`seed`) and dealt round-robin, the deal
    carrying on from one class to the next so fold sizes differ by at most one row overall as well as
    within a class."""
    rng = random.Random(seed)
    fold, turn = [None] * len(labels), 0
    for i in range(len(names)):
        rows = [j for j, label in enumerate(labels) if label == i]
        rng.shuffle(rows)
        for j in rows:
            fold[j], turn = turn % k, turn + 1
    return fold


def predict(model_dir, texts, names):
    """predict returns, for each of `texts`, a label index, the probabilities over `names` in
    `names` order, and the seconds taken. Each text is classified in a call of its own, so the
    seconds are what one message costs and compare with `classify` in llm.py, which also sends one
    message at a time."""
    preds, probs, seconds = [], [], []

    def sklearn(clf, x):
        # `classes_` holds the training label indices, so a column is mapped through it rather
        # than taken as a position in `names`.
        row = clf.predict_proba(x)[0]
        vector = [0.0] * len(names)
        for label, p in zip(clf.classes_, row):
            vector[int(label)] = float(p)
        return int(clf.classes_[row.argmax()]), vector

    tfidf = os.path.join(model_dir, "tfidf.joblib")
    if os.path.exists(tfidf):
        import joblib

        clf = joblib.load(tfidf)
        for text in texts:
            start = time.perf_counter()
            pred, vector = sklearn(clf, [text])
            seconds.append(time.perf_counter() - start)
            preds.append(pred)
            probs.append(vector)
        return preds, probs, seconds

    frozen = os.path.join(model_dir, "embed.joblib")
    if os.path.exists(frozen):
        import joblib

        from train import EMBED_PROMPT, NO_PROMPT, embed, load_embedder

        clf = joblib.load(frozen)
        prompt = "" if os.path.exists(os.path.join(model_dir, NO_PROMPT)) else EMBED_PROMPT
        tok, model, device = load_embedder()
        for text in texts:
            start = time.perf_counter()
            pred, vector = sklearn(clf, embed([text], prompt, tok, model, device))
            seconds.append(time.perf_counter() - start)
            preds.append(pred)
            probs.append(vector)
        return preds, probs, seconds

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
            row = torch.softmax(model(**batch).logits[0].float(), -1).tolist()
            seconds.append(time.perf_counter() - start)
            vector = [0.0] * len(names)
            for i, p in enumerate(row):
                vector[to_ours[i]] = p
            preds.append(to_ours[max(range(len(row)), key=row.__getitem__)])
            probs.append(vector)
    return preds, probs, seconds


def report(classifier, split, gold, preds, seconds, names, invalid=None, probs=None, tokens=None):
    """report formats the scores. `invalid` is the number of `preds` that are -1, an mlx-lm reply
    that is no class name, or None for a classifier that cannot give one; the line is printed only
    for the former. `probs` is the probability a Laya or `--reply logits` answer gave its choice,
    one per message, and adds the accuracy on the messages where it is above `HIGH_CONFIDENCE`.
    `tokens` is a `--jev` request's billed input tokens, one per message, and adds their mean and
    price."""
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
    if tokens is not None:
        mean = statistics.mean(tokens)
        lines.append(f"tokens    {mean:.1f} (mean billed input tokens per message)")
        lines.append(
            f"dollars   {mean * DOLLARS_PER_MILLION_TOKENS:.2f} per million messages "
            f"({sum(tokens) * DOLLARS_PER_MILLION_TOKENS / 1e6:.6f} for these "
            f"{len(tokens)}, at ${DOLLARS_PER_MILLION_TOKENS} per million input tokens)"
        )
    if invalid is not None:
        lines.append(f"invalid   {invalid} (replies that are not a class name; counted as wrong)")
    if probs is not None:
        high = [i for i, p in enumerate(probs) if p > HIGH_CONFIDENCE]
        acc = accuracy_score([gold[i] for i in high], [preds[i] for i in high]) if high else 0.0
        lines.append(
            f"p > {HIGH_CONFIDENCE}   accuracy {acc:.4f} on {len(high)} of {len(gold)} messages "
            "(top probability above it)"
        )
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
    ap.add_argument("--split", choices=["dev", "test", "rest", "file"], default="test")
    ap.add_argument("--train-file", metavar="JSONL", help="with --split rest: the file whose rows are left out")
    ap.add_argument("--val-file", metavar="JSONL", help="with --split file: the rows to score")
    ap.add_argument("--model", help="directory written by train.py")
    ap.add_argument("--llm", metavar="MODEL", help="Ollama model to score in place of --model")
    ap.add_argument("--laya", action="store_true", help="score the Laya decision model zero-shot")
    ap.add_argument("--jev", action="store_true", help="score TypeSafe System One, jev.py")
    ap.add_argument("--d3", metavar="MODEL", help="Decision 3.0 model to score, d3.py")
    ap.add_argument("--prompt", choices=list(PROMPTS), help="required with --llm and --laya; --jev, --d3 and --reply logits default to definitions")
    ap.add_argument("--mlx", metavar="MODEL", help=f"mlx-lm model to score, {MLX_PROMPT} prompt")
    ap.add_argument("--adapter", metavar="DIR", help="LoRA adapter from finetune.py, with --mlx")
    ap.add_argument(
        "--reply",
        choices=["generate", "logits"],
        default="generate",
        help="with --mlx: generate a class name, or read a class letter off one forward pass",
    )
    ap.add_argument("--write-train", metavar="JSONL", help="export the restricted train split")
    ap.add_argument("--no-dev", action="store_true", help="with --write-train: leave out the dev rows")
    ap.add_argument("--write-folds", type=int, metavar="K", help="write K cross-validation folds of the train split")
    ap.add_argument("--fold-seed", type=int, default=0, metavar="S", help="seed of the --write-folds deal")
    ap.add_argument(
        "--dump",
        metavar="JSONL",
        help="write {text, label, pred, prob} per message; prob is over the class set in order",
    )
    args = ap.parse_args()
    names = list(CLASS_SETS[args.set])

    if args.no_dev and not args.write_train:
        ap.error("--no-dev needs --write-train")
    if args.write_train:
        texts, labels, ids = banking77("train", names)
        if args.no_dev:
            dev = set(dev_and_shots(names)[3])
            keep = [i for i, mid in enumerate(ids) if mid not in dev]
            texts, labels = [texts[i] for i in keep], [labels[i] for i in keep]
        os.makedirs(os.path.dirname(args.write_train) or ".", exist_ok=True)
        with open(args.write_train, "w") as f:
            for text, label in zip(texts, labels):
                f.write(json.dumps({"text": text, "label": names[label]}) + "\n")
        print(f"wrote {len(texts)} rows to {args.write_train}", file=sys.stderr)
    if args.write_folds is not None:
        if args.write_folds < 2:
            ap.error("--write-folds needs at least 2 folds")
        texts, labels, _ = banking77("train", names)
        fold = deal_folds(labels, names, args.write_folds, args.fold_seed)
        os.makedirs(os.path.join("data", args.set), exist_ok=True)
        for i in range(args.write_folds):
            for part, keep in [("train", lambda f: f != i), ("val", lambda f: f == i)]:
                path = os.path.join("data", args.set, f"banking77_fold{i}_{part}.jsonl")
                rows = [j for j in range(len(texts)) if keep(fold[j])]
                with open(path, "w") as f:
                    for j in rows:
                        f.write(json.dumps({"text": texts[j], "label": names[labels[j]]}) + "\n")
                print(f"wrote {len(rows)} rows to {path}", file=sys.stderr)
    if bool(args.model) + bool(args.llm) + bool(args.mlx) + args.laya + args.jev + bool(args.d3) != 1:
        if (args.write_train or args.write_folds) and not (args.model or args.llm or args.mlx or args.laya or args.jev or args.d3):
            return
        ap.error("give exactly one of --model, --llm, --mlx, --laya, --jev and --d3, or --write-train or --write-folds alone")
    logits = args.mlx and args.reply == "logits"
    if (args.jev or args.d3 or logits) and not args.prompt:
        args.prompt = "definitions"
    if bool(args.llm or args.laya or args.jev or args.d3 or logits) != bool(args.prompt):
        ap.error("--prompt goes with --llm, --laya, --jev, --d3 or --reply logits, and --llm and --laya need it")
    if logits and args.prompt not in ("definitions", "fewshot"):
        ap.error("--reply logits takes --prompt definitions or fewshot")
    if args.jev and args.prompt not in JEV_PROMPTS:
        ap.error(f"--jev takes --prompt {' or '.join(JEV_PROMPTS)}")
    if args.d3 and args.prompt not in D3_PROMPTS:
        ap.error(f"--d3 takes --prompt {' or '.join(D3_PROMPTS)}")
    if args.laya and args.prompt not in LAYA_PROMPTS:
        ap.error(f"--laya takes --prompt {' or '.join(LAYA_PROMPTS)}")
    if args.adapter and not args.mlx:
        ap.error("--adapter needs --mlx")
    if args.reply == "logits" and (not args.mlx or args.adapter):
        ap.error("--reply logits needs --mlx and takes no --adapter")
    if args.dump and (args.llm or (args.mlx and args.reply != "logits")):
        ap.error("--dump needs probabilities: --model, --laya, --jev, --d3 or --mlx --reply logits")
    if (args.split == "rest") != bool(args.train_file):
        ap.error("--split rest and --train-file go together")
    if (args.split == "file") != bool(args.val_file):
        ap.error("--split file and --val-file go together")
    if args.split in ("rest", "file") and not (args.model or args.jev or args.d3 or logits):
        ap.error(f"--split {args.split} scores a --model, --jev, --d3 or --mlx --reply logits only")

    shots = None
    if args.split == "dev":
        texts, gold, shots, ids = dev_and_shots(names)
    elif args.split in ("rest", "file"):
        if args.split == "rest":
            texts, gold, ids = rest(args.set, args.train_file, names)
        else:
            texts, gold, ids = from_file(args.val_file, names)
        if args.prompt == "fewshot":
            # An example is in the question, so its row would be scored with its own answer given.
            _, _, shots, _ = dev_and_shots(names)
            examples = {text for group in shots.values() for text in group}
            keep = [i for i, text in enumerate(texts) if text not in examples]
            print(f"left out {len(texts) - len(keep)} rows that are fewshot examples", file=sys.stderr)
            texts, gold, ids = [texts[i] for i in keep], [gold[i] for i in keep], [ids[i] for i in keep]
    else:
        texts, gold, ids = banking77("test", names)
        if args.prompt == "fewshot":
            _, _, shots, _ = dev_and_shots(names)

    # `vectors` is each message's probabilities over `names`, from the classifiers that have them;
    # `probs` is the top one of each, which only the Laya and `--reply logits` reports print.
    invalid = probs = vectors = tokens = None
    if args.jev:
        answers = classify_jev(texts, ids, args.set, args.prompt, shots)
        answered = sorted({m for *_, m in answers})
        classifier = f"TypeSafe System One {' '.join(answered)} prompt {args.prompt}"
        stem = f"jev-{args.prompt}"
        preds = [names.index(label) for label, *_ in answers]
        vectors = [v for _, v, *_ in answers]
        probs = [v[p] for v, p in zip(vectors, preds)]
        seconds = [s for _, _, s, _, _ in answers]
        tokens = [t for *_, t, _ in answers]
    elif args.d3:
        classifier = f"{args.d3} prompt {args.prompt}"
        stem = f"{os.path.basename(os.path.normpath(args.d3))}-{args.prompt}"
        answers = classify_d3(texts, ids, args.set, args.prompt, shots, args.d3)
        preds = [names.index(label) for label, _, _ in answers]
        vectors = [v for _, v, _ in answers]
        probs = [v[p] for v, p in zip(vectors, preds)]
        seconds = [s for _, _, s in answers]
    elif args.laya:
        classifier = f"{LAYA_MODEL} prompt {args.prompt}"
        stem = f"laya-{args.prompt}"
        answers = classify_laya(texts, args.set, args.prompt)
        preds = [names.index(label) for label, _, _ in answers]
        vectors = [v for _, v, _ in answers]
        probs = [v[p] for v, p in zip(vectors, preds)]
        seconds = [s for _, _, s in answers]
    elif args.mlx and args.reply == "logits":
        classifier = f"{args.mlx} prompt {args.prompt} reply logits"
        stem = f"{os.path.basename(os.path.normpath(args.mlx))}-{args.prompt}-logits"
        examples = shots if args.prompt == "fewshot" else None
        answers = classify_mlx_logits(texts, args.set, args.mlx, examples)
        preds = [names.index(label) for label, _, _ in answers]
        vectors = [v for _, v, _ in answers]
        probs = [v[p] for v, p in zip(vectors, preds)]
        seconds = [s for _, _, s in answers]
    elif args.mlx:
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
        preds, vectors, seconds = predict(args.model, texts, names)

    split = args.split
    if args.split == "file":
        split = os.path.splitext(os.path.basename(args.val_file))[0]
    text = report(classifier, split, gold, preds, seconds, names, invalid, probs, tokens)
    sys.stdout.write(text)
    out = os.path.join("results", args.set, split, stem + ".txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out}", file=sys.stderr)
    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        with open(args.dump, "w") as f:
            for text, label, pred, vector in zip(texts, gold, preds, vectors):
                row = {"text": text, "label": names[label], "pred": names[pred], "prob": vector}
                f.write(json.dumps(row) + "\n")
        print(f"wrote {len(texts)} rows to {args.dump}", file=sys.stderr)


if __name__ == "__main__":
    main()
