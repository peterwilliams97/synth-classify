"""human.py scores human labellers on the `confusable` test split, from label sheets written by
`score.py --set confusable --write-label-sheet` and filled in, against data/confusable/test_key.csv,
the answer key the same run wrote. Each sheet gets a report in the format of score.py's, with no
seconds line, plus its errors, the errors that chose the other intent of the true class's pair, and
its most frequent confusion, printed and written to results/confusable/test/human-<sheet stem>.txt.
A blank label, or one that is not a class name, counts as wrong and on the `invalid` line. Then one
summary line: mean, min and max accuracy over the sheets, and for each two sheets the share of
messages on which both gave the same label, a blank agreeing with nothing.

It reads no Banking77: the key carries the true labels."""

import argparse
import csv
import itertools
import os
import statistics
import sys

from sklearn.metrics import accuracy_score, confusion_matrix

from classes import CLASS_SETS
from score import report

SET = "confusable"
KEY = os.path.join("data", SET, "test_key.csv")


def read_sheet(path):
    """read_sheet returns {id: label} for the rows after the sheet's id, text, label header row,
    with the label stripped of surrounding spaces. Exits if the header row is missing."""
    with open(path, newline="") as f:
        rows = iter(csv.reader(f))
        for row in rows:
            if row[:3] == ["id", "text", "label"]:
                return {row[0]: (row[2] if len(row) > 2 else "").strip() for row in rows if row}
    sys.exit(f"{path}: no id, text, label header row")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True, metavar="CSV", help="filled label sheets")
    args = ap.parse_args()
    names = list(CLASS_SETS[SET])
    ids = range(len(names))
    width = max(len(name) for name in names)

    with open(KEY, newline="") as f:
        key = {r["id"]: r["label"] for r in csv.DictReader(f)}
    if not key:
        sys.exit(f"{KEY}: no rows")
    order = list(key)
    gold = [names.index(key[mid]) for mid in order]

    answers, accuracies = {}, []
    for path in args.sheets:
        sheet = read_sheet(path)
        if set(sheet) != set(key):
            sys.exit(f"{path}: {len(set(key) - set(sheet))} key ids missing, {len(set(sheet) - set(key))} not in {KEY}")
        labels = [sheet[mid] for mid in order]
        answers[path] = labels
        preds = [names.index(label) if label in names else -1 for label in labels]
        invalid = preds.count(-1)
        accuracies.append(accuracy_score(gold, preds))

        # `confusable` lists its classes pair by pair, so classes 2k and 2k+1 are one pair.
        errors = sum(g != p for g, p in zip(gold, preds))
        in_pair = sum(p != -1 and g != p and g // 2 == p // 2 for g, p in zip(gold, preds))
        matrix = confusion_matrix(gold, preds, labels=ids)
        cells = [(matrix[i][j], i, j) for i in ids for j in ids if i != j]
        n, i, j = max(cells)
        stem = os.path.splitext(os.path.basename(path))[0]
        text = report(path, "test", gold, preds, None, names, invalid)
        text += f"errors    {errors} of {len(gold)}, {in_pair} inside a pair\n"
        text += f"most frequent confusion  {n}: {names[i]} labelled {names[j]}\n" if n else ""
        sys.stdout.write(text)
        out = os.path.join("results", SET, "test", f"human-{stem}.txt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(text)
        print(f"wrote {out}", file=sys.stderr)

    agreement = [
        f"{os.path.basename(a)}/{os.path.basename(b)} "
        f"{statistics.mean(x != '' and x == y for x, y in zip(answers[a], answers[b])):.4f}"
        for a, b in itertools.combinations(args.sheets, 2)
    ]
    print(
        f"{len(args.sheets)} sheets: accuracy mean {statistics.mean(accuracies):.4f}, "
        f"min {min(accuracies):.4f}, max {max(accuracies):.4f}; "
        f"agreement {', '.join(agreement) or 'needs two sheets'}"
    )


if __name__ == "__main__":
    main()
