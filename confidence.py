"""confidence.py asks whether a classifier's score can route messages: answer the ones it is sure
of, send the rest to a person. It reads two `score.py --dump` files of one classifier, dev and
test, and takes the raw score of a message to be its top probability. Discrimination is measured
on test with nothing fitted; calibration and the routing thresholds are fitted on dev alone and
applied to test, so dev is the only labelled data the report fits on. It writes the report it
prints to results/<set>/confidence/<name>.txt."""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from classes import CLASS_SETS

ALPHAS = [0.02, 0.05, 0.10]
ECE_BINS = 10
# One-sided confidence level of the Clopper-Pearson upper bound on dev error behind `tau_cp`.
CP_LEVEL = 0.90


def read_dump(path, names):
    """read_dump returns raw scores, correctness and true label indices from a `--dump` file whose
    probability vectors are over `names`."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        sys.exit(f"{path}: no rows")
    for r in rows:
        if len(r["prob"]) != len(names) or r["label"] not in names:
            sys.exit(f"{path}: row not over the {len(names)} classes of this set: {r['text']!r}")
    score = np.array([max(r["prob"]) for r in rows])
    correct = np.array([r["pred"] == r["label"] for r in rows], dtype=float)
    label = np.array([names.index(r["label"]) for r in rows])
    return score, correct, label


def ece(conf, correct, raw):
    """ece is the expected calibration error over `ECE_BINS` bins of equal count, messages ordered
    by `conf` with ties broken by `raw`, so a plateau of the isotonic fit is split by the raw score
    rather than by file order."""
    order = np.lexsort((raw, conf))
    total = 0.0
    for b in np.array_split(order, ECE_BINS):
        total += len(b) / len(order) * abs(conf[b].mean() - correct[b].mean())
    return total


def cp_upper(errors, n):
    return 1.0 if errors == n else float(beta.ppf(CP_LEVEL, errors + 1, n - errors))


def threshold(score, correct, alpha, bound):
    """threshold is the lowest dev score `tau` at which the dev error among messages scoring at
    least `tau` — or, with `bound`, its Clopper-Pearson upper bound — is at most `alpha`. It is
    infinite when no dev score qualifies, which accepts nothing."""
    for tau in np.unique(score):
        kept = score >= tau
        errors, n = int((1 - correct[kept]).sum()), int(kept.sum())
        if (cp_upper(errors, n) if bound else errors / n) <= alpha:
            return tau
    return np.inf


def routed(score, correct, tau):
    """routed returns the coverage of `tau` and the error among the messages it accepts, None when
    it accepts none."""
    kept = score >= tau
    return kept.mean(), (1 - correct[kept]).mean() if kept.any() else None


def fmt(x, spec=".4f"):
    return "-" if x is None else format(x, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(CLASS_SETS), default="distinct")
    ap.add_argument("--model-name", required=True, help="report name, results/<set>/confidence/")
    ap.add_argument("--dev", required=True, help="score.py --split dev --dump file")
    ap.add_argument("--test", required=True, help="score.py --split test --dump file")
    args = ap.parse_args()
    names = list(CLASS_SETS[args.set])
    width = max(len(name) for name in names)

    dev_s, dev_c, _ = read_dump(args.dev, names)
    test_s, test_c, test_y = read_dump(args.test, names)
    if test_c.all() or not test_c.any():
        sys.exit(f"{args.test}: every prediction is {'right' if test_c.all() else 'wrong'}")

    iso = IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(dev_s, dev_c)
    calibrated = iso.predict(test_s)

    lines = [
        f"model     {args.model_name}",
        f"dev       {args.dev} ({len(dev_s)} rows, accuracy {dev_c.mean():.4f})",
        f"test      {args.test} ({len(test_s)} rows, accuracy {test_c.mean():.4f})",
        "score     top probability of each message",
        "discrimination on test, nothing fitted",
        f"  AUROC of score against correctness  {roc_auc_score(test_c, test_s):.4f}",
        f"  mean score, right predictions       {test_s[test_c == 1].mean():.4f}"
        f" ({int(test_c.sum())})",
        f"  mean score, wrong predictions       {test_s[test_c == 0].mean():.4f}"
        f" ({int(len(test_c) - test_c.sum())})",
        f"calibration: isotonic regression of correctness on score, fitted on dev",
        f"  test ECE, {ECE_BINS} equal-count bins  before {ece(test_s, test_c, test_s):.4f}"
        f"  after {ece(calibrated, test_c, test_s):.4f}",
        "routing: accept a message when its score is at least tau, chosen on dev",
        f"  tau     lowest dev score whose accepted dev error is at most alpha",
        f"  tau_cp  the same with the dev error's one-sided {CP_LEVEL:.0%} Clopper-Pearson upper"
        " bound",
    ]
    for alpha in ALPHAS:
        tau = threshold(dev_s, dev_c, alpha, bound=False)
        tau_cp = threshold(dev_s, dev_c, alpha, bound=True)
        lines.append(f"  alpha {alpha:.2f}")
        for label, t in [("tau   ", tau), ("tau_cp", tau_cp)]:
            dev_cov, dev_err = routed(dev_s, dev_c, t)
            cov, err = routed(test_s, test_c, t)
            lines.append(
                f"    {label} {fmt(None if np.isinf(t) else t, '.6f'):<8}  "
                f"dev coverage {dev_cov:.4f} error {fmt(dev_err)}  "
                f"test coverage {cov:.4f} error {fmt(err)}"
            )
        lines.append("    test recall among accepted at tau, per true class (accepted of total)")
        kept = test_s >= tau
        for i, name in enumerate(names):
            mine = test_y == i
            n = int((kept & mine).sum())
            recall = test_c[kept & mine].mean() if n else None
            lines.append(f"      {name:<{width}} {fmt(recall)} ({n} of {int(mine.sum())})")

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    out = os.path.join("results", args.set, "confidence", args.model_name + ".txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
