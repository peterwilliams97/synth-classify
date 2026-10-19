#!/bin/bash
# run-d3-lora.sh fine-tunes vllm-sr/d3-flash with LoRA (finetune_d3.py) on the confusable subsets
# banking77_train_nodev_<k>.jsonl, k = 30 100 200 400, and routes each model as d3-flash itself is
# routed in the validation-size table: a test dump, a `--split rest` dump of the rows its training
# file left out, and confidence.py with --dev-size 400 at dev seeds 0, 1 and 2. Before the four, a
# sanity run trains on the 30-row file for 1 epoch, and the script stops unless that lowers the mean
# readout loss on its own 30 rows and changes the probabilities on 10 test messages; the check is
# written to results/confusable/train/d3-flash-lora-sanity-check.txt. `run-d3-lora.sh sanity` runs
# the check alone. Training and scoring need Metal, so this runs outside the Claude Code sandbox.

set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME=$PWD/data/hf
export HF_HUB_OFFLINE=1

S=confusable
D=results/$S/confidence/dumps
PY=.venv-d3/bin/python

$PY finetune_d3.py --train data/$S/banking77_train_nodev_30.jsonl --out models/$S/d3-flash-lora-sanity --epochs 1
$PY - <<'EOF'
import math
import sys

from classes import CLASS_SETS
from d3 import classify_d3
from score import banking77, dev_and_shots, from_file

S = "confusable"
names = list(CLASS_SETS[S])
_, _, shots, _ = dev_and_shots(names)
train = from_file(f"data/{S}/banking77_train_nodev_30.jsonl", names)
test = [column[:10] for column in banking77("test", names)]
tuned = f"models/{S}/d3-flash-lora-sanity"


def run(model, rows):
    texts, labels, ids = rows
    answers = classify_d3(texts, ids, S, "fewshot", shots, model)
    return [v for _, v, _ in answers], labels


def loss(vectors, labels):
    return sum(-math.log(max(v[y], 1e-12)) for v, y in zip(vectors, labels)) / len(labels)


base_train, labels = run("vllm-sr/d3-flash", train)
tuned_train, _ = run(tuned, train)
base_test, _ = run("vllm-sr/d3-flash", test)
tuned_test, _ = run(tuned, test)
before, after = loss(base_train, labels), loss(tuned_train, labels)
change = max(abs(a - b) for u, w in zip(base_test, tuned_test) for a, b in zip(u, w))
text = (
    f"model     {tuned} (data/{S}/banking77_train_nodev_30.jsonl, 1 epoch)\n"
    f"loss      {before:.4f} base, {after:.4f} tuned (mean readout cross-entropy on the 30 training rows, fewshot)\n"
    f"change    {change:.4f} (largest absolute probability difference, base against tuned, first 10 test messages)\n"
)
sys.stdout.write(text)
with open(f"results/{S}/train/d3-flash-lora-sanity-check.txt", "w") as f:
    f.write(text)
if not after < before or change == 0:
    sys.exit("sanity check failed: the readout loss did not fall or the probabilities did not change")
EOF
[ "${1:-}" = sanity ] && exit

for k in 30 100 200 400; do
    M=models/$S/d3-flash-lora-$k
    T=data/$S/banking77_train_nodev_$k.jsonl
    $PY finetune_d3.py --train $T --out $M
    $PY score.py --set $S --d3 $M --prompt fewshot --dump "$D/d3-flash-lora-$k-fewshot-test.jsonl"
    $PY score.py --set $S --split rest --train-file $T --d3 $M --prompt fewshot \
        --dump "$D/d3-flash-lora-$k-fewshot-rest.jsonl"
    for s in 0 1 2; do
        $PY confidence.py --set $S --model-name d3-flash-lora-$k-fewshot-dev400-ds$s \
            --dev "$D/d3-flash-lora-$k-fewshot-rest.jsonl" --test "$D/d3-flash-lora-$k-fewshot-test.jsonl" \
            --dev-size 400 --dev-seed $s
    done
done
