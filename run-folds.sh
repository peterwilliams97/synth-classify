#!/bin/bash
# run-folds.sh routes classifiers by 4-fold cross-validation over the confusable train split: the
# folds from `score.py --write-folds 4 --fold-seed 0`, each fold's val file as the rows the
# threshold is set on, and the 400 test messages untouched. UltraBERT-Embed-150M + LR is trained
# on each fold's train file; d3-flash and Gemma 4 31B letter logits, both with `fewshot`, are
# scored on each val file and routed against their existing test dumps. They share the GPU, so
# they run one after another, and need Metal, so this runs outside the Claude Code sandbox.
# TypeSafe System One is `run-folds.sh jev`, an API call, which can run beside the others.

set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME=$PWD/data/hf
export HF_HUB_OFFLINE=1

S=confusable
D=results/$S/confidence/dumps
route() { # route <model name> <test dump>
    .venv/bin/python confidence.py --set $S --model-name "$1-fold$i" \
        --dev "$D/$1-fold$i-val.jsonl" --test "$2"
}

if [ "${1:-}" = jev ]; then
    : "${TYPESAFE_API_KEY:?set TYPESAFE_API_KEY to an OpenRouter key (sk-or-v1-…)}"
    for i in 0 1 2 3; do
        .venv/bin/python score.py --set $S --split file --val-file data/$S/banking77_fold${i}_val.jsonl \
            --jev --prompt fewshot --dump "$D/jev-fewshot-fold$i-val.jsonl"
        route jev-fewshot "$D/jev-fewshot-test.jsonl"
    done
    exit
fi

for i in 0 1 2 3; do
    M=models/$S/ultrabert-embed-fold$i
    .venv/bin/python train.py --set $S --train data/$S/banking77_fold${i}_train.jsonl --out $M --embed
    .venv/bin/python score.py --set $S --split file --val-file data/$S/banking77_fold${i}_val.jsonl \
        --model $M --dump "$D/ultrabert-embed-fold$i-val.jsonl"
    .venv/bin/python score.py --set $S --model $M --dump "$D/ultrabert-embed-fold$i-test.jsonl"
    .venv/bin/python confidence.py --set $S --model-name ultrabert-embed-fold$i \
        --dev "$D/ultrabert-embed-fold$i-val.jsonl" --test "$D/ultrabert-embed-fold$i-test.jsonl"
done
for i in 0 1 2 3; do
    .venv-d3/bin/python score.py --set $S --split file --val-file data/$S/banking77_fold${i}_val.jsonl \
        --d3 vllm-sr/d3-flash --prompt fewshot --dump "$D/d3-flash-fewshot-fold$i-val.jsonl"
    route d3-flash-fewshot "$D/d3-flash-fewshot-test.jsonl"
done
for i in 0 1 2 3; do
    .venv/bin/python score.py --set $S --split file --val-file data/$S/banking77_fold${i}_val.jsonl \
        --mlx mlx-community/gemma-4-31b-it-4bit --reply logits --prompt fewshot \
        --dump "$D/gemma-4-31b-it-4bit-fewshot-logits-fold$i-val.jsonl"
    route gemma-4-31b-it-4bit-fewshot-logits "$D/gemma-4-31b-it-4bit-fewshot-logits-test.jsonl"
done
