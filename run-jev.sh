#!/bin/bash
# run-jev.sh scores TypeSafe System One (jev.py) on confusable dev and test through OpenRouter,
# with the `definitions` and `fewshot` prompts, dumping its probabilities, then runs confidence.py
# on each prompt's two dumps. Responses are cached in
# results/jev/confusable.jsonl, so a re-run after a failure resends only the missing messages.

set -euo pipefail
cd "$(dirname "$0")"

: "${TYPESAFE_API_KEY:?set TYPESAFE_API_KEY to an OpenRouter key (sk-or-v1-…)}"

export HF_HOME=$PWD/data/hf
export HF_HUB_OFFLINE=1

D=results/confusable/confidence/dumps
for p in definitions fewshot; do
    .venv/bin/python score.py --set confusable --split dev --jev --prompt $p --dump "$D/jev-$p-dev.jsonl"
    .venv/bin/python score.py --set confusable --jev --prompt $p --dump "$D/jev-$p-test.jsonl"
    .venv/bin/python confidence.py --set confusable --model-name jev-$p \
        --dev "$D/jev-$p-dev.jsonl" --test "$D/jev-$p-test.jsonl"
done
