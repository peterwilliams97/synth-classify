#!/bin/bash
# run-jev.sh scores TypeSafe System One (jev.py) on confusable dev and test through OpenRouter,
# dumping its probabilities, then runs confidence.py on the two dumps. Responses are cached in
# results/jev/confusable.jsonl, so a re-run after a failure resends only the missing messages.

set -euo pipefail
cd "$(dirname "$0")"

: "${TYPESAFE_API_KEY:?set TYPESAFE_API_KEY to an OpenRouter key (sk-or-v1-…)}"

export HF_HOME=$PWD/data/hf
export HF_HUB_OFFLINE=1

D=results/confusable/confidence/dumps
.venv/bin/python score.py --set confusable --split dev --jev --dump "$D/jev-definitions-dev.jsonl"
.venv/bin/python score.py --set confusable --jev --dump "$D/jev-definitions-test.jsonl"
.venv/bin/python confidence.py --set confusable --model-name jev-definitions \
    --dev "$D/jev-definitions-dev.jsonl" --test "$D/jev-definitions-test.jsonl"
