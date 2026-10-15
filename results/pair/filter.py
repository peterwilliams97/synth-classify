"""filter.py writes data/pair/synthetic.jsonl: the rows of data/confusable/synthetic.jsonl whose
label is in `CLASS_SETS["pair"]`, unchanged and in order. It is the `pair` baseline with no new
generation — the same `gemma4:e4b` messages the `confusable` synthetic models were trained on. Run
from the repo root."""

import json
import os
import sys

sys.path.insert(0, ".")
from classes import CLASS_SETS

SRC = "data/confusable/synthetic.jsonl"
OUT = "data/pair/synthetic.jsonl"

names = CLASS_SETS["pair"]
with open(SRC) as f:
    rows = [line for line in f if line.strip() and json.loads(line)["label"] in names]
counts = {name: sum(json.loads(r)["label"] == name for r in rows) for name in names}
if not all(counts.values()):
    sys.exit(f"{SRC}: no rows for {[n for n, c in counts.items() if not c]}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.writelines(rows)
print(f"wrote {len(rows)} rows to {OUT}: {counts}", file=sys.stderr)
