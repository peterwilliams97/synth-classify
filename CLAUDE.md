# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## What this is

A worked example, not a library: train an intent classifier on LLM-generated messages, score it on
real Banking77, once per class set in `CLASS_SETS` (`classes.py`): `distinct`, 10 intents on
unrelated topics, and `confusable`, 5 pairs of intents on the same topic. Four scripts and a results
table in `README.md`. Python, not Go — the Go rules in
`~/CLAUDE.md` (layout, file headers, `test.sh`) do not apply here; its Python / ML environment
section does.

## Commands

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

S=distinct   # or confusable; every script takes --set, default distinct
.venv/bin/python generate.py --set $S                          # ~7 min; needs Ollama on :11434
.venv/bin/python score.py --set $S --write-train data/$S/banking77_train.jsonl
.venv/bin/python train.py --set $S --train data/$S/synthetic.jsonl --out models/$S/modernbert-synthetic
.venv/bin/python train.py --set $S --train data/$S/synthetic.jsonl --out models/$S/tfidf-synthetic --tfidf
.venv/bin/python score.py --set $S --model models/$S/modernbert-synthetic
```

The same two `train.py` lines with `data/$S/banking77_train.jsonl` give the real-data models.
`score.py` prints its report and writes the same text to `results/$S/<model directory name>.txt`. A
ModernBERT fine-tune is about 3.5 minutes on MPS; the TF-IDF model is seconds.

There is no test suite, linter or `test.sh`. The check on a change is the end-to-end run, compared
against the table in `README.md`.

## Running inside the Claude Code sandbox

Two things fail in the Bash sandbox and work outside it; neither is a bug in the code:

- `torch.backends.mps.is_available()` returns `False`, so `train.py` silently falls back to CPU.
  Check for `device: mps` on stderr before trusting a timing or calling a run an MPS run.
- Hugging Face downloads fail with `OSStatus -26276` (TLS trust lookup), surfacing as "Couldn't
  reach … on the Hub".

`~/.cache/huggingface` is not sandbox-writable either; the runs recorded in `README.md` used
`HF_HOME=$PWD/data/hf`, which `.gitignore` already covers.

## Invariants that span files

- **`generate.py` must not read Banking77.** It imports `classes` and nothing from `datasets` or
  `score`. The LLM sees a class name and its one-sentence definition only; anything else voids the
  synthetic-vs-real comparison. `score.py` is the only file that loads the dataset.
- **Label order is `list(CLASS_SETS[<set>])`**, computed independently as `names` in `main` of
  `train.py` and of `score.py`. `score.py`'s `predict` maps a ModernBERT model's outputs through the
  saved `id2label` names, so reordering a class set does not break saved ModernBERT models; a saved
  `tfidf.joblib` stores bare indices and must be retrained.
- **`--set` must be the same for a model's train and score runs.** Scoring a ModernBERT model
  under the other set fails on the `id2label` lookup; a `tfidf.joblib` holds bare indices, so it
  scores without error and the numbers are meaningless.
- **Changing a `CLASS_SETS` entry invalidates everything downstream of that set**:
  `data/<set>/synthetic.jsonl`, the exported train split, every model under `models/<set>/`, every
  file under `results/<set>/`, and that set's rows in the README table. Class names must match
  `label_text` values in `mteb/banking77` exactly — `banking77` in `score.py` exits if a class has
  no rows.
- **A model directory is one of two shapes.** `train.py --tfidf` writes `tfidf.joblib`; without the
  flag it writes a `transformers` checkpoint. `score.py` picks the loader by whether
  `tfidf.joblib` exists.
- **Nothing is tuned on the test split.** `train.py` has no validation set and fixed defaults
  (5 epochs, lr 5e-5, batch 16, seed 0). Keep it that way, or the README numbers stop meaning what
  they say.

## Data

`mteb/banking77` is the Banking77 copy, because `PolyAI/banking77` is a dataset script and
`datasets` 5.x refuses those. Restricted to `distinct` it is 1188 train and 399 test rows;
restricted to `confusable`, 1284 and 400.

`data/<set>/synthetic.jsonl` are the only tracked files under `data/`; they are the `gemma4:e4b`
output the README table was computed from, and `results/<set>/*.txt` are the `score.py` reports
behind each row. Regenerating a synthetic file changes the committed training set, so re-run and
update the table in the same commit. `requirements.txt` is unpinned — the README numbers came from
torch 2.14.1, transformers 5.19.0, datasets 5.1.0.
