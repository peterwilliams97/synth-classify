# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## What this is

A worked example, not a library: train an intent classifier on LLM-generated messages, score it on
real Banking77, once per class set in `CLASS_SETS` (`classes.py`): `distinct`, 10 intents on
unrelated topics, and `confusable`, 5 pairs of intents on the same topic. Local Ollama models
prompted with the class list (`llm.py`) are scored beside the trained ones. Five scripts and two
results tables in `README.md`. All Python — the repository layout, file header and `test.sh` rules
in `~/CLAUDE.md` do not apply here; its Python / ML environment section does.

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
.venv/bin/python score.py --set $S --split dev --llm gemma4:e4b --prompt names   # ~2 min
```

The same two `train.py` lines with `data/$S/banking77_train.jsonl` give the real-data models.
`score.py` prints its report and writes the same text to `results/$S/<split>/<name>.txt`, where
`<name>` is the model directory name, or for `--llm` the Ollama model with `:` replaced by `-`,
then `-<prompt>`. A ModernBERT fine-tune is about 3.5 minutes on MPS; the TF-IDF model is seconds.
An LLM run is one Ollama request per message: about 0.5 s each for `gemma4:e4b` and 1.5–2.5 s for
`qwen3.6:27b-q4_K_M`, so 400 test messages take 3 to 17 minutes.

`score.py` overwrites the tracked report at that path. Its `model` line is the `--model` argument
as typed, so pass `models/$S/<name>` with no `./` and no trailing slash, and its `seconds` line
differs on every run — a re-run that changes only `seconds` has reproduced the report. `<name>`
drops the parent directory: two model directories with one basename write the same report file.

There is no test suite, linter or `test.sh`. The check on a change is the end-to-end run, compared
against the test table in `README.md`, and for `llm.py` or `dev_and_shots` the dev table as well.

## Running inside the Claude Code sandbox

Two things fail in the Bash sandbox and work outside it; neither is a bug in the code:

- `torch.backends.mps.is_available()` returns `False`, so `train.py` and `predict` in `score.py`
  fall back to CPU. Both print `device:` on stderr; check for `device: mps` before trusting a timing
  or calling a run an MPS run.
- Hugging Face downloads fail with `OSStatus -26276` (TLS trust lookup), surfacing as "Couldn't
  reach … on the Hub".

`~/.cache/huggingface` is not sandbox-writable either; the runs recorded in `README.md` used
`HF_HOME=$PWD/data/hf`, which `.gitignore` already covers. With the dataset cached there,
`HF_HUB_OFFLINE=1` lets `score.py --llm` run inside the sandbox; Ollama on `localhost:11434` is
reachable from it.

## Invariants that span files

- **`generate.py` must not read Banking77.** It imports `classes` and nothing from `datasets` or
  `score`. The LLM sees a class name and its one-sentence definition only; anything else voids the
  synthetic-vs-real comparison. `score.py` is the only file that loads the dataset.
- **`llm.py` does not load the dataset either.** The `fewshot` examples are real Banking77 train
  messages, chosen by `dev_and_shots` in `score.py` and passed to `classify` as `examples`.
- **dev, the `fewshot` examples and test are disjoint.** `dev_and_shots` shuffles each class's
  train rows with `random.Random(0)` and takes the first `DEV_PER_CLASS` (20) as dev and the next
  `SHOTS_PER_CLASS` (3) as examples; test is the Banking77 test split. Changing the seed, either
  constant or the class order changes dev and the examples, and invalidates every `--llm` report
  and both README tables' LLM rows.
- **Prompts are chosen on dev, never on test.** For each Ollama model and class set, test is run
  with `fewshot` and with whichever of `names` and `definitions` has the higher dev accuracy,
  `names` on a tie. Do not run the third prompt on test.
- **dev is drawn from the train split, so it scores only `--llm` classifiers.** The real-data
  models were trained on those rows; `--split dev --model` runs, and the number means nothing.
- **Seconds per message are one message per call**, in `predict` (`score.py`) as in `classify`
  (`llm.py`), so the two are comparable. Time them with nothing else on the GPU: Ollama and MPS
  share it.
- **Label order is `list(CLASS_SETS[<set>])`**, computed independently as `names` in `main` of
  `train.py` and of `score.py`. `score.py`'s `predict` maps a ModernBERT model's outputs through the
  saved `id2label` names, so reordering a class set does not break saved ModernBERT models; a saved
  `tfidf.joblib` stores bare indices and must be retrained.
- **`--set` must be the same for a model's train and score runs.** Scoring a ModernBERT model
  under the other set fails on the `id2label` lookup; a `tfidf.joblib` holds bare indices, so it
  scores without error and the numbers are meaningless.
- **Changing a `CLASS_SETS` entry invalidates everything downstream of that set**:
  `data/<set>/synthetic.jsonl`, the exported train split, every model under `models/<set>/`, every
  file under `results/<set>/`, and that set's rows in both README tables. A definition is also
  part of the `definitions` and `fewshot` prompts. Class names must match
  `label_text` values in `mteb/banking77` exactly — `banking77` in `score.py` exits if a class has
  no rows.
- **`max_length=128` is written twice**, in `train_modernbert` (`train.py`) and in `predict`
  (`score.py`). Change both or the model is scored on inputs truncated differently from the ones
  it was trained on.
- **Ollama is called through `/api/chat` with `urllib`, in two places**: `ask` in `generate.py`
  and `classify` in `llm.py`. `requirements.txt` has no Ollama client. Both send `"think": False`
  and a JSON schema as `format`; `classify`'s schema is an `enum` of the class names, which is
  what restricts an answer to one class.
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
output the README test table's synthetic rows were computed from, and
`results/<set>/<split>/*.txt` are the `score.py` reports behind each row of the dev and test
tables. Regenerating a synthetic file changes the committed training set, so retrain, re-score and
update the test table's synthetic rows in the same commit. `requirements.txt` is
unpinned — the README numbers came from torch 2.14.1, transformers 5.19.0, datasets 5.1.0, and
Ollama 0.32.5.
