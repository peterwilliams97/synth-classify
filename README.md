# synth-classify

A small working example of training a text classifier on LLM-generated data and scoring it on real
data. A local Ollama model writes 100 customer messages for each of 10 Banking77 intents, seeing
only the intent's name and a one-sentence definition — never any Banking77 text. A ModernBERT-base
fine-tune and a TF-IDF + logistic regression baseline are each trained once on that synthetic set
and once on the real Banking77 train split, and all four are scored on the real Banking77 test
split.

## Files

- `classes.py` — `CLASSES`: the 10 intent names and their definitions, shared by every script.
- `generate.py` — asks an Ollama model (`--model`, default `gemma4:e4b`) for 100 messages per class
  and writes `data/synthetic.jsonl` with fields `text`, `label`.
- `train.py` — fine-tunes `answerdotai/ModernBERT-base` on a jsonl file (`--train`) and saves it to
  `--out`; runs on Apple MPS. `--tfidf` fits the TF-IDF + logistic regression baseline instead.
- `score.py` — scores a saved model on the Banking77 test split restricted to `CLASSES`: accuracy,
  macro F1, per-class recall. `--write-train` exports the restricted train split as jsonl.

Banking77 is loaded from [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77), a
parquet copy. The original `PolyAI/banking77` is a dataset script, which `datasets` 5.1.0 refuses
with "Dataset scripts are no longer supported".

## Run

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate

python generate.py                                    # needs `ollama serve`; data/synthetic.jsonl
python score.py --write-train data/banking77_train.jsonl

python train.py --train data/synthetic.jsonl       --out models/modernbert-synthetic
python train.py --train data/banking77_train.jsonl --out models/modernbert-real
python train.py --train data/synthetic.jsonl       --out models/tfidf-synthetic --tfidf
python train.py --train data/banking77_train.jsonl --out models/tfidf-real      --tfidf

for m in models/*; do python score.py --model $m; done
```

## Results

Banking77 test split restricted to `CLASSES`, 399 messages. Training sets: 1000 synthetic messages
from `gemma4:e4b`, or the 1188 real Banking77 train messages for the same classes. One run each,
seed 0, 5 epochs for ModernBERT; no hyperparameter was tuned against the test split.

| Model | Training set | Accuracy | Macro F1 |
|---|---|---|---|
| ModernBERT-base | synthetic (`gemma4:e4b`) | 0.9925 | 0.9925 |
| ModernBERT-base | real Banking77 train | 0.9950 | 0.9950 |
| TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 0.9373 | 0.9375 |
| TF-IDF + logistic regression | real Banking77 train | 0.9875 | 0.9874 |
