# synth-classify

A small working example of training a text classifier on LLM-generated data and scoring it on real
data. A local Ollama model writes 100 customer messages for each of 10 Banking77 intents, seeing
only the intent's name and a one-sentence definition — never any Banking77 text. A ModernBERT-base
fine-tune and a TF-IDF + logistic regression baseline are each trained once on that synthetic set
and once on the real Banking77 train split, and all four are scored on the real Banking77 test
split.

That is done for two class sets, chosen with `--set`: `distinct`, 10 intents on unrelated topics,
and `confusable`, 5 pairs of intents on the same topic.

## Files

- `classes.py` — `CLASS_SETS`: for each class set, the 10 intent names and their definitions,
  shared by every script.
- `generate.py` — asks an Ollama model (`--model`, default `gemma4:e4b`) for 100 messages per class
  and writes `data/<set>/synthetic.jsonl` with fields `text`, `label`.
- `train.py` — fine-tunes `answerdotai/ModernBERT-base` on a jsonl file (`--train`) and saves it to
  `--out`; runs on Apple MPS. `--tfidf` fits the TF-IDF + logistic regression baseline instead.
- `score.py` — scores a saved model on the Banking77 test split restricted to the class set:
  accuracy, macro F1, per-class recall and a confusion matrix, printed and written to
  `results/<set>/<model>.txt`. `--write-train` exports the restricted train split as jsonl.

Banking77 is loaded from [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77), a
parquet copy. The original `PolyAI/banking77` is a dataset script, which `datasets` 5.1.0 refuses
with "Dataset scripts are no longer supported".

## Run

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate

S=distinct   # or confusable

python generate.py --set $S                           # needs `ollama serve`
python score.py --set $S --write-train data/$S/banking77_train.jsonl

python train.py --set $S --train data/$S/synthetic.jsonl       --out models/$S/modernbert-synthetic
python train.py --set $S --train data/$S/banking77_train.jsonl --out models/$S/modernbert-real
python train.py --set $S --train data/$S/synthetic.jsonl       --out models/$S/tfidf-synthetic --tfidf
python train.py --set $S --train data/$S/banking77_train.jsonl --out models/$S/tfidf-real      --tfidf

for m in models/$S/*; do python score.py --set $S --model $m; done
```

## Class sets

`distinct` is `card_arrival`, `lost_or_stolen_card`, `exchange_rate`, `change_pin`,
`terminate_account`, `age_limit`, `atm_support`, `apple_pay_or_google_pay`, `request_refund` and
`edit_personal_details`.

`confusable` is 5 pairs. Each intent's definition in `classes.py` states what separates it from the
other intent in its pair:

| Pair                                                   | What separates them |
|--------------------------------------------------------|---------------------|
| `card_arrival` / `card_delivery_estimate`              | a card that is late or missing / a question about delivery time |
| `declined_transfer` / `failed_transfer`                | refused when attempted / sent, then failed |
| `card_payment_wrong_exchange_rate` / `wrong_exchange_rate_for_cash_withdrawal` | a card purchase / an ATM withdrawal |
| `getting_virtual_card` / `get_disposable_virtual_card` | the reusable virtual card / the single-use one |
| `verify_my_identity` / `why_verify_identity`           | how to verify / why it is required |

## Results

Banking77 test split restricted to the class set: 399 messages for `distinct`, 400 for
`confusable`. Training sets: 1000 synthetic messages from `gemma4:e4b`, or the real Banking77 train
messages for the same classes (1188 for `distinct`, 1284 for `confusable`). One run each, seed 0,
5 epochs for ModernBERT; no hyperparameter was tuned against the test split. Each row's full report
is `results/<class set>/<model>.txt`.

| Class set    | Model           | Training set                        | Accuracy | Macro F1 |
|--------------|-----------------|---------------------------------------|--------|--------|
| `distinct`   | ModernBERT-base | synthetic (`gemma4:e4b`)              | 0.9925 | 0.9925 |
| `distinct`   | ModernBERT-base | real Banking77 train                  | 0.9950 | 0.9950 |
| `distinct`   | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 0.9373 | 0.9375 |
| `distinct`   | TF-IDF + logistic regression | real Banking77 train     | 0.9875 | 0.9874 |
| `confusable` | ModernBERT-base | synthetic (`gemma4:e4b`)              | 0.8425 | 0.8413 |
| `confusable` | ModernBERT-base | real Banking77 train                  | 0.9375 | 0.9376 |
| `confusable` | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 0.7900 | 0.7849 |
| `confusable` | TF-IDF + logistic regression | real Banking77 train     | 0.9250 | 0.9251 |

On `confusable`, 54 of the synthetic-trained ModernBERT's 63 errors and 71 of the synthetic-trained
TF-IDF model's 84 fall inside a pair. The largest single cell for both is `why_verify_identity`
predicted as `verify_my_identity`: 15 of 40 for ModernBERT, 23 of 40 for TF-IDF.
