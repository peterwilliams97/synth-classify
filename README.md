# synth-classify

A small working example of training a text classifier on LLM-generated data and scoring it on real
data. A local Ollama model writes 100 customer messages for each of 10 Banking77 intents, seeing
only the intent's name and a one-sentence definition — never any Banking77 text. A ModernBERT-base
fine-tune and a TF-IDF + logistic regression baseline are each trained once on that synthetic set
and once on the real Banking77 train split, and all four are scored on the real Banking77 test
split.

That is done for two class sets, chosen with `--set`: `distinct`, 10 intents on unrelated topics,
and `confusable`, 5 pairs of intents on the same topic.

Beside the trained models, two local Ollama models are scored as classifiers with no training: each
message is sent with a prompt listing the classes, and the answer is restricted to one class name.

## Files

- `classes.py` — `CLASS_SETS`: for each class set, the 10 intent names and their definitions,
  shared by every script.
- `generate.py` — asks an Ollama model (`--model`, default `gemma4:e4b`) for 100 messages per class
  and writes `data/<set>/synthetic.jsonl` with fields `text`, `label`.
- `train.py` — fine-tunes `answerdotai/ModernBERT-base` on a jsonl file (`--train`) and saves it to
  `--out`; runs on Apple MPS. `--tfidf` fits the TF-IDF + logistic regression baseline instead.
- `llm.py` — `classify` sends each message to an Ollama model at temperature 0, with structured
  output restricted to an enum of the class names. `PROMPTS` holds the three prompts: `names` (class
  names only), `definitions` (names and the definitions from `CLASS_SETS`) and `fewshot`
  (definitions plus 3 real Banking77 train messages per class).
- `score.py` — scores a saved model (`--model`) or an Ollama model (`--llm MODEL --prompt NAME`) on
  Banking77 restricted to the class set: accuracy, macro F1, median seconds per message, per-class
  recall and a confusion matrix, printed and written to `results/<set>/<split>/<name>.txt`.
  `--split test` (the default) is the Banking77 test split. `--split dev` is 20 messages per class
  sampled with a fixed seed from the Banking77 train split; the `fewshot` examples are train
  messages outside dev. `--write-train` exports the restricted train split as jsonl.

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

for p in names definitions fewshot; do
  python score.py --set $S --split dev --llm gemma4:e4b --prompt $p
done
python score.py --set $S --llm gemma4:e4b --prompt fewshot      # test, with the prompts chosen on dev
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

### Prompts on dev

dev is 200 messages per class set: 20 per class from the Banking77 train split. Accuracy of each
Ollama model with each prompt; full reports are `results/<class set>/dev/<model>-<prompt>.txt`.

| Class set    | Model                | `names` | `definitions` | `fewshot` |
|--------------|----------------------|---------|---------------|-----------|
| `distinct`   | `gemma4:e4b`         | 0.9800  | 0.9800        | 0.9850    |
| `distinct`   | `qwen3.6:27b-q4_K_M` | 0.9900  | 0.9900        | 1.0000    |
| `confusable` | `gemma4:e4b`         | 0.8500  | 0.8750        | 0.9000    |
| `confusable` | `qwen3.6:27b-q4_K_M` | 0.8650  | 0.8500        | 0.8950    |

The prompts run on test were chosen on dev: for each model and class set, `fewshot` and whichever
of `names` and `definitions` scored higher here. On `distinct` the two tie for both models, in
macro F1 as well as accuracy, and `names`, the shorter prompt, was taken. The third prompt was not
run on test.

### Test

Banking77 test split restricted to the class set: 399 messages for `distinct`, 400 for
`confusable`. Training sets: 1000 synthetic messages from `gemma4:e4b`, or the real Banking77 train
messages for the same classes (1188 for `distinct`, 1284 for `confusable`). One run each, seed 0,
5 epochs for ModernBERT; no hyperparameter was tuned against the test split. The Ollama models are
not trained; a `fewshot` prompt contains 30 real train messages, the other two prompts none. Each
row's full report is `results/<class set>/test/<model>.txt`.

Seconds is the median time to classify one message, sent one per call, on an M5 Pro with 48 GB:
ModernBERT on MPS, TF-IDF on CPU, the Ollama models fully on the GPU.

| Class set    | Model                | Training set or prompt   | Accuracy | Macro F1 | Seconds |
|--------------|----------------------|--------------------------|----------|----------|---------|
| `distinct`   | ModernBERT-base      | synthetic (`gemma4:e4b`) | 0.9925   | 0.9925   | 0.0073  |
| `distinct`   | ModernBERT-base      | real Banking77 train     | 0.9950   | 0.9950   | 0.0073  |
| `distinct`   | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 0.9373 | 0.9375 | 0.00015 |
| `distinct`   | TF-IDF + logistic regression | real Banking77 train     | 0.9875 | 0.9874 | 0.00015 |
| `distinct`   | `gemma4:e4b`         | `names` prompt           | 0.9825   | 0.9822   | 0.49    |
| `distinct`   | `gemma4:e4b`         | `fewshot` prompt         | 0.9875   | 0.9874   | 0.44    |
| `distinct`   | `qwen3.6:27b-q4_K_M` | `names` prompt           | 0.9900   | 0.9899   | 1.6     |
| `distinct`   | `qwen3.6:27b-q4_K_M` | `fewshot` prompt         | 0.9925   | 0.9925   | 2.4     |
| `confusable` | ModernBERT-base      | synthetic (`gemma4:e4b`) | 0.8425   | 0.8413   | 0.0073  |
| `confusable` | ModernBERT-base      | real Banking77 train     | 0.9375   | 0.9376   | 0.0073  |
| `confusable` | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 0.7900 | 0.7849 | 0.00015 |
| `confusable` | TF-IDF + logistic regression | real Banking77 train     | 0.9250 | 0.9251 | 0.00015 |
| `confusable` | `gemma4:e4b`         | `definitions` prompt     | 0.9125   | 0.9126   | 0.63    |
| `confusable` | `gemma4:e4b`         | `fewshot` prompt         | 0.9300   | 0.9302   | 0.47    |
| `confusable` | `qwen3.6:27b-q4_K_M` | `names` prompt           | 0.8775   | 0.8756   | 1.5     |
| `confusable` | `qwen3.6:27b-q4_K_M` | `fewshot` prompt         | 0.9325   | 0.9329   | 2.5     |

On `confusable`, 54 of the synthetic-trained ModernBERT's 63 errors and 71 of the synthetic-trained
TF-IDF model's 84 fall inside a pair. The largest single cell for both is `why_verify_identity`
predicted as `verify_my_identity`: 15 of 40 for ModernBERT, 23 of 40 for TF-IDF.

On `confusable`, `gemma4:e4b` with the `definitions` prompt scores 0.9125, against 0.8425 for
ModernBERT trained on the messages that same model wrote from those definitions. With the `fewshot`
prompt both Ollama models are within 3 messages of 400 of ModernBERT trained on the real train
split, at about 60 and 330 times its seconds per message. On `distinct` every classifier but the
synthetic-trained TF-IDF model is within 5 messages of 399 of every other.
