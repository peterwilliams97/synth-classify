# synth-classify

Trained classifiers and local LLMs prompted with the class list are each scored on one task:
assigning a Banking77 customer message to one of 10 intents, on the Banking77 test split. That is
done for two class sets in `CLASS_SETS`, chosen with `--set`: `distinct`, 10 intents on unrelated
topics, and `confusable`, 5 pairs of intents on the same topic. The test split restricted to the
class set is 399 messages for `distinct` and 400 for `confusable`, 39 or 40 per class.

Every row below is one run with seed 0, and the binomial standard error √(p(1−p)/n) of the
accuracies reported is 0.012 to 0.020 on `confusable` (5 to 8 messages of 400) and 0.003 to 0.012
on `distinct` (1 to 5 messages of 399).

## On `distinct` every LLM classifier is within 5 messages of ModernBERT trained on real data; on `confusable` only the `fewshot` prompts are

| Classifier                                          | Real labelled examples it used | `distinct` | `confusable` |
|-----------------------------------------------------|--------------------------------|------------|--------------|
| TF-IDF + logistic regression, synthetic train       | 0                              | 0.9373     | 0.7900       |
| TF-IDF + logistic regression, real train            | 1188 / 1284                    | 0.9875     | 0.9250       |
| ModernBERT-base, synthetic train                    | 0                              | 0.9925     | 0.8425       |
| ModernBERT-base, real train                         | 1188 / 1284                    | 0.9950     | 0.9375       |
| `gemma4:e4b`, `names` / `definitions` prompt        | 0                              | 0.9825     | 0.9125       |
| `gemma4:e4b`, `fewshot` prompt                      | 30                             | 0.9875     | 0.9300       |
| `qwen3.6:27b-q4_K_M`, `names` prompt                | 0                              | 0.9900     | 0.8775       |
| `qwen3.6:27b-q4_K_M`, `fewshot` prompt              | 30                             | 0.9925     | 0.9325       |
| Gemma 4 E4B (MLX), no adapter                       | 0                              | 0.9975     | 0.8500       |
| Gemma 4 E4B (MLX) + LoRA adapter, synthetic train   | 0                              | 0.9950     | 0.8500       |
| Gemma 4 E4B (MLX) + LoRA adapter, real train        | 1188 / 1284                    | 0.9950     | 0.8575       |

The numbers are test accuracy, and a cell with two values gives `distinct` first, `confusable`
second. A synthetic train set is 1000 messages, 100 per class, that `gemma4:e4b` wrote from the
class's name and definition in `CLASS_SETS`; a real train set is the Banking77 train rows of the
same classes. `gemma4:e4b` and `qwen3.6:27b-q4_K_M` are Ollama models and are not trained; the
prompt without examples is whichever of `names` and `definitions` scored higher on dev, a choice
that used 200 real labelled messages the column does not count. Gemma 4 E4B (MLX) is
`mlx-community/gemma-4-e4b-it-4bit` run with mlx-lm on the `definitions` prompt. ModernBERT trained
on real data makes 2 errors of 399 on `distinct`, where the LLM classifiers make 1 to 7, and 25 of
400 on `confusable`, where the `fewshot` prompts make 27 and 28 and the other LLM classifiers 35 to
60.

## Real labelled examples: every classifier above 0.92 on `confusable` used them

| Classifier                              | Real labelled examples | Accuracy | Errors | Inside a pair | Largest cell (true → predicted), of 40 |
|-----------------------------------------|------|--------|----|----|-------------------------------------------------|
| TF-IDF, synthetic train                 | 0    | 0.7900 | 84 | 71 | `why_verify_identity` → `verify_my_identity`, 23 |
| TF-IDF, real train                      | 1284 | 0.9250 | 30 | 26 | `verify_my_identity` → `why_verify_identity`, 7  |
| ModernBERT-base, synthetic train        | 0    | 0.8425 | 63 | 54 | `why_verify_identity` → `verify_my_identity`, 15 |
| ModernBERT-base, real train             | 1284 | 0.9375 | 25 | 24 | `card_arrival` → `card_delivery_estimate`, 5     |
| `gemma4:e4b`, `definitions`             | 0    | 0.9125 | 35 | 32 | `card_arrival` → `card_delivery_estimate`, 13    |
| `gemma4:e4b`, `fewshot`                 | 30   | 0.9300 | 28 | 25 | `card_arrival` → `card_delivery_estimate`, 7     |
| `qwen3.6:27b-q4_K_M`, `names`           | 0    | 0.8775 | 49 | 47 | `verify_my_identity` → `why_verify_identity`, 17 |
| `qwen3.6:27b-q4_K_M`, `fewshot`         | 30   | 0.9325 | 27 | 21 | `card_arrival` → `card_delivery_estimate`, 6     |
| Gemma 4 E4B (MLX), no adapter           | 0    | 0.8500 | 60 | 26 | `card_arrival` → `card_delivery_estimate`, 16    |
| Gemma 4 E4B (MLX) + LoRA, synthetic train | 0  | 0.8500 | 60 | 34 | `card_arrival` → `card_delivery_estimate`, 18    |
| Gemma 4 E4B (MLX) + LoRA, real train    | 1284 | 0.8575 | 57 | 56 | `declined_transfer` → `failed_transfer`, 24      |

An error is inside a pair when the predicted class is the other intent of the true class's pair,
and the largest cell is the largest off-diagonal count in the confusion matrix of
`results/confusable/test/<name>.txt`, where each class has 40 messages. With real labelled examples
in place of none, errors go from 84 to 30 for TF-IDF, 63 to 25 for ModernBERT, 35 to 28 for
`gemma4:e4b` and 49 to 27 for `qwen3.6:27b-q4_K_M`. The LoRA adapter trained on the real train set
is the one classifier that used real labelled examples and scored below 0.92: 24 of its 57 errors
are `declined_transfer` predicted as `failed_transfer` and 10 are `verify_my_identity` predicted as
`why_verify_identity`. Of the 60 errors with no adapter, the 34 outside a pair are invalid replies;
of the 60 with the adapter trained on the synthetic set, the 26 outside a pair are all predictions
of `getting_virtual_card`. Without real examples the largest cell is `why_verify_identity`
predicted as `verify_my_identity` for TF-IDF and ModernBERT, the reverse for
`qwen3.6:27b-q4_K_M`, and `card_arrival` predicted as `card_delivery_estimate` for every Gemma 4
E4B run.

## Restricting the reply and real examples in the prompt raised accuracy by 7 to 25 messages; the 27B model and the LoRA adapters by 0 to 3

| Change                                                                              | Before          | After           | Messages     |
|-------------------------------------------------------------------------------------|-----------------|-----------------|--------------|
| Reply restricted to the class names: mlx-lm reply (34 invalid) → Ollama enum (none can be), `definitions` | 0.8500 | 0.9125 | +25 of 400 |
| 3 real examples per class in the prompt: dev-chosen prompt → `fewshot`              | 0.9125 / 0.8775 | 0.9300 / 0.9325 | +7 / +22 of 400 |
| Prompt chosen on dev: the lower of `names` and `definitions` → the higher, on dev   | 0.8500 / 0.8500 | 0.8750 / 0.8650 | +5 / +3 of 200 |
| `qwen3.6:27b-q4_K_M` in place of `gemma4:e4b`, `fewshot`                            | 0.9300          | 0.9325          | +1 of 400    |
| LoRA adapter trained on the synthetic set (invalid replies 34 → 0)                  | 0.8500          | 0.8500          | 0 of 400     |
| LoRA adapter trained on the real train set (invalid replies 34 → 0)                 | 0.8500          | 0.8575          | +3 of 400    |

Every number is `confusable` test accuracy except the dev row, which is accuracy on the 200 dev
messages because the prompt not chosen was not run on test; a cell with two values gives
`gemma4:e4b` first, `qwen3.6:27b-q4_K_M` second. The first row compares two runtimes as well as two
reply rules — Ollama's `gemma4:e4b` is `google/gemma-4-E4B-it` at Q4_K_M and
`mlx-community/gemma-4-e4b-it-4bit` is a 4-bit quantisation of it — and on the 366 messages the
mlx-lm model answered with a class name it is right on 340. With the dev-chosen prompts,
`qwen3.6:27b-q4_K_M` scores 0.8775 with `names` against 0.9125 for `gemma4:e4b` with
`definitions`, 14 messages lower. The LoRA configuration is rank 8, scale 20, dropout 0 on every
linear layer of the last 16 of 42 transformer blocks, Adam at learning rate 1e-4, batch size 4, 2
passes over the rows, seed 0. It is the only configuration tried, so the last two rows are what
that configuration gives.

## An LLM classifier takes 19 to 335 times ModernBERT's seconds per message, and a LoRA adapter 5 to 9 times its training minutes

| Classifier                             | Real labelled examples needed | Training minutes | Seconds per message | × ModernBERT |
|----------------------------------------|-------------------------------|------------------|---------------------|--------------|
| TF-IDF + logistic regression           | 0 or 1188 / 1284              | not recorded     | 0.00015 / 0.00015   | 0.02 / 0.02  |
| ModernBERT-base                        | 0 or 1188 / 1284              | 1.6 / 1.7 or 1.8 / 2.0 | 0.0073 / 0.0073 | 1 / 1      |
| `gemma4:e4b`, dev-chosen prompt        | 200, to choose the prompt     | none             | 0.49 / 0.63         | 67 / 85      |
| `gemma4:e4b`, `fewshot`                | 30                            | none             | 0.44 / 0.47         | 60 / 63      |
| `qwen3.6:27b-q4_K_M`, dev-chosen prompt | 200, to choose the prompt    | none             | 1.6 / 1.5           | 222 / 197    |
| `qwen3.6:27b-q4_K_M`, `fewshot`        | 30                            | none             | 2.4 / 2.5           | 332 / 335    |
| Gemma 4 E4B (MLX), no adapter          | 0                             | none             | 0.141 / 0.258       | 19 / 35      |
| Gemma 4 E4B (MLX) + LoRA adapter       | 0 or 1188 / 1284              | 7.8 / 13.6 or 9.1 / 17.9 | 0.163 / 0.235 or 0.167 / 0.235 | 22 / 32 or 23 / 32 |

A pair of values is `distinct` / `confusable`, and where a cell has two pairs the first is for the
synthetic train set and the second for the real one. Seconds is the median time to classify one
test message, sent one per call, on an M5 Pro with 48 GB, and the last column divides the seconds
in the row's report by those in `results/<set>/test/modernbert-real.txt` (0.0073 and 0.00735).
Training minutes are the wall time of one `train.py` or `finetune.py` run, from
`results/<set>/train/<name>.txt`. No file under `results/` records the TF-IDF model's training
time or the minutes `generate.py` took to write a synthetic set, which every synthetic-trained row
needs first.

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

## How it was measured

**Training sets.** `generate.py` asks `gemma4:e4b` for 100 messages per class, showing it the class
name and its one-sentence definition from `CLASS_SETS` and never any Banking77 text; the 1000
messages are `data/<set>/synthetic.jsonl`. The real train set is the Banking77 train split
restricted to the class set: 1188 messages for `distinct`, 1284 for `confusable`.

**dev and test.** Test is the Banking77 test split restricted to the class set. dev is 200 messages
per class set: 20 per class sampled with a fixed seed from the Banking77 train split. The 3
examples per class in a `fewshot` prompt, 30 in all, are train messages outside dev. dev scores
only the Ollama and mlx-lm models; the classifiers trained on the real train set saw those rows.

**How prompts were chosen.** Each Ollama model was run on dev with every prompt in `PROMPTS`. Test
was run with `fewshot` and with whichever of `names` and `definitions` scored higher on dev. On
`distinct` the two tie for both models, in macro F1 as well as accuracy, and `names`, the shorter
prompt, was taken. The third prompt was not run on test. The mlx-lm rows all use `definitions`,
which is not chosen on dev for those rows: it is the prompt the adapters are trained with.

**Fixed before any test run.** ModernBERT: 5 epochs, learning rate 5e-5, batch size 16, seed 0, no
validation set. The LoRA adapters: the configuration stated above, with nothing tuned — there is no
held-out split to choose another on, dev being part of the real train split. The Ollama models run
at temperature 0 with the answer restricted to an enum of the class names. Nothing restricts an
mlx-lm reply: the reply is what follows the model's reasoning, if it wrote any, and a reply that is
not exactly a class name is invalid and counted as wrong. The 512-token limit on a reply and that
rule were settled on dev. With no adapter on `confusable`, 34 of 400 test replies are invalid, all
of them reasoning cut off at 512 tokens; the other five mlx-lm test runs have none.

**Seconds.** The median time to classify one message, sent one per call, on an M5 Pro with 48 GB:
ModernBERT on MPS, TF-IDF on CPU, the Ollama and mlx-lm models fully on the GPU. The median hides
the 34 cut-off replies, which take several seconds each.

## All results

### Test

Each row's full report is `results/<class set>/test/<model>.txt`.

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
| `distinct`   | Gemma 4 E4B (MLX)    | `definitions` prompt     | 0.9975   | 0.9975   | 0.141   |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 0.9950 | 0.9950 | 0.163   |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | real Banking77 train     | 0.9950 | 0.9950 | 0.167   |
| `confusable` | ModernBERT-base      | synthetic (`gemma4:e4b`) | 0.8425   | 0.8413   | 0.0073  |
| `confusable` | ModernBERT-base      | real Banking77 train     | 0.9375   | 0.9376   | 0.0073  |
| `confusable` | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 0.7900 | 0.7849 | 0.00015 |
| `confusable` | TF-IDF + logistic regression | real Banking77 train     | 0.9250 | 0.9251 | 0.00015 |
| `confusable` | `gemma4:e4b`             | `definitions` prompt     | 0.9125   | 0.9126   | 0.63    |
| `confusable` | `gemma4:e4b`             | `fewshot` prompt         | 0.9300   | 0.9302   | 0.47    |
| `confusable` | `qwen3.6:27b-q4_K_M`     | `names` prompt           | 0.8775   | 0.8756   | 1.5     |
| `confusable` | `qwen3.6:27b-q4_K_M`     | `fewshot` prompt         | 0.9325   | 0.9329   | 2.5     |
| `confusable` | Gemma 4 E4B (MLX)        | `definitions` prompt     | 0.8500   | 0.8858   | 0.258   |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 0.8500 | 0.8510 | 0.235   |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | real Banking77 train     | 0.8575 | 0.8513 | 0.235   |

### Prompts on dev

Accuracy of each Ollama model with each prompt on the 200 dev messages; full reports are
`results/<class set>/dev/<model>-<prompt>.txt`.

| Class set    | Model                | `names` | `definitions` | `fewshot` |
|--------------|----------------------|---------|---------------|-----------|
| `distinct`   | `gemma4:e4b`         | 0.9800  | 0.9800        | 0.9850    |
| `distinct`   | `qwen3.6:27b-q4_K_M` | 0.9900  | 0.9900        | 1.0000    |
| `confusable` | `gemma4:e4b`         | 0.8500  | 0.8750        | 0.9000    |
| `confusable` | `qwen3.6:27b-q4_K_M` | 0.8650  | 0.8500        | 0.8950    |

### Training minutes

Wall time of one `train.py` or `finetune.py` run on the same M5 Pro, from loading the base model to
saving, with nothing else on the GPU; files are `results/<class set>/train/<model>.txt`. The
ModernBERT minutes are from re-running the training commands into a scratch directory; the models
scored in the test table were not replaced.

| Class set    | Model                    | Training set             | Rows | Minutes |
|--------------|--------------------------|--------------------------|------|---------|
| `distinct`   | ModernBERT-base          | synthetic (`gemma4:e4b`) | 1000 | 1.6     |
| `distinct`   | ModernBERT-base          | real Banking77 train     | 1188 | 1.8     |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 1000 | 7.8     |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | real Banking77 train     | 1188 | 9.1     |
| `confusable` | ModernBERT-base          | synthetic (`gemma4:e4b`) | 1000 | 1.7     |
| `confusable` | ModernBERT-base          | real Banking77 train     | 1284 | 2.0     |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 1000 | 13.6    |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | real Banking77 train     | 1284 | 17.9    |

ModernBERT runs 5 passes over the rows and the adapters 2. The `confusable` prompt is about twice
the length of the `distinct` one (2060 characters against 1012), and every training example for an
adapter carries the whole prompt.

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
  With `mlx=True` it runs an mlx-lm model instead, with an optional LoRA adapter; nothing restricts
  that reply, so one that is not a class name is returned as invalid.
- `finetune.py` — trains a LoRA adapter for `mlx-community/gemma-4-e4b-it-4bit` on a jsonl file
  (`--train`) and saves it to `--out`. Each row is a chat: the `definitions` prompt and the message
  as the user turn, the class name as the assistant turn.
- `score.py` — scores a saved model (`--model`), an Ollama model (`--llm MODEL --prompt NAME`) or
  an mlx-lm model with the `definitions` prompt (`--mlx MODEL`, and `--adapter DIR` for a LoRA
  adapter) on Banking77 restricted to the class set: accuracy, macro F1, median seconds per
  message, per-class recall and a confusion matrix, printed and written to
  `results/<set>/<split>/<name>.txt`. An `--mlx` report also counts the invalid replies.
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

M=mlx-community/gemma-4-e4b-it-4bit
python finetune.py --set $S --train data/$S/synthetic.jsonl       --out models/$S/gemma-4-e4b-lora-synthetic
python finetune.py --set $S --train data/$S/banking77_train.jsonl --out models/$S/gemma-4-e4b-lora-real
python score.py --set $S --mlx $M
python score.py --set $S --mlx $M --adapter models/$S/gemma-4-e4b-lora-synthetic
python score.py --set $S --mlx $M --adapter models/$S/gemma-4-e4b-lora-real
```

`train.py` and `finetune.py` also write the minutes a run took to
`results/<set>/train/<name>.txt`.
