# synth-classify

Trained classifiers and local LLMs prompted with the class list are each scored on one task:
assigning a Banking77 customer message to one of 10 intents, on the Banking77 test split. That is
done for two class sets in `CLASS_SETS`, chosen with `--set`: `distinct`, 10 intents on unrelated
topics, and `confusable`, 5 pairs of intents on the same topic. The test split restricted to the
class set is 399 messages for `distinct` and 400 for `confusable`, 39 or 40 per class. A third set,
`pair`, is the `verify_my_identity` / `why_verify_identity` pair of `confusable` alone, a pilot for
generation methods that no table below includes. Its reports are under `results/pair/`, and the
`contrast` method it piloted was superseded by the `confusable` diagnosed run.

Every row below is one run with seed 0, and the binomial standard error √(p(1−p)/n) of the
accuracies reported is 0.012 to 0.020 on `confusable` (5 to 8 messages of 400) and 0.003 to 0.012
on `distinct` (1 to 5 messages of 399).

## Classifiers

Seven kinds are scored. A **synthetic train set** is 1000 messages, 100 per class, that `gemma4:e4b`
wrote from each class's name and definition in `CLASS_SETS`, never seeing Banking77. A **real train
set** is the Banking77 train rows of the same classes: 1188 for `distinct`, 1284 for `confusable`.
A **diagnosed train set**, `confusable` only, is 1000 messages from the same generator with the
definition of the other class in the pair added to each request and, for the class in each pair
that loses more of the 200 dev messages to the other in the confusion matrix of
UltraBERT-Embed-150M + LR trained on the synthetic set, one note saying what that class's messages
must cover. The
generator is shown no dev or train text; the 200 dev labels enter only as counts in that matrix.

- **TF-IDF + logistic regression** — a bag-of-words baseline, trained on a synthetic, diagnosed
  or real train set.
- **ModernBERT-base** — an encoder fine-tuned on a synthetic, diagnosed or real train set.
- **UltraBERT-Embed-150M + logistic regression** — `dmis-lab/UltraBERT-Embed-150M` embeddings,
  frozen, with logistic regression trained on them from a synthetic, diagnosed or real train set;
  "+ LR" in
  the tables.
- **An Ollama model with a prompt from `PROMPTS`** — `gemma4:e4b` or `qwen3.6:27b-q4_K_M`, not
  trained. The prompt is `names` (the class names), `definitions` (names and definitions) or
  `fewshot` (definitions plus 3 real train messages per class, 30 in all), and the reply is
  restricted to a class name.
- **Gemma 4 E4B (MLX), no adapter** — `mlx-community/gemma-4-e4b-it-4bit` run with mlx-lm on the
  `definitions` prompt, not trained. Nothing restricts the reply; one that is not a class name is
  invalid and counted as wrong.
- **Gemma 4 E4B (MLX) + LoRA adapter** — the same model and prompt, with a LoRA adapter trained on
  a synthetic or a real train set.
- **Laya** — `convaiinnovations/laya`, asked one `choice` question per message with the class
  names as the options, not trained.

## Findings

- [LLM classifiers against the baselines](#llm-classifiers-against-the-baselines): on `confusable`
  logistic regression on frozen UltraBERT-Embed-150M embeddings trained on real data makes 17
  errors of 400, ModernBERT trained on real data 25, the two `fewshot` prompts 27 and 28, and the
  other LLM classifiers 35 to 60.
- [Classes that are close together](#classes-that-are-close-together): 200 real labels used as
  counts took the classifiers trained on generated messages from 61 to 35 errors of 400
  (UltraBERT-Embed-150M + LR), 63 to 53 (ModernBERT) and 84 to 83 (TF-IDF), against 17, 25 and 30
  with 1284 to train on and 27 and 28 with 30 in the prompt; every classifier above 0.92 on
  `confusable` had the 1284 or the 30.
- [What raised an LLM classifier's accuracy](#what-raised-an-llm-classifiers-accuracy): restricting
  the reply to a class name (+25 messages of 400) and 3 real examples per class in the prompt (+7
  and +22); the 27B model and the LoRA adapters, one configuration with nothing tuned, added 0 to
  3.
- [Costs](#costs): on `confusable` an LLM classifier takes 32 to 335 times ModernBERT's seconds per
  message, and a LoRA adapter 13.6 to 17.9 training minutes against ModernBERT's 1.7 to 2.0.

## LLM classifiers against the baselines

**On `confusable` only the two `fewshot` prompts come within 3 errors of ModernBERT trained on real
data; every other LLM classifier makes at least 10 more.**

| Classifier                       | Training set or prompt | Real labelled examples used | Accuracy | Errors of 400 |
|----------------------------------|------------------------|-----------------------------|----------|---------------|
| TF-IDF + logistic regression     | synthetic              | 0                           | 0.7900   | 84            |
| TF-IDF + logistic regression     | diagnosed              | 200, as counts              | 0.7925   | 83            |
| TF-IDF + logistic regression     | real                   | 1284                        | 0.9250   | 30            |
| ModernBERT-base                  | synthetic              | 0                           | 0.8425   | 63            |
| ModernBERT-base                  | diagnosed              | 200, as counts              | 0.8675   | 53            |
| ModernBERT-base                  | real                   | 1284                        | 0.9375   | 25            |
| UltraBERT-Embed-150M + LR        | synthetic              | 0                           | 0.8475   | 61            |
| UltraBERT-Embed-150M + LR        | diagnosed              | 200, as counts              | 0.9125   | 35            |
| UltraBERT-Embed-150M + LR        | real                   | 1284                        | 0.9575   | 17            |
| `gemma4:e4b`                     | `definitions`          | 0                           | 0.9125   | 35            |
| `gemma4:e4b`                     | `fewshot`              | 30                          | 0.9300   | 28            |
| `qwen3.6:27b-q4_K_M`             | `names`                | 0                           | 0.8775   | 49            |
| `qwen3.6:27b-q4_K_M`             | `fewshot`              | 30                          | 0.9325   | 27            |
| Gemma 4 E4B (MLX), no adapter    | `definitions`          | 0                           | 0.8500   | 60            |
| Gemma 4 E4B (MLX) + LoRA adapter | synthetic              | 0                           | 0.8500   | 60            |
| Gemma 4 E4B (MLX) + LoRA adapter | real                   | 1284                        | 0.8575   | 57            |

Every classifier above 0.92 used real labelled examples; the LoRA adapter trained on the real train
set is the one that used them and scored lower. On `distinct` every classifier but TF-IDF trained
on the synthetic set (0.9373) scores 0.9825 to 0.9975: 1 to 7 errors of 399, against 2 for
ModernBERT trained on real data.

## Classes that are close together

**200 real labels used as counts took UltraBERT-Embed-150M + LR trained on generated messages from
61 errors of 400 to 35, short of the 17 it makes with 1284 to train on and of the 27 and 28 of the
two prompts with 30 in them; every classifier above 0.92 on `confusable` had the 1284 or the 30.**

| Classifier                       | Training set or prompt | Real labelled examples used | Errors | Inside a pair | Most frequent error (true → predicted)        | Messages, of 40 |
|----------------------------------|------------------------|-----------------------------|--------|---------------|-----------------------------------------------|-----------------|
| TF-IDF + logistic regression     | synthetic              | 0                           | 84     | 71            | `why_verify_identity` → `verify_my_identity`  | 23              |
| TF-IDF + logistic regression     | diagnosed              | 200, as counts              | 83     | 71            | `why_verify_identity` → `verify_my_identity`  | 24              |
| TF-IDF + logistic regression     | real                   | 1284                        | 30     | 26            | `verify_my_identity` → `why_verify_identity`  | 7               |
| ModernBERT-base                  | synthetic              | 0                           | 63     | 54            | `why_verify_identity` → `verify_my_identity`  | 15              |
| ModernBERT-base                  | diagnosed              | 200, as counts              | 53     | 48            | `why_verify_identity` → `verify_my_identity`  | 14              |
| ModernBERT-base                  | real                   | 1284                        | 25     | 24            | `card_arrival` → `card_delivery_estimate`     | 5               |
| UltraBERT-Embed-150M + LR        | synthetic              | 0                           | 61     | 53            | `why_verify_identity` → `verify_my_identity`  | 13              |
| UltraBERT-Embed-150M + LR        | diagnosed              | 200, as counts              | 35     | 31            | `card_arrival` → `card_delivery_estimate`     | 10              |
| UltraBERT-Embed-150M + LR        | real                   | 1284                        | 17     | 15            | `declined_transfer` → `failed_transfer`       | 3               |
| `gemma4:e4b`                     | `definitions`          | 0                           | 35     | 32            | `card_arrival` → `card_delivery_estimate`     | 13              |
| `gemma4:e4b`                     | `fewshot`              | 30                          | 28     | 25            | `card_arrival` → `card_delivery_estimate`     | 7               |
| `qwen3.6:27b-q4_K_M`             | `names`                | 0                           | 49     | 47            | `verify_my_identity` → `why_verify_identity`  | 17              |
| `qwen3.6:27b-q4_K_M`             | `fewshot`              | 30                          | 27     | 21            | `card_arrival` → `card_delivery_estimate`     | 6               |
| Gemma 4 E4B (MLX), no adapter    | `definitions`          | 0                           | 60     | 26            | `card_arrival` → `card_delivery_estimate`     | 16              |
| Gemma 4 E4B (MLX) + LoRA adapter | synthetic              | 0                           | 60     | 34            | `card_arrival` → `card_delivery_estimate`     | 18              |
| Gemma 4 E4B (MLX) + LoRA adapter | real                   | 1284                        | 57     | 56            | `declined_transfer` → `failed_transfer`       | 24              |

Real examples took the errors of 400 from 84 to 30 for TF-IDF, 63 to 25 for ModernBERT, 61 to 17
for UltraBERT-Embed-150M + LR, 35 to 28 for `gemma4:e4b` and 49 to 27 for `qwen3.6:27b-q4_K_M`.
The LoRA adapter trained on the real train
set is the exception at 57. With no adapter, the 34 errors outside a pair are the invalid replies.
56 of the real-trained adapter's 57 errors are inside a pair, 24 of them `declined_transfer`
predicted as `failed_transfer`. The synthetic-trained adapter's 26 errors outside a pair are all
predictions of `getting_virtual_card`. From the synthetic set to the diagnosed one,
UltraBERT-Embed-150M + LR's errors inside a pair went from 13 to 1 for the identity pair, 13 to 7
for the transfers, 9 to 2 for the virtual cards and 5 to 3 for the exchange rates, and from 13 to
18 for the card pair, where `card_arrival` → `card_delivery_estimate` rose from 4 to 10. The notes
are one draft, written once and not revised after any score.

## What raised an LLM classifier's accuracy

**Restricting the reply to a class name and putting real examples in the prompt raised accuracy by
7 to 25 messages of 400; the 27B model and the LoRA adapters, one configuration with nothing tuned,
by 0 to 3.**

| Change                                                                   | Before | After  | Messages | Scored on |
|--------------------------------------------------------------------------|--------|--------|----------|-----------|
| Reply restricted to a class name: mlx-lm with no adapter → Ollama `gemma4:e4b`, `definitions` | 0.8500 | 0.9125 | +25 | test, 400 |
| 3 real examples per class, `gemma4:e4b`: `definitions` → `fewshot`       | 0.9125 | 0.9300 | +7       | test, 400 |
| 3 real examples per class, `qwen3.6:27b-q4_K_M`: `names` → `fewshot`     | 0.8775 | 0.9325 | +22      | test, 400 |
| Prompt chosen on dev, `gemma4:e4b`: `names` → `definitions`              | 0.8500 | 0.8750 | +5       | dev, 200  |
| Prompt chosen on dev, `qwen3.6:27b-q4_K_M`: `definitions` → `names`      | 0.8500 | 0.8650 | +3       | dev, 200  |
| `qwen3.6:27b-q4_K_M` in place of `gemma4:e4b`, `fewshot`                 | 0.9300 | 0.9325 | +1       | test, 400 |
| LoRA adapter trained on the synthetic set, against no adapter            | 0.8500 | 0.8500 | 0        | test, 400 |
| LoRA adapter trained on the real train set, against no adapter           | 0.8500 | 0.8575 | +3       | test, 400 |

The first row changes the runtime and the quantisation as well as the reply rule, and the mlx-lm
model is right on 340 of the 366 messages it answered with a class name. The two dev rows are
scored on dev because the prompt not chosen was not run on test. With the dev-chosen prompts,
`qwen3.6:27b-q4_K_M` is 14 messages below `gemma4:e4b`, 0.8775 against 0.9125. Both adapters take
the invalid replies from 34 to 0, and theirs is the only LoRA configuration tried.

## Costs

**On `confusable` an LLM classifier takes 32 to 335 times ModernBERT's seconds per message, and a
LoRA adapter 13.6 to 17.9 training minutes against ModernBERT's 1.7 to 2.0.**

| Classifier                       | Training set or prompt | Training minutes | Seconds per message | × ModernBERT |
|----------------------------------|------------------------|------------------|---------------------|--------------|
| TF-IDF + logistic regression     | synthetic              | 0.71 s           | 0.00015             | 0.02         |
| TF-IDF + logistic regression     | diagnosed              | 0.85 s           | 0.00016             | 0.02         |
| TF-IDF + logistic regression     | real                   | 0.65 s           | 0.00015             | 0.02         |
| ModernBERT-base                  | synthetic              | 1.7              | 0.0073              | 1            |
| ModernBERT-base                  | diagnosed              | 1.8              | 0.0077              | 1.0          |
| ModernBERT-base                  | real                   | 2.0              | 0.0073              | 1            |
| UltraBERT-Embed-150M + LR        | synthetic              | 5.86 s           | 0.0107              | 1.5          |
| UltraBERT-Embed-150M + LR        | diagnosed              | 5.50 s           | 0.0109              | 1.5          |
| UltraBERT-Embed-150M + LR        | real                   | 6.52 s           | 0.0109              | 1.5          |
| `gemma4:e4b`                     | `definitions`          | none             | 0.63                | 85           |
| `gemma4:e4b`                     | `fewshot`              | none             | 0.47                | 63           |
| `qwen3.6:27b-q4_K_M`             | `names`                | none             | 1.5                 | 197          |
| `qwen3.6:27b-q4_K_M`             | `fewshot`              | none             | 2.5                 | 335          |
| Gemma 4 E4B (MLX), no adapter    | `definitions`          | none             | 0.258               | 35           |
| Gemma 4 E4B (MLX) + LoRA adapter | synthetic              | 13.6             | 0.235               | 32           |
| Gemma 4 E4B (MLX) + LoRA adapter | real                   | 17.9             | 0.235               | 32           |

× ModernBERT divides the seconds in the row's report by the 0.00735 in
`results/confusable/test/modernbert-real.txt`. On `distinct` the LLM classifiers take 19 to 332
times ModernBERT's seconds and a LoRA adapter 7.8 to 9.1 minutes. No file under `results/` records
the minutes `generate.py` took to write a synthetic set.

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

**The diagnosed notes.** `generate.py --method diagnosed` sends the synthetic set's request plus
the other class's definition, under "Do not write messages with this other intent", and for one
class in each `confusable` pair a note from `DIAGNOSED_NOTES` in `generate.py`. That class is the
one losing more of its 20 dev messages to the other in
`results/confusable/dev/ultrabert-embed-synthetic.txt`, UltraBERT-Embed-150M + LR trained on the
synthetic set; the comment above each note gives the two dev cells it came from. Each note was
written from those counts and that class's rows in `data/confusable/synthetic.jsonl`, not from any
Banking77 message, names the kind of message those rows lack, and asks the model to avoid no words.
The requests are `data/confusable/synthetic-diagnosed.prompt.txt`, the 1000 messages
`data/confusable/synthetic-diagnosed.jsonl`.

**dev and test.** Test is the Banking77 test split restricted to the class set. dev is 200 messages
per class set: 20 per class sampled with a fixed seed from the Banking77 train split. The 3 examples
per class in a `fewshot` prompt, 30 in all, are train messages outside dev. dev scores only the
Ollama and mlx-lm models, Laya, and UltraBERT-Embed-150M + LR trained on the synthetic set, for the
diagnosed notes; the classifiers trained on the real train set saw those rows.

**How prompts were chosen.** Each Ollama model was run on dev with every prompt in `PROMPTS`. Test
was run with `fewshot` and with whichever of `names` and `definitions` scored higher on dev. On
`distinct` the two tie for both models, in macro F1 as well as accuracy, and `names`, the shorter
prompt, was taken. The third prompt was not run on test. That choice used the 200 labelled dev
messages, which the Real labelled examples column does not count. The mlx-lm rows all use
`definitions`, which is not chosen on dev for those rows: it is the prompt the adapters are trained
with.

**Fixed before any test run.** ModernBERT: 5 epochs, learning rate 5e-5, batch size 16, seed 0, no
validation set. UltraBERT-Embed-150M: revision `367d534`, float32 with SDPA attention, each
message prefixed with MTEB's Banking77Classification instruction, the position-0 embedding of the
last layer, and logistic regression at scikit-learn's default regularisation. The LoRA adapters:
rank 8, scale 20, dropout 0 on every linear layer of the last 16
of 42 transformer blocks, Adam at learning rate 1e-4, batch size 4, 2 passes over the rows, seed 0,
with nothing tuned — there is no held-out split to choose another configuration on, dev being part
of the real train split. The Ollama models run
at temperature 0 with the answer restricted to an enum of the class names. Nothing restricts an
mlx-lm reply: the reply is what follows the model's reasoning, if it wrote any, and a reply that is
not exactly a class name is invalid and counted as wrong. The 512-token limit on a reply and that
rule were settled on dev. With no adapter on `confusable`, 34 of 400 test replies are invalid, all
of them reasoning cut off at 512 tokens; the other five mlx-lm test runs have none.

**Ollama against mlx-lm.** The two Gemma 4 E4B classifiers without an adapter differ in runtime as
well as in whether the reply is restricted: Ollama's `gemma4:e4b` is `google/gemma-4-E4B-it` at
Q4_K_M and `mlx-community/gemma-4-e4b-it-4bit` is a 4-bit quantisation of it. On the 366
`confusable` test messages the mlx-lm model answered with a class name, it is right on 340.

**What the embedding model was trained on.** UltraBERT-Embed-150M's model card lists its
classification training data as the synthetic classification set of
`nvidia/embed-nemotron-dataset-v1`, Amazon polarity and toxicity datasets. That set's card, its subset configs, the Llama-Embed-Nemotron
paper (arXiv 2511.07025) and the UltraBERT paper (arXiv 2610.15028) do not mention Banking77. Of its
1,044,212 classification examples, none names Banking77, none offers more than one Banking77 label
name as a candidate, and 18 have a Banking77 text verbatim, generic ones such as "How do I reset my
password?", one of them a `confusable` or `distinct` test message ("I want a refund for my
purchase", `request_refund`). `results/contamination/scan_nemotron.py` is the scan and
`results/contamination/embed-nemotron-dataset-v1.txt` its output. The card's other training data
was not checked.

**Accuracy and macro F1.** Accuracy is the fraction of messages given their true class. Macro F1 is
the unweighted mean over the 10 classes of each class's F1, the harmonic mean of its precision and
recall. The four sections above Class sets report accuracy, because classes have 39 or 40 messages
each, and an invalid reply counts as wrong in accuracy but against no class's precision in macro F1
(0.8858 against 0.8500 for the mlx-lm model with no adapter).

**Seconds and minutes.** Seconds is the median time to classify one test message, sent one per
call, on an M5 Pro with 48 GB: ModernBERT and UltraBERT-Embed-150M on MPS, TF-IDF and the
logistic regression on CPU, the Ollama and mlx-lm models fully on the GPU. The median hides
the 34 cut-off replies, which take several seconds each. Training minutes are the wall time of one
`train.py` or `finetune.py` run, from `results/<set>/train/<name>.txt`.

## All results

### Test

Each row's full report is `results/<class set>/test/<model>.txt`. The "no prompt" rows
(`ultrabert-embed-noprompt-*`) embed each message without MTEB's Banking77Classification
instruction before it, to measure what that instruction adds to the UltraBERT-Embed-150M + LR rows.

| Class set    | Model                | Training set or prompt   | Accuracy | Macro F1 | Seconds |
|--------------|----------------------|--------------------------|----------|----------|---------|
| `distinct`   | ModernBERT-base      | synthetic (`gemma4:e4b`) | 0.9925   | 0.9925   | 0.0073  |
| `distinct`   | ModernBERT-base      | real Banking77 train     | 0.9950   | 0.9950   | 0.0073  |
| `distinct`   | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 0.9373 | 0.9375 | 0.00015 |
| `distinct`   | TF-IDF + logistic regression | real Banking77 train     | 0.9875 | 0.9874 | 0.00015 |
| `distinct`   | UltraBERT-Embed-150M + LR | synthetic (`gemma4:e4b`) | 0.9925 | 0.9925 | 0.0112 |
| `distinct`   | UltraBERT-Embed-150M + LR | real Banking77 train     | 0.9975 | 0.9975 | 0.0106 |
| `distinct`   | UltraBERT-Embed-150M + LR, no prompt | synthetic (`gemma4:e4b`) | 0.9950 | 0.9950 | 0.0100 |
| `distinct`   | UltraBERT-Embed-150M + LR, no prompt | real Banking77 train     | 0.9975 | 0.9975 | 0.0107 |
| `distinct`   | `gemma4:e4b`         | `names` prompt           | 0.9825   | 0.9822   | 0.49    |
| `distinct`   | `gemma4:e4b`         | `fewshot` prompt         | 0.9875   | 0.9874   | 0.44    |
| `distinct`   | `qwen3.6:27b-q4_K_M` | `names` prompt           | 0.9900   | 0.9899   | 1.6     |
| `distinct`   | `qwen3.6:27b-q4_K_M` | `fewshot` prompt         | 0.9925   | 0.9925   | 2.4     |
| `distinct`   | Gemma 4 E4B (MLX)    | `definitions` prompt     | 0.9975   | 0.9975   | 0.141   |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 0.9950 | 0.9950 | 0.163   |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | real Banking77 train     | 0.9950 | 0.9950 | 0.167   |
| `distinct`   | Gemma 4 E4B (MLX), letter logits | `definitions` prompt | 0.9799 | 0.9797 | 0.147 |
| `confusable` | ModernBERT-base      | synthetic (`gemma4:e4b`) | 0.8425   | 0.8413   | 0.0073  |
| `confusable` | ModernBERT-base      | diagnosed (`gemma4:e4b`) | 0.8675   | 0.8674   | 0.0077  |
| `confusable` | ModernBERT-base      | real Banking77 train     | 0.9375   | 0.9376   | 0.0073  |
| `confusable` | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 0.7900 | 0.7849 | 0.00015 |
| `confusable` | TF-IDF + logistic regression | diagnosed (`gemma4:e4b`) | 0.7925 | 0.7887 | 0.00016 |
| `confusable` | TF-IDF + logistic regression | contrast (`gemma4:e4b`)  | 0.6825 | 0.6681 | 0.00015 |
| `confusable` | TF-IDF + logistic regression | real Banking77 train     | 0.9250 | 0.9251 | 0.00015 |
| `confusable` | UltraBERT-Embed-150M + LR | synthetic (`gemma4:e4b`) | 0.8475 | 0.8456 | 0.0107 |
| `confusable` | UltraBERT-Embed-150M + LR | diagnosed (`gemma4:e4b`) | 0.9125 | 0.9134 | 0.0109 |
| `confusable` | UltraBERT-Embed-150M + LR | contrast (`gemma4:e4b`)  | 0.8375 | 0.8353 | 0.0109 |
| `confusable` | UltraBERT-Embed-150M + LR | real Banking77 train     | 0.9575 | 0.9576 | 0.0109 |
| `confusable` | UltraBERT-Embed-150M + LR, no prompt | synthetic (`gemma4:e4b`) | 0.8275 | 0.8245 | 0.0100 |
| `confusable` | UltraBERT-Embed-150M + LR, no prompt | real Banking77 train     | 0.9325 | 0.9326 | 0.0102 |
| `confusable` | `gemma4:e4b`             | `definitions` prompt     | 0.9125   | 0.9126   | 0.63    |
| `confusable` | `gemma4:e4b`             | `fewshot` prompt         | 0.9300   | 0.9302   | 0.47    |
| `confusable` | `qwen3.6:27b-q4_K_M`     | `names` prompt           | 0.8775   | 0.8756   | 1.5     |
| `confusable` | `qwen3.6:27b-q4_K_M`     | `fewshot` prompt         | 0.9325   | 0.9329   | 2.5     |
| `confusable` | Gemma 4 E4B (MLX)        | `definitions` prompt     | 0.8500   | 0.8858   | 0.258   |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 0.8500 | 0.8510 | 0.235   |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | real Banking77 train     | 0.8575 | 0.8513 | 0.235   |
| `confusable` | Gemma 4 E4B (MLX), letter logits | `definitions` prompt | 0.8675 | 0.8648 | 0.338 |
| `distinct`   | Laya                     | `names` prompt           | 0.9123   | 0.9139   | 0.0225  |
| `confusable` | Laya                     | `names` prompt           | 0.7075   | 0.6750   | 0.0241  |

On `confusable` Laya scores below TF-IDF + logistic regression trained on the synthetic set, 0.7075
against 0.7900, so it has no rows in the four sections above All results. Of its answers whose top
probability is above 0.9, 338 of 399 on `distinct` are 0.9734 accurate and 273 of 400 on
`confusable` 0.7912.

The letter-logits rows read the class from the probability of its letter at the last position in one
forward pass, thinking off; they score below the Ollama `gemma4:e4b` row with the prompt chosen on
dev on both sets, 0.8675 against 0.9125 (`definitions`) on `confusable` and 0.9799 against 0.9825
(`names`) on `distinct`, and their top probability is above 0.9 on 378 of 400 `confusable` answers
at 0.8783 accuracy and 397 of 399 `distinct` at 0.9798, so they have no rows in the four sections
above All results and the confidence does not separate right from wrong answers.

The contrast rows are `generate.py --method contrast`, which shows both definitions in a pair and
asks for at least half the messages to avoid the words of the class name; that rule made both
classifiers worse than the synthetic set, 84 to 127 errors of 400 for TF-IDF and 61 to 65 for
UltraBERT-Embed-150M + LR, so they have no rows in the four sections above All results.

### Prompts on dev

Accuracy of each Ollama model and of Laya with each prompt on the 200 dev messages; full reports
are `results/<class set>/dev/<model>-<prompt>.txt`. Laya takes no `fewshot` prompt.

| Class set    | Model                | `names` | `definitions` | `fewshot` |
|--------------|----------------------|---------|---------------|-----------|
| `distinct`   | `gemma4:e4b`         | 0.9800  | 0.9800        | 0.9850    |
| `distinct`   | `qwen3.6:27b-q4_K_M` | 0.9900  | 0.9900        | 1.0000    |
| `confusable` | `gemma4:e4b`         | 0.8500  | 0.8750        | 0.9000    |
| `confusable` | `qwen3.6:27b-q4_K_M` | 0.8650  | 0.8500        | 0.8950    |
| `distinct`   | Laya                 | 0.8650  | 0.7600        | —         |
| `confusable` | Laya                 | 0.6700  | 0.2950        | —         |

### Training minutes

Wall time of one `train.py` or `finetune.py` run on the same M5 Pro, from loading the base model to
saving, with nothing else on the GPU; files are `results/<class set>/train/<model>.txt`. A file
whose `model` line is under `xtmp/` is a re-run of the training command into a scratch directory,
and the model scored in the test table was not replaced; one under `models/` is the run whose model
was scored. The UltraBERT-Embed-150M times are mostly embedding the train set. A time under a
minute is in seconds.

| Class set    | Model                    | Training set             | Rows | Minutes |
|--------------|--------------------------|--------------------------|------|---------|
| `distinct`   | ModernBERT-base          | synthetic (`gemma4:e4b`) | 1000 | 1.6     |
| `distinct`   | ModernBERT-base          | real Banking77 train     | 1188 | 1.8     |
| `distinct`   | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 1000 | 0.65 s  |
| `distinct`   | TF-IDF + logistic regression | real Banking77 train     | 1188 | 0.60 s  |
| `distinct`   | UltraBERT-Embed-150M + LR | synthetic (`gemma4:e4b`) | 1000 | 6.92 s  |
| `distinct`   | UltraBERT-Embed-150M + LR | real Banking77 train     | 1188 | 6.40 s  |
| `distinct`   | UltraBERT-Embed-150M + LR, no prompt | synthetic (`gemma4:e4b`) | 1000 | 5.19 s  |
| `distinct`   | UltraBERT-Embed-150M + LR, no prompt | real Banking77 train     | 1188 | 6.86 s  |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 1000 | 7.8     |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | real Banking77 train     | 1188 | 9.1     |
| `confusable` | ModernBERT-base          | synthetic (`gemma4:e4b`) | 1000 | 1.7     |
| `confusable` | ModernBERT-base          | diagnosed (`gemma4:e4b`) | 1000 | 1.8     |
| `confusable` | ModernBERT-base          | real Banking77 train     | 1284 | 2.0     |
| `confusable` | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 1000 | 0.71 s  |
| `confusable` | TF-IDF + logistic regression | diagnosed (`gemma4:e4b`) | 1000 | 0.85 s  |
| `confusable` | TF-IDF + logistic regression | real Banking77 train     | 1284 | 0.65 s  |
| `confusable` | UltraBERT-Embed-150M + LR | synthetic (`gemma4:e4b`) | 1000 | 5.86 s  |
| `confusable` | UltraBERT-Embed-150M + LR | diagnosed (`gemma4:e4b`) | 1000 | 5.50 s  |
| `confusable` | UltraBERT-Embed-150M + LR | real Banking77 train     | 1284 | 6.52 s  |
| `confusable` | UltraBERT-Embed-150M + LR, no prompt | synthetic (`gemma4:e4b`) | 1000 | 4.60 s  |
| `confusable` | UltraBERT-Embed-150M + LR, no prompt | real Banking77 train     | 1284 | 5.57 s  |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 1000 | 13.6    |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | real Banking77 train     | 1284 | 17.9    |

ModernBERT runs 5 passes over the rows and the adapters 2. The `confusable` prompt is about twice
the length of the `distinct` one (2060 characters against 1012), and every training example for an
adapter carries the whole prompt.

## Files

- `classes.py` — `CLASS_SETS`: for each class set, the intent names and their definitions, shared
  by every script.
- `generate.py` — asks an Ollama model (`--model`, default `gemma4:e4b`) for 100 messages per class
  and writes `data/<set>/synthetic.jsonl` with fields `text`, `label`; `--method contrast` or
  `diagnosed` changes the request and writes `synthetic-<method>.jsonl`.
- `train.py` — fine-tunes `answerdotai/ModernBERT-base` on a jsonl file (`--train`) and saves it to
  `--out`; runs on Apple MPS. `--tfidf` fits the TF-IDF + logistic regression baseline instead, and
  `--embed` logistic regression on frozen `dmis-lab/UltraBERT-Embed-150M` embeddings.
- `llm.py` — `classify` sends each message to an Ollama model at temperature 0, with structured
  output restricted to an enum of the class names. `PROMPTS` holds the three prompts: `names` (class
  names only), `definitions` (names and the definitions from `CLASS_SETS`) and `fewshot`
  (definitions plus 3 real Banking77 train messages per class).
  With `mlx=True` it runs an mlx-lm model instead, with an optional LoRA adapter; nothing restricts
  that reply, so one that is not a class name is returned as invalid. `laya_question` builds the
  one `choice` question Laya is asked from the `names` or `definitions` prompt, and `classify_laya`
  asks it of each message and returns the highest-probability option.
- `finetune.py` — trains a LoRA adapter for `mlx-community/gemma-4-e4b-it-4bit` on a jsonl file
  (`--train`) and saves it to `--out`. Each row is a chat: the `definitions` prompt and the message
  as the user turn, the class name as the assistant turn.
- `score.py` — scores a saved model (`--model`), an Ollama model (`--llm MODEL --prompt NAME`), an
  mlx-lm model with the `definitions` prompt (`--mlx MODEL`, and `--adapter DIR` for a LoRA adapter,
  or `--reply logits` for letter logits, `--mlx` only, no adapter) or Laya
  (`--laya --prompt names|definitions`) on Banking77 restricted to the class set: accuracy, macro F1,
  median seconds per message, per-class recall and a confusion matrix, printed and written to
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
python generate.py --set confusable --method diagnosed   # confusable only
python train.py --set confusable --train data/confusable/synthetic-diagnosed.jsonl --out models/confusable/modernbert-diagnosed
python train.py --set confusable --train data/confusable/synthetic-diagnosed.jsonl --out models/confusable/tfidf-diagnosed --tfidf
python train.py --set confusable --train data/confusable/synthetic-diagnosed.jsonl --out models/confusable/ultrabert-embed-diagnosed --embed
python score.py --set $S --write-train data/$S/banking77_train.jsonl

python train.py --set $S --train data/$S/synthetic.jsonl       --out models/$S/modernbert-synthetic
python train.py --set $S --train data/$S/banking77_train.jsonl --out models/$S/modernbert-real
python train.py --set $S --train data/$S/synthetic.jsonl       --out models/$S/tfidf-synthetic --tfidf
python train.py --set $S --train data/$S/banking77_train.jsonl --out models/$S/tfidf-real      --tfidf
python train.py --set $S --train data/$S/synthetic.jsonl       --out models/$S/ultrabert-embed-synthetic --embed
python train.py --set $S --train data/$S/banking77_train.jsonl --out models/$S/ultrabert-embed-real      --embed
python train.py --set $S --train data/$S/synthetic.jsonl       --out models/$S/ultrabert-embed-noprompt-synthetic --embed --no-prompt
python train.py --set $S --train data/$S/banking77_train.jsonl --out models/$S/ultrabert-embed-noprompt-real      --embed --no-prompt

for m in models/$S/*; do python score.py --set $S --model $m; done

for p in names definitions fewshot; do
  python score.py --set $S --split dev --llm gemma4:e4b --prompt $p
done
python score.py --set $S --llm gemma4:e4b --prompt fewshot      # test, with the prompts chosen on dev

python score.py --set $S --split dev --laya --prompt names
python score.py --set $S --split dev --laya --prompt definitions
python score.py --set $S --laya --prompt names                  # test, with the prompt chosen on dev

M=mlx-community/gemma-4-e4b-it-4bit
python finetune.py --set $S --train data/$S/synthetic.jsonl       --out models/$S/gemma-4-e4b-lora-synthetic
python finetune.py --set $S --train data/$S/banking77_train.jsonl --out models/$S/gemma-4-e4b-lora-real
python score.py --set $S --mlx $M
python score.py --set $S --mlx $M --adapter models/$S/gemma-4-e4b-lora-synthetic
python score.py --set $S --mlx $M --adapter models/$S/gemma-4-e4b-lora-real
```

`train.py` and `finetune.py` also write the minutes a run took, or the seconds if under a minute,
to `results/<set>/train/<name>.txt`.
