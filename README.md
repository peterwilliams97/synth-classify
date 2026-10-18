# synth-classify

I wanted to know three things about a text classifier that runs on a laptop:

1. How much of the work can it do without a person checking?
2. How accurate is it?
3. How many hand-labelled examples does it need?

This repo answers them on [Banking77](https://huggingface.co/datasets/mteb/banking77), a public set
of customer messages to a bank, sorted into 10 categories. Five of the categories come in pairs that
are hard to tell apart ("how do I verify my identity" and "why must I verify my identity"), and
those pairs are where the methods differ.

The short answers, on the 400 Banking77 test messages of the `confusable` class set:

1. UltraBERT-Embed-150M + logistic regression, trained on the diagnosed set, handles 40% of
   messages with no human check at under 5% error. Nothing else passes 24%.
2. The same model trained on 1284 real messages: 96%. `gemma4:e4b` with the category definitions
   and no labels: 91%.
3. 0 labels gets 91% accuracy and no safe automation; 200 gets the 40% above; 1284 gets 96%.

## Aim

Banking77 has 77 categories in all. This repo uses two sets of 10 of them, called class sets. The
code calls a category a class.

- `distinct` is 10 categories on unrelated topics.
- `confusable` is 5 pairs of categories. The two categories in a pair share a topic.

[Class sets](#class-sets) lists both.

Each of the three questions becomes one measurement.

1. **Coverage at 5% error**, on `confusable` only. Coverage is the share of test messages that a
   classifier labels on its own. Every other message goes to a person. Error is the share of the
   labelled-on-its-own messages whose label is wrong.
2. **Accuracy**: the share of test messages given their true category.
3. **Hand labels**: the number of Banking77 messages, with their true category, that a method
   uses. A message written by an LLM is not a hand label.

Accuracy alone does not answer question 1. A classifier at 91% accuracy labels 9% of messages
wrong. Accuracy does not say which 9%. So without coverage a person checks every message.

## Method

**Test and dev.** The test set is the Banking77 test split restricted to one class set. It has 399
messages for `distinct` and 400 for `confusable`. Each classifier is scored on it once. The dev set
is 200 messages per class set, 20 per category, drawn from the Banking77 train split. Every choice
made before a test run is made on dev. Those choices are the prompt, the notes that steer the
generator, and the routing threshold.

**Training sets.** Three kinds are used.

- **synthetic**: 1000 messages, 100 per category. `gemma4:e4b`, a Gemma 4 LLM run locally with
  Ollama, wrote them. It was shown each category's name and one-sentence definition from
  `CLASS_SETS` in `classes.py`. It never saw a Banking77 message.
- **diagnosed** (`confusable` only): 1000 messages from the same generator. Each request adds the
  definition of the other category in the pair. For one category in each pair, the request also
  adds a note saying what that category's messages must cover. That category is the one that loses
  more dev messages to its partner. Those losses are counted from UltraBERT-Embed-150M + logistic
  regression trained on the synthetic set. The generator sees no dev message. The 200 dev labels
  reach it only as counts.
- **real**: the Banking77 train rows of the class set. There are 1188 for `distinct` and 1284 for
  `confusable`.

**Classifiers.** Eight kinds are scored.

- **TF-IDF + logistic regression**: logistic regression on word-frequency features. It is the
  baseline. It is trained on a synthetic, diagnosed or real set.
- **ModernBERT-base**: a text encoder, `answerdotai/ModernBERT-base`, fine-tuned in full on a
  synthetic, diagnosed or real set.
- **UltraBERT-Embed-150M + logistic regression**: logistic regression on embeddings from
  `dmis-lab/UltraBERT-Embed-150M`. An embedding is a fixed-length vector computed from a message.
  The embedding model is not trained here. Only the logistic regression is trained, on a synthetic,
  diagnosed or real set. The tables write it "UltraBERT-Embed-150M + LR".
- **An Ollama model with a prompt from `PROMPTS`** (`llm.py`): `gemma4:e4b` or
  `qwen3.6:27b-q4_K_M`, not trained. The `names` prompt lists the category names. The
  `definitions` prompt adds each category's definition. The `fewshot` prompt adds 3 real train
  messages per category to `definitions`, 30 in all. Ollama restricts the reply to a category name.
- **Gemma 4 E4B (MLX), no adapter**: `mlx-community/gemma-4-e4b-it-4bit` run with mlx-lm on the
  `definitions` prompt, not trained. mlx-lm is Apple's LLM runner for its own GPUs. Nothing
  restricts its reply. A reply that is not a category name is invalid and counts as wrong.
- **Gemma 4 E4B (MLX) + LoRA adapter**: the same model and prompt with a LoRA adapter. A LoRA
  adapter is a small set of extra weights trained while the model's own weights stay fixed. Each
  adapter is trained on a synthetic or a real set.
- **Laya**: `convaiinnovations/laya`, a model that picks one option from a list. It is asked one
  question per message with the category names as the options. It is not trained.
- **TypeSafe System One**: `typesafe/jev-1.13`, an API reached through OpenRouter that picks one
  option from a list and gives a probability for each. It is asked Laya's question with the
  `definitions` prompt, one per message. It is not trained, and it does not run on the laptop.

**Routing.** Routing keeps a classifier's label when the classifier's top probability is at least
a threshold. A message below the threshold goes to a person. The threshold is the lowest dev
probability at which the error among dev messages above it is at most the target. Coverage and
error are then measured on test. Only a classifier that gives a probability for each category can
be routed.

**Uncertainty.** Every row is one run with seed 0. The binomial standard error of an accuracy p
on n messages is √(p(1−p)/n). On the 400 `confusable` test messages it is 0.012 to 0.020, which is
5 to 8 messages. On the 399 `distinct` test messages it is 0.003 to 0.012, which is 1 to 5 messages.

Every number in this file is in a file under `results/`.

## Results

Every table in this section is on `confusable`.

### 1. How much of the work can it do without a person checking?

**UltraBERT-Embed-150M + logistic regression trained on the diagnosed set labels 161 of the 400
test messages on its own, and 6 of those 161 are wrong.**

In the table, α is the error target. AUROC is the chance that a randomly chosen right answer gets a
higher top probability than a randomly chosen wrong one. ECE is the mean gap between the top
probability and the observed accuracy, over 10 bins of 40 messages. Isotonic calibration is a
monotone mapping of the top probability onto accuracy, fitted on dev.

| Classifier                       | Training set or prompt | AUROC  | ECE after isotonic calibration | Test coverage at α 0.05 | Test error at α 0.05 |
|----------------------------------|------------------------|--------|--------------------------------|-------------------------|----------------------|
| TF-IDF + logistic regression     | diagnosed              | 0.7855 | 0.0319                         | 0.2375                  | 0.0526               |
| ModernBERT-base                  | diagnosed              | 0.8394 | 0.0564                         | 0.1575                  | 0.0159               |
| UltraBERT-Embed-150M + LR        | synthetic              | 0.8405 | 0.0495                         | 0.1825                  | 0.0274               |
| UltraBERT-Embed-150M + LR        | diagnosed              | 0.7794 | 0.0333                         | 0.4025                  | 0.0373               |
| Gemma 4 E4B (MLX), letter logits | `definitions`          | 0.7770 | 0.0608                         | 0.0375                  | 0.0000               |
| Laya                             | `names`                | 0.6753 | 0.0705                         | 0.0000                  | —                    |
| TypeSafe System One              | `definitions`          | 0.7867 | 0.0534                         | 0.0000                  | —                    |

TF-IDF + logistic regression's 0.2375 misses the target: 5 of the 95 messages it keeps are wrong.

A stricter threshold replaces the dev error with its one-sided 90% Clopper-Pearson upper bound. The
bound is the highest error rate that the dev count does not rule out at 90% confidence. Under that
bound at a 5% target, no classifier keeps any test message. Under that bound at a 10% target,
UltraBERT-Embed-150M + logistic regression trained on the diagnosed set keeps 287 of the 400 test
messages, 0.7175. 11 of those 287 are wrong, 0.0383.

Three kinds of classifier are not in the table. The Ollama models return a category name and no
probability, so they cannot be routed. A classifier trained on the real train set cannot be routed
here, because dev is part of the data it was trained on. The LoRA adapters return a reply and no
probability.

### 2. How accurate is it?

**UltraBERT-Embed-150M + logistic regression trained on the real train set makes the fewest
errors: 17 of 400.**

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
| TypeSafe System One              | `definitions`          | 0                           | 0.9050   | 38            |

Every classifier above 0.92 used real labelled examples. The Gemma 4 E4B (MLX) + LoRA adapter
trained on the real train set also used them, and stayed at 0.8575. The two `fewshot` prompts come
within 3 errors of ModernBERT-base trained on the real train set. Every other LLM classifier makes
at least 10 more errors than ModernBERT-base trained on the real train set.

On `distinct`, the choice of classifier matters little. Every classifier in the test table under
[All results](#all-results) scores 0.9799 to 0.9975 there. The two exceptions are TF-IDF + logistic
regression trained on the synthetic set, at 0.9373, and Laya, at 0.9123.

**Most errors fall inside a pair.** An error inside a pair is one whose true and predicted
categories are the two categories of one pair.

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
| TypeSafe System One              | `definitions`          | 0                           | 38     | 35            | `card_delivery_estimate` → `card_arrival` and `why_verify_identity` → `verify_my_identity` | 8 each |

Gemma 4 E4B (MLX) with no adapter is the one classifier with most of its errors outside a pair.
Those 34 errors outside a pair are its invalid replies. The 26 errors outside a pair of the
Gemma 4 E4B (MLX) + LoRA adapter trained on the synthetic set are all predictions of
`getting_virtual_card`.

### 3. How many hand-labelled examples does it need?

**With 200 hand labels, UltraBERT-Embed-150M + logistic regression matches the accuracy that
`gemma4:e4b` reaches with none.**

| Hand labels                                       | Accuracy | Method |
|---------------------------------------------------|----------|--------|
| 0                                                 | 0.9125   | `gemma4:e4b` with the class definitions in the prompt, reply restricted to a class name |
| 30, as examples in the prompt                     | 0.9300 / 0.9325   | `gemma4:e4b` or `qwen3.6:27b-q4_K_M` with the `fewshot` prompt |
| 200, as confusion counts that steer the generator | 0.9125, at about 1/60 of the LLM's time per message | UltraBERT-Embed-150M + LR trained on the diagnosed set |
| 1284, to train on                                 | 0.9575   | UltraBERT-Embed-150M + LR |

The 200 hand labels are the dev set. They do two jobs. They count the losses that steer the
diagnosed set. They also set the routing threshold of question 1. So 200 hand labels buy both the
0.9125 and the 40% coverage.

With 0 hand labels no classifier can be routed, because the threshold is set on labelled dev
messages.

The diagnosed notes did not help every classifier equally. From the synthetic set to the diagnosed
set, UltraBERT-Embed-150M + logistic regression went from 61 errors to 35. ModernBERT-base went
from 63 to 53. TF-IDF + logistic regression went from 84 to 83.

The notes changed UltraBERT-Embed-150M + logistic regression's errors inside each pair as follows.

- `verify_my_identity` / `why_verify_identity`: 13 to 1.
- `declined_transfer` / `failed_transfer`: 13 to 7.
- `getting_virtual_card` / `get_disposable_virtual_card`: 9 to 2.
- The exchange-rate pair: 5 to 3.
- `card_arrival` / `card_delivery_estimate`: 13 to 18. Within that, `card_arrival` predicted as
  `card_delivery_estimate` rose from 4 to 10.

The notes are one draft each. They were written once and not revised after any score.

### What raised an LLM classifier's accuracy

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

The first row changes three things at once: the reply rule, the runtime and the quantisation. The
mlx-lm model answered 366 of the 400 test messages with a category name. It is right on 340 of
those 366. The two dev rows are scored on dev because the prompt not chosen was never run on test.
Without examples in the prompt, `qwen3.6:27b-q4_K_M` is 14 messages below `gemma4:e4b`. Both LoRA
adapters take the invalid replies from 34 to 0. Theirs is the only LoRA configuration tried.

### Costs

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
| TypeSafe System One              | `definitions`          | none             | 0.53                | 72           |

Seconds per message is the median time to classify one test message. "× ModernBERT" divides a
row's seconds per message by 0.00735 s, ModernBERT-base trained on the real train set
(`results/confusable/test/modernbert-real.txt`). On `distinct`, an LLM classifier takes 19 to 332
times ModernBERT-base's seconds per message. On `distinct`, a LoRA adapter takes 7.8 to 9.1
training minutes. No file under `results/` records how long `generate.py` took to write a training
set.

TypeSafe System One's seconds include the round trip to OpenRouter. It is the one classifier
billed per message: a mean of 787 input tokens per message at $0.042 per million input tokens is
$33 per million messages, or $0.013 for the 400 (`results/confusable/test/jev-definitions.txt`).

### What did not help

- **Training on LLM-written text with nothing to steer the generator.** On `confusable`, the
  synthetic set cost TF-IDF + logistic regression 13.5 points of accuracy against the real train
  set. It cost ModernBERT-base 9.5 points. It cost UltraBERT-Embed-150M + logistic regression 11.0
  points.
- **A prompt rule telling the generator to avoid the words of the category name** (the contrast
  set). TF-IDF + logistic regression went from 84 errors of 400 to 127. UltraBERT-Embed-150M +
  logistic regression went from 61 to 65.
- **One LoRA configuration.** It removed the invalid replies. It left accuracy within 3 messages of
  400.
- **Laya, zero-shot.** It scored 0.7075 on `confusable` test. That is below TF-IDF + logistic
  regression trained on the synthetic set, at 0.7900.
- **Reading the category from letter logits.** Letter logits are the scores the LLM gives to each
  option letter at the last position, in one forward pass with no reasoning. Gemma 4 E4B (MLX) read
  this way scored 0.8675 on `confusable` test. Its top probability was above 0.9 on 378 of the 400
  test messages.

## Discussion

Accuracy and coverage rank the classifiers differently. UltraBERT-Embed-150M + logistic regression
trained on the diagnosed set and `gemma4:e4b` with the `definitions` prompt both make 35 errors of
400. Of the two, only UltraBERT-Embed-150M + logistic regression gives a probability that a
threshold can use. So of the two, only UltraBERT-Embed-150M + logistic regression can take work
away from a person.

UltraBERT-Embed-150M + logistic regression trained on the diagnosed set does not rank its answers
best. Its AUROC of 0.7794 is below the 0.8405 of the same classifier trained on the synthetic set.
Its higher coverage goes with a lower error rate. On dev it is right on 0.8850 of messages, against
0.8150 for the synthetic-trained one. A classifier with fewer errors overall can keep more messages
before its error passes 5%.

The thresholds sit very close to 1. For UltraBERT-Embed-150M + logistic regression the 5%
thresholds are 0.999821 (diagnosed) and 0.999993 (synthetic). For ModernBERT-base trained on the
diagnosed set it is 0.999994. For the letter logits it is 1.000000. TF-IDF + logistic regression's
threshold, 0.904269, is not close to 1. A threshold of 0.9 does not separate the
letter-logits answers: the 378 above it are 0.8783 accurate.

Routing needs a labelled dev set as well as a classifier. With 0 hand labels nothing sets the
threshold, so a person checks every message. The 96% classifier, UltraBERT-Embed-150M + logistic
regression trained on the real train set, has no coverage figure here. Measuring one needs labelled
messages outside its training data.

200 dev messages are too few to certify 5% error. At a 5% target the Clopper-Pearson bound admits
no message for any classifier. The plain 5% thresholds rest on dev error alone. On test they held
for every classifier except TF-IDF + logistic regression. TypeSafe System One has no 5% threshold:
111 of the 200 dev messages score exactly 1.00 and 7 of those are wrong, so no 5% cut-off exists;
its probabilities arrive in steps of 0.01.

The pairs are the whole difficulty. On `distinct` every classifier but two scores at least 0.9799.
On `confusable` most errors of every classifier but one are inside a pair.

The diagnosed notes cut errors on four pairs out of five. On the card pair the errors rose. The note
for that pair was written once from counts and not revised.

UltraBERT-Embed-150M's own training data could hold Banking77 text. That would make its results
look better than they would on new categories. A scan of its classification data found 18 Banking77
texts verbatim. One is a test message here. The scan is described under
[Details of the method](#details-of-the-method).

A gap of 1 to 3 messages of 400 is smaller than one standard error. The `fewshot` prompts' 27 and
28 errors against ModernBERT-base's 25 are such a gap.

## Conclusion

- With 0 hand labels, `gemma4:e4b` with the `definitions` prompt labels 91% of `confusable` test
  messages correctly. A person still checks every message.
- With 200 hand labels, UltraBERT-Embed-150M + logistic regression trained on the diagnosed set
  labels 40% of those messages on its own. Under 5% of the 40% are wrong. It takes about 0.01 s to
  classify one message.
- With 1284 hand labels, UltraBERT-Embed-150M + logistic regression reaches 96% accuracy. Its
  coverage was not measured.
- On 10 unrelated categories, the method matters little. Every classifier but two scores at least
  0.9799.

## Class sets

`distinct` is `card_arrival`, `lost_or_stolen_card`, `exchange_rate`, `change_pin`,
`terminate_account`, `age_limit`, `atm_support`, `apple_pay_or_google_pay`, `request_refund` and
`edit_personal_details`.

`confusable` is 5 pairs. Each category's definition in `classes.py` states what separates it from
the other category in its pair.

| Pair                                                   | What separates them |
|--------------------------------------------------------|---------------------|
| `card_arrival` / `card_delivery_estimate`              | a card that is late or missing / a question about delivery time |
| `declined_transfer` / `failed_transfer`                | refused when attempted / sent, then failed |
| `card_payment_wrong_exchange_rate` / `wrong_exchange_rate_for_cash_withdrawal` | a card purchase / an ATM withdrawal |
| `getting_virtual_card` / `get_disposable_virtual_card` | the reusable virtual card / the single-use one |
| `verify_my_identity` / `why_verify_identity`           | how to verify / why it is required |

## Details of the method

**The synthetic set.** `generate.py` asks `gemma4:e4b` for 100 messages per category. It shows the
model the category name and its definition from `CLASS_SETS`. It shows no Banking77 text. The 1000
messages are `data/<set>/synthetic.jsonl`.

**The real train set.** It is the Banking77 train split restricted to the class set.

**The diagnosed notes.** `generate.py --method diagnosed` starts from the synthetic set's request.
It adds the other category's definition under "Do not write messages with this other intent". For
one category in each `confusable` pair it also adds a note from `DIAGNOSED_NOTES` in `generate.py`.
That category is the one losing more of its 20 dev messages to its partner in
`results/confusable/dev/ultrabert-embed-synthetic.txt`. The comment above each note gives the two
dev cells it came from. Each note was written from those counts and from that category's rows in
`data/confusable/synthetic.jsonl`. No note was written from a Banking77 message. Each note names
the kind of message those rows lack. No note asks the model to avoid any word. The requests are
`data/confusable/synthetic-diagnosed.prompt.txt`. The 1000 messages are
`data/confusable/synthetic-diagnosed.jsonl`.

**The pair pilot.** A third class set, `pair`, is the `verify_my_identity` / `why_verify_identity`
pair of `confusable` alone. It was a pilot for generation methods. No table above includes it. Its
reports are under `results/pair/`. The `contrast` method it piloted was superseded by the
`confusable` diagnosed run.

**dev and test.** `dev_and_shots` in `score.py` draws dev with a fixed seed: 20 messages per
category from the Banking77 train split. The 3 examples per category in a `fewshot` prompt are train
messages outside dev. dev scores the Ollama models, the mlx-lm models and Laya. It also scores
UltraBERT-Embed-150M + logistic regression trained on the synthetic set, which is where the
diagnosed counts come from. dev does not score a classifier trained on the real train set, because
that classifier saw the dev rows.

**How prompts were chosen.** Each Ollama model was run on dev with every prompt in `PROMPTS`. Test
was run with `fewshot`. Test was also run with whichever of `names` and `definitions` scored higher
on dev. On `distinct` the two tie for both models, in macro F1 as well as accuracy. On a tie `names`
was taken, being the shorter prompt. The third prompt was not run on test. The choice used the 200
labelled dev messages. The Real labelled examples column does not count them. The mlx-lm rows all
use `definitions`. That prompt was not chosen on dev for them. It is the prompt the adapters are
trained with.

**Fixed before any test run.**

- ModernBERT-base: 5 epochs, learning rate 5e-5, batch size 16, seed 0, no validation set.
- UltraBERT-Embed-150M: revision `367d534`, float32 with SDPA attention. Each message is prefixed
  with MTEB's Banking77Classification instruction. The embedding is the position-0 vector of the
  last layer. The logistic regression uses scikit-learn's default regularisation.
- LoRA adapters: rank 8, scale 20, dropout 0, on every linear layer of the last 16 of 42
  transformer blocks. Adam at learning rate 1e-4, batch size 4, 2 passes over the rows, seed 0.
  Nothing was tuned. There is no held-out split to choose another configuration on, because dev is
  part of the real train split.
- Ollama models: temperature 0. The answer is restricted to an enum of the category names.
- mlx-lm replies: nothing restricts them. The reply is the text after the model's reasoning, if it
  wrote any. A reply that is not exactly a category name is invalid and counts as wrong. The
  512-token limit on a reply was settled on dev. So was that rule. With no adapter on `confusable`,
  34 of 400 test replies are invalid. All 34 are reasoning cut off at 512 tokens. The other five
  mlx-lm test runs have no invalid replies.

**Ollama against mlx-lm.** The two Gemma 4 E4B classifiers without an adapter differ in runtime.
They also differ in whether the reply is restricted. Ollama's `gemma4:e4b` is
`google/gemma-4-E4B-it` at Q4_K_M. `mlx-community/gemma-4-e4b-it-4bit` is a 4-bit quantisation of the same model.

**What UltraBERT-Embed-150M was trained on.** Its model card lists its classification training
data. That data is the synthetic classification set of `nvidia/embed-nemotron-dataset-v1`, Amazon
polarity and toxicity datasets. Four sources were read for any mention of Banking77: that set's
card, its subset configs, the Llama-Embed-Nemotron paper (arXiv 2511.07025) and the UltraBERT paper
(arXiv 2610.15028). None mentions it. The set has 1,044,212 classification examples. None names
Banking77. None offers more than one Banking77 label name as a candidate. 18 contain a Banking77
text verbatim. These are generic, such as "How do I reset my password?". One of the 18 is a
`confusable` or `distinct` test message: "I want a refund for my purchase", `request_refund`.
`results/contamination/scan_nemotron.py` is the scan.
`results/contamination/embed-nemotron-dataset-v1.txt` is its output. The card's other training data was not checked.

**Accuracy and macro F1.** Accuracy is the fraction of messages given their true category. A
category's F1 is the harmonic mean of its precision and recall. Macro F1 is the unweighted mean of
the 10 categories' F1. Results reports accuracy, because each category has 39 or 40 test messages.
An invalid reply counts as wrong in accuracy. An invalid reply counts against no category's
precision in macro F1. So Gemma 4 E4B (MLX) with no adapter has macro F1 0.8858 and accuracy
0.8500.

**Seconds and minutes.** Seconds is the median time to classify one test message, one message per
call. The machine is an M5 Pro with 48 GB. ModernBERT-base and UltraBERT-Embed-150M run on MPS,
PyTorch's Apple GPU backend. TF-IDF and the logistic regressions run on CPU. The Ollama and mlx-lm
models run fully on the GPU. The median hides the 34 cut-off mlx-lm replies, which take several
seconds each. Training minutes are the wall time of one `train.py` or `finetune.py` run, from
`results/<set>/train/<name>.txt`.

**Routing.** `confidence.py` reads a classifier's `score.py --dump` files of dev and test. A
message's score is its top probability over the class set. AUROC is of that score against
correctness on test, with nothing fitted. Isotonic regression of correctness on the score is fitted
on the 200 dev messages and applied to test. ECE is over 10 bins of equal count. For each α the
threshold is the lowest dev score at which the dev error among messages scoring at least that is at
most α. The bounded threshold uses the error's one-sided 90% Clopper-Pearson upper bound in place
of the error. Test coverage is the share of test messages at or above the threshold. Test error is the
error among them. Reports are `results/confusable/confidence/<name>.txt`. They are computed from the
dumps under `results/confusable/confidence/dumps/`.

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
| `confusable` | TypeSafe System One      | `definitions` prompt     | 0.9050   | 0.9053   | 0.53    |

On `confusable` Laya scores below TF-IDF + logistic regression trained on the synthetic set, 0.7075
against 0.7900, so it has no rows in the Results tables but the routing table. Of its answers
whose top probability is above 0.9, 338 of 399 on `distinct` are 0.9734 accurate and 273 of 400 on
`confusable` 0.7912.

The letter-logits rows read the class from the probability of its letter at the last position in one
forward pass, thinking off; they score below the Ollama `gemma4:e4b` row with the prompt chosen on
dev on both sets, 0.8675 against 0.9125 (`definitions`) on `confusable` and 0.9799 against 0.9825
(`names`) on `distinct`, and their top probability is above 0.9 on 378 of 400 `confusable` answers
at 0.8783 accuracy and 397 of 399 `distinct` at 0.9798, so they have no rows in the Results tables
but the routing table, and a threshold of 0.9 does not separate right from wrong answers; the
routing table under question 1 of Results gives the AUROC.

The contrast rows are `generate.py --method contrast`, which shows both definitions in a pair and
asks for at least half the messages to avoid the words of the class name; that rule made both
classifiers worse than the synthetic set, 84 to 127 errors of 400 for TF-IDF and 61 to 65 for
UltraBERT-Embed-150M + LR, so they have no rows in the sections above All results.

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
- `jev.py` — `classify_jev` asks TypeSafe System One Laya's `definitions` question for each message,
  with the key from `TYPESAFE_API_KEY`, and caches every response in `results/jev/<set>.jsonl`, so a
  re-run of `score.py --jev` sends nothing and reproduces its report.
- `finetune.py` — trains a LoRA adapter for `mlx-community/gemma-4-e4b-it-4bit` on a jsonl file
  (`--train`) and saves it to `--out`. Each row is a chat: the `definitions` prompt and the message
  as the user turn, the class name as the assistant turn.
- `score.py` — scores a saved model (`--model`), an Ollama model (`--llm MODEL --prompt NAME`), an
  mlx-lm model with the `definitions` prompt (`--mlx MODEL`, and `--adapter DIR` for a LoRA adapter,
  or `--reply logits` for letter logits, `--mlx` only, no adapter), Laya
  (`--laya --prompt names|definitions`) or TypeSafe System One (`--jev`) on Banking77 restricted to
  the class set: accuracy, macro F1, median seconds per message, per-class recall and a confusion
  matrix, printed and written to `results/<set>/<split>/<name>.txt`. An `--mlx` report also counts the invalid replies.
  `--split test` (the default) is the Banking77 test split. `--split dev` is 20 messages per class
  sampled with a fixed seed from the Banking77 train split; the `fewshot` examples are train
  messages outside dev. `--write-train` exports the restricted train split as jsonl. `--dump FILE`
  also writes each message's probability over the class set, for `--model`, `--laya`, `--jev` and
  `--mlx --reply logits`.
- `confidence.py` — from a classifier's dev and test `--dump` files: AUROC of the top probability
  against correctness, ECE before and after isotonic calibration fitted on dev, and test coverage
  and error at thresholds chosen on dev for α 0.02, 0.05 and 0.10, with and without a
  Clopper-Pearson bound, written to `results/<set>/confidence/<name>.txt`.

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

D=results/confusable/confidence/dumps   # confusable only
for n in tfidf-diagnosed modernbert-diagnosed ultrabert-embed-synthetic ultrabert-embed-diagnosed; do
  python score.py --set confusable --split dev --model models/confusable/$n --dump $D/$n-dev.jsonl
  python score.py --set confusable             --model models/confusable/$n --dump $D/$n-test.jsonl
done
python score.py --set confusable --split dev --mlx $M --reply logits --dump $D/gemma-4-e4b-it-4bit-definitions-logits-dev.jsonl
python score.py --set confusable             --mlx $M --reply logits --dump $D/gemma-4-e4b-it-4bit-definitions-logits-test.jsonl
python score.py --set confusable --split dev --laya --prompt names --dump $D/laya-names-dev.jsonl
python score.py --set confusable             --laya --prompt names --dump $D/laya-names-test.jsonl
python score.py --set confusable --split dev --jev --dump $D/jev-definitions-dev.jsonl   # needs TYPESAFE_API_KEY
python score.py --set confusable             --jev --dump $D/jev-definitions-test.jsonl
for d in $D/*-dev.jsonl; do
  n=$(basename $d -dev.jsonl)
  python confidence.py --set confusable --model-name $n --dev $d --test $D/$n-test.jsonl
done
```

`train.py` and `finetune.py` also write the minutes a run took, or the seconds if under a minute,
to `results/<set>/train/<name>.txt`.
