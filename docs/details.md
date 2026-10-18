# synth-classify: details

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

**Train, validation and test.** The test set is the Banking77 test split restricted to one class
set. It has 399 messages for `distinct` and 400 for `confusable`. Each classifier is scored on it
once. The validation set is 200 messages per class set, 20 per category, drawn from the Banking77
train split. The train set is the rest of that split. Every choice made before a test run is made on
validation. Those choices are the prompt, the notes that steer the
generator, and the routing threshold.

**Training sets.** Four kinds are used.

- **synthetic**: 1000 messages, 100 per category. `gemma4:e4b`, a Gemma 4 LLM run locally with
  Ollama, wrote them. It was shown each category's name and one-sentence definition from
  `CLASS_SETS` in `classes.py`. It never saw a Banking77 message.
- **diagnosed** (`confusable` only): 1000 messages from the same generator. Each request adds the
  definition of the other category in the pair. For one category in each pair, the request also
  adds a note saying what that category's messages must cover. That category is the one that loses
  more validation messages to its partner. Those losses are counted from UltraBERT-Embed-150M +
  logistic regression trained on the synthetic set. The generator sees no validation message. The
  200 validation labels reach it only as counts.
- **train**: the Banking77 train rows of the class set minus the 200 validation rows, 1084 for
  `confusable`. A classifier trained on it can have its threshold set on validation.
- **train + validation**: all the Banking77 train rows of the class set, 1188 for `distinct` and
  1284 for `confusable`.

The tables under questions 1, 2 and 3 do not use these names. Their Trained on column gives a
row's training set as its rows and source: synthetic is "1000 LLM-written", diagnosed "1000
LLM-written, steered by 200 validation counts", train "1084 Banking77 train" and train + validation
"1284 Banking77 train + validation". An untrained classifier is "not trained". Their Prompt column
names the prompt a model is given, and is "—" for a classifier given none.

**Classifiers.** Eight kinds are scored.

- **TF-IDF + logistic regression**: logistic regression on word-frequency features. It is the
  baseline. It is trained on a synthetic, diagnosed or train + validation set.
- **ModernBERT-base**: a text encoder, `answerdotai/ModernBERT-base`, fine-tuned in full on a
  synthetic, diagnosed or train + validation set.
- **UltraBERT-Embed-150M + logistic regression**: logistic regression on embeddings from
  `dmis-lab/UltraBERT-Embed-150M`. An embedding is a fixed-length vector computed from a message.
  The embedding model is not trained here. Only the logistic regression is trained, on a synthetic,
  diagnosed, train or train + validation set. The tables write it "UltraBERT-Embed-150M + LR".
- **An Ollama model with a prompt from `PROMPTS`** (`llm.py`): `gemma4:e4b` or
  `qwen3.6:27b-q4_K_M`, not trained. The `names` prompt lists the category names. The
  `definitions` prompt adds each category's definition. The `fewshot` prompt adds 3 real train
  messages per category to `definitions`, 30 in all. Ollama restricts the reply to a category name.
- **Gemma 4 E4B (MLX), no adapter**: `mlx-community/gemma-4-e4b-it-4bit` run with mlx-lm on the
  `definitions` prompt, not trained. mlx-lm is Apple's LLM runner for its own GPUs. Nothing
  restricts its reply. A reply that is not a category name is invalid and counts as wrong.
- **Gemma 4 E4B (MLX) + LoRA adapter**: the same model and prompt with a LoRA adapter. A LoRA
  adapter is a small set of extra weights trained while the model's own weights stay fixed. Each
  adapter is trained on a synthetic or a train + validation set.
- **Laya**: `convaiinnovations/laya`, a model that picks one option from a list. It is asked one
  question per message with the category names as the options. It is not trained.
- **TypeSafe System One**: `typesafe/jev-1.13`, an API reached through OpenRouter that picks one
  option from a list and gives a probability for each. It is asked Laya's question with the
  `definitions` prompt, one per message, or with the `fewshot` prompt: the same options, each
  definition followed by the 30 examples the Ollama models see, 3 per category. It is not trained,
  and it does not run on the laptop. Under the Clopper-Pearson bound at α 0.05, with the `fewshot`
  prompt, its certified coverage matches UltraBERT-Embed-150M + LR's at the same validation size,
  with 30 examples against 400 to 1084 training rows, at $54 per million messages and 0.53 s a
  message against free and 0.01 s.

**Routing.** Routing keeps a classifier's label when the classifier's top probability is at least a
threshold. A message below the threshold goes to a person. The threshold is the lowest validation
probability at which the one-sided 90% Clopper-Pearson upper bound on the error among validation
messages above it is at most the target, 5%. Coverage and error are then measured on test. Only a
classifier that gives a probability for each category can be routed.

**Uncertainty.** Every row is one run with seed 0. The binomial standard error of an accuracy p
on n messages is √(p(1−p)/n). On the 400 `confusable` test messages it is 0.012 to 0.020, which is
5 to 8 messages. On the 399 `distinct` test messages it is 0.003 to 0.012, which is 1 to 5 messages.

Every number in this file is in a file under `results/`.

## 1. How much of the work can it do without a person checking?

**Under the Clopper-Pearson bound at α 0.05 with 400 validation rows, TypeSafe System One with the
`fewshot` prompt's 30 examples and UltraBERT-Embed-150M + logistic regression trained on 400
Banking77 train rows both certify about 85% of the 400 test messages at 3% error: 0.8658 at 0.0297
and 0.8500 at 0.0300, means over the runs of the validation-size table under question 3.
UltraBERT-Embed-150M + LR trained on train, 1084 rows, certifies 86% on its one draw of 200
validation rows, 344 of the 400 with 9 wrong.** No other classifier in the table labels any message
on that draw. System One costs $54 per million messages and 0.53 s a message against free and
0.01 s.

AUROC is the chance that a randomly chosen right answer gets a higher top probability than a
randomly chosen wrong one. ECE is the mean gap between the top probability and the observed
accuracy, over 10 bins of 40 messages. Isotonic calibration is a monotone mapping of the top
probability onto accuracy, fitted on validation. The bound is the one-sided 90% Clopper-Pearson
upper bound on the validation error: the highest error rate that the validation count does not rule
out at 90% confidence.

| Classifier                       | Trained on                                         | Prompt        | AUROC  | ECE after isotonic calibration | Test coverage under the bound at α 0.05, 200 validation rows, one draw | Test error under the bound at α 0.05, 200 validation rows, one draw | Test coverage under the bound at α 0.05, 400 validation rows | Test error under the bound at α 0.05, 400 validation rows | Test coverage under the bound at α 0.10 | Test error under the bound at α 0.10 |
|----------------------------------|----------------------------------------------------|---------------|--------|--------------------------------|------------------------------------------------------------------------|---------------------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------|--------------------------------------|
| TF-IDF + logistic regression     | 1000 LLM-written, steered by 200 validation counts | —             | 0.7855 | 0.0319                         | 0.0000                                                                 | —                                                                   | not measured                                                 | not measured                                              | 0.0000                                  | —                                    |
| ModernBERT-base                  | 1000 LLM-written, steered by 200 validation counts | —             | 0.8394 | 0.0564                         | 0.0000                                                                 | —                                                                   | not measured                                                 | not measured                                              | 0.0000                                  | —                                    |
| UltraBERT-Embed-150M + LR        | 1000 LLM-written                                   | —             | 0.8405 | 0.0495                         | 0.0000                                                                 | —                                                                   | not measured                                                 | not measured                                              | 0.1550                                  | 0.0323                               |
| UltraBERT-Embed-150M + LR        | 1000 LLM-written, steered by 200 validation counts | —             | 0.7794 | 0.0333                         | 0.0000                                                                 | —                                                                   | not measured                                                 | not measured                                              | 0.7175                                  | 0.0383                               |
| UltraBERT-Embed-150M + LR        | 1084 Banking77 train                               | —             | 0.8641 | 0.0240                         | 0.8600                                                                 | 0.0262                                                              | not measured                                                 | not measured                                              | 1.0000                                  | 0.0475                               |
| UltraBERT-Embed-150M + LR        | 400 Banking77 train                                | —             | —      | —                              | —                                                                      | —                                                                   | 0.8500 (0.7200–0.9300)                                       | 0.0300 (0.0168–0.0490)                                    | —                                       | —                                    |
| Gemma 4 E4B (MLX), letter logits | not trained                                        | `definitions` | 0.7770 | 0.0608                         | 0.0000                                                                 | —                                                                   | not measured                                                 | not measured                                              | 0.0000                                  | —                                    |
| Gemma 4 E4B (MLX), letter logits | not trained                                        | `fewshot`     | 0.7941 | 0.0463                         | 0.0000                                                                 | —                                                                   | not measured                                                 | not measured                                              | 0.0000                                  | —                                    |
| Laya                             | not trained                                        | `names`       | 0.6753 | 0.0705                         | 0.0000                                                                 | —                                                                   | not measured                                                 | not measured                                              | 0.0000                                  | —                                    |
| TypeSafe System One              | not trained                                        | `definitions` | 0.7867 | 0.0534                         | 0.0000                                                                 | —                                                                   | not measured                                                 | not measured                                              | 0.8050                                  | 0.0404                               |
| TypeSafe System One              | not trained                                        | `fewshot`     | 0.8214 | 0.0452                         | 0.0000                                                                 | —                                                                   | 0.8658 (0.8400–0.9100)                                       | 0.0297 (0.0265–0.0357)                                    | 0.9500                                  | 0.0421                               |

Under the bound, certifying α needs about 2.3/α accepted validation messages with no error among
them, so 2% needs 115 of the 200 and no classifier here has an error-free prefix that long; 200
validation rows cannot certify 2% for any model.

α is the error a buyer accepts among messages labelled without a person; 0.05 is the one this repo
quotes.

The plain threshold, set on the validation error itself, is not quoted, because it does not hold its
target: in the 36 validation-size runs under question 3 its test error is over 5% in 12, against 1
of the 28 runs in which the bound keeps any message (`results/confusable/confidence/`).

Three kinds of classifier are not in the table. The Ollama models return a category name and no
probability, so they cannot be routed. A classifier trained on train + validation cannot be routed
here, because validation is part of the data it was trained on; the train row is
UltraBERT-Embed-150M + logistic regression trained without the 200 validation rows. The LoRA
adapters return a reply and no probability.

## Details of the method

**The synthetic set.** `generate.py` asks `gemma4:e4b` for 100 messages per category. It shows the
model the category name and its definition from `CLASS_SETS`. It shows no Banking77 text. The 1000
messages are `data/<set>/synthetic.jsonl`.

**Train and validation.** train + validation is the Banking77 train split restricted to the class
set; validation is 200 of its rows and train the rest. Files, flags and result paths keep the names
the code uses: train is `data/<set>/banking77_train_nodev.jsonl` (`--no-dev`, the
`ultrabert-embed-real-nodev-*` models), validation is dev (`--split dev`, `results/<set>/dev/`) and
train + validation is `data/<set>/banking77_train.jsonl`.

**The diagnosed notes.** `generate.py --method diagnosed` starts from the synthetic set's request.
It adds the other category's definition under "Do not write messages with this other intent". For
one category in each `confusable` pair it also adds a note from `DIAGNOSED_NOTES` in `generate.py`.
That category is the one losing more of its 20 validation messages to its partner in
`results/confusable/dev/ultrabert-embed-synthetic.txt`. The comment above each note gives the two
validation cells it came from. Each note was written from those counts and from that category's rows
in `data/confusable/synthetic.jsonl`. No note was written from a Banking77 message. Each note names
the kind of message those rows lack. No note asks the model to avoid any word. The requests are
`data/confusable/synthetic-diagnosed.prompt.txt`. The 1000 messages are
`data/confusable/synthetic-diagnosed.jsonl`.

**The pair pilot.** A third class set, `pair`, is the `verify_my_identity` / `why_verify_identity`
pair of `confusable` alone. It was a pilot for generation methods. No table above includes it. Its
reports are under `results/pair/`. The `contrast` method it piloted was superseded by the
`confusable` diagnosed run.

**Validation and test.** `dev_and_shots` in `score.py` draws validation with a fixed seed: 20
messages per category from the Banking77 train split. The 3 examples per category in a `fewshot`
prompt are train messages, outside validation. Validation scores the Ollama models, the mlx-lm
models and Laya. It also scores UltraBERT-Embed-150M + logistic regression trained on the synthetic
set, which is where the diagnosed counts come from. Validation does not score a classifier trained
on train + validation, because that classifier saw the validation rows.

**How prompts were chosen.** Each Ollama model was run on validation with every prompt in `PROMPTS`.
Test was run with `fewshot`. Test was also run with whichever of `names` and `definitions` scored
higher on validation. On `distinct` the two tie for both models, in macro F1 as well as accuracy. On
a tie `names` was taken, being the shorter prompt. The third prompt was not run on test. The choice
used the 200 labelled validation messages. The Real labelled examples column does not count them.
The mlx-lm rows all use `definitions`. That prompt was not chosen on validation for them. It is the
prompt the adapters are trained with.

**Fixed before any test run.**

- ModernBERT-base: 5 epochs, learning rate 5e-5, batch size 16, seed 0, no early stopping on
  validation.
- UltraBERT-Embed-150M: revision `367d534`, float32 with SDPA attention. Each message is prefixed
  with MTEB's Banking77Classification instruction. The embedding is the position-0 vector of the
  last layer. The logistic regression uses scikit-learn's default regularisation.
- LoRA adapters: rank 8, scale 20, dropout 0, on every linear layer of the last 16 of 42
  transformer blocks. Adam at learning rate 1e-4, batch size 4, 2 passes over the rows, seed 0.
  Nothing was tuned. There is no held-out split to choose another configuration on, because
  validation is part of train + validation.
- Ollama models: temperature 0. The answer is restricted to an enum of the category names.
- mlx-lm replies: nothing restricts them. The reply is the text after the model's reasoning, if it
  wrote any. A reply that is not exactly a category name is invalid and counts as wrong. The
  512-token limit on a reply was settled on validation. So was that rule. With no adapter on
  `confusable`, 34 of 400 test replies are invalid. All 34 are reasoning cut off at 512 tokens. The
  other five mlx-lm test runs have no invalid replies.

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
the 10 categories' F1. Question 2 reports accuracy, because each category has 39 or 40 test
messages. An invalid reply counts as wrong in accuracy. An invalid reply counts against no
category's precision in macro F1. So Gemma 4 E4B (MLX) with no adapter has macro F1 0.8858 and
accuracy 0.8500.

**Seconds and minutes.** Seconds is the median time to classify one test message, one message per
call. The machine is an M5 Pro with 48 GB. ModernBERT-base and UltraBERT-Embed-150M run on MPS,
PyTorch's Apple GPU backend. TF-IDF and the logistic regressions run on CPU. The Ollama and mlx-lm
models run fully on the GPU. The median hides the 34 cut-off mlx-lm replies, which take several
seconds each. Training minutes are the wall time of one `train.py` or `finetune.py` run, from
`results/<set>/train/<name>.txt`.

**Routing.** `confidence.py` reads a classifier's `score.py --dump` files of validation and test. A
message's score is its top probability over the class set. AUROC is of that score against
correctness on test, with nothing fitted. Isotonic regression of correctness on the score is fitted
on the 200 validation messages and applied to test. ECE is over 10 bins of equal count. For each α
the threshold is the lowest validation score at which the validation error among messages scoring at
least that is at most α. The bounded threshold uses the error's one-sided 90% Clopper-Pearson upper
bound in place of the error. Test coverage is the share of test messages at or above the threshold.
Test error is the error among them. Reports are `results/confusable/confidence/<name>.txt`. They are
computed from the dumps under `results/confusable/confidence/dumps/`.

## All results

### Test

Each row's full report is `results/<class set>/test/<model>.txt`. The "no prompt" rows
(`ultrabert-embed-noprompt-*`) embed each message without MTEB's Banking77Classification
instruction before it, to measure what that instruction adds to the UltraBERT-Embed-150M + LR rows.
The train + validation rows include the 200 validation rows, so they cannot be routed.

| Class set    | Model                | Training set or prompt   | Accuracy | Macro F1 | Seconds |
|--------------|----------------------|--------------------------|----------|----------|---------|
| `distinct`   | ModernBERT-base      | synthetic (`gemma4:e4b`) | 0.9925   | 0.9925   | 0.0073  |
| `distinct`   | ModernBERT-base      | train + validation | 0.9950   | 0.9950   | 0.0073  |
| `distinct`   | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 0.9373 | 0.9375 | 0.00015 |
| `distinct`   | TF-IDF + logistic regression | train + validation | 0.9875 | 0.9874 | 0.00015 |
| `distinct`   | UltraBERT-Embed-150M + LR | synthetic (`gemma4:e4b`) | 0.9925 | 0.9925 | 0.0112 |
| `distinct`   | UltraBERT-Embed-150M + LR | train + validation | 0.9975 | 0.9975 | 0.0106 |
| `distinct`   | UltraBERT-Embed-150M + LR, no prompt | synthetic (`gemma4:e4b`) | 0.9950 | 0.9950 | 0.0100 |
| `distinct`   | UltraBERT-Embed-150M + LR, no prompt | train + validation | 0.9975 | 0.9975 | 0.0107 |
| `distinct`   | `gemma4:e4b`         | `names` prompt           | 0.9825   | 0.9822   | 0.49    |
| `distinct`   | `gemma4:e4b`         | `fewshot` prompt         | 0.9875   | 0.9874   | 0.44    |
| `distinct`   | `qwen3.6:27b-q4_K_M` | `names` prompt           | 0.9900   | 0.9899   | 1.6     |
| `distinct`   | `qwen3.6:27b-q4_K_M` | `fewshot` prompt         | 0.9925   | 0.9925   | 2.4     |
| `distinct`   | Gemma 4 E4B (MLX)    | `definitions` prompt     | 0.9975   | 0.9975   | 0.141   |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 0.9950 | 0.9950 | 0.163   |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | train + validation | 0.9950 | 0.9950 | 0.167   |
| `distinct`   | Gemma 4 E4B (MLX), letter logits | `definitions` prompt | 0.9799 | 0.9797 | 0.147 |
| `confusable` | ModernBERT-base      | synthetic (`gemma4:e4b`) | 0.8425   | 0.8413   | 0.0073  |
| `confusable` | ModernBERT-base      | diagnosed (`gemma4:e4b`) | 0.8675   | 0.8674   | 0.0077  |
| `confusable` | ModernBERT-base      | train + validation | 0.9375   | 0.9376   | 0.0073  |
| `confusable` | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 0.7900 | 0.7849 | 0.00015 |
| `confusable` | TF-IDF + logistic regression | diagnosed (`gemma4:e4b`) | 0.7925 | 0.7887 | 0.00016 |
| `confusable` | TF-IDF + logistic regression | contrast (`gemma4:e4b`)  | 0.6825 | 0.6681 | 0.00015 |
| `confusable` | TF-IDF + logistic regression | train + validation | 0.9250 | 0.9251 | 0.00015 |
| `confusable` | UltraBERT-Embed-150M + LR | synthetic (`gemma4:e4b`) | 0.8475 | 0.8456 | 0.0107 |
| `confusable` | UltraBERT-Embed-150M + LR | diagnosed (`gemma4:e4b`) | 0.9125 | 0.9134 | 0.0109 |
| `confusable` | UltraBERT-Embed-150M + LR | contrast (`gemma4:e4b`)  | 0.8375 | 0.8353 | 0.0109 |
| `confusable` | UltraBERT-Embed-150M + LR | train + validation | 0.9575 | 0.9576 | 0.0109 |
| `confusable` | UltraBERT-Embed-150M + LR | train | 0.9525 | 0.9526 | 0.0102 |
| `confusable` | UltraBERT-Embed-150M + LR | 50 of train, seed 0 | 0.8900 | 0.8893 | 0.0101 |
| `confusable` | UltraBERT-Embed-150M + LR | 100 of train, seed 0 | 0.9250 | 0.9252 | 0.00995 |
| `confusable` | UltraBERT-Embed-150M + LR | 200 of train, seed 0 | 0.9200 | 0.9199 | 0.00973 |
| `confusable` | UltraBERT-Embed-150M + LR | 400 of train, seed 0 | 0.9325 | 0.9325 | 0.0100 |
| `confusable` | UltraBERT-Embed-150M + LR | 795 of train, seed 0 | 0.9475 | 0.9477 | 0.0100 |
| `confusable` | UltraBERT-Embed-150M + LR | 50 of train, seed 1 | 0.8825 | 0.8823 | 0.0102 |
| `confusable` | UltraBERT-Embed-150M + LR | 100 of train, seed 1 | 0.9250 | 0.9251 | 0.0100 |
| `confusable` | UltraBERT-Embed-150M + LR | 200 of train, seed 1 | 0.9400 | 0.9400 | 0.00981 |
| `confusable` | UltraBERT-Embed-150M + LR | 400 of train, seed 1 | 0.9375 | 0.9376 | 0.0101 |
| `confusable` | UltraBERT-Embed-150M + LR | 50 of train, seed 2 | 0.8350 | 0.8267 | 0.0100 |
| `confusable` | UltraBERT-Embed-150M + LR | 100 of train, seed 2 | 0.9075 | 0.9075 | 0.0100 |
| `confusable` | UltraBERT-Embed-150M + LR | 200 of train, seed 2 | 0.9225 | 0.9226 | 0.0100 |
| `confusable` | UltraBERT-Embed-150M + LR | 400 of train, seed 2 | 0.9250 | 0.9251 | 0.0100 |
| `confusable` | UltraBERT-Embed-150M + LR, no prompt | synthetic (`gemma4:e4b`) | 0.8275 | 0.8245 | 0.0100 |
| `confusable` | UltraBERT-Embed-150M + LR, no prompt | train + validation | 0.9325 | 0.9326 | 0.0102 |
| `confusable` | `gemma4:e4b`             | `definitions` prompt     | 0.9125   | 0.9126   | 0.63    |
| `confusable` | `gemma4:e4b`             | `fewshot` prompt         | 0.9300   | 0.9302   | 0.47    |
| `confusable` | `qwen3.6:27b-q4_K_M`     | `names` prompt           | 0.8775   | 0.8756   | 1.5     |
| `confusable` | `qwen3.6:27b-q4_K_M`     | `fewshot` prompt         | 0.9325   | 0.9329   | 2.5     |
| `confusable` | Gemma 4 E4B (MLX)        | `definitions` prompt     | 0.8500   | 0.8858   | 0.258   |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 0.8500 | 0.8510 | 0.235   |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | train + validation | 0.8575 | 0.8513 | 0.235   |
| `confusable` | Gemma 4 E4B (MLX), letter logits | `definitions` prompt | 0.8675 | 0.8648 | 0.338 |
| `confusable` | Gemma 4 E4B (MLX), letter logits | `fewshot` prompt     | 0.8600 | 0.8582 | 0.471 |
| `confusable` | Gemma 4 31B (MLX), letter logits | `fewshot` prompt     | 0.9375 | 0.9379 | 4.28  |
| `distinct`   | Laya                     | `names` prompt           | 0.9123   | 0.9139   | 0.0225  |
| `confusable` | Laya                     | `names` prompt           | 0.7075   | 0.6750   | 0.0241  |
| `confusable` | TypeSafe System One      | `definitions` prompt     | 0.9050   | 0.9053   | 0.53    |
| `confusable` | TypeSafe System One      | `fewshot` prompt         | 0.9450   | 0.9449   | 0.53    |

On `confusable` Laya scores below TF-IDF + logistic regression trained on the synthetic set, 0.7075
against 0.7900, so it has no rows in the tables under [question 2](#2-how-accurate-is-it) and
[question 3](#3-how-many-hand-labelled-examples-does-it-need) but one in [question
1](#1-how-much-of-the-work-can-it-do-without-a-person-checking)'s table. Of its answers whose top
probability is above 0.9, 338 of 399 on `distinct` are 0.9734 accurate and 273 of 400 on
`confusable` 0.7912.

The `definitions` letter-logits rows read the class from the probability of its letter at the last
position in one forward pass, thinking off; they score below the Ollama `gemma4:e4b` row with the
prompt chosen on validation on both sets, 0.8675 against 0.9125 (`definitions`) on `confusable` and
0.9799 against 0.9825 (`names`) on `distinct`, and their top probability is above 0.9 on 378 of 400
`confusable` answers at 0.8783 accuracy and 397 of 399 `distinct` at 0.9798, so they have no rows in
the tables under [question 2](#2-how-accurate-is-it) and [question
3](#3-how-many-hand-labelled-examples-does-it-need) but one in [question
1](#1-how-much-of-the-work-can-it-do-without-a-person-checking)'s table, and a threshold of 0.9 does
not separate right from wrong answers; [question
1](#1-how-much-of-the-work-can-it-do-without-a-person-checking)'s table gives the AUROC. The
`fewshot` letter-logits row adds the 30 `fewshot` examples under each lettered category; it scores
0.8600 on `confusable`, 3 messages below `definitions`, and is in the routing and accuracy tables.

The contrast rows are `generate.py --method contrast`, which shows both definitions in a pair and
asks for at least half the messages to avoid the words of the class name; that rule made both
classifiers worse than the synthetic set, 84 to 127 errors of 400 for TF-IDF and 61 to 65 for
UltraBERT-Embed-150M + LR, so they have no rows in the sections above All results.

### Prompts on validation

Accuracy of each Ollama model, of Laya and of TypeSafe System One with each prompt on the 200
validation messages; full reports are `results/<class set>/dev/<model>-<prompt>.txt`. Laya takes no
`fewshot` prompt and System One no `names` prompt. System One's two prompts tie on validation, and
both were run on test.

| Class set    | Model                | `names` | `definitions` | `fewshot` |
|--------------|----------------------|---------|---------------|-----------|
| `distinct`   | `gemma4:e4b`         | 0.9800  | 0.9800        | 0.9850    |
| `distinct`   | `qwen3.6:27b-q4_K_M` | 0.9900  | 0.9900        | 1.0000    |
| `confusable` | `gemma4:e4b`         | 0.8500  | 0.8750        | 0.9000    |
| `confusable` | `qwen3.6:27b-q4_K_M` | 0.8650  | 0.8500        | 0.8950    |
| `distinct`   | Laya                 | 0.8650  | 0.7600        | —         |
| `confusable` | Laya                 | 0.6700  | 0.2950        | —         |
| `confusable` | TypeSafe System One  | —       | 0.9000        | 0.9000    |

### Training minutes

Wall time of one `train.py` or `finetune.py` run on the same M5 Pro, from loading the base model to
saving, with nothing else on the GPU; files are `results/<class set>/train/<model>.txt`. A file
whose `model` line is under `xtmp/` is a re-run of the training command into a scratch directory,
and the model scored in the test table was not replaced; one under `models/` is the run whose model
was scored. The UltraBERT-Embed-150M times are mostly embedding the training set. A time under a
minute is in seconds. The train + validation rows include the 200 validation rows, so their models
cannot be routed.

| Class set    | Model                    | Training set             | Rows | Minutes |
|--------------|--------------------------|--------------------------|------|---------|
| `distinct`   | ModernBERT-base          | synthetic (`gemma4:e4b`) | 1000 | 1.6     |
| `distinct`   | ModernBERT-base          | train + validation | 1188 | 1.8     |
| `distinct`   | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 1000 | 0.65 s  |
| `distinct`   | TF-IDF + logistic regression | train + validation | 1188 | 0.60 s  |
| `distinct`   | UltraBERT-Embed-150M + LR | synthetic (`gemma4:e4b`) | 1000 | 6.92 s  |
| `distinct`   | UltraBERT-Embed-150M + LR | train + validation | 1188 | 6.40 s  |
| `distinct`   | UltraBERT-Embed-150M + LR, no prompt | synthetic (`gemma4:e4b`) | 1000 | 5.19 s  |
| `distinct`   | UltraBERT-Embed-150M + LR, no prompt | train + validation | 1188 | 6.86 s  |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 1000 | 7.8     |
| `distinct`   | Gemma 4 E4B (MLX) + LoRA | train + validation | 1188 | 9.1     |
| `confusable` | ModernBERT-base          | synthetic (`gemma4:e4b`) | 1000 | 1.7     |
| `confusable` | ModernBERT-base          | diagnosed (`gemma4:e4b`) | 1000 | 1.8     |
| `confusable` | ModernBERT-base          | train + validation | 1284 | 2.0     |
| `confusable` | TF-IDF + logistic regression | synthetic (`gemma4:e4b`) | 1000 | 0.71 s  |
| `confusable` | TF-IDF + logistic regression | diagnosed (`gemma4:e4b`) | 1000 | 0.85 s  |
| `confusable` | TF-IDF + logistic regression | train + validation | 1284 | 0.65 s  |
| `confusable` | UltraBERT-Embed-150M + LR | synthetic (`gemma4:e4b`) | 1000 | 5.86 s  |
| `confusable` | UltraBERT-Embed-150M + LR | diagnosed (`gemma4:e4b`) | 1000 | 5.50 s  |
| `confusable` | UltraBERT-Embed-150M + LR | train + validation | 1284 | 6.52 s  |
| `confusable` | UltraBERT-Embed-150M + LR | train | 1084 | 7.50 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 50 of train, seed 0 | 50   | 5.01 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 100 of train, seed 0 | 100  | 2.92 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 200 of train, seed 0 | 200  | 3.15 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 400 of train, seed 0 | 400  | 3.65 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 795 of train, seed 0 | 795  | 4.69 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 50 of train, seed 1 | 50   | 5.25 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 100 of train, seed 1 | 100  | 2.76 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 200 of train, seed 1 | 200  | 3.04 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 400 of train, seed 1 | 400  | 3.55 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 50 of train, seed 2 | 50   | 2.68 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 100 of train, seed 2 | 100  | 3.06 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 200 of train, seed 2 | 200  | 3.21 s  |
| `confusable` | UltraBERT-Embed-150M + LR | 400 of train, seed 2 | 400  | 3.74 s  |
| `confusable` | UltraBERT-Embed-150M + LR, no prompt | synthetic (`gemma4:e4b`) | 1000 | 4.60 s  |
| `confusable` | UltraBERT-Embed-150M + LR, no prompt | train + validation | 1284 | 5.57 s  |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | synthetic (`gemma4:e4b`) | 1000 | 13.6    |
| `confusable` | Gemma 4 E4B (MLX) + LoRA | train + validation | 1284 | 17.9    |

ModernBERT runs 5 passes over the rows and the adapters 2. The `confusable` prompt is about twice
the length of the `distinct` one (2060 characters against 1012), and every training example for an
adapter carries the whole prompt.

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

## 2. How accurate is it?

**UltraBERT-Embed-150M + logistic regression trained on train makes 19 errors of 400, the fewest
of any classifier that did not see the validation rows.**

| Classifier                       | Trained on                                         | Prompt        | Real labelled examples used | Accuracy | Errors of 400 |
|----------------------------------|----------------------------------------------------|---------------|-----------------------------|----------|---------------|
| TF-IDF + logistic regression     | 1000 LLM-written                                   | —             | 0                           | 0.7900   | 84            |
| TF-IDF + logistic regression     | 1000 LLM-written, steered by 200 validation counts | —             | 200, as counts              | 0.7925   | 83            |
| TF-IDF + logistic regression     | 1284 Banking77 train + validation                  | —             | 1284                        | 0.9250   | 30            |
| ModernBERT-base                  | 1000 LLM-written                                   | —             | 0                           | 0.8425   | 63            |
| ModernBERT-base                  | 1000 LLM-written, steered by 200 validation counts | —             | 200, as counts              | 0.8675   | 53            |
| ModernBERT-base                  | 1284 Banking77 train + validation                  | —             | 1284                        | 0.9375   | 25            |
| UltraBERT-Embed-150M + LR        | 1000 LLM-written                                   | —             | 0                           | 0.8475   | 61            |
| UltraBERT-Embed-150M + LR        | 1000 LLM-written, steered by 200 validation counts | —             | 200, as counts              | 0.9125   | 35            |
| UltraBERT-Embed-150M + LR        | 1284 Banking77 train + validation                  | —             | 1284                        | 0.9575   | 17            |
| UltraBERT-Embed-150M + LR        | 1084 Banking77 train                               | —             | 1084                        | 0.9525   | 19            |
| `gemma4:e4b`                     | not trained                                        | `definitions` | 0                           | 0.9125   | 35            |
| `gemma4:e4b`                     | not trained                                        | `fewshot`     | 30                          | 0.9300   | 28            |
| `qwen3.6:27b-q4_K_M`             | not trained                                        | `names`       | 0                           | 0.8775   | 49            |
| `qwen3.6:27b-q4_K_M`             | not trained                                        | `fewshot`     | 30                          | 0.9325   | 27            |
| Gemma 4 E4B (MLX), no adapter    | not trained                                        | `definitions` | 0                           | 0.8500   | 60            |
| Gemma 4 E4B (MLX) + LoRA adapter | 1000 LLM-written                                   | `definitions` | 0                           | 0.8500   | 60            |
| Gemma 4 E4B (MLX) + LoRA adapter | 1284 Banking77 train + validation                  | `definitions` | 1284                        | 0.8575   | 57            |
| Gemma 4 E4B (MLX), letter logits | not trained                                        | `fewshot`     | 30                          | 0.8600   | 56            |
| Gemma 4 31B (MLX), letter logits | not trained                                        | `fewshot`     | 30                          | 0.9375   | 25            |
| TypeSafe System One              | not trained                                        | `definitions` | 0                           | 0.9050   | 38            |
| TypeSafe System One              | not trained                                        | `fewshot`     | 30                          | 0.9450   | 22            |

Every classifier above 0.92 used real labelled examples. The Gemma 4 E4B (MLX) + LoRA adapter
trained on train + validation also used them, and stayed at 0.8575. The `fewshot` prompts of
`gemma4:e4b` and `qwen3.6:27b-q4_K_M` come within 3 errors of ModernBERT-base trained on train +
validation, and TypeSafe System One with the `fewshot` prompt makes 3 fewer; under the bound of
question 1 its certified coverage matches UltraBERT-Embed-150M + logistic regression's at the same
validation size, with 30 examples against 400 to 1084 training rows, at $54 per million messages and
0.53 s a message against free and 0.01 s. Every other LLM classifier makes at least 10 more errors
than ModernBERT-base trained on train + validation.

On `distinct`, the choice of classifier matters little. Every classifier in the test table under
[All results](#all-results) scores 0.9799 to 0.9975 there. The two exceptions are TF-IDF + logistic
regression trained on the synthetic set, at 0.9373, and Laya, at 0.9123.

**Most errors fall inside a pair.** An error inside a pair is one whose true and predicted
categories are the two categories of one pair.

| Classifier                       | Trained on                                         | Prompt        | Real labelled examples used | Errors | Inside a pair | Most frequent error (true → predicted)        | Messages, of 40 |
|----------------------------------|----------------------------------------------------|---------------|-----------------------------|--------|---------------|-----------------------------------------------|-----------------|
| TF-IDF + logistic regression     | 1000 LLM-written                                   | —             | 0                           | 84     | 71            | `why_verify_identity` → `verify_my_identity`  | 23              |
| TF-IDF + logistic regression     | 1000 LLM-written, steered by 200 validation counts | —             | 200, as counts              | 83     | 71            | `why_verify_identity` → `verify_my_identity`  | 24              |
| TF-IDF + logistic regression     | 1284 Banking77 train + validation                  | —             | 1284                        | 30     | 26            | `verify_my_identity` → `why_verify_identity`  | 7               |
| ModernBERT-base                  | 1000 LLM-written                                   | —             | 0                           | 63     | 54            | `why_verify_identity` → `verify_my_identity`  | 15              |
| ModernBERT-base                  | 1000 LLM-written, steered by 200 validation counts | —             | 200, as counts              | 53     | 48            | `why_verify_identity` → `verify_my_identity`  | 14              |
| ModernBERT-base                  | 1284 Banking77 train + validation                  | —             | 1284                        | 25     | 24            | `card_arrival` → `card_delivery_estimate`     | 5               |
| UltraBERT-Embed-150M + LR        | 1000 LLM-written                                   | —             | 0                           | 61     | 53            | `why_verify_identity` → `verify_my_identity`  | 13              |
| UltraBERT-Embed-150M + LR        | 1000 LLM-written, steered by 200 validation counts | —             | 200, as counts              | 35     | 31            | `card_arrival` → `card_delivery_estimate`     | 10              |
| UltraBERT-Embed-150M + LR        | 1284 Banking77 train + validation                  | —             | 1284                        | 17     | 15            | `declined_transfer` → `failed_transfer`       | 3               |
| `gemma4:e4b`                     | not trained                                        | `definitions` | 0                           | 35     | 32            | `card_arrival` → `card_delivery_estimate`     | 13              |
| `gemma4:e4b`                     | not trained                                        | `fewshot`     | 30                          | 28     | 25            | `card_arrival` → `card_delivery_estimate`     | 7               |
| `qwen3.6:27b-q4_K_M`             | not trained                                        | `names`       | 0                           | 49     | 47            | `verify_my_identity` → `why_verify_identity`  | 17              |
| `qwen3.6:27b-q4_K_M`             | not trained                                        | `fewshot`     | 30                          | 27     | 21            | `card_arrival` → `card_delivery_estimate`     | 6               |
| Gemma 4 E4B (MLX), no adapter    | not trained                                        | `definitions` | 0                           | 60     | 26            | `card_arrival` → `card_delivery_estimate`     | 16              |
| Gemma 4 E4B (MLX) + LoRA adapter | 1000 LLM-written                                   | `definitions` | 0                           | 60     | 34            | `card_arrival` → `card_delivery_estimate`     | 18              |
| Gemma 4 E4B (MLX) + LoRA adapter | 1284 Banking77 train + validation                  | `definitions` | 1284                        | 57     | 56            | `declined_transfer` → `failed_transfer`       | 24              |
| TypeSafe System One              | not trained                                        | `definitions` | 0                           | 38     | 35            | `card_delivery_estimate` → `card_arrival` and `why_verify_identity` → `verify_my_identity` | 8 each |
| TypeSafe System One              | not trained                                        | `fewshot`     | 30                          | 22     | 22            | `card_arrival` → `card_delivery_estimate` and `declined_transfer` → `failed_transfer` | 5 each |

Gemma 4 E4B (MLX) with no adapter is the one classifier with most of its errors outside a pair.
Those 34 errors outside a pair are its invalid replies. The 26 errors outside a pair of the
Gemma 4 E4B (MLX) + LoRA adapter trained on the synthetic set are all predictions of
`getting_virtual_card`.

## 3. How many hand-labelled examples does it need?

**With 200 hand labels, UltraBERT-Embed-150M + logistic regression matches the accuracy that
`gemma4:e4b` reaches with none.**

| Hand labels                                       | Accuracy | Method |
|---------------------------------------------------|----------|--------|
| 0                                                 | 0.9125   | `gemma4:e4b` with the class definitions in the prompt, reply restricted to a class name |
| 30, as examples in the prompt                     | 0.9300 / 0.9325 / 0.9450 | `gemma4:e4b`, `qwen3.6:27b-q4_K_M` or TypeSafe System One with the `fewshot` prompt |
| 200, as confusion counts that steer the generator | 0.9125, at about 1/60 of the LLM's time per message | UltraBERT-Embed-150M + LR trained on the diagnosed set |
| 1084, to train on                                 | 0.9525   | UltraBERT-Embed-150M + LR trained on train |

The 200 hand labels are the validation set. They do two jobs. They count the losses that steer the
diagnosed set. They also set the routing threshold of question 1. So 200 hand labels buy the
0.9125, and under the bound no message labelled without a person.

With 0 hand labels no classifier can be routed, because the threshold is set on labelled validation
messages.

The label curve trains UltraBERT-Embed-150M + logistic regression on stratified subsets of train: 5,
10, 20, 40 and 80 messages per class, each subset inside the next one drawn with the same seed. 50,
100, 200 and 400 are drawn with seeds 0, 1 and 2, and each cell gives the mean over the three draws,
then the min–max. 800 is drawn with seed 0 alone, and 1084 is the whole set, so those two rows are
single runs. The 200 validation labels set every threshold, on top of the training rows.
`getting_virtual_card` and `get_disposable_virtual_card` have 78 and 77 rows in train, so the 800
subset holds 795. The last column is the coverage on the 400 test messages under the Clopper-Pearson
bound of question 1 at α 0.05.

| Training rows | Draws | Accuracy               | Test coverage under the Clopper-Pearson bound at α 0.05 |
|---------------|-------|------------------------|----------------------------------------------------------|
| 50            | 3     | 0.8692 (0.8350–0.8900) | 0.0000 (0.0000–0.0000) |
| 100           | 3     | 0.9192 (0.9075–0.9250) | 0.1125 (0.0000–0.3375) |
| 200           | 3     | 0.9275 (0.9200–0.9400) | 0.5150 (0.0000–0.7975) |
| 400           | 3     | 0.9317 (0.9250–0.9375) | 0.4508 (0.0000–0.6800) |
| 800 (795)     | 1     | 0.9475                 | 0.8775 |
| 1084          | 1     | 0.9525                 | 0.8600 |

Mean accuracy rises with the training rows, from 0.8692 at 50 to 0.9317 at 400. Coverage under the
bound does not rise with every step, 0.5150 at 200 and 0.4508 at 400, and it varies more between
draws of one size than between sizes: at 200 one draw keeps nothing and another 0.7975, at 400
one keeps nothing and another 0.6800.

How much the threshold depends on the number of validation labels is measured on the three 400-row
models (seeds 0, 1 and 2). Each model's validation set is replaced by its rest: the 684 rows of
train that it was not trained on. From the rest, N rows are drawn, N / 10 per class, with validation
seeds 0, 1 and 2, and the α 0.05 thresholds are chosen on them; 3 models × 3 draws is 9 runs per N.
The rest holds 38 `getting_virtual_card` and 37 `get_disposable_virtual_card` rows, so N = 400 draws
395. Each cell is the mean over the 9 runs, then the min–max, all on the 400 test messages. Error
under the bound is over the runs that keep any test message, counted in brackets.

| Validation rows | Test coverage under the Clopper-Pearson bound | Test error under the Clopper-Pearson bound |
|-----------------|-----------------------------------------------|--------------------------------------------|
| **UltraBERT-Embed-150M + LR, 400-row models, 9 runs per N** | | |
| 100             | 0.3342 (0.0000–0.9750)                        | 0.0261 (0.0113–0.0641), 4 runs             |
| 200             | 0.7142 (0.0000–0.9450)                        | 0.0246 (0.0137–0.0450), 8 runs             |
| 400 (395)       | 0.8500 (0.7200–0.9300)                        | 0.0300 (0.0168–0.0490), 9 runs             |
| **TypeSafe System One, `fewshot`, 3 runs per N** | | |
| 100             | 0.2825 (0.0000–0.8475)                        | 0.0265, 1 run                              |
| 200             | 0.8025 (0.6325–0.9350)                        | 0.0263 (0.0119–0.0401), 3 runs             |
| 400 (394)       | 0.8658 (0.8400–0.9100)                        | 0.0297 (0.0265–0.0357), 3 runs             |
| **Gemma 4 31B (MLX), letter logits, `fewshot`, 3 runs per N** | | |
| 100             | 0.0000 (0.0000–0.0000)                        | —, 0 runs                                  |
| 200             | 0.3100 (0.0000–0.9300)                        | 0.0403, 1 run                              |
| 400 (394)       | 0.3042 (0.0000–0.9125)                        | 0.0384, 1 run                              |
| **UltraBERT-Embed-150M + LR at 400 validation rows, by training rows, 9 runs each** | | |
| 30 training     | 0.3278 (0.0000–0.6100)                        | 0.0267 (0.0071–0.0574), 8 runs             |
| 50 training     | 0.3347 (0.0000–0.7050)                        | 0.0296 (0.0122–0.0426), 7 runs             |
| 100 training    | 0.7681 (0.4625–0.8975)                        | 0.0396 (0.0270–0.0543), 9 runs             |
| 200 training    | 0.9103 (0.7825–1.0000)                        | 0.0453 (0.0288–0.0600), 9 runs             |

With 400 validation rows, the 30- and 50-row models keep a third of the test messages under the
bound, the 100-row models 0.7681 and the 200-row models 0.9103, against System One's 0.8658 at 394,
though at a mean test error of 0.0453 against System One's 0.0297 and above 0.05 in some runs
(`results/confusable/confidence/ultrabert-embed-real-nodev-<training rows>[-s<seed>]-dev400-ds<validation seed>.txt`).

More validation rows raise the coverage under the bound, from 0.3342 at 100 to 0.8500 at 395, and
narrow its range. At 100 validation rows the bound keeps nothing in 5 of 9
runs, at 200 in 1, at 400 in none. Under the bound, test error stays under 0.05 in every run that
keeps a message but one, 0.0641 at N = 100. Each run's report is
`results/confusable/confidence/ultrabert-embed-real-nodev-400[-s<seed>]-dev<N>-ds<validation seed>.txt`.

TypeSafe System One's rest is the 684 rows of the seed-0 400-row model's rest less the 15 that are
`fewshot` examples, 669 rows, and under the bound it keeps test messages in every run with 200 or
more validation rows, 0.8025 of them at 200 and 0.8658 at 394, at a test error of at most 0.0401,
where on the 200 validation messages of question 1 it keeps none
(`results/confusable/confidence/jev-fewshot-dev<N>-ds<validation seed>.txt`). With 200 validation
rows the bound's coverage ranges from 0 to 0.95 for both classifiers, with 400 it ranges from 0.72
to 0.93.

Gemma 4 31B (MLX) with letter logits and the `fewshot` prompt is drawn from the same 669
rows. 650 of them score above 0.9, at 0.9246 accuracy
(`results/confusable/rest/gemma-4-31b-it-4bit-fewshot-logits.txt`), so the bound keeps nothing
in all 3 runs at 100 validation rows and in 2 of 3 at 200 and 400
(`results/confusable/confidence/gemma-4-31b-it-4bit-fewshot-logits-dev<N>-ds<validation seed>.txt`).

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

## What raised an LLM classifier's accuracy

| Change                                                                   | Before | After  | Messages | Scored on |
|--------------------------------------------------------------------------|--------|--------|----------|-----------|
| Reply restricted to a class name: mlx-lm with no adapter → Ollama `gemma4:e4b`, `definitions` | 0.8500 | 0.9125 | +25 | test, 400 |
| 3 real examples per class, `gemma4:e4b`: `definitions` → `fewshot`       | 0.9125 | 0.9300 | +7       | test, 400 |
| 3 real examples per class, `qwen3.6:27b-q4_K_M`: `names` → `fewshot`     | 0.8775 | 0.9325 | +22      | test, 400 |
| 3 real examples per class, TypeSafe System One: `definitions` → `fewshot` | 0.9050 | 0.9450 | +16     | test, 400 |
| Prompt chosen on validation, `gemma4:e4b`: `names` → `definitions`              | 0.8500 | 0.8750 | +5       | validation, 200 |
| Prompt chosen on validation, `qwen3.6:27b-q4_K_M`: `definitions` → `names`      | 0.8500 | 0.8650 | +3       | validation, 200 |
| `qwen3.6:27b-q4_K_M` in place of `gemma4:e4b`, `fewshot`                 | 0.9300 | 0.9325 | +1       | test, 400 |
| LoRA adapter trained on 1000 LLM-written messages, against no adapter    | 0.8500 | 0.8500 | 0        | test, 400 |
| LoRA adapter trained on 1284 Banking77 train + validation, against no adapter | 0.8500 | 0.8575 | +3       | test, 400 |

The first row changes three things at once: the reply rule, the runtime and the quantisation. The
mlx-lm model answered 366 of the 400 test messages with a category name. It is right on 340 of those
366. The two validation rows are scored on validation because the prompt not chosen was never run on
test. Without examples in the prompt, `qwen3.6:27b-q4_K_M` is 14 messages below `gemma4:e4b`. Both
LoRA adapters take the invalid replies from 34 to 0. Theirs is the only LoRA configuration tried.

## Costs

| Classifier                       | Trained on                                         | Prompt        | Training minutes | Seconds per message | × ModernBERT |
|----------------------------------|----------------------------------------------------|---------------|------------------|---------------------|--------------|
| TF-IDF + logistic regression     | 1000 LLM-written                                   | —             | 0.71 s           | 0.00015             | 0.02         |
| TF-IDF + logistic regression     | 1000 LLM-written, steered by 200 validation counts | —             | 0.85 s           | 0.00016             | 0.02         |
| TF-IDF + logistic regression     | 1284 Banking77 train + validation                  | —             | 0.65 s           | 0.00015             | 0.02         |
| ModernBERT-base                  | 1000 LLM-written                                   | —             | 1.7              | 0.0073              | 1            |
| ModernBERT-base                  | 1000 LLM-written, steered by 200 validation counts | —             | 1.8              | 0.0077              | 1.0          |
| ModernBERT-base                  | 1284 Banking77 train + validation                  | —             | 2.0              | 0.0073              | 1            |
| UltraBERT-Embed-150M + LR        | 1000 LLM-written                                   | —             | 5.86 s           | 0.0107              | 1.5          |
| UltraBERT-Embed-150M + LR        | 1000 LLM-written, steered by 200 validation counts | —             | 5.50 s           | 0.0109              | 1.5          |
| UltraBERT-Embed-150M + LR        | 1284 Banking77 train + validation                  | —             | 6.52 s           | 0.0109              | 1.5          |
| `gemma4:e4b`                     | not trained                                        | `definitions` | none             | 0.63                | 85           |
| `gemma4:e4b`                     | not trained                                        | `fewshot`     | none             | 0.47                | 63           |
| `qwen3.6:27b-q4_K_M`             | not trained                                        | `names`       | none             | 1.5                 | 197          |
| `qwen3.6:27b-q4_K_M`             | not trained                                        | `fewshot`     | none             | 2.5                 | 335          |
| Gemma 4 E4B (MLX), no adapter    | not trained                                        | `definitions` | none             | 0.258               | 35           |
| Gemma 4 E4B (MLX) + LoRA adapter | 1000 LLM-written                                   | `definitions` | 13.6             | 0.235               | 32           |
| Gemma 4 E4B (MLX) + LoRA adapter | 1284 Banking77 train + validation                  | `definitions` | 17.9             | 0.235               | 32           |
| Gemma 4 31B (MLX), letter logits | not trained                                        | `fewshot`     | none             | 4.28                | 582          |
| TypeSafe System One              | not trained                                        | `definitions` | none             | 0.53                | 72           |
| TypeSafe System One              | not trained                                        | `fewshot`     | none             | 0.53                | 71           |

Seconds per message is the median time to classify one test message. "× ModernBERT" divides a
row's seconds per message by 0.00735 s, ModernBERT-base trained on train + validation
(`results/confusable/test/modernbert-real.txt`). On `distinct`, an LLM classifier takes 19 to 332
times ModernBERT-base's seconds per message. On `distinct`, a LoRA adapter takes 7.8 to 9.1
training minutes. No file under `results/` records how long `generate.py` took to write a training
set.

TypeSafe System One's seconds include the round trip to OpenRouter. Under the bound of question 1,
with the `fewshot` prompt, its certified coverage matches UltraBERT-Embed-150M + logistic
regression's at the same validation size, with 30 examples against 400 to 1084 training rows, at $54
per million messages and 0.53 s a message against free and 0.01 s. It is the one classifier billed
per message: a mean of 787 input tokens per message at $0.042 per million input tokens is $33 per
million messages, or $0.013 for the 400 (`results/confusable/test/jev-definitions.txt`). With the
`fewshot` prompt it is a mean of 1294 input tokens, $54 per million messages, or $0.022 for the 400
(`results/confusable/test/jev-fewshot.txt`).

## What did not help

- **Training on LLM-written text with nothing to steer the generator.** On `confusable`, the
  synthetic set cost TF-IDF + logistic regression 13.5 points of accuracy against train +
  validation. It cost ModernBERT-base 9.5 points. It cost UltraBERT-Embed-150M + logistic regression
  11.0 points.
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
     threshold can use. Under the bound, its threshold set on 200 validation messages keeps no test
     message, so neither takes work away from a person here.

UltraBERT-Embed-150M + logistic regression trained on the diagnosed set does not rank its answers
best. Its AUROC of 0.7794 is below the 0.8405 of the same classifier trained on the synthetic set.
It makes fewer errors: on validation it is right on 0.8850 of messages, against 0.8150 for the
synthetic-trained one. A classifier with fewer errors overall can keep more messages before its
error passes 5%, even when it ranks its answers worse.

Under the bound the one 5% threshold in [question
1](#1-how-much-of-the-work-can-it-do-without-a-person-checking)'s table, that of
UltraBERT-Embed-150M + logistic regression trained on train, is 0.992611. A threshold of 0.9 does
not separate the `definitions` letter-logits answers: the 378 above it are 0.8783 accurate.

Routing needs a labelled validation set as well as a classifier. With 0 hand labels nothing sets
the threshold, so a person checks every message. The 96% classifier, UltraBERT-Embed-150M +
logistic regression trained on train + validation, has no coverage figure here. Trained on train,
without the 200 validation rows, the same classifier scores 0.9525 and under the bound keeps
0.8600 of test messages at 0.0262 error.

200 validation messages are too few to certify 5% error for most classifiers. At a 5% target the
Clopper-Pearson bound admits no message for any classifier in [question
1](#1-how-much-of-the-work-can-it-do-without-a-person-checking)'s table but UltraBERT-Embed-150M +
logistic regression trained on train; of the label curve under question 3, it admits the 800 model
and some draws at 100, 200 and 400. TypeSafe System One's probabilities arrive in steps of 0.01, and
its highest step holds wrong answers: with the `definitions` prompt 111 of the 200 validation
messages score exactly 1.00 and 7 of those are wrong, with the `fewshot` prompt 118 and 5. The bound
on those 1.00 answers alone is over 5%, so on this draw no threshold qualifies. On three other draws
of 200, with the `fewshot` prompt, its certified coverage matches UltraBERT-Embed-150M + logistic
regression's at the same validation size, with 30 examples against 400 to 1084 training rows, at $54
per million messages and 0.53 s a message against free and 0.01 s.

The pairs are the whole difficulty. On `distinct` every classifier but two scores at least 0.9799.
On `confusable` most errors of every classifier but one are inside a pair.

The diagnosed notes cut errors on four pairs out of five. On the card pair the errors rose. The note
for that pair was written once from counts and not revised.

UltraBERT-Embed-150M's own training data could hold Banking77 text. That would make its results
look better than they would on new categories. A scan of its classification data found 18 Banking77
texts verbatim. One is a test message here. The scan is described under
[Details of the method](#details-of-the-method).

A gap of 1 to 3 messages of 400 is smaller than one standard error. The `fewshot` prompts' 27, 28
and 22 errors against ModernBERT-base's 25 are such a gap.

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
python score.py --set $S --llm gemma4:e4b --prompt fewshot      # test, with the prompts chosen on validation

python score.py --set $S --split dev --laya --prompt names
python score.py --set $S --split dev --laya --prompt definitions
python score.py --set $S --laya --prompt names                  # test, with the prompt chosen on validation

M=mlx-community/gemma-4-e4b-it-4bit
python finetune.py --set $S --train data/$S/synthetic.jsonl       --out models/$S/gemma-4-e4b-lora-synthetic
python finetune.py --set $S --train data/$S/banking77_train.jsonl --out models/$S/gemma-4-e4b-lora-real
python score.py --set $S --mlx $M
python score.py --set $S --mlx $M --adapter models/$S/gemma-4-e4b-lora-synthetic
python score.py --set $S --mlx $M --adapter models/$S/gemma-4-e4b-lora-real

D=results/confusable/confidence/dumps   # confusable only
python score.py --set confusable --write-train data/confusable/banking77_train_nodev.jsonl --no-dev
python train.py --set confusable --train data/confusable/banking77_train_nodev.jsonl --out models/confusable/ultrabert-embed-real-nodev --embed
# The label curve: per class in CLASS_SETS order, that class's rows shuffled with
# random.Random(seed) and the first k kept; seed 0 has no suffix and also draws k = 80.
python -c '
import json, random
from classes import CLASS_SETS
rows = [json.loads(l) for l in open("data/confusable/banking77_train_nodev.jsonl")]
for seed, ks in ((0, (5, 10, 20, 40, 80)), (1, (5, 10, 20, 40)), (2, (5, 10, 20, 40))):
    by = {}
    for name in CLASS_SETS["confusable"]:
        by[name] = [r for r in rows if r["label"] == name]
        random.Random(seed).shuffle(by[name])
    for k in ks:
        suffix = f"_s{seed}" if seed else ""
        with open(f"data/confusable/banking77_train_nodev_{10 * k}{suffix}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for name in by for r in by[name][:k])
'
for n in ultrabert-embed-real-nodev-{50,100,200,400,800} ultrabert-embed-real-nodev-{50,100,200,400}-s{1,2}; do
  k=${n#ultrabert-embed-real-nodev-}; k=${k/-s/_s}
  python train.py --set confusable --train data/confusable/banking77_train_nodev_$k.jsonl --out models/confusable/$n --embed
done
for n in tfidf-diagnosed modernbert-diagnosed ultrabert-embed-synthetic ultrabert-embed-diagnosed ultrabert-embed-real-nodev \
         ultrabert-embed-real-nodev-{50,100,200,400,800} ultrabert-embed-real-nodev-{50,100,200,400}-s{1,2}; do
  python score.py --set confusable --split dev --model models/confusable/$n --dump $D/$n-dev.jsonl
  python score.py --set confusable             --model models/confusable/$n --dump $D/$n-test.jsonl
done
python score.py --set confusable --split dev --mlx $M --reply logits --dump $D/gemma-4-e4b-it-4bit-definitions-logits-dev.jsonl
python score.py --set confusable             --mlx $M --reply logits --dump $D/gemma-4-e4b-it-4bit-definitions-logits-test.jsonl
python score.py --set confusable --split dev --mlx $M --reply logits --prompt fewshot --dump $D/gemma-4-e4b-it-4bit-fewshot-logits-dev.jsonl
python score.py --set confusable             --mlx $M --reply logits --prompt fewshot --dump $D/gemma-4-e4b-it-4bit-fewshot-logits-test.jsonl
python score.py --set confusable --split dev --laya --prompt names --dump $D/laya-names-dev.jsonl
python score.py --set confusable             --laya --prompt names --dump $D/laya-names-test.jsonl
python score.py --set confusable --split dev --jev --dump $D/jev-definitions-dev.jsonl   # needs TYPESAFE_API_KEY
python score.py --set confusable             --jev --dump $D/jev-definitions-test.jsonl
python score.py --set confusable --split dev --jev --prompt fewshot --dump $D/jev-fewshot-dev.jsonl
python score.py --set confusable             --jev --prompt fewshot --dump $D/jev-fewshot-test.jsonl
for d in $D/*-dev.jsonl; do
  n=$(basename $d -dev.jsonl)
  python confidence.py --set confusable --model-name $n --dev $d --test $D/$n-test.jsonl
done
# Validation size: thresholds of the 400-row models chosen on N of their rest rows.
for s in "" -s1 -s2; do
  m=ultrabert-embed-real-nodev-400$s
  python score.py --set confusable --split rest --train-file data/confusable/banking77_train_nodev_400${s/-/_}.jsonl \
    --model models/confusable/$m --dump $D/$m-rest.jsonl
  for n in 100 200 400; do for d in 0 1 2; do
    python confidence.py --set confusable --model-name $m-dev$n-ds$d --dev $D/$m-rest.jsonl \
      --test $D/$m-test.jsonl --dev-size $n --dev-seed $d
  done; done
done
# The same at 400 validation rows for the 30-, 50-, 100- and 200-row models; the 30-row subsets are
# drawn as the label curve's are, with k = 3 and seeds 0, 1 and 2.
for n in ultrabert-embed-real-nodev-30 ultrabert-embed-real-nodev-30-s{1,2}; do
  k=${n#ultrabert-embed-real-nodev-}; k=${k/-s/_s}
  python train.py --set confusable --train data/confusable/banking77_train_nodev_$k.jsonl --out models/confusable/$n --embed
  python score.py --set confusable --model models/confusable/$n --dump $D/$n-test.jsonl
done
for k in 30 50 100 200; do for s in "" -s1 -s2; do
  m=ultrabert-embed-real-nodev-$k$s
  python score.py --set confusable --split rest --train-file data/confusable/banking77_train_nodev_$k${s/-/_}.jsonl \
    --model models/confusable/$m --dump $D/$m-rest.jsonl
  for d in 0 1 2; do
    python confidence.py --set confusable --model-name $m-dev400-ds$d --dev $D/$m-rest.jsonl \
      --test $D/$m-test.jsonl --dev-size 400 --dev-seed $d
  done
done; done
python score.py --set confusable --split rest --train-file data/confusable/banking77_train_nodev_400.jsonl \
  --jev --prompt fewshot --dump $D/jev-fewshot-rest.jsonl
for n in 100 200 400; do for d in 0 1 2; do
  python confidence.py --set confusable --model-name jev-fewshot-dev$n-ds$d --dev $D/jev-fewshot-rest.jsonl \
    --test $D/jev-fewshot-test.jsonl --dev-size $n --dev-seed $d
done; done
M31=mlx-community/gemma-4-31b-it-4bit
python score.py --set confusable --mlx $M31 --reply logits --prompt fewshot \
  --dump $D/gemma-4-31b-it-4bit-fewshot-logits-test.jsonl
python score.py --set confusable --split rest --train-file data/confusable/banking77_train_nodev_400.jsonl \
  --mlx $M31 --reply logits --prompt fewshot --dump $D/gemma-4-31b-it-4bit-fewshot-logits-rest.jsonl
for n in 100 200 400; do for d in 0 1 2; do
  python confidence.py --set confusable --model-name gemma-4-31b-it-4bit-fewshot-logits-dev$n-ds$d \
    --dev $D/gemma-4-31b-it-4bit-fewshot-logits-rest.jsonl \
    --test $D/gemma-4-31b-it-4bit-fewshot-logits-test.jsonl --dev-size $n --dev-seed $d
done; done
```

`train.py` and `finetune.py` also write the minutes a run took, or the seconds if under a minute,
to `results/<set>/train/<name>.txt`.
