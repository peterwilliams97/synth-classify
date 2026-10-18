# synth-classify

I wanted to know three things about a text classifier for customer messages:

1. How much of the work can it do without a person checking?
2. How accurate is it?
3. How many hand-labelled examples does it need?

## Method

The data is [Banking77](https://huggingface.co/datasets/mteb/banking77), customer messages to a
bank, restricted to the `confusable` class set: 10 categories in 5 pairs, the two in a pair sharing
a topic, such as `verify_my_identity` / `why_verify_identity`. Test is its 400 Banking77 test
messages, train 1084 Banking77 train messages and validation 200 more; below, validation is 400 rows
drawn from the train rows a classifier was not trained on. A classifier labels a message with no
human check when its top probability is at least a threshold: the lowest validation probability at
which the one-sided 90% Clopper-Pearson upper bound on the validation error above it is at most α,
0.05. The plain threshold, set on the validation error itself, is not used because it does not hold
its target: its test error is over 5% in 12 of 36 runs, against 1 of the 28 in which the bound keeps
a message.

**Fraction of messages labelled with no human check, certified under 5% error, 400 validation rows**

| Classifier                                                 | Labels to build | Certified coverage     | Error                  | Seconds per message | $ per million messages |
|------------------------------------------------------------|-----------------|------------------------|------------------------|---------------------|------------------------|
| TypeSafe System One, `fewshot` (30 examples)               | 30 + 400        | 0.8658 (0.8400–0.9100) | 0.0297 (0.0265–0.0357) | 0.53                | 54                     |
| UltraBERT-Embed-150M + LR, 30 training rows                | 30 + 400        | 0.3278 (0.0000–0.6100) | 0.0267 (0.0071–0.0574) | 0.01                | free                   |
| UltraBERT-Embed-150M + LR, 100 training rows               | 100 + 400       | 0.7681 (0.4625–0.8975) | 0.0396 (0.0270–0.0543) | 0.01                | free                   |
| UltraBERT-Embed-150M + LR, 200 training rows               | 200 + 400       | 0.9103 (0.7825–1.0000) | 0.0453 (0.0288–0.0600) | 0.01                | free                   |
| UltraBERT-Embed-150M + LR, 400 training rows               | 400 + 400       | 0.8500 (0.7200–0.9300) | 0.0300 (0.0168–0.0490) | 0.01                | free                   |
| UltraBERT-Embed-150M + LR, 1084 (200 validation rows, one draw) | 1084 + 200 | 0.8600                 | 0.0262                 | 0.01                | free                   |

UltraBERT-Embed-150M + LR is logistic regression on frozen `dmis-lab/UltraBERT-Embed-150M`
embeddings, run on a laptop; TypeSafe System One is an API given the 30 examples in its question.
Labels to build is training or example labels + validation labels. Each cell is the mean, then the
min–max, over 9 runs for an UltraBERT-Embed-150M + LR row (3 training draws × 3 validation draws)
and 3 for System One; at 400 the validation draws hold 395 and 394 rows. Error is over the runs that
keep any message.

**Accuracy on the 400 test messages**

| Classifier                                     | Labels in training or prompt | Accuracy               |
|------------------------------------------------|------------------------------|------------------------|
| TypeSafe System One, `fewshot` (30 examples)   | 30                           | 0.9450                 |
| UltraBERT-Embed-150M + LR, 30 training rows    | 30                           | 0.8550 (0.8400–0.8850) |
| UltraBERT-Embed-150M + LR, 100 training rows   | 100                          | 0.9192 (0.9075–0.9250) |
| UltraBERT-Embed-150M + LR, 200 training rows   | 200                          | 0.9275 (0.9200–0.9400) |
| UltraBERT-Embed-150M + LR, 400 training rows   | 400                          | 0.9317 (0.9250–0.9375) |
| UltraBERT-Embed-150M + LR, 1084 training rows  | 1084                         | 0.9525                 |
| `gemma4:e4b`, `definitions`                    | 0                            | 0.9125                 |

The 30- to 400-row cells are the mean, then the min–max, over 3 training draws.

**Certified coverage against validation rows**

| Validation rows | UltraBERT-Embed-150M + LR, 400 training rows, 9 runs | TypeSafe System One, `fewshot`, 3 runs |
|-----------------|------------------------------------------------------|----------------------------------------|
| 100             | 0.3342 (0.0000–0.9750)                               | 0.2825 (0.0000–0.8475)                 |
| 200             | 0.7142 (0.0000–0.9450)                               | 0.8025 (0.6325–0.9350)                 |
| 400             | 0.8500 (0.7200–0.9300)                               | 0.8658 (0.8400–0.9100)                 |

## Conclusion

- With 30 labels as examples in its question and 400 to set the threshold, TypeSafe System One
  labels 87% of test messages with no human check at 3% error, and is 94.5% accurate. It is an API,
  $54 per million messages and 0.53 s a message.
- UltraBERT-Embed-150M + LR needs 200 training rows and 400 validation rows to do the same, 91% at
  4.5% error; with 30 training rows it certifies 33%. It is free and 0.01 s a message.
- Both need 400 validation rows. With 200, certified coverage ranges from 0 to 0.95 between draws;
  with 400, from 0.72 to 0.93.
- With 0 labels, `gemma4:e4b` is 91% accurate and nothing can be certified.

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
- `jev.py` — `classify_jev` asks TypeSafe System One the `jev_question` for each message, Laya's
  `definitions` question or the same options with the `fewshot` examples after each definition,
  with the key from `TYPESAFE_API_KEY`, and caches every response in `results/jev/<set>.jsonl`, so a
  re-run of `score.py --jev` sends nothing and reproduces its report.
- `finetune.py` — trains a LoRA adapter for `mlx-community/gemma-4-e4b-it-4bit` on a jsonl file
  (`--train`) and saves it to `--out`. Each row is a chat: the `definitions` prompt and the message
  as the user turn, the class name as the assistant turn.
- `score.py` — scores a saved model (`--model`), an Ollama model (`--llm MODEL --prompt NAME`), an
  mlx-lm model with the `definitions` prompt (`--mlx MODEL`, and `--adapter DIR` for a LoRA adapter,
  or `--reply logits` for letter logits, `--mlx` only, no adapter, with `--prompt
  definitions|fewshot`, default `definitions`), Laya (`--laya --prompt names|definitions`) or
  TypeSafe System One (`--jev`, `--prompt definitions|fewshot`, default `definitions`) on Banking77
  restricted to the class set: accuracy, macro F1, median seconds per message, per-class recall and
  a confusion matrix, printed and written to `results/<set>/<split>/<name>.txt`. An `--mlx` report
  also counts the invalid replies. `--split test` (the default) is the Banking77 test split.
  `--split dev` is validation, 20 messages per class sampled with a fixed seed from the Banking77
  train split; the `fewshot` examples are train messages outside validation. `--write-train` exports
  train + validation as jsonl, and with `--no-dev` leaves out the 200 validation rows, giving train.
  `--split rest --train-file JSONL` is the rows of `data/<set>/banking77_train_nodev.jsonl` not in
  `JSONL`, for a `--model` trained on a subset of it, or for `--jev`, less the `fewshot` examples
  with `--prompt fewshot`. `--dump FILE` also writes each message's
  probability over the class set, for `--model`, `--laya`, `--jev` and `--mlx --reply logits`.
- `confidence.py` — from a classifier's validation (`--dev`) and test `--dump` files: AUROC of the
  top probability against correctness, ECE before and after isotonic calibration fitted on
  validation, and test coverage and error at thresholds chosen on validation for α 0.02, 0.05 and
  0.10, with and without a Clopper-Pearson bound, written to `results/<set>/confidence/<name>.txt`.
  `--dev-size N --dev-seed S` fits on N rows of the validation file, N / 10 per class, drawn with
  seed S.

Banking77 is loaded from [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77), a
parquet copy. The original `PolyAI/banking77` is a dataset script, which `datasets` 5.1.0 refuses
with "Dataset scripts are no longer supported".

The other classifiers, every table, the class sets, the full method and how to run it are in [docs/details.md](docs/details.md).
