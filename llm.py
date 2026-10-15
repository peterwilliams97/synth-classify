"""llm.py classifies messages with a local LLM: `PROMPTS` builds the instructions, `classify` sends
them with one message at a time and gets back one class name. The LLM is an Ollama model, or an
mlx-lm model with an optional LoRA adapter trained by finetune.py. `classify_laya` asks the same of
the Laya decision model, which scores options instead of generating text. Like generate.py it must
not import `datasets` or read Banking77 — the `fewshot` examples are chosen by score.py and passed
in."""

import json
import sys
import time
import urllib.request

from classes import CLASS_SETS

INTRO = (
    "You sort messages sent to a banking app's customer support by intent. "
    "Answer with the one intent that fits the message best."
)


def names(classes, examples):
    return INTRO + "\n\nIntents:\n" + "\n".join(f"- {name}" for name in classes)


def definitions(classes, examples):
    lines = [f"- {name}: {definition}" for name, definition in classes.items()]
    return INTRO + "\n\nIntents:\n" + "\n".join(lines)


def fewshot(classes, examples):
    blocks = []
    for name, definition in classes.items():
        shots = "\n".join(f"    example: {text}" for text in examples[name])
        blocks.append(f"- {name}: {definition}\n{shots}")
    return INTRO + "\n\nIntents:\n" + "\n".join(blocks)


# Each builder takes a `CLASS_SETS` entry and {class name: [example message]}; only `fewshot`
# reads the examples.
PROMPTS = {"names": names, "definitions": definitions, "fewshot": fewshot}

# The `PROMPTS` key finetune.py trains adapters with, and so the one score.py gives every mlx-lm
# classifier, adapter or not.
MLX_PROMPT = "definitions"

# Reply length limit for an mlx-lm model, in tokens. The longest class name in `CLASS_SETS` is 11;
# the rest is room for the reasoning Gemma 4 sometimes writes before it answers.
MAX_REPLY_TOKENS = 512


def user_turn(instructions, text):
    """user_turn is the single user message an mlx-lm model sees: a `PROMPTS` result, then the
    message to classify. finetune.py trains on it and `classify_mlx` sends it, so both call this."""
    return f"{instructions}\n\nMessage: {text}"


# The Laya checkpoint `classify_laya` loads: the English one at the repo root, pinned at laya's own
# reviewed commit (`PINNED_REVISIONS` in the `laya` package).
LAYA_MODEL = "convaiinnovations/laya"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

# Laya's token budgets, per its model card's advice for questions with many options. Its defaults
# (512 and 192) cut every `definitions` option to 17 tokens, dropping the clause that separates the
# two intents of a `confusable` pair; at these values no option or message is cut.
LAYA_MAX_LEN = 1024
LAYA_HEAD_MAX_LEN = 512

# The `PROMPTS` keys `laya_question` builds; Laya has no examples to give, so no `fewshot`.
LAYA_PROMPTS = ["names", "definitions"]


def laya_question(classes, prompt):
    """laya_question is the one `choice` question Laya is asked about each message, for a
    `CLASS_SETS` entry and a `LAYA_PROMPTS` key: `names` gives the class names as the options,
    `definitions` each class name with its definition, which Laya renders as `name: definition`."""
    criteria = {
        name: definition if prompt == "definitions" else None
        for name, definition in classes.items()
    }
    return {
        "type": "choice",
        "instructions": "Which intent does this message sent to a banking app's customer support express?",
        "criteria": criteria,
    }


def classify(
    texts,
    class_set,
    model,
    prompt,
    examples=None,
    url="http://localhost:11434",
    mlx=False,
    adapter=None,
):
    """classify returns a (class name, seconds) pair for each of `texts`. `class_set` and `prompt`
    are keys of `CLASS_SETS` and `PROMPTS`. `model` is an Ollama model, or with `mlx` an mlx-lm
    model path or Hugging Face repo, loaded with the LoRA adapter in directory `adapter` if one is
    given. The seconds are the wall time of that message's request, so the first one includes
    Ollama loading `model`. An Ollama answer is always a class name; an mlx-lm reply that is not
    one comes back as None."""
    classes = CLASS_SETS[class_set]
    system = PROMPTS[prompt](classes, examples)
    if mlx:
        return classify_mlx(texts, classes, system, model, adapter)
    schema = {
        "type": "object",
        "properties": {"intent": {"type": "string", "enum": list(classes)}},
        "required": ["intent"],
    }
    out = []
    for text in texts:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 0},
        }
        req = urllib.request.Request(
            url + "/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as resp:
            content = json.load(resp)["message"]["content"]
        seconds = time.perf_counter() - start
        label = json.loads(content)["intent"]
        if label not in classes:
            raise ValueError(f"{model} answered {label!r}, not a class in {class_set}")
        out.append((label, seconds))
    return out


def classify_mlx(texts, classes, instructions, model, adapter):
    """classify_mlx is `classify` for an mlx-lm model. Nothing restricts the reply to a class name
    as Ollama's `format` schema does, and nothing stops the model reasoning first as Ollama's
    `"think": False` does. The reply is the text after the tokenizer's end-of-reasoning marker if
    there is one, stripped of surrounding whitespace, and it must equal a class name; any other
    reply, reasoning cut off at `MAX_REPLY_TOKENS` included, is printed on stderr and returned as
    None. Decoding is greedy, and the whole prompt is processed for every message — no prompt
    cache is kept between them."""
    # Imported here because importing mlx fails where there is no Metal device.
    from mlx_lm import generate, load

    lm, tok = load(model, adapter_path=adapter)
    out = []
    for text in texts:
        start = time.perf_counter()
        tokens = tok.apply_chat_template(
            [{"role": "user", "content": user_turn(instructions, text)}],
            add_generation_prompt=True,
        )
        reply = generate(lm, tok, tokens, max_tokens=MAX_REPLY_TOKENS)
        if tok.think_end:
            reply = reply.split(tok.think_end)[-1]
        reply = reply.strip()
        seconds = time.perf_counter() - start
        if reply not in classes:
            print(f"invalid reply {reply[-80:]!r} to {text!r}", file=sys.stderr)
            reply = None
        out.append((reply, seconds))
    return out


def classify_laya(texts, class_set, prompt):
    """classify_laya returns a (class name, probability, seconds) triple for each of `texts`: the
    option of `laya_question` with the highest probability, that probability, and the wall time of
    the `predict` call for that one message. Laya picks MPS when there is one."""
    import laya

    question = laya_question(CLASS_SETS[class_set], prompt)
    agent = laya.load(LAYA_MODEL, revision=LAYA_REVISION)
    print(f"device: {agent.device}", file=sys.stderr)
    out = []
    for text in texts:
        start = time.perf_counter()
        result = agent.predict(
            text, {"intent": question}, max_len=LAYA_MAX_LEN, head_max_len=LAYA_HEAD_MAX_LEN
        )
        seconds = time.perf_counter() - start
        answer = result["answers"]["intent"]
        out.append((answer["choice"], answer["probabilities"][answer["choice"]], seconds))
    return out
