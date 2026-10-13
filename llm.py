"""llm.py classifies messages with a local Ollama model and no training: `PROMPTS` builds the
instructions, `classify` sends them with one message at a time and gets back one class name. Like
generate.py it must not import `datasets` or read Banking77 — the `fewshot` examples are chosen by
score.py and passed in."""

import json
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


def classify(texts, class_set, model, prompt, examples=None, url="http://localhost:11434"):
    """classify returns a (class name, seconds) pair for each of `texts`. `class_set` and `prompt`
    are keys of `CLASS_SETS` and `PROMPTS`. The seconds are the wall time of that message's
    request, so the first one includes Ollama loading `model`."""
    classes = CLASS_SETS[class_set]
    system = PROMPTS[prompt](classes, examples)
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
