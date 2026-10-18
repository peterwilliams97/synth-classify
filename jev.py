"""jev.py classifies messages with TypeSafe's System One API, which answers a `choice` question
with a probability for every option. Each message is one POST /v1/systemone request: the message
as `state`, and as the one question `jev_question` for a `JEV_PROMPTS` key. Like llm.py it must
not import `datasets` or read Banking77; score.py passes the messages, their ids and the `fewshot`
examples in.

Every response is cached in results/jev/<set>.jsonl under the message's id and prompt, so dev and
test are
each sent once and a re-run of score.py reproduces its report, `seconds` line included, from the
cache. The key comes from `TYPESAFE_API_KEY`. `TYPESAFE_BASE_URL` and `TYPESAFE_MODEL` override
`BASE_URL` and `MODEL`."""

import json
import os
import sys
import time
import urllib.error
import urllib.request

from classes import CLASS_SETS
from llm import laya_question

# System One through OpenRouter: every row of results/jev/confusable.jsonl was asked of these.
BASE_URL = "https://openrouter.ai/api"
MODEL = "typesafe/jev-1.13"
# The `PROMPTS` keys `jev_question` builds; the report name is jev-<key>.
JEV_PROMPTS = ["definitions", "fewshot"]
# Input tokens are billed at this rate; output tokens are free.
DOLLARS_PER_MILLION_TOKENS = 0.042


def cache_path(class_set):
    return os.path.join("results", "jev", class_set + ".jsonl")


def jev_question(classes, prompt, examples):
    """jev_question is the `choice` question System One is asked about each message: for
    `definitions` it is llm.py's `laya_question`, Laya's, and for `fewshot` the same options with
    each definition followed by that class's `examples` as llm.py's `fewshot` lists them, so the
    examples are the ones the Ollama models see."""
    question = laya_question(classes, "definitions")
    if prompt == "fewshot":
        question["criteria"] = {
            name: definition + "".join(f"\nexample: {text}" for text in examples[name])
            for name, definition in classes.items()
        }
    return question


def read_cache(path):
    """read_cache returns {(message id, prompt): cached row}; a later row for a key replaces an
    earlier one. A row with no `prompt` field is a `definitions` row: those were cached before
    `fewshot` was asked."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["id"], r.get("prompt", "definitions")): r for r in rows}


def classify_jev(texts, ids, class_set, prompt, examples):
    """classify_jev returns a (class name, probabilities, seconds, input tokens, model) tuple for
    each of `texts`: the option System One chose, every option's probability as a list in
    `CLASS_SETS` order, the wall time of that message's request, its billed input tokens and the
    model that answered. `ids` names each message in the cache. `prompt` is a `JEV_PROMPTS` key and
    `examples` the `fewshot` examples, {class name: [message]}. A cached row is used when it was
    asked of the same model and the same question; otherwise the message is sent and the response
    appended to the cache."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("TYPESAFE_API_KEY is not set")
    url = os.environ.get("TYPESAFE_BASE_URL", BASE_URL).rstrip("/") + "/v1/systemone"
    model = os.environ.get("TYPESAFE_MODEL", MODEL)
    names = list(CLASS_SETS[class_set])
    question = jev_question(CLASS_SETS[class_set], prompt, examples)
    path = cache_path(class_set)
    cache = read_cache(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = []
    for text, mid in zip(texts, ids):
        row = cache.get((mid, prompt))
        if row is None or row["requested"] != model or row["question"] != question:
            body = {"model": model, "state": text, "questions": {"intent": question}}
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            )
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    response = json.load(resp)
            except urllib.error.HTTPError as e:
                sys.exit(f"{url}: HTTP {e.code} on {mid}: {e.read().decode(errors='replace')}")
            seconds = time.perf_counter() - start
            row = {
                "id": mid,
                "prompt": prompt,
                "requested": model,
                "question": question,
                "seconds": seconds,
                "response": response,
            }
            # Appended per message, so an interrupted run resumes where it stopped.
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
        response = row["response"]
        answer = response["answers"]["intent"]
        if answer["choice"] not in names:
            raise ValueError(f"{mid}: System One chose {answer['choice']!r}, not a {class_set} class")
        vector = [float(answer["probabilities"][name]) for name in names]
        tokens = response["usage"]["input_tokens"]
        out.append((answer["choice"], vector, row["seconds"], tokens, response["model"]))
    return out
