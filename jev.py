"""jev.py classifies messages with TypeSafe's System One API, which answers a `choice` question
with a probability for every option. Each message is one POST /v1/systemone request: the message
as `state`, and as the one question llm.py's `laya_question` with the `definitions` prompt, the
question Laya is asked. Like llm.py it must not import `datasets` or read Banking77; score.py
passes the messages and their ids in.

Every response is cached in results/jev/<set>.jsonl under the message's id, so dev and test are
each sent once and a re-run of score.py reproduces its report, `seconds` line included, from the
cache. The key comes from `TYPESAFE_API_KEY`, the base URL from `TYPESAFE_BASE_URL` and the model
from `TYPESAFE_MODEL`."""

import json
import os
import sys
import time
import urllib.error
import urllib.request

from classes import CLASS_SETS
from llm import laya_question

BASE_URL = "https://api.typesafe.ai"
MODEL = "jev-latest"
# The `PROMPTS` key of the question asked; the report name is jev-<this>.
JEV_PROMPT = "definitions"
# Input tokens are billed at this rate; output tokens are free.
DOLLARS_PER_MILLION_TOKENS = 0.042


def cache_path(class_set):
    return os.path.join("results", "jev", class_set + ".jsonl")


def read_cache(path):
    """read_cache returns {message id: cached row}; a later row for an id replaces an earlier one."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: r for r in rows}


def classify_jev(texts, ids, class_set):
    """classify_jev returns a (class name, probabilities, seconds, input tokens, model) tuple for
    each of `texts`: the option System One chose, every option's probability as a list in
    `CLASS_SETS` order, the wall time of that message's request, its billed input tokens and the
    model that answered. `ids` names each message in the cache. A cached row is used when it was
    asked of the same `TYPESAFE_MODEL` and the same question; otherwise the message is sent and the
    response appended to the cache."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("TYPESAFE_API_KEY is not set")
    url = os.environ.get("TYPESAFE_BASE_URL", BASE_URL).rstrip("/") + "/v1/systemone"
    model = os.environ.get("TYPESAFE_MODEL", MODEL)
    names = list(CLASS_SETS[class_set])
    question = laya_question(CLASS_SETS[class_set], JEV_PROMPT)
    path = cache_path(class_set)
    cache = read_cache(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = []
    for text, mid in zip(texts, ids):
        row = cache.get(mid)
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
