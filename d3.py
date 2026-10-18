"""d3.py classifies messages with a Decision 3.0 model from vLLM Semantic Router, such as
`vllm-sr/d3-flash`, run locally through transformers: the model answers a `choice` question with a
probability for every option, as TypeSafe's System One API does, and is asked jev.py's
`jev_question`. Like llm.py it must not import `datasets` or read Banking77; score.py passes the
messages, their ids and the `fewshot` examples in.

The model's code runs with `trust_remote_code` and needs transformers 5.17.0, which is not the
version in requirements.txt, so it runs in its own venv, `.venv-d3`, made from requirements-d3.txt.

Every response is cached in results/d3/<set>.jsonl under the message's id, the prompt and the model
id, so a re-run of score.py reproduces its report, `seconds` line included, without loading the
model."""

import json
import os
import sys
import time

from classes import CLASS_SETS
from jev import JEV_PROMPTS, jev_question

# The prompts `classify_d3` takes are jev.py's; the report name is <model basename>-<key>.
D3_PROMPTS = JEV_PROMPTS
# The commit each model's code and weights are loaded at, since the code runs with
# `trust_remote_code`; a model not listed loads at its main branch.
REVISIONS = {"vllm-sr/d3-flash": "13dc1cc8ef23caefa765907e1d764751e5629f7d"}


def cache_path(class_set):
    return os.path.join("results", "d3", class_set + ".jsonl")


def read_cache(path):
    """read_cache returns {(message id, prompt, model id): cached row}; a later row for a key
    replaces an earlier one."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["id"], r["prompt"], r["model"]): r for r in rows}


def classify_d3(texts, ids, class_set, prompt, examples, model_id):
    """classify_d3 returns a (class name, probabilities, seconds) tuple for each of `texts`: the
    option the model chose, every option's probability as a list in `CLASS_SETS` order, and the wall
    time of that message's `system_one` call. `ids` names each message in the cache. `prompt` is a
    `D3_PROMPTS` key and `examples` the `fewshot` examples, {class name: [message]}. A cached row is
    used when it was asked of the same revision and the same question; the model is loaded, on MPS
    in bfloat16, only if some message is not cached."""
    names = list(CLASS_SETS[class_set])
    question = jev_question(CLASS_SETS[class_set], prompt, examples)
    revision = REVISIONS.get(model_id)
    path = cache_path(class_set)
    cache = read_cache(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = None
    out = []
    for text, mid in zip(texts, ids):
        row = cache.get((mid, prompt, model_id))
        if row is None or row["revision"] != revision or row["question"] != question:
            if model is None:
                model = load(model_id, revision)
            start = time.perf_counter()
            response = model.system_one(state=text, questions={"intent": question})
            seconds = time.perf_counter() - start
            row = {
                "id": mid,
                "prompt": prompt,
                "model": model_id,
                "revision": revision,
                "question": question,
                "seconds": seconds,
                "response": response,
            }
            # Appended per message, so an interrupted run resumes where it stopped.
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
        answer = row["response"]["answers"]["intent"]
        if "error" in answer:
            sys.exit(f"{mid}: {model_id} answered {answer['error']!r}")
        if answer["choice"] not in names:
            raise ValueError(f"{mid}: {model_id} chose {answer['choice']!r}, not a {class_set} class")
        vector = [float(answer["probabilities"][name]) for name in names]
        out.append((answer["choice"], vector, row["seconds"]))
    return out


def load(model_id, revision):
    """load returns the model on MPS in bfloat16, and exits rather than fall back to the CPU."""
    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModel

    if not torch.backends.mps.is_available():
        sys.exit("d3.py: MPS is not available; it does not fall back to the CPU")
    print("device: mps", file=sys.stderr)
    # The model's runtime calls snapshot_download for a Hub id, which lists the repository even
    # under HF_HUB_OFFLINE; a cached snapshot is passed as its directory, which it reads as is.
    cached = try_to_load_from_cache(model_id, "config.json", revision=revision)
    source = os.path.dirname(cached) if isinstance(cached, str) else model_id
    return AutoModel.from_pretrained(
        source, trust_remote_code=True, revision=revision, device="mps", dtype=torch.bfloat16
    )
