"""generate.py writes data/<set>/synthetic.jsonl: customer messages a local Ollama model invents for
each class in the `CLASS_SETS` entry named by `--set`. The model sees class names, definitions and
the instructions in this file and nothing else — this file must not import `datasets` or read any
Banking77 text, or the synthetic-vs-real comparison is void.

`--method` picks the request: `plain` (`plain_prompt`) gives one class's name and definition;
`contrast` (`contrast_prompt`) gives the class and the other one in its pair, for a set listed pair
by pair (`confusable`, `pair`), writes data/<set>/synthetic-contrast.jsonl, and writes every distinct request text beside
it as synthetic-contrast.prompt.txt."""

import argparse
import json
import os
import sys
import urllib.request

from classes import CLASS_SETS

BATCH = 20

# One style is added to each request so the batches for a class differ from each other.
STYLES = [
    "short and blunt, a few words to one sentence",
    "polite and complete sentences",
    "casual, lowercase, the way people type on a phone",
    "phrased as direct questions",
    "frustrated or worried in tone",
    "a little rambling, with some background before the point",
    "phrased as statements of a problem, not questions",
    "from someone unsure of the right banking terms",
]

SCHEMA = {
    "type": "object",
    "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
    "required": ["messages"],
}


# CONTRAST_NOTES is what `contrast_prompt` adds for a class, keyed by class set and then class name:
# the ways a message of that class can be told from the other one in its pair. Only `pair` has
# notes; `confusable` gets the general prompt alone, so its other four pairs are not shaped by
# anything written for the identity pair.
CONTRAST_NOTES = {
    "pair": {
        "why_verify_identity": (
            "The messages without those words should question whether verification is needed, "
            "complain about it, or ask what happens without it."
        ),
        "verify_my_identity": (
            "A message may mention that verification was asked for, but it asks only how to do it, "
            "what to submit, or where the customer is in the process."
        ),
    },
}


def plain_prompt(class_set, name, style):
    """plain_prompt asks for messages of class `name` of `CLASS_SETS[class_set]` given only its
    name, definition and `style`."""
    classes = CLASS_SETS[class_set]
    return (
        f"Write {BATCH} different messages that customers of a banking app might send to "
        f"customer support.\n"
        f"Every message has this intent: {name} — {classes[name]}\n"
        f"Style for this batch: {style}.\n"
        f"Vary the wording, length and situation. One message per array item, no numbering."
    )


def contrast_prompt(class_set, name, style):
    """contrast_prompt asks for messages of class `name` that could not be read as the other class
    in its pair, showing both definitions, and for at least half of them to avoid the words that
    signal the class most obviously, the words of its name among them."""
    classes = CLASS_SETS[class_set]
    names = list(classes)
    # Assumes the set lists its classes pair by pair, as `confusable` and `pair` do, so the other
    # class in a pair is the neighbour at index `i ^ 1`.
    other = names[names.index(name) ^ 1]
    note = CONTRAST_NOTES.get(class_set, {}).get(name, "")
    return (
        f"Write {BATCH} different messages that customers of a banking app might send to "
        f"customer support.\n"
        f"These two intents are easy to confuse:\n"
        f"- {name}: {classes[name]}\n"
        f"- {other}: {classes[other]}\n"
        f"Every message has the intent {name}. Each message must belong to {name} and could not be "
        f"read as {other}.\n"
        f"At least half of the messages must express the intent without the words that most "
        f"obviously signal it, in particular the words that make up the name {name}.\n"
        + (f"{note}\n" if note else "")
        + f"Vary the length from 4 to 30 words.\n"
        f"Style for this batch: {style}.\n"
        f"Vary the wording and situation. One message per array item, no numbering."
    )


METHODS = {"plain": plain_prompt, "contrast": contrast_prompt}


def ask(url, model, prompt, seed):
    """ask makes one `/api/chat` request with `prompt`, which asks for `BATCH` messages. `SCHEMA`
    as `format` makes the reply a JSON object whose `messages` is an array of strings, and `seed`
    varies across a class's requests so they don't repeat one another at temperature 1.0. Returns
    the stripped, non-empty messages, which can repeat; `main` drops repeats."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "options": {"temperature": 1.0, "seed": seed},
    }
    req = urllib.request.Request(
        url + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        content = json.load(resp)["message"]["content"]
    return [m.strip() for m in json.loads(content)["messages"] if m.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--set", choices=list(CLASS_SETS), default="distinct")
    ap.add_argument("--method", choices=list(METHODS), default="plain")
    ap.add_argument("--out", help="default data/<set>/synthetic.jsonl, or synthetic-<method>.jsonl")
    ap.add_argument("--url", default="http://localhost:11434")
    args = ap.parse_args()
    if args.method == "contrast" and args.set == "distinct":
        ap.error("--method contrast needs a set listed pair by pair; distinct is not")
    suffix = "" if args.method == "plain" else f"-{args.method}"
    out = args.out or f"data/{args.set}/synthetic{suffix}.jsonl"

    rows, prompts = [], {}
    for name in CLASS_SETS[args.set]:
        seen = {}  # lowercased text -> text, so near-identical repeats are dropped
        for attempt in range(4 * args.per_class // BATCH):
            if len(seen) >= args.per_class:
                break
            style = STYLES[attempt % len(STYLES)]
            prompt = METHODS[args.method](args.set, name, style)
            prompts.setdefault(prompt)
            for text in ask(args.url, args.model, prompt, seed=attempt):
                seen.setdefault(text.lower(), text)
            print(f"{name}: {min(len(seen), args.per_class)}/{args.per_class}", file=sys.stderr)
        if len(seen) < args.per_class:
            sys.exit(f"{name}: only {len(seen)} distinct messages, wanted {args.per_class}")
        rows += [{"text": t, "label": name} for t in list(seen.values())[: args.per_class]]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}", file=sys.stderr)
    if args.method != "plain":
        path = os.path.splitext(out)[0] + ".prompt.txt"
        with open(path, "w") as f:
            f.write("\n\n----\n\n".join(prompts) + "\n")
        print(f"wrote {len(prompts)} distinct prompts to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
