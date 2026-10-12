"""generate.py writes data/synthetic.jsonl: customer messages a local Ollama model invents for each
class in `CLASSES`. The model sees the class name and definition and nothing else — this file must
not import `datasets` or read any Banking77 text, or the synthetic-vs-real comparison is void."""

import argparse
import json
import sys
import urllib.request

from classes import CLASSES

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


def ask(url, model, name, definition, style, seed):
    prompt = (
        f"Write {BATCH} different messages that customers of a banking app might send to "
        f"customer support.\n"
        f"Every message has this intent: {name} — {definition}\n"
        f"Style for this batch: {style}.\n"
        f"Vary the wording, length and situation. One message per array item, no numbering."
    )
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
    ap.add_argument("--out", default="data/synthetic.jsonl")
    ap.add_argument("--url", default="http://localhost:11434")
    args = ap.parse_args()

    rows = []
    for name, definition in CLASSES.items():
        seen = {}  # lowercased text -> text, so near-identical repeats are dropped
        for attempt in range(4 * args.per_class // BATCH):
            if len(seen) >= args.per_class:
                break
            style = STYLES[attempt % len(STYLES)]
            for text in ask(args.url, args.model, name, definition, style, seed=attempt):
                seen.setdefault(text.lower(), text)
            print(f"{name}: {min(len(seen), args.per_class)}/{args.per_class}", file=sys.stderr)
        if len(seen) < args.per_class:
            sys.exit(f"{name}: only {len(seen)} distinct messages, wanted {args.per_class}")
        rows += [{"text": t, "label": name} for t in list(seen.values())[: args.per_class]]

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
