"""scan_nemotron.py looks for Banking77 in nvidia/embed-nemotron-dataset-v1's
SyntheticClassificationData, the classification data dmis-lab/UltraBERT-Embed-150M's model card
says it was trained on. Its one argument is a directory holding that subset's queries.parquet and
corpus.parquet, downloaded from the dataset repo; run from the repo root with HF_HOME=$PWD/data/hf
HF_HUB_OFFLINE=1. Output is embed-nemotron-dataset-v1.txt beside this file."""

import re, sys, collections
import pyarrow.parquet as pq
from datasets import load_dataset
sys.path.insert(0, ".")
from classes import CLASS_SETS

pqdir = sys.argv[1]
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
b77 = {}
for split in ["train", "test"]:
    for r in load_dataset("mteb/banking77", split=split):
        b77.setdefault(norm(r["text"]), set()).add((split, r["label_text"]))
ours = {n for s in CLASS_SETS.values() for n in s}
labels = {r for v in b77.values() for _, r in v}
print("banking77 unique texts", len(b77), "labels", len(labels))

q = pq.read_table(f"{pqdir}/SyntheticClassificationData/queries.parquet", columns=["question", "pos_doc", "neg_doc"]).to_pylist()
corpus = {r["id"]: r["text"] for r in pq.read_table(f"{pqdir}/SyntheticClassificationData/corpus.parquet").to_pylist()}
print("queries", len(q), "corpus", len(corpus))

pat = re.compile(r"banking ?77|banking_data|polyai", re.I)
print("literal banking77/banking_data/polyai in queries:", sum(bool(pat.search(r["question"])) for r in q),
      "in corpus:", sum(bool(pat.search(t)) for t in corpus.values()))

exact = collections.Counter(); bankq = []; hits = []
for r in q:
    instr, _, query = r["question"].partition("\nQuery: ")
    if norm(query) in b77:
        hits.append((r["question"], b77[norm(query)]))
    if re.search(r"\bbank", instr, re.I):
        bankq.append((instr, query, corpus.get(r["pos_doc"][0]["id"])))
print("exact Banking77 text matches:", len(hits))
for h in hits: print("  ", h)
test_hits = [h for h in hits if any(s == "test" for s, _ in h[1])]
our_test = [h for h in test_hits if any(s == "test" and l in ours for s, l in h[1])]
print("  of which Banking77 test:", len(test_hits), " test in a CLASS_SETS class:", len(our_test))

print("queries with 'bank' in the instruction:", len(bankq), "distinct instructions:", len({b[0] for b in bankq}))
for i, c in collections.Counter(b[0] for b in bankq).most_common(15): print(f"  {c:5} {i[:150]!r}")

lab_forms = {l: {norm(l), norm(l.replace("_", " "))} for l in labels}
corp_norm = {norm(t) for t in corpus.values()}
lab_hits = sorted(l for l, forms in lab_forms.items() if forms & corp_norm)
print("Banking77 label names present as corpus label texts:", len(lab_hits), lab_hits)

# 8-gram overlap of Banking77 test texts (our classes) against queries under bank instructions
def grams(s, n=8):
    w = norm(s).split(); return {tuple(w[i:i+n]) for i in range(len(w) - n + 1)}
bg = set().union(*(grams(b[1]) for b in bankq)) if bankq else set()
test_ours = [t for t, v in b77.items() if any(s == "test" and l in ours for s, l in v)]
ov = [t for t in test_ours if grams(t) & bg]
print(f"our-class test texts sharing an 8-gram with a bank-instruction query: {len(ov)} of {len(test_ours)}")
for t in ov[:10]: print("  ", t)

print("our-class Banking77 test texts matched exactly:", our_test)

# A Banking77 task would show up as a query whose candidate labels include several Banking77 names.
ids = {i for i, t in corpus.items() if any(norm(t) in f for f in lab_forms.values())}
print("corpus label texts equal to a Banking77 label name:", sorted({corpus[i] for i in ids}))
per_query = collections.Counter()
for r in q:
    docs = {d["id"] for d in r["pos_doc"] + r["neg_doc"]}
    per_query[len({norm(corpus[d]).replace(" ", "_") for d in docs & ids})] += 1
print("queries by number of distinct Banking77 label names among their candidates:", dict(per_query))
