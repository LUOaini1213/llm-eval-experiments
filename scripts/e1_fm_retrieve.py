"""Retrieve passages for the fm-knowledge-assistant questions (read-only use of that repository).

Run with a Python that has fm-knowledge-assistant's requirements (duckdb, sentence-transformers):

    FMKA_ROOT=../fm-knowledge-assistant python scripts/e1_fm_retrieve.py

Two retrieval configurations are stored per question:

- "top6":     the six best hybrid-search passages, as the assistant itself uses them;
- "drop3":    passages ranked 4 to 9, i.e. the best three removed. This simulates a retrieval miss, which is how
              a RAG system produces wrong but plausible answers (the model uses a number from a nearby clause).

The passage text comes from SCDF/BCA/NEA codes whose terms do not allow republication, so the output goes to
local/ (git-ignored) and never into the repository. Only the model answers generated from it are kept.
"""
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # do not leave __pycache__ in the other repository
ROOT = Path(__file__).resolve().parents[1]
FMKA = Path(os.environ.get("FMKA_ROOT", Path(__file__).resolve().parents[2] / "fm-knowledge-assistant"))
sys.path.insert(0, str(FMKA / "src"))
os.chdir(FMKA)  # fmka resolves its database relative to its own tree

from fmka.index import Index  # noqa: E402
from fmka.search import search  # noqa: E402


def main():
    ix = Index()
    qs = [json.loads(line) for line in (ROOT / "data" / "e1" / "fm_gold.jsonl").read_text(encoding="utf-8").splitlines()]
    out = ROOT / "local" / "fm_passages.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in qs:
            hits = search(ix, q["question"], k=9, mode="hybrid")
            rows = [ix.row(i) for i, _, _ in hits]
            texts = [f"({ix.titles.get(r['doc_id'], r['doc_id'])}, clause {r['clause'] or '-'}, page {r['page']}) "
                     f"{r['heading']} {r['text']}".strip() for r in rows]
            f.write(json.dumps({"id": q["id"], "top6": texts[:6], "drop3": texts[3:9],
                                "top6_ids": [int(i) for i, _, _ in hits[:6]],
                                "drop3_ids": [int(i) for i, _, _ in hits[3:9]]}, ensure_ascii=False) + "\n")
    print("wrote", out, len(qs))


if __name__ == "__main__":
    main()
