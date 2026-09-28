"""Remove long verbatim excerpts of the building codes from the committed E1 files.

Some RAG answers copy whole sentences from the retrieved passages. The SCDF and BCA terms do not allow their codes
to be republished, so any fm answer that repeats 20 or more consecutive words of its retrieved passages is replaced,
in every committed file, by a placeholder that records the run length and the SHA-256 of the original text.
The labels and judge replies are unchanged, so every result still recomputes. The unredacted files stay in local/.

Run locally after the judges (it needs local/fm_passages.jsonl):  python scripts/redact_code_text.py
"""
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab.items import load_jsonl, write_jsonl  # noqa: E402

MIN_RUN = 20
PREFIX = "[not published:"


def words(t):
    return re.findall(r"\w+", t.lower())


def longest_run(a: str, b: str) -> int:
    A, B = words(a), words(b)
    pos = {}
    for j, w in enumerate(B):
        pos.setdefault(w, []).append(j)
    best = 0
    for i, w in enumerate(A):
        for j in pos.get(w, ()):
            k = 0
            while i + k < len(A) and j + k < len(B) and A[i + k] == B[j + k]:
                k += 1
            best = max(best, k)
    return best


def placeholder(text: str, run: int) -> str:
    return (f"{PREFIX} this answer repeats {run} consecutive words of a building code whose terms do not allow "
            f"republication; sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}]")


def main():
    passages = {r["id"]: " ".join(r["top6"] + r["drop3"]) for r in load_jsonl(ROOT / "local" / "fm_passages.jsonl")}
    backup = ROOT / "local" / "unredacted"
    backup.mkdir(parents=True, exist_ok=True)
    memo = {}

    def fix(qid, text):
        if not text or text.startswith(PREFIX) or qid not in passages:
            return text
        key = (qid, text)
        if key not in memo:
            run = longest_run(text, passages[qid])
            memo[key] = placeholder(text, run) if run >= MIN_RUN else text
        return memo[key]

    d = ROOT / "data" / "e1"
    targets = [(ROOT / "results" / "e1" / "candidates.jsonl", ["answer"]), (d / "pool.jsonl", ["candidate"]),
               (d / "items_pilot.jsonl", ["candidate"]), (d / "items_main_order.jsonl", ["candidate"]),
               (d / "pairs.jsonl", ["first", "second"])]
    for path, fields in targets:
        if not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
        rows = load_jsonl(path)
        for r in rows:
            if r.get("source") == "fm":
                for f in fields:
                    r[f] = fix(r["qid"], r[f])
        write_jsonl(path, rows)
    amb = d / "ambiguous_items.csv"
    if not (backup / amb.name).exists():
        shutil.copy2(amb, backup / amb.name)
    with amb.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for r in rows[1:]:
        if r[0].startswith("fm-"):
            r[3] = fix(r[0].split("|")[0], r[3])
    with amb.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    n = sum(1 for v in memo.values() if v.startswith(PREFIX))
    print(f"redacted {n} distinct answers (runs of {MIN_RUN}+ words)")


if __name__ == "__main__":
    main()
