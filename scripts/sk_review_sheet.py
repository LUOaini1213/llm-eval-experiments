"""List the shown replies that still need a review label (data/sk/review_labels.jsonl), with the detector's verdict.

    python scripts/sk_review_sheet.py        # writes local/review_todo.jsonl (git-ignored) and prints a count

Identical texts for the same item and configuration share one label, so repeats that reproduce run 1 need no
new review.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab import sk_analysis as A  # noqa: E402


def main():
    data = A.Data.load(ROOT)
    todo, seen = [], set()
    for (config, run, iid), r in sorted(data.responses.items()):
        it = data.items[iid]
        if it["test"] not in A.REVIEWED_TESTS or not r["shown"]:
            continue
        k = (iid, config, A.text_key(r["draft"]))
        if k in data.reviews or k in seen:
            continue
        seen.add(k)
        todo.append({"id": iid, "config": config, "text_key": k[2], "test": it["test"], "prompt": it["prompt"],
                     "detector": A.detector_hit(r, it, "shown") if it["test"] != "type and ease of elicitation" else None,
                     "reply": r["draft"]})
    out = ROOT / "local" / "review_todo.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in todo), encoding="utf-8")
    print(len(todo), "replies to review ->", out)


if __name__ == "__main__":
    main()
