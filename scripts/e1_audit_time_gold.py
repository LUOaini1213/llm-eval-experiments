"""Recheck the frozen E1 time labels without resampling an already judged experiment.

Writes a correction audit only. Original pool, split order, judge replies and the
first-run manifest remain historical evidence. Newly eligible answers need a new
judging run before they can support a revised benchmark result.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab import analysis as A, gold as G  # noqa: E402
from evalab.items import label_sgbus, load_jsonl  # noqa: E402


def audit(root=ROOT):
    specs = {s["id"]: s for s in load_jsonl(root / "data/e3/sgbus_items.jsonl") if s["kind"] == "time"}
    pool = load_jsonl(root / "data/e1/pool.jsonl")

    def label(row):
        return "refusal" if G.is_refusal(row["candidate"]) else label_sgbus(specs[row["qid"]], row["candidate"])

    changes = []
    corrected = []
    for row in pool:
        now = label(row) if row["source"] == "sgbus" and row["qid"] in specs else row["gold"]
        corrected.append(dict(row, gold=now))
        if now != row["gold"]:
            changes.append({"item_id": row["item_id"], "qid": row["qid"], "candidate": row["candidate"],
                            "reference": row["reference"], "before": row["gold"], "after": now})

    impact = {}
    for name, rows in (
        ("pilot", load_jsonl(root / "data/e1/items_pilot.jsonl")),
        ("main", A.load_main_items(root)),
        ("main_order", load_jsonl(root / "data/e1/items_main_order.jsonl")),
        ("robust", load_jsonl(root / "data/e1/items_robust.jsonl")),
    ):
        relevant = [r for r in rows if r["source"] == "sgbus" and r["qid"] in specs]
        impact[name] = {"items": len(rows), "time_labels_checked": len(relevant),
                        "changed_item_ids": [r["item_id"] for r in relevant if label(r) != r["gold"]]}
    pairs = load_jsonl(root / "data/e1/pairs.jsonl")
    relevant = [p for p in pairs if p["source"] == "sgbus" and p["qid"] in specs]
    impact["pairs"] = {"items": len(pairs), "time_labels_checked": 2 * len(relevant), "changed_item_ids": []}
    for p in relevant:
        for side, want in (("first", "correct"), ("second", "incorrect")):
            if label(dict(p, candidate=p[side])) != want:
                impact["pairs"]["changed_item_ids"].append(p["pair_id"] + "|" + side)
    paths = ["data/e1/pool.jsonl", "data/e1/items_main_order.jsonl", "results/e1/run_manifest.json"]
    return {
        "scope": "Post-hoc lexical time-label correction; frozen experimental samples are not resampled.",
        "input_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
        "pool_size": len(pool),
        "time_candidates_checked": sum(r["source"] == "sgbus" and r["qid"] in specs for r in pool),
        "original_pool_labels": dict(sorted(Counter(r["gold"] for r in pool).items())),
        "corrected_pool_labels": dict(sorted(Counter(r["gold"] for r in corrected).items())),
        "changes": changes,
        "frozen_set_impact": impact,
        "limitations": [
            "These are deterministic lexical labels, not new human annotations or semantic entailment judgments.",
            "Tentative statements (for example 'typically' or 'around') can still match the expected clock value.",
            "Originally excluded candidates have no E1 judge replies; no scores are imputed for them.",
        ],
    }


if __name__ == "__main__":
    report = audit()
    (ROOT / "results/e1/time_gold_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("time_candidates_checked", "original_pool_labels",
                                            "corrected_pool_labels", "frozen_set_impact")}, indent=1))
