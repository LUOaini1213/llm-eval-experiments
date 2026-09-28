"""Label every candidate answer with its deterministic gold and split into pilot, main, robustness and pair sets.

Inputs:  results/e1/candidates.jsonl (natural answers), data/e1/fm_gold.jsonl, data/e3/sgbus_items.jsonl,
         sg-transit-geo-assistant's cached eval results (read-only; SG_TRANSIT_ROOT to override).
Outputs (data/e1/):
  pool.jsonl              every natural candidate with its label (correct / incorrect / ambiguous / refusal)
  ambiguous_items.csv     candidates the deterministic check could not decide (excluded from analysis)
  items_pilot.jsonl       pilot items (disjoint from main; used only for timing, judge choice and power analysis)
  items_main_order.jsonl  every remaining non-ambiguous natural item in a seeded, stratified order; the main set is
                          the first N rows, N fixed in docs/PREREGISTRATION.md before any main-set judging
  items_robust.jsonl      controlled 2x2 perturbations (terse/verbose x correct/wrong)
  pairs.jsonl             one correct and one wrong natural answer to the same question
"""
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab.gold import is_refusal  # noqa: E402
from evalab.items import (fm_wrong_reference, label_fm, label_sgbus, load_jsonl, perturbations,  # noqa: E402
                          sgbus_reference, sgbus_wrong_reference, write_jsonl)

SEED = 20260928
TRANSIT = Path(os.environ.get("SG_TRANSIT_ROOT", Path(__file__).resolve().parents[2] / "sg-transit-geo-assistant"))
OUT = ROOT / "data" / "e1"
PILOT_PER_STRATUM = 8           # 3 sources x 2 labels x 8 = 48 pilot items
ROBUST_SGBUS_QUESTIONS = 60


def sql_schema_text() -> str:
    """Compact schema from sg-transit-geo-assistant's backend/app/schema.py (tables and columns only)."""
    sys.dont_write_bytecode = True
    import importlib.util
    spec = importlib.util.spec_from_file_location("transit_schema", TRANSIT / "backend" / "app" / "schema.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return "\n".join(f"{t}({', '.join(c for c in v['columns'])})" for t, v in mod.TABLES.items())


def sql_candidates() -> list[dict]:
    """Every distinct (question, SQL) the sg-transit evaluation produced, with its execution-accuracy label."""
    seen = {}
    for f in sorted((TRANSIT / "eval" / "results").glob("results*.json")):
        for r in json.loads(f.read_text(encoding="utf-8"))["records"]:
            if r.get("kind") != "answerable" or r.get("status") != "answered" or not r.get("sql"):
                continue
            key = (f.stem.startswith("results_heldout"), r["question"], re.sub(r"\s+", " ", r["sql"].strip().rstrip(";")).lower())
            if key in seen:
                if seen[key]["gold"] != ("correct" if r["correct"] else "incorrect"):
                    seen[key]["conflict"] = True
                continue
            seen[key] = {"qid": f"sql-{'h' if 'heldout' in f.stem else 'm'}-{r['id']}", "source": "sql", "task": "sql", "question": r["question"],
                         "candidate": r["sql"].strip(), "reference": r["gold_sql"],
                         "gold": "correct" if r["correct"] else "incorrect",
                         "generator": r.get("engine"), "config": f.stem, "conflict": False}
    rows = [v for v in seen.values() if not v["conflict"]]
    return rows


def main():
    rng = random.Random(SEED)
    fm = {q["id"]: q for q in load_jsonl(OUT / "fm_gold.jsonl")}
    bus = {q["id"]: q for q in load_jsonl(ROOT / "data" / "e3" / "sgbus_items.jsonl")}
    cands = load_jsonl(ROOT / "results" / "e1" / "candidates.jsonl")
    if any(c["answer"].startswith("[not published:") for c in cands):
        raise SystemExit("candidates.jsonl is the redacted, published copy (see scripts/redact_code_text.py); "
                         "labels must be computed from the unredacted answers in local/unredacted/")

    pool = []
    for c in cands:
        if c["source"] == "fm":
            spec = fm[c["qid"]]
            q, ref = spec["question"], spec["reference"]
            label = "refusal" if is_refusal(c["answer"]) else label_fm(spec, c["answer"])
        else:
            it = bus[c["qid"]]
            q, ref = it["question"], sgbus_reference(it)
            label = "refusal" if is_refusal(c["answer"]) else label_sgbus(it, c["answer"])
        pool.append({"item_id": c["cand_id"], "qid": c["qid"], "source": c["source"], "task": "qa", "question": q,
                     "reference": ref, "candidate": c["answer"], "gold": label, "generator": c["generator"],
                     "config": c["config"], "style": "natural"})
    schema = sql_schema_text()
    for s in sql_candidates():
        s.pop("conflict")
        pool.append({"item_id": f"{s['qid']}|{s['generator']}|{s['config']}", "style": "natural",
                     "schema": schema, **s})
    # identical answers to the same question are judged once
    dedup, seen = [], set()
    for p in pool:
        k = (p["qid"], re.sub(r"\s+", " ", p["candidate"].strip().lower()))
        if k in seen:
            continue
        seen.add(k)
        dedup.append(p)
    pool = dedup
    write_jsonl(OUT / "pool.jsonl", pool)
    with (OUT / "ambiguous_items.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["item_id", "question", "reference", "candidate"])
        for p in pool:
            if p["gold"] == "ambiguous":
                w.writerow([p["item_id"], p["question"], p["reference"], p["candidate"]])

    usable = [p for p in pool if p["gold"] in ("correct", "incorrect")]
    strata = defaultdict(list)
    for p in usable:
        strata[(p["source"], p["gold"])].append(p)
    for k in sorted(strata):
        rng.shuffle(strata[k])
    pilot = []
    for k in sorted(strata):
        pilot += strata[k][:PILOT_PER_STRATUM]
        strata[k] = strata[k][PILOT_PER_STRATUM:]
    pilot_q = {p["qid"] for p in pilot}
    # interleave strata so that any prefix of the order is close to balanced by source and label
    order, keys = [], sorted(strata)
    while any(strata[k] for k in keys):
        for k in keys:
            if strata[k]:
                order.append(strata[k].pop())
    write_jsonl(OUT / "items_pilot.jsonl", pilot)
    write_jsonl(OUT / "items_main_order.jsonl", order)

    # controlled perturbations: all fm questions and a template-stratified sample of bus questions
    by_t = defaultdict(list)
    for it in bus.values():
        by_t[it["template"]].append(it)
    chosen = []
    for t in sorted(by_t):
        r = random.Random(f"{SEED}-{t}")
        chosen += r.sample(by_t[t], min(len(by_t[t]), ROBUST_SGBUS_QUESTIONS // len(by_t)))
    robust = []
    for spec in fm.values():
        q = {"id": spec["id"], "reference": spec["reference"], "wrong": fm_wrong_reference(spec)}
        for v in perturbations(q):
            lab = label_fm(spec, v["answer"])
            assert lab == ("correct" if v["correct"] else "incorrect"), (spec["id"], v, lab)
            robust.append({"item_id": f"{spec['id']}|{v['style']}|{'right' if v['correct'] else 'wrong'}",
                           "qid": spec["id"], "source": "fm", "task": "qa", "question": spec["question"],
                           "reference": spec["reference"], "candidate": v["answer"], "gold": lab,
                           "style": v["style"], "generator": "template", "config": "perturbation"})
    for it in chosen:
        q = {"id": it["id"], "reference": sgbus_reference(it), "wrong": sgbus_wrong_reference(it)}
        for v in perturbations(q):
            lab = label_sgbus(it, v["answer"])
            assert lab == ("correct" if v["correct"] else "incorrect"), (it["id"], v, lab)
            robust.append({"item_id": f"{it['id']}|{v['style']}|{'right' if v['correct'] else 'wrong'}",
                           "qid": it["id"], "source": "sgbus", "task": "qa", "question": it["question"],
                           "reference": q["reference"], "candidate": v["answer"], "gold": lab,
                           "style": v["style"], "generator": "template", "config": "perturbation"})
    write_jsonl(OUT / "items_robust.jsonl", robust)

    # pairs: one correct and one wrong natural answer to the same question, not from pilot questions
    byq = defaultdict(lambda: {"correct": [], "incorrect": []})
    for p in usable:
        if p["qid"] not in pilot_q:
            byq[p["qid"]][p["gold"]].append(p)
    pairs = []
    for qid in sorted(byq):
        g = byq[qid]
        if g["correct"] and g["incorrect"]:
            r = random.Random(f"{SEED}-pair-{qid}")
            a, b = r.choice(g["correct"]), r.choice(g["incorrect"])
            pairs.append({"pair_id": qid, "qid": qid, "source": a["source"], "question": a["question"],
                          "reference": a["reference"], "first": a["candidate"], "second": b["candidate"],
                          "first_id": a["item_id"], "second_id": b["item_id"]})
    write_jsonl(OUT / "pairs.jsonl", pairs)

    summary = {"pool": Counter((p["source"], p["gold"]) for p in pool),
               "pilot": Counter((p["source"], p["gold"]) for p in pilot),
               "main_order": Counter((p["source"], p["gold"]) for p in order),
               "robust": Counter((p["source"], p["style"], p["gold"]) for p in robust),
               "pairs": Counter((p["source"],) for p in pairs)}
    summary = {k: {"|".join(kk): v for kk, v in c.items()} for k, c in summary.items()}
    (ROOT / "results" / "e1").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "e1" / "item_counts.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
