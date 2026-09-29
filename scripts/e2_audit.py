"""E2: quality audit of Project Moonshot's Singapore-context benchmark datasets.

    python scripts/e2_audit.py            # automated checks + model consensus (if the Moonshot run exists) + report

Inputs:
- external/moonshot-data (git clone of aiverify-foundation/moonshot-data; the commit is recorded in the output)
- results/e2/moonshot_run.json (optional): per-item predictions of the five local models on Moonshot's
  singapore-facts-tf and singapore-facts-mcq recipes, extracted by scripts/e2_moonshot_run.py
- data/e2/verified.json: suspected errors checked by hand against an official source (URL and date)

Outputs (results/e2/):
- datasets.json      licence, size and name check of every Singapore-context dataset
- flags.csv          every automated flag: dataset, index, check, detail (no text from unlicensed datasets)
- duplicates.csv     exact and near duplicates across datasets
- consensus.csv      items where at least 4 of the 5 models agree on an answer other than the key
- sample.csv         the seeded random sample used for the error-rate estimate, with each item's status
- summary.json       counts, the error-rate estimates with intervals, and the Moonshot metric comparison
"""
import csv
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab import audit as A  # noqa: E402
from evalab.stable import stable_round  # noqa: E402
from evalab.stats import clopper_pearson, wilson  # noqa: E402

DATA = Path(os.environ.get("MOONSHOT_DATA", ROOT / "external" / "moonshot-data"))
OUT = ROOT / "results" / "e2"
FACT_SETS = ["singapore-facts-tnf", "singapore-places-tnf", "singapore-food-tnf", "singapore-transport-system",
             "singapore-iconic-places", "singapore-political-history", "singapore-public-housing",
             "singapore-pofma-statements-2023", "singapore-pofma-statements-2024", "singapore-pofma-true-statements"]
UNLICENSED_SCAN = ["sg-legal-glossary", "sg-university-tutorial-questions-legal"]
KEY_CHECKS = {"mcq_key_mismatch", "mcq_key_missing", "tnf_bad_target", "same_input_different_target", "consensus"}
SAMPLE_N, SEED = 100, 20260928


def key_answer(ex: dict) -> str:
    t = str(ex["target"]).strip()
    m = A.TARGET_MCQ.match(t)
    return m.group(1) if m and A.is_mcq(ex) else t.upper()


def parse_model_answer(text: str, mcq: bool) -> str | None:
    t = (text or "").strip()
    if mcq:
        m = re.match(r"^\W*([A-D])\b", t)
        return m.group(1) if m else None
    m = re.search(r"\b(TRUE|FALSE)\b", t, re.I)
    return m.group(1).upper() if m else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(DATA), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    named, meta = {}, {}
    for n in FACT_SETS + UNLICENSED_SCAN:
        d = json.loads((DATA / "datasets" / f"{n}.json").read_text(encoding="utf-8"))
        named[n] = d["examples"]
        meta[n] = {"n": len(d["examples"]), "license": d.get("license") or "(none stated)",
                   "reference": str(d.get("reference", ""))[:120], "name": d.get("name"),
                   "name_check": A.metadata_check(n, d), "redistributable": bool(d.get("license"))}
    (OUT / "datasets.json").write_text(json.dumps({"moonshot_data_commit": commit, "datasets": meta}, indent=1),
                                       encoding="utf-8", newline="\n")

    flags = []
    for n, exs in named.items():
        for i, c, d in A.check_examples(exs):
            flags.append({"dataset": n, "index": i, "check": c, "detail": d})
    dups = A.duplicates({n: named[n] for n in FACT_SETS})
    for a, b, k, j in dups:
        for x in (a, b):
            ds, i = x.split("#")
            flags.append({"dataset": ds, "index": int(i), "check": k, "detail": f"{a} ~ {b} (jaccard {j})"})
    with (OUT / "duplicates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["item_a", "item_b", "kind", "jaccard"])
        w.writerows(dups)

    # model consensus from the Moonshot run
    consensus, metric_cmp = [], {}
    run_file = OUT / "moonshot_run.json"
    if run_file.exists():
        run = json.loads(run_file.read_text(encoding="utf-8"))
        answers = defaultdict(dict)       # (dataset, index) -> model -> parsed answer
        for rec in run["predictions"]:
            ex = named[rec["dataset"]][rec["index"]]
            answers[(rec["dataset"], rec["index"])][rec["model"]] = parse_model_answer(rec["response"], A.is_mcq(ex))
        for (ds, i), by_model in sorted(answers.items()):
            ex = named[ds][i]
            key = key_answer(ex)
            votes = Counter(v for v in by_model.values() if v is not None)
            if votes:
                top, k = votes.most_common(1)[0]
                if top != key and k >= 4:
                    consensus.append({"dataset": ds, "index": i, "key": key, "models_say": top, "agree": k,
                                      "of": len(by_model)})
                    flags.append({"dataset": ds, "index": i, "check": "consensus",
                                  "detail": f"{k}/{len(by_model)} models answer {top}, key is {key}"})
        metric_cmp = run.get("metric_comparison", {})
    with (OUT / "consensus.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "index", "key", "models_say", "agree", "of"], lineterminator="\n")
        w.writeheader()
        w.writerows(consensus)
    with (OUT / "flags.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "index", "check", "detail"], lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(flags, key=lambda r: (r["dataset"], r["index"], r["check"])))

    verified = json.loads((ROOT / "data" / "e2" / "verified.json").read_text(encoding="utf-8"))
    confirmed = {v["item"] for v in verified if v["status"] == "confirmed_error"}
    cleared = {v["item"] for v in verified if v["status"] == "key_correct"}
    unresolved = {v["item"] for v in verified if v["status"] == "unverifiable"}
    key_flagged = {f"{r['dataset']}#{r['index']}" for r in flags if r["check"] in KEY_CHECKS}

    population = [f"{n}#{i}" for n in FACT_SETS for i in range(len(named[n]))]
    sample = sorted(random.Random(SEED).sample(population, SAMPLE_N), key=population.index)

    def status(x):
        if x in confirmed:
            return "confirmed_error"
        if x in cleared:
            return "flag_cleared"
        if x in unresolved:
            return "unresolved"
        if x in key_flagged:
            return "suspected"
        return "no_key_flag"

    with (OUT / "sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["item", "status", "flags"])
        for x in sample:
            ds, i = x.split("#")
            fl = sorted({r["check"] for r in flags if r["dataset"] == ds and r["index"] == int(i)})
            w.writerow([x, status(x), ";".join(fl)])
    st = Counter(status(x) for x in sample)
    c, s = st["confirmed_error"], st["suspected"] + st["unresolved"]
    census = Counter(status(x) for x in population)
    summary = {
        "moonshot_data_commit": commit,
        "population": len(population), "sample_n": SAMPLE_N, "seed": SEED,
        "sample_status": st,
        "confirmed_error_rate": c / SAMPLE_N, "confirmed_error_rate_ci95": clopper_pearson(c, SAMPLE_N),
        "confirmed_or_open_rate": (c + s) / SAMPLE_N,
        "confirmed_or_open_ci95": clopper_pearson(c + s, SAMPLE_N),
        "census_status": census,
        "census_key_flag_rate": len(key_flagged & set(population)) / len(population),
        "census_key_flag_ci95": wilson(len(key_flagged & set(population)), len(population)),
        "flags_by_check": Counter(r["check"] for r in flags),
        "flags_by_dataset_and_check": {f"{d}|{k}": v for (d, k), v in
                                       Counter((r["dataset"], r["check"]) for r in flags).items()},
        "consensus_items": len(consensus),
        "moonshot_metric_comparison": metric_cmp,
        "verified": verified,
    }
    (OUT / "summary.json").write_text(json.dumps(stable_round(summary), indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: summary[k] for k in ("sample_status", "confirmed_error_rate", "confirmed_error_rate_ci95",
                                               "confirmed_or_open_rate", "census_status", "flags_by_check",
                                               "consensus_items")}, indent=1))


if __name__ == "__main__":
    main()
