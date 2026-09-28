"""E1 analysis. Every number in the README's E1 section comes from the files this script writes.

    python scripts/e1_analyze.py pilot    # judge choice and power analysis for the pre-registration
    python scripts/e1_analyze.py main     # the pre-registered analysis (plus robustness and pairwise sets)

Reads data/e1/*.jsonl and results/e1/judgments_*.jsonl only, so it runs without a model (and in CI).
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab import analysis as A  # noqa: E402
from evalab import judge as J  # noqa: E402
from evalab import stats as S  # noqa: E402
from evalab.items import load_jsonl  # noqa: E402

JUDGES = ["qwen2.5:3b", "llama3.2:3b", "phi4-mini:3.8b", "gemma3:4b", "qwen3.5:4b"]
OUT = ROOT / "results" / "e1"
MDE = 0.05          # smallest accuracy difference worth detecting (pre-registered)
ALPHA, POWER = 0.05, 0.8


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8", newline="\n")


def pipelines(items, idx, best, jury3):
    ps = {}
    for cond in J.POINTWISE:
        for j in JUDGES:
            ps[f"{j}|{cond}"] = A.build_pipeline(f"{j}|{cond}", [j], cond, "single", items, idx)
    for cond in ("bin_ref", "strict_ref", "bin_free"):
        for name, members in (("J5", JUDGES), ("J3", jury3)):
            for rule in ("majority", "unanimity"):
                ps[f"{name}-{rule}|{cond}"] = A.build_pipeline(f"{name}-{rule}|{cond}", members, cond, rule, items, idx)
    for name, members in (("J5", JUDGES), ("J3", jury3)):
        ps[f"{name}-mean_score|score_ref"] = A.build_pipeline(f"{name}-mean_score|score_ref", members, "score_ref",
                                                             "mean_score", items, idx)
    return ps


def pilot():
    items = load_jsonl(ROOT / "data" / "e1" / "items_pilot.jsonl")
    idx = A.index_judgments(load_jsonl(OUT / "judgments_pilot.jsonl"))
    single = {j: A.metrics(A.build_pipeline(j, [j], "bin_ref", "single", items, idx), items) for j in JUDGES}
    best, jury3 = A.choose_best_and_jury(single, JUDGES)
    ps = pipelines(items, idx, best, jury3)
    m = {k: A.metrics(p, items) for k, p in ps.items()}
    # power: discordance between the J5 majority and the best single judge on the pilot
    a, b = ps["J5-majority|bin_ref"], ps[f"{best}|bin_ref"]
    xa, xb = A.right(a, items), A.right(b, items)
    bb, cc = S.discordant(xa, xb)
    p_disc_obs = (bb + cc) / len(items)
    # plan with the larger of the observed discordance and 0.10, since a 48-item pilot estimates it poorly
    p_disc = max(p_disc_obs, 0.10)
    n_needed = S.mcnemar_sample_size(p_disc, MDE, ALPHA, POWER)
    avail = len(load_jsonl(ROOT / "data" / "e1" / "items_main_order.jsonl"))
    grid = {n: S.mcnemar_power_sim(n, p_disc, MDE, ALPHA, sims=3000, seed=1) for n in (200, 300, 400, 500, 600, 800)
            if n <= avail} | {avail: S.mcnemar_power_sim(avail, p_disc, MDE, ALPHA, sims=3000, seed=1)}
    secs = {j: single[j]["seconds_per_item"] for j in JUDGES}
    dump("pilot_metrics.json", m)
    dump("pilot_choices.json", {"best_single": best, "jury3": jury3, "pilot_accuracy_bin_ref":
                                {j: single[j]["accuracy"] for j in JUDGES}, "seconds_per_item_bin_ref": secs})
    dump("power.json", {"pilot_n": len(items), "discordant_only_jury_right": bb, "discordant_only_best_right": cc,
                        "p_disc_observed": p_disc_obs, "p_disc_planning": p_disc, "mde": MDE, "alpha": ALPHA,
                        "power": POWER, "n_needed_connor": n_needed, "available_main_items": avail,
                        "simulated_power_by_n": grid})
    print(json.dumps({"best": best, "jury3": jury3, "p_disc_obs": p_disc_obs, "n_needed": n_needed, "avail": avail,
                      "grid": grid, "acc": {j: round(single[j]["accuracy"], 3) for j in JUDGES},
                      "secs": secs}, indent=1))


def main_analysis():
    choices = json.loads((OUT / "pilot_choices.json").read_text(encoding="utf-8"))
    best, jury3 = choices["best_single"], choices["jury3"]
    items = load_jsonl(ROOT / "data" / "e1" / "items_main_order.jsonl")
    rows = load_jsonl(OUT / "judgments_main.jsonl")
    n = len({r["item_id"] for r in rows})
    items = items[:n]
    idx = A.index_judgments(rows)
    ps = pipelines(items, idx, best, jury3)
    m = {k: A.metrics(p, items) for k, p in ps.items()}
    dump("main_metrics.json", m)

    # ---- primary hypothesis
    h1 = A.compare(ps["J5-majority|bin_ref"], ps[f"{best}|bin_ref"], items)
    # ---- secondary family A (pipeline design), Holm-adjusted together
    fam_a = {
        "A1 reference-guided vs reference-free (J5 majority)": A.compare(ps["J5-majority|bin_ref"],
                                                                        ps["J5-majority|bin_free"], items),
        "A2 unanimity vs majority, false accepts (J5)": A.compare(ps["J5-unanimity|bin_ref"],
                                                                  ps["J5-majority|bin_ref"], items, "incorrect"),
        "A3 unanimity vs majority, false rejects (J5)": A.compare(ps["J5-unanimity|bin_ref"],
                                                                  ps["J5-majority|bin_ref"], items, "correct"),
        "A4 mean score vs majority vote (J5)": A.compare(ps["J5-mean_score|score_ref"], ps["J5-majority|bin_ref"],
                                                         items),
        "A5 three-judge vs five-judge majority": A.compare(ps["J3-majority|bin_ref"], ps["J5-majority|bin_ref"],
                                                           items),
        "A6 strict rubric vs basic rubric (best single judge)": A.compare(ps[f"{best}|strict_ref"],
                                                                          ps[f"{best}|bin_ref"], items),
    }
    for k, v in zip(fam_a, S.holm([v["p_mcnemar"] for v in fam_a.values()])):
        fam_a[k]["p_holm"] = v

    # ---- robustness set: verbosity/confidence bias
    rob_items = load_jsonl(ROOT / "data" / "e1" / "items_robust.jsonl")
    ridx = A.index_judgments(load_jsonl(OUT / "judgments_robust.jsonl"))
    byq = defaultdict(dict)
    for it in rob_items:
        byq[it["qid"]][(it["style"], it["gold"])] = it["item_id"]
    rob = {}
    for cond in ("bin_ref", "strict_ref"):
        for name, members, rule in [(j, [j], "single") for j in JUDGES] + [("J5-majority", JUDGES, "majority"),
                                                                           ("J3-majority", jury3, "majority")]:
            p = A.build_pipeline(f"{name}|{cond}", members, cond, rule, rob_items, ridx)
            res = {}
            for gold, what in (("incorrect", "false_accept"), ("correct", "false_reject")):
                v = [p.decisions[byq[q][("verbose", gold)]] for q in sorted(byq)]
                t = [p.decisions[byq[q][("terse", gold)]] for q in sorted(byq)]
                if what == "false_reject":
                    v, t = [not x for x in v], [not x for x in t]
                bb, cc = S.discordant(v, t)
                d, lo, hi = S.paired_bootstrap_diff([float(x) for x in v], [float(x) for x in t], b=4000)
                res[what] = {"verbose_rate": sum(v) / len(v), "terse_rate": sum(t) / len(t), "diff": d,
                             "diff_ci": [lo, hi], "only_verbose": bb, "only_terse": cc,
                             "p_mcnemar": S.mcnemar_exact(bb, cc), "n_questions": len(v)}
            rob[f"{name}|{cond}"] = res

    # ---- pairwise set: position bias
    pairs = load_jsonl(ROOT / "data" / "e1" / "pairs.jsonl")
    prow = defaultdict(dict)
    for r in load_jsonl(OUT / "judgments_pairs.jsonl"):
        prow[(r["judge"], r["condition"], r["item_id"])][r["order"]] = r
    pw = {}
    for cond in J.PAIRWISE:
        for j in JUDGES:
            outs = [J.pair_outcome(prow[(j, cond, p["pair_id"])]["AB"]["parsed"],
                                   prow[(j, cond, p["pair_id"])]["BA"]["parsed"]) for p in pairs]
            parsed = [c for p in pairs for c in (prow[(j, cond, p["pair_id"])]["AB"]["parsed"],
                                                 prow[(j, cond, p["pair_id"])]["BA"]["parsed"]) if c is not None]
            first = sum(1 for c in parsed if c == "A")
            incons = [o for o in outs if not o["consistent"]]
            first_inc = sum(o["first_slot"] for o in incons)
            slots_inc = sum(2 for o in incons)
            pw[f"{j}|{cond}"] = {
                "pairs": len(pairs), "accuracy_per_order": (sum(o["right_ab"] for o in outs) +
                                                            sum(o["right_ba"] for o in outs)) / (2 * len(pairs)),
                "both_orders_right": sum(o["both_right"] for o in outs) / len(pairs),
                "consistency": sum(o["consistent"] for o in outs) / len(pairs),
                "first_slot_share": first / max(1, len(parsed)), "parsed": len(parsed),
                "p_first_slot": S.binom_test_two_sided(first, len(parsed)) if parsed else 1.0,
                "inconsistent_pairs": len(incons),
                "first_slot_share_inconsistent": first_inc / max(1, slots_inc)}
    # ---- secondary family B (biases), Holm-adjusted together
    fam_b = {}
    for j in JUDGES + ["J5-majority"]:
        r = rob[f"{j}|bin_ref"]["false_accept"]
        fam_b[f"B verbosity: false accepts verbose vs terse, {j}"] = {**r, "p": r["p_mcnemar"]}
    for j in JUDGES:
        r = pw[f"{j}|pair_free"]
        fam_b[f"B position: first-slot share, {j}, no reference"] = {**r, "p": r["p_first_slot"]}
    for k, v in zip(fam_b, S.holm([v["p"] for v in fam_b.values()])):
        fam_b[k]["p_holm"] = v

    # ---- agreement, calibration, per-source, self-preference (exploratory)
    agree = {c: A.agreement(items, idx, JUDGES, c) for c in ("bin_ref", "strict_ref", "bin_free", "score_ref")}
    calib = {}
    for j in JUDGES:
        sc = [idx.get((j, "score_ref", it["item_id"]), {}).get("parsed") for it in items]
        pr = [((s or 1) - 1) / 4 for s in sc]
        y = [it["gold"] == "correct" for it in items]
        calib[j] = {"ece": S.ece(pr, y), "brier": S.brier(pr, y), "auroc": S.auroc(pr, y),
                    "unparsed": sum(s is None for s in sc),
                    "score_hist": {str(k): sum(1 for s in sc if s == k) for k in (1, 2, 3, 4, 5, None)}}
    mean5 = [sum((idx.get((j, "score_ref", it["item_id"]), {}).get("parsed") or 1) for j in JUDGES) / 5 for it in items]
    calib["J5-mean"] = {"ece": S.ece([(x - 1) / 4 for x in mean5], [it["gold"] == "correct" for it in items]),
                        "brier": S.brier([(x - 1) / 4 for x in mean5], [it["gold"] == "correct" for it in items]),
                        "auroc": S.auroc(mean5, [it["gold"] == "correct" for it in items])}
    per_source = {}
    for src in sorted({it["source"] for it in items}):
        sub = [it for it in items if it["source"] == src]
        for k in [f"{j}|bin_ref" for j in JUDGES] + ["J5-majority|bin_ref", "J5-unanimity|bin_ref",
                                                        "J3-majority|bin_ref", "J5-majority|bin_free"]:
            per_source[f"{src}|{k}"] = A.metrics(ps[k], sub)
    selfpref = {}
    for j in JUDGES:
        own = [it for it in items if it.get("generator") == j]
        other = [it for it in items if it.get("generator") != j and it["source"] != "sql"]
        if own:
            p = ps[f"{j}|bin_ref"]
            mo, mt = A.metrics(p, own), A.metrics(p, other)
            selfpref[j] = {"own_n": len(own), "own_false_accept": mo["false_accept_rate"],
                           "own_wrong_n": mo["n_wrong_answers"], "other_false_accept": mt["false_accept_rate"],
                           "other_wrong_n": mt["n_wrong_answers"], "own_accuracy": mo["accuracy"],
                           "other_accuracy": mt["accuracy"]}
    # sensitivity (exploratory, added after the main run): the three fm questions whose gold is a phrase rather than
    # a number ("1 hour", "KPI", "Class 1") can mislabel a paraphrase ("measurable performance metrics") as wrong
    phrase_q = {"fm-q05", "fm-q33", "fm-q37"}
    sub = [it for it in items if it["qid"] not in phrase_q]
    sensitivity = {"excluded_questions": sorted(phrase_q), "n": len(sub),
                   "metrics": {k: A.metrics(ps[k], sub) for k in
                               [f"{j}|bin_ref" for j in JUDGES] + ["J5-majority|bin_ref", "J3-majority|bin_ref",
                                                                   "J5-unanimity|bin_ref"]},
                   "primary_H1": A.compare(ps["J5-majority|bin_ref"], ps[f"{best}|bin_ref"], sub)}
    dump("sensitivity_phrase_gold.json", sensitivity)
    parse_fail = {f"{j}|{c}": sum(1 for it in items if idx.get((j, c, it["item_id"]), {}).get("parsed") is None)
                  for j in JUDGES for c in J.POINTWISE}

    # ---- trade-off and recommendation (pre-registered rule, see docs/PREREGISTRATION.md)
    pts = [m[k] for k in m]
    front = A.pareto(pts)
    top = max(pts, key=lambda x: (x["accuracy"], -x["seconds_per_item"]))
    candidates = []
    for x in sorted(pts, key=lambda x: x["seconds_per_item"]):
        if x["pipeline"] == top["pipeline"]:
            candidates.append((x, None))
            break
        c = A.compare(ps[x["pipeline"]], ps[top["pipeline"]], items)
        if c["diff_ci"][0] > -0.02 and x["seconds_per_item"] <= 0.5 * top["seconds_per_item"]:
            candidates.append((x, c))
            break
    rec, rec_cmp = candidates[0]
    safe = [x for x in pts if x["false_reject_rate"] <= 0.20]
    low_far = min(safe, key=lambda x: (x["false_accept_rate"], x["seconds_per_item"])) if safe else None
    recommendation = {"most_accurate": top["pipeline"], "recommended": rec["pipeline"],
                      "recommended_vs_most_accurate": rec_cmp, "pareto_front_seconds": front,
                      "lowest_false_accept_with_false_reject_at_most_20pct": low_far["pipeline"] if low_far else None}

    dump("tests.json", {"primary_H1": h1, "family_A": fam_a, "family_B": fam_b})
    dump("robustness.json", rob)
    dump("pairwise.json", pw)
    dump("agreement.json", agree)
    dump("calibration.json", calib)
    dump("per_source.json", per_source)
    dump("self_preference.json", selfpref)
    dump("parse_failures.json", parse_fail)
    dump("recommendation.json", recommendation)
    with (OUT / "pipelines.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pipeline", "n", "accuracy", "acc_lo", "acc_hi", "false_accept", "fa_lo", "fa_hi",
                    "false_reject", "fr_lo", "fr_hi", "kappa_vs_gold", "calls_per_item", "tokens_per_item",
                    "seconds_per_item"])
        for k, x in sorted(m.items(), key=lambda kv: -kv[1]["accuracy"]):
            w.writerow([k, x["n"], f"{x['accuracy']:.4f}", f"{x['accuracy_ci'][0]:.4f}", f"{x['accuracy_ci'][1]:.4f}",
                        f"{x['false_accept_rate']:.4f}", f"{x['false_accept_ci'][0]:.4f}",
                        f"{x['false_accept_ci'][1]:.4f}", f"{x['false_reject_rate']:.4f}",
                        f"{x['false_reject_ci'][0]:.4f}", f"{x['false_reject_ci'][1]:.4f}",
                        f"{x['kappa_vs_gold']:.4f}", f"{x['calls_per_item']:.2f}", f"{x['tokens_per_item']:.1f}",
                        f"{x['seconds_per_item']:.3f}"])
    print(json.dumps({"H1": h1, "recommendation": recommendation}, indent=1, default=float))


if __name__ == "__main__":
    {"pilot": pilot, "main": main_analysis}[sys.argv[1]]()
