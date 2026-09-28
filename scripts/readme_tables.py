"""Print the README's result tables as Markdown, straight from the result files (so no number is typed by hand).

    python scripts/readme_tables.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R1, R2, R3 = (ROOT / "results" / x for x in ("e1", "e2", "e3"))
JUDGES = ["qwen2.5:3b", "llama3.2:3b", "phi4-mini:3.8b", "gemma3:4b", "qwen3.5:4b"]


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x):
    return f"{100 * x:.1f}"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def e1():
    m = load(R1 / "main_metrics.json")
    ch = load(R1 / "pilot_choices.json")
    rows = [(f"{j}", f"{j}|bin_ref") for j in JUDGES] + [
        (f"J3 majority ({', '.join(ch['jury3'])})", "J3-majority|bin_ref"),
        ("J5 majority", "J5-majority|bin_ref"), ("J5 unanimity", "J5-unanimity|bin_ref"),
        ("J5 mean 1-5 score >= 4", "J5-mean_score|score_ref"), ("J5 majority, strict rubric", "J5-majority|strict_ref"),
        ("qwen3.5:4b, strict rubric", "qwen3.5:4b|strict_ref"), ("J5 majority, no reference", "J5-majority|bin_free"),
        ("qwen3.5:4b, no reference", "qwen3.5:4b|bin_free")]
    print("| Pipeline | Accuracy % [95% CI] | False accepts % [CI] | False rejects % [CI] | Calls / item | "
          "GPU s / item |")
    print("|---|---|---|---|---|---|")
    for label, k in rows:
        x = m[k]
        print(f"| {label} | {pct(x['accuracy'])} {ci(x['accuracy_ci'])} | {pct(x['false_accept_rate'])} "
              f"{ci(x['false_accept_ci'])} | {pct(x['false_reject_rate'])} {ci(x['false_reject_ci'])} | "
              f"{x['calls_per_item']:.0f} | {x['seconds_per_item']:.2f} |")
    t = load(R1 / "tests.json")
    h = t["primary_H1"]
    print(f"\nH1: J5 majority right on {h['a_right']}, best pilot judge on {h['b_right']} of {h['n']}; "
          f"discordant {h['only_a_right']} vs {h['only_b_right']}; diff {pct(h['diff'])} pp {ci(h['diff_ci'])}; "
          f"OR {h['discordant_odds_ratio']:.2f} [{h['odds_ratio_ci'][0]:.2f}, {h['odds_ratio_ci'][1]:.2f}]; "
          f"p = {h['p_mcnemar']:.3f}")
    print("\n| Test | n | Difference, pp [95% CI] | Discordant (a only / b only) | p | Holm p |")
    print("|---|---|---|---|---|---|")
    for k, v in t["family_A"].items():
        print(f"| {k} | {v['n']} | {100 * v['diff']:+.1f} {ci(v['diff_ci'])} | {v['only_a_right']} / "
              f"{v['only_b_right']} | {v['p_mcnemar']:.2g} | {v['p_holm']:.2g} |")
    print("\n| Family B test | Rate (verbose / terse, or first-slot share) | p | Holm p |")
    print("|---|---|---|---|")
    for k, v in t["family_B"].items():
        rate = (f"{pct(v['verbose_rate'])} / {pct(v['terse_rate'])}" if "verbose_rate" in v
                else f"{pct(v['first_slot_share'])}")
        print(f"| {k} | {rate} | {v['p']:.2g} | {v['p_holm']:.2g} |")
    rob = load(R1 / "robustness.json")
    print("\n| Judge | Verbose right answers failed, basic rubric % | ... strict rubric % | Terse right answers failed % |")
    print("|---|---|---|---|")
    for j in JUDGES + ["J5-majority", "J3-majority"]:
        b, s = rob[f"{j}|bin_ref"]["false_reject"], rob[f"{j}|strict_ref"]["false_reject"]
        print(f"| {j} | {pct(b['verbose_rate'])} | {pct(s['verbose_rate'])} | "
              f"{pct(max(b['terse_rate'], s['terse_rate']))} |")
    pw = load(R1 / "pairwise.json")
    print("\n| Judge | No reference: right per order % | consistent % | first-slot % | With reference: right % | "
          "consistent % | first-slot % |")
    print("|---|---|---|---|---|---|---|")
    for j in JUDGES:
        a, b = pw[f"{j}|pair_free"], pw[f"{j}|pair_ref"]
        print(f"| {j} | {pct(a['accuracy_per_order'])} | {pct(a['consistency'])} | {pct(a['first_slot_share'])} | "
              f"{pct(b['accuracy_per_order'])} | {pct(b['consistency'])} | {pct(b['first_slot_share'])} |")
    ps = load(R1 / "per_source.json")
    print("\n| Source | n | " + " | ".join(JUDGES) + " | J3 majority | J5 majority |")
    print("|---|---|" + "---|" * (len(JUDGES) + 2))
    for src in ("fm", "sgbus", "sql"):
        cells = [pct(ps[f"{src}|{j}|bin_ref"]["accuracy"]) for j in JUDGES]
        cells += [pct(ps[f"{src}|J3-majority|bin_ref"]["accuracy"]), pct(ps[f"{src}|J5-majority|bin_ref"]["accuracy"])]
        print(f"| {src} | {ps[f'{src}|J5-majority|bin_ref']['n']} | " + " | ".join(cells) + " |")
    ag = load(R1 / "agreement.json")
    print("\nFleiss kappa: " + ", ".join(f"{c} {v['fleiss_kappa']:.2f}" for c, v in ag.items()))
    cal = load(R1 / "calibration.json")
    print("Calibration: " + ", ".join(f"{k} ECE {v['ece']:.2f} AUROC {v['auroc']:.2f}" for k, v in cal.items()))
    print("Recommendation:", json.dumps(load(R1 / "recommendation.json")))
    s = load(R1 / "sensitivity_phrase_gold.json")
    print("Sensitivity:", s["n"], {k: pct(v["accuracy"]) for k, v in s["metrics"].items()},
          f"H1 p {s['primary_H1']['p_mcnemar']:.3f}")


def pts(x):
    return f"{100 * x:.1f}"


def plabel(k):
    """README label of an E1 pipeline key, e.g. 'J5-majority|bin_free' -> 'J5 majority, no reference'."""
    j, cond = k.split("|")
    j = j.replace("-majority", " majority").replace("-unanimity", " unanimity")
    return j + {"bin_ref": "", "bin_free": ", no reference"}.get(cond, f", {cond}")


def e1_amendment():
    """Amendment 1 (exploratory): benchmark rankings (A) and reliability-weighted juries (B)."""
    pr = R1 / "ranking.json"
    if pr.exists():
        r = load(pr)
        d = r["design"]
        print(f"\nA design: {d['n_systems']} systems {d['systems_per_source']}, pairs {d['pairs_per_source']}, "
              f"common questions per pair {d['common_questions_per_pair']}")
        print("\n| Pipeline | Mean abs. score error, points [95% CI] | tau-b, building codes [CI] | tau-b, bus [CI] | "
              "Flips / ordered pairs [CI] | Pairs the judge ties | Flips without phrase-gold questions |")
        print("|---|---|---|---|---|---|---|")
        for k, v in r["pipelines"].items():
            bs = v["by_source"]
            nop = v["without_phrase_gold_questions"]
            print(f"| {plabel(k)} | {pts(v['mae'])} [{pts(v['mae_ci'][0])}, {pts(v['mae_ci'][1])}] | "
                  f"{bs['fm']['tau_b']:.2f} [{bs['fm']['tau_b_ci'][0]:.2f}, {bs['fm']['tau_b_ci'][1]:.2f}] | "
                  f"{bs['sgbus']['tau_b']:.2f} [{bs['sgbus']['tau_b_ci'][0]:.2f}, {bs['sgbus']['tau_b_ci'][1]:.2f}] | "
                  f"{v['flips']} / {v['gold_untied_pairs']} [{v['flips_ci'][0]:.0f}, {v['flips_ci'][1]:.0f}] | "
                  f"{v['judge_ties']} | "
                  f"{nop['flips']} / {nop['gold_untied_pairs']} |")
        print("\n| Gold gap (points) | Pairs | " + " | ".join(plabel(k) for k in r["pipelines"] if "bin_free" not in k) + " |")
        print("|---|---|" + "---|" * sum("bin_free" not in k for k in r["pipelines"]))
        keys = [k for k in r["pipelines"] if "bin_free" not in k]
        for i, b in enumerate(r["pipelines"][keys[0]]["real_pairs_by_gap"]):
            print(f"| {b['lo']}-{b['hi']} | {b['pairs']} | " +
                  " | ".join(str(r["pipelines"][k]["real_pairs_by_gap"][i]["flips"]) for k in keys) + " |")
        for k in ("qwen3.5:4b|bin_ref", "J3-majority|bin_ref", "gemma3:4b|bin_ref"):
            e = {n: v["error"] for n, v in r["pipelines"][k]["error_by_system"].items()}
            worst = max(e, key=lambda n: abs(e[n]))
            print(k, "signed score errors: too high", sum(x > 1e-9 for x in e.values()), "too low",
                  sum(x < -1e-9 for x in e.values()), "min", pts(min(e.values())), "largest", worst, pts(e[worst]))
        for k in ("qwen3.5:4b|bin_ref", "J5-majority|bin_ref"):
            print(k, "flipped pairs:", [(f["a"], f["b"], pts(f["gold_gap"]), pts(f["judge_gap"]))
                                        for s_ in ("fm", "sgbus", "sql")
                                        for f in r["pipelines"][k]["by_source"][s_]["flipped_pairs"]])
        sim = r["simulation"]
        print(f"\n{sim['label']}; {sim['answers_per_candidate']} answers per candidate, "
              f"{sim['pairs_per_source']} pairs per source")
        print("\n| Pipeline | Flip rate %, gap 1-4 points | 5-9 | 10-20 | 21 or more |")
        print("|---|---|---|---|---|")
        for k, c in sim["curves"].items():
            bb = c["by_band"]
            print(f"| {plabel(k)} | " + " | ".join(pts(bb[x]["all"]) for x in ("1-4", "5-9", "10-20", "21-100")) + " |")
        for k in ("qwen3.5:4b|bin_ref", "J5-majority|bin_ref"):
            print(k, "by source, 1-4 and 5-9:", {x: {s_: pts(v) for s_, v in sim["curves"][k]["by_band"][x].items()
                                                     if s_ != "pairs"} for x in ("1-4", "5-9")})
    pa = R1 / "aggregation.json"
    if pa.exists():
        a = load(pa)
        print("\n| Method | Accuracy % [95% CI] | False accepts % [CI] | False rejects % [CI] | "
              "Accuracy vs qwen3.5, points [CI] | False accepts vs qwen3.5 [CI] | Discordant (this / qwen3.5) | "
              "McNemar p | Holm p |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k, v in a["methods"].items():
            vs = v["vs_best_single"]
            holm = f"{vs['p_holm']:.2g}" if "p_holm" in vs else "-"
            print(f"| {v['label']} | {pct(v['accuracy'])} {ci(v['accuracy_ci'])} | {pct(v['false_accept_rate'])} "
                  f"{ci(v['false_accept_ci'])} | {pct(v['false_reject_rate'])} {ci(v['false_reject_ci'])} | "
                  f"{100 * vs['accuracy']['diff']:+.1f} {ci(vs['accuracy']['diff_ci'])} | "
                  f"{100 * vs['false_accept_rate']['diff']:+.1f} {ci(vs['false_accept_rate']['diff_ci'])} | "
                  f"{vs['only_this_right']} / {vs['only_best_single_right']} | {vs['p_mcnemar']:.2g} | {holm} |")
        print("\n| Method | building codes | bus | SQL |")
        print("|---|---|---|---|")
        for k, v in a["methods"].items():
            print(f"| {v['label']} | " + " | ".join(pct(v["by_source"][s_]["accuracy"]) for s_ in ("fm", "sgbus", "sql"))
                  + " |")
        print("Split sensitivity:", {k: f"{pct(v['mean'])} ({pct(v['min'])}-{pct(v['max'])})"
                                     for k, v in a["split_sensitivity"].items()})
        print("CV best single per fold:", a["design"]["cv_best_single_by_fold"])
        import csv
        with (R1 / "aggregation_predictions.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        folds = sorted({r_["fold"] for r_ in rows})
        tr = [sum(1 for r_ in rows if r_["source"] == "sql" and r_["gold"] == "incorrect" and r_["fold"] != f_)
              for f_ in folds]
        print("SQL wrong answers per training fold:", min(tr), "-", max(tr))
        pm = a["parameters"]
        print("\n| Judge | Sensitivity, gold | Sensitivity, Dawid-Skene | Specificity, gold | "
              "Specificity, Dawid-Skene |")
        print("|---|---|---|---|---|")
        for j in JUDGES:
            g, ds = pm["two_coin"][j], pm["dawid_skene_transductive"][j]
            print(f"| {j} | {pct(g['sensitivity'])} | {pct(ds['sensitivity'])} | {pct(g['specificity'])} | "
                  f"{pct(ds['specificity'])} |")
        print("SQL two-coin (all items, display only):", {j: (pct(v["sensitivity"]), pct(v["specificity"]))
                                                           for j, v in pm["two_coin_per_source"]["sql"].items()})


def e2():
    s = load(R2 / "summary.json")
    print("\nE2:", json.dumps({k: s[k] for k in ("sample_status", "confirmed_error_rate", "confirmed_error_rate_ci95",
                                                 "confirmed_or_open_rate", "confirmed_or_open_ci95",
                                                 "census_status", "census_key_flag_rate", "census_key_flag_ci95",
                                                 "flags_by_check", "consensus_items")}))
    for k, v in sorted(s["moonshot_metric_comparison"].items()):
        ex = v["exactstrmatch_by_dataset"]
        print(f"| {k} | {v['n']} | {pct(v['lenient_accuracy'])} | " +
              ", ".join(f"{d.replace('singapore-', '')} {a:.1f}" for d, a in ex.items()) + " |")


def e3():
    p = R3 / "moonshot_run.json"
    if not p.exists():
        return
    s = load(p)
    print("\n| Target model | n | Rule score % [95% CI] | Jury score % | Jury false accepts / rule-wrong | "
          "Jury false rejects / rule-right |")
    print("|---|---|---|---|---|---|")
    for k, v in s["models"].items():
        print(f"| {k} | {v['n']} | {pct(v['rule_accuracy'])} {ci(v['rule_accuracy_ci95'])} | {pct(v['jury_accuracy'])} | "
              f"{v['jury_false_accepts']} / {v['rule_wrong']} | {v['jury_false_rejects']} / {v['rule_right']} |")
    print("usage", s["usage"], "duration", s["duration_s"])


if __name__ == "__main__":
    e1()
    e1_amendment()
    if (R2 / "summary.json").exists():
        e2()
    e3()
