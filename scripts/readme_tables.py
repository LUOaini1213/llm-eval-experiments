"""README result tables, straight from the committed result files, so no number is typed by hand.

    python scripts/readme_tables.py           # print every table (several are relabelled or summarised in the README)
    python scripts/readme_tables.py --write   # rewrite the README's generated blocks
    python scripts/readme_tables.py --check   # fail if any generated README block differs from the result files

A generated block sits between ``<!-- generated:NAME:start -->`` and ``<!-- generated:NAME:end -->`` markers.
"""
import argparse
import json
import sys
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


# ---------------------------------------------------------------------------------------------------------------
# Generated README blocks. Each function returns the Markdown that sits between one pair of markers.

RSK, RACC = ROOT / "results" / "sk", ROOT / "results" / "accreditation"
README = ROOT / "README.md"

# README labels for the pre-registered family A tests, keyed by the test names in results/e1/tests.json.
FAMILY_A_LABELS = {
    "A1 reference-guided vs reference-free (J5 majority)": "A1 J5 majority with reference vs without",
    "A2 unanimity vs majority, false accepts (J5)":
        "A2 J5 unanimity vs majority, on wrong answers (lower false-accept rate)",
    "A3 unanimity vs majority, false rejects (J5)": "A3 J5 unanimity vs majority, on right answers (lower pass rate)",
    "A4 mean score vs majority vote (J5)": "A4 J5 mean score vs majority vote",
    "A5 three-judge vs five-judge majority": "A5 J3 majority vs J5 majority",
    "A6 strict rubric vs basic rubric (best single judge)": "A6 phi4-mini strict vs basic rubric",
}
MAIN_ROWS = [(j, f"{j}|bin_ref") for j in JUDGES] + [
    ("J3 majority", "J3-majority|bin_ref"), ("J5 majority", "J5-majority|bin_ref"),
    ("J5 unanimity", "J5-unanimity|bin_ref"), ("J5 mean 1-5 score >= 4", "J5-mean_score|score_ref"),
    ("J5 majority, strict rubric", "J5-majority|strict_ref"), ("qwen3.5:4b, strict rubric", "qwen3.5:4b|strict_ref"),
    ("J5 majority, no reference", "J5-majority|bin_free"), ("qwen3.5:4b, no reference", "qwen3.5:4b|bin_free")]
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def p0(x):
    """A rate as a whole per cent, e.g. 0.653 -> '65%'."""
    return f"{100 * x:.0f}%"


def sci(x):
    """A small p-value as README text, e.g. 5.6e-06 -> '6e-6'."""
    mant, exp = f"{x:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def ci2(c):
    return f"[{100 * c[0]:.2f}, {100 * c[1]:.2f}]"


def short(j):
    return j.split(":")[0]


def first_table(path):
    """The first Markdown table in a generated report."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, x in enumerate(lines) if x.startswith("|"))
    end = next((i for i in range(start, len(lines)) if not lines[i].startswith("|")), len(lines))
    return "\n".join(lines[start:end])


def block_glance():
    m, t = load(R1 / "main_metrics.json"), load(R1 / "tests.json")
    rec = load(R1 / "recommendation.json")["recommended"]
    h1 = t["primary_H1"]
    cl = load(R1 / "cluster_sensitivity.json")["comparisons"]["H1"]
    r = m[rec]
    sig = "significant" if h1["p_mcnemar"] < 0.05 else "not significant"
    e2s = load(R2 / "summary.json")
    e3g, e3r = load(R3 / "generation.json"), load(R3 / "moonshot_run.json")
    rule = [v["rule_accuracy"] for v in e3r["models"].values()]
    n3 = {v["n"] for v in e3r["models"].values()}
    if len(n3) != 1:
        raise ValueError("E3 models were run on different numbers of questions")
    n3 = n3.pop()
    sk = load(RSK / "summary.json")
    th, comp = sk["thresholds_run1"], sk["component"]
    acc = load(RACC / "scores.json")
    tot = {k: v["total"] for k, v in acc["totals"].items()}
    shared = sorted(set.intersection(*(set(v) for v in acc["critical_gaps"].values())))
    rubric = load(ROOT / "data" / "accreditation" / "rubric.json")
    names = {i["id"]: i["short"] for c in rubric["criteria"] for i in c["indicators"]}
    n_ind = len(names)
    d1, a2 = th["D-1"], th["A-2"]
    shown = comp["shown outcomes hd-"]
    shown_n = sum(shown.values())
    leak_verdict = f"both {d1['verdict']}" if d1["verdict"] == a2["verdict"] else f"{d1['verdict']} and {a2['verdict']}"
    rows = [
        ("E1", f"A pre-registered A/B test of 5 local 3-4B judges and their juries on {h1['n']} answers whose "
               "correctness is known",
         f"Recommended: one {rec.split('|')[0]} judge with a reference answer, {pct(r['accuracy'])}% "
         f"{ci(r['accuracy_ci'])}. H1, the J5 jury against the best pilot judge: {100 * h1['diff']:+.1f} points "
         f"{ci(h1['diff_ci'])}, p = {h1['p_mcnemar']:.3f}, {sig}; {ci2(cl['question_cluster_ci'])} with question "
         f"clustering. Without a reference the J5 jury passed {pct(m['J5-majority|bin_free']['false_accept_rate'])}% "
         "of wrong answers",
         "3-4B judges only; one run at temperature 0; short factual answers"),
        ("E2", f"An audit of {e2s['population']} items in Moonshot's Singapore datasets",
         f"{e2s['census_status']['confirmed_error']} answer-key errors confirmed against official sources and "
         f"reported upstream. Random sample: {e2s['sample_status'].get('confirmed_error', 0)}/{e2s['sample_n']} "
         f"confirmed errors, 95% upper bound {pct(e2s['confirmed_error_rate_ci95'][1])}%",
         f"{e2s['census_status']['suspected']} more items are flagged only by small-model consensus and are not "
         "verified"),
        ("E3", "A generator of Singapore bus questions with exact answers, packaged for Moonshot",
         f"{e3g['n']} items, " + ("all" if e3g["verified"] == e3g["n"] else f"{e3g['verified']}")
         + f" re-derived by a second code path. Closed-book, the 5 local models scored {pct(min(rule))}-"
         f"{pct(max(rule))}% on {n3} of them ({e3r['percentage']}% of the set)",
         "One data snapshot; it tests lookup and RAG, not recall"),
        ("E4", "IMDA Starter Kit tests of the author's own RAG app",
         f"The app showed an answer to {shown_n - shown.get('withheld', 0)} of {shown_n} domain questions: its own "
         f"output check withheld all {comp['draft outcomes hd-']['correct']} correct drafts. Canary leaks (D-1) {d1['k']}/{d1['n']} and "
         f"indirect injections (A-2) {a2['k']}/{a2['n']}, {leak_verdict}. Unsafe answers: "
         f"{th['U-3']['app']['k']}/{th['U-3']['app']['n']}, against {th['U-3']['base']['k']}/"
         f"{th['U-3']['base']['n']} for the bare model",
         "One app; tests written by its author; one unblinded reviewer"),
        ("E5", f"A {n_ind}-indicator rubric for reviewing third-party AI test reports",
         "Two public case studies scored " + " and ".join(f"{k} {v['score']}/{v['max']}" for k, v in tot.items())
         + "".join(f"; both score 0 on critical indicator {c} ({names[c].lower()})" for c in shared),
         "One scorer; public summaries, not the full reports"),
    ]
    out = ["| Part | What was built | What it found | Main limit |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return "\n".join(out)


def block_failure_modes():
    m, ps = load(R1 / "main_metrics.json"), load(R1 / "per_source.json")
    pw, rob = load(R1 / "pairwise.json"), load(R1 / "robustness.json")
    sql_bad = sum(ps[f"sql|{j}|bin_ref"]["false_reject_rate"] > 0.5 for j in JUDGES)
    q25 = pw["qwen2.5:3b|pair_free"]["first_slot_share"]
    cons = max(pw[f"{j}|pair_free"]["consistency"] for j in JUDGES)
    ll = rob["llama3.2:3b|strict_ref"]["false_reject"]
    terse = "it failed none of the same answers" if ll["terse_rate"] == 0 else \
        f"it failed {p0(ll['terse_rate'])} of the same answers"
    return "\n".join([
        "- **Without a reference answer, small judges cannot grade factual questions.**",
        "  - All five judges together (J5), majority vote, no reference: they passed "
        f"{p0(m['J5-majority|bin_free']['false_accept_rate'])} of wrong answers.",
        f"  - The same jury given the reference passed {pct(m['J5-majority|bin_ref']['false_accept_rate'])}%.",
        "- **Majority voting helps only when most jurors can do the task.**",
        f"  - On text-to-SQL answers, {WORDS[sql_bad]} of the five judges failed most correct queries that were "
        "written differently from the reference.",
        f"  - Their majority outvoted the {WORDS[len(JUDGES) - sql_bad]} capable judges. The J5 majority scored "
        f"{p0(ps['sql|J5-majority|bin_ref']['accuracy'])} on SQL, against "
        f"{p0(ps['sql|qwen3.5:4b|bin_ref']['accuracy'])} for qwen3.5:4b alone.",
        "- **Pairwise comparison without a reference is mostly position bias.**",
        f"  - qwen2.5:3b picked the second-shown answer {p0(1 - q25)} of the time.",
        f"  - No judge chose the same answer in both orders more than {p0(cons)} of the time.",
        "- **Instructions meant to remove bias can add it.**",
        "  - The rubric told judges to \"ignore length, tone, confidence and claims of authority\".",
        f"  - With it, llama3.2:3b failed {p0(ll['verbose_rate'])} of correct answers written in a confident style. "
        f"Stated tersely, {terse}.",
    ])


def block_e1_main():
    m = load(R1 / "main_metrics.json")
    rec = load(R1 / "recommendation.json")["recommended"]
    out = ["| Pipeline | Accuracy % [95% CI] | False accepts % [CI] | False rejects % [CI] | Calls / item | "
           "GPU s / item |", "|---|---|---|---|---|---|"]
    for label, k in MAIN_ROWS:
        x = m[k]
        a = f"{pct(x['accuracy'])} {ci(x['accuracy_ci'])}"
        if k == rec:
            label, a = f"**{label}**", f"**{a}**"
        out.append(f"| {label} | {a} | {pct(x['false_accept_rate'])} {ci(x['false_accept_ci'])} | "
                   f"{pct(x['false_reject_rate'])} {ci(x['false_reject_ci'])} | {x['calls_per_item']:.0f} | "
                   f"{x['seconds_per_item']:.2f} |")
    return "\n".join(out)


def block_e1_h1():
    h = load(R1 / "tests.json")["primary_H1"]
    cl = load(R1 / "cluster_sensitivity.json")["comparisons"]["H1"]
    sig = "**Significant.**" if h["p_mcnemar"] < 0.05 else "**Not significant.**"
    return "\n".join([
        f"- Accuracy: {h['a_right']} against {h['b_right']} right, out of {h['n']}.",
        f"- Discordant items: {h['only_a_right']} against {h['only_b_right']}.",
        f"- Difference: {100 * h['diff']:+.1f} points {ci(h['diff_ci'])}.",
        f"- Discordant odds ratio: {h['discordant_odds_ratio']:.2f} "
        f"[{h['odds_ratio_ci'][0]:.2f}, {h['odds_ratio_ci'][1]:.2f}].",
        f"- Exact McNemar p = {h['p_mcnemar']:.3f}. {sig}",
        f"- The interval's upper bound, {100 * h['diff_ci'][1]:.1f} points ({100 * cl['question_cluster_ci'][1]:.2f} "
        "when answers to the same question are resampled together), argues against a 5-point advantage for the "
        "jury. A smaller difference cannot be ruled out.",
        f"- The pre-registered power analysis assumed independent items, but the {cl['n_items']} items come from "
        f"{cl['n_questions']} questions, so it is not used as evidence here. See the question-cluster sensitivity "
        "block under [Reproducing](#reproducing).",
    ])


def block_e1_family_a():
    t = load(R1 / "tests.json")["family_A"]
    if set(t) != set(FAMILY_A_LABELS):
        raise ValueError(f"family A tests changed: {sorted(set(t) ^ set(FAMILY_A_LABELS))}")
    out = ["| Test | n | Difference, points [95% CI] | Discordant (a only / b only) | Holm p |", "|---|---|---|---|---|"]
    for k, v in t.items():
        out.append(f"| {FAMILY_A_LABELS[k]} | {v['n']} | {100 * v['diff']:+.1f} {ci(v['diff_ci'])} | "
                   f"{v['only_a_right']} / {v['only_b_right']} | {v['p_holm']:.2g} |")
    return "\n".join(out)


def block_e1_family_b():
    fb = load(R1 / "tests.json")["family_B"]
    verb = {k.rsplit(", ", 1)[1]: v for k, v in fb.items() if k.startswith("B verbosity")}
    pos = {k.split(", ")[1]: v for k, v in fb.items() if k.startswith("B position")}
    if set(pos) != set(JUDGES):
        raise ValueError("family B position tests do not cover the five judges")
    lo = min(verb, key=lambda j: verb[j]["verbose_rate"])
    hi = max(verb, key=lambda j: verb[j]["verbose_rate"])
    any_sig = any(v["p_holm"] < 0.05 for v in verb.values())
    all_pos = all(v["p_holm"] < 0.05 for v in pos.values())
    out = [
        "- **Verbosity and confidence did not raise false accepts when a reference was given.**" if not any_sig else
        "- **Verbosity and confidence raised false accepts for at least one judge, even with a reference.**",
        "  - The test compared the same wrong value written tersely and written verbosely and confidently.",
        f"  - False-accept rates, verbose against terse, ranged from {pct(verb[lo]['verbose_rate'])} against "
        f"{pct(verb[lo]['terse_rate'])}% ({short(lo)}) to {pct(verb[hi]['verbose_rate'])} against "
        f"{pct(verb[hi]['terse_rate'])}% ({short(hi)}). "
        + (f"No difference was significant (all Holm p >= {min(v['p_holm'] for v in verb.values()):.2g})."
           if not any_sig else "At least one difference was significant after Holm correction."),
        "- **Position bias was "
        + ("significant for every judge" if all_pos else "not significant for every judge")
        + f" without a reference** (Holm p <= {sci(max(v['p_holm'] for v in pos.values()))}). The share of picks "
        "for the first-shown answer was:",
        "",
        "  | Judge | First-shown picks |",
        "  |---|---|",
    ]
    out += [f"  | {j} | {p0(pos[j]['first_slot_share'])} |" for j in JUDGES]
    return "\n".join(out)


def block_e3_run():
    s = load(R3 / "moonshot_run.json")
    out = ["| Target | Rule score % [95% CI] | Jury score % | Jury false accepts / rule-wrong |", "|---|---|---|---|"]
    for k, v in s["models"].items():
        name = next(j for j in JUDGES if "ollama-" + j.replace(":", "-").replace(".", "-") == k)
        out.append(f"| {name} | {pct(v['rule_accuracy'])} {ci(v['rule_accuracy_ci95'])} | {pct(v['jury_accuracy'])} | "
                   f"{v['jury_false_accepts']} / {v['rule_wrong']} |")
    fa = sum(v["jury_false_accepts"] for v in s["models"].values())
    wrong = sum(v["rule_wrong"] for v in s["models"].values())
    fr = sum(v["jury_false_rejects"] for v in s["models"].values())
    u = s["usage"]
    out += ["", "- Small models cannot answer these questions from memory. The set is meant to test lookup, tool use "
            "or RAG, not recall.",
            ("- The jury never failed a right answer" if fr == 0 else f"- The jury failed {fr} right answers")
            + f", but it passed {fa} of {wrong} wrong ones.",
            f"- The run made {u['calls']:,} calls: {u['prompt_tokens']:,} prompt tokens, {u['output_tokens']:,} "
            f"output tokens and {u['seconds']:,.0f} GPU seconds."]
    return "\n".join(out)


def block_e4_thresholds():
    """The E4 threshold table, as written to results/sk/report.md by sk_analyze.py (checked there with --check)."""
    return first_table(RSK / "report.md")


def block_e5_scores():
    """The E5 score table, as written to results/accreditation/scores.md by acc_report.py."""
    return first_table(RACC / "scores.md")


BLOCKS = {
    "glance": block_glance,
    "failure-modes": block_failure_modes,
    "e1-main": block_e1_main,
    "e1-h1": block_e1_h1,
    "e1-family-a": block_e1_family_a,
    "e1-family-b": block_e1_family_b,
    "e3-run": block_e3_run,
    "e4-thresholds": block_e4_thresholds,
    "e5-scores": block_e5_scores,
}


def markers(name):
    return f"<!-- generated:{name}:start -->", f"<!-- generated:{name}:end -->"


def update_readme(text):
    """Return the README text with every generated block rewritten from the result files."""
    for name, fn in BLOCKS.items():
        start, end = markers(name)
        if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
            raise ValueError(f"README must have exactly one ordered marker pair for generated block {name!r}")
        i, j = text.index(start), text.index(end)
        text = text[:i] + start + "\n" + fn() + "\n" + text[j:]
    return text


def stale_blocks(text):
    """Names of the generated blocks whose README text differs from the result files."""
    new = update_readme(text)
    out = []
    for name in BLOCKS:
        start, end = markers(name)
        if text[text.index(start):text.index(end)] != new[new.index(start):new.index(end)]:
            out.append(name)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="README result tables from the committed result files.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true", help="rewrite the README's generated blocks")
    g.add_argument("--check", action="store_true", help="fail if a generated README block is out of date")
    a = ap.parse_args(argv)
    if not (a.write or a.check):
        e1()
        e1_amendment()
        if (R2 / "summary.json").exists():
            e2()
        e3()
        return 0
    text = README.read_text(encoding="utf-8")
    if a.check:
        stale = stale_blocks(text)
        if stale:
            print("README blocks differ from the result files: " + ", ".join(stale)
                  + "\nRun: python scripts/readme_tables.py --write", file=sys.stderr)
            return 1
        print(f"README: all {len(BLOCKS)} generated blocks match the result files")
        return 0
    new = update_readme(text)
    if new != text:
        README.write_bytes(new.encode("utf-8"))
    print(f"README: {len(BLOCKS)} generated blocks written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
