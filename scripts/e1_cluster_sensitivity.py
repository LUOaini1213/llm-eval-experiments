"""Offline, post-hoc E1 question-cluster sensitivity; never overwrites the original analysis files.

    python scripts/e1_cluster_sensitivity.py          # write the new JSON and Markdown reports
    python scripts/e1_cluster_sensitivity.py --check  # independently recompute and compare both reports
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from e1_analyze import JUDGES, pipelines  # noqa: E402
from evalab import analysis as A  # noqa: E402
from evalab import judge as J  # noqa: E402
from evalab.cluster_sensitivity import build_report, endpoint_stability  # noqa: E402
from evalab.items import load_jsonl  # noqa: E402
from evalab.stable import stable_round  # noqa: E402

OUT = ROOT / "results" / "e1"
REPORT_JSON = OUT / "cluster_sensitivity.json"
REPORT_MD = OUT / "cluster_sensitivity.md"
B, SEED = 4000, 0
STABILITY_REPLICATES, STABILITY_SEEDS = (B, 4 * B), (SEED, 1, 2, 3, 4)
INPUTS = ["data/e1/items_main_order.jsonl", "results/e1/judgments_main.jsonl", "results/e1/pilot_choices.json",
          "results/e1/run_manifest.json", "results/e1/main_metrics.json", "results/e1/tests.json",
          "docs/PREREGISTRATION.md", "requirements.txt"]
CODE = ["src/evalab/stats.py", "src/evalab/cluster_sensitivity.py", "src/evalab/analysis.py",
        "src/evalab/judge.py", "scripts/e1_analyze.py", "scripts/e1_cluster_sensitivity.py"]
README_START, README_END = "<!-- cluster-sensitivity:start -->", "<!-- cluster-sensitivity:end -->"


def load_main(root=ROOT):
    """Require the declared frozen sample and complete judgment matrix, before any result is written."""
    out = root / "results" / "e1"
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    prereg = (root / "docs" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    declared = int(re.search(r"Main-set size:\s*\**\s*(\d+)", prereg).group(1))
    if manifest["n_items"] != declared:
        raise ValueError("manifest sample size differs from pre-registration")
    all_items = load_jsonl(root / "data" / "e1" / "items_main_order.jsonl")
    items = all_items[:declared]
    ids = {it["item_id"] for it in items}
    if len(items) != declared or len(ids) != declared:
        raise ValueError("declared main sample is incomplete or has duplicate item IDs")
    if any(it["gold"] not in {"correct", "incorrect"} for it in items):
        raise ValueError("main sample has an unsupported gold label")
    rows = load_jsonl(out / "judgments_main.jsonl")
    expected = {(judge, condition, item_id) for judge in JUDGES for condition in J.POINTWISE for item_id in ids}
    actual = [(r["judge"], r["condition"], r["item_id"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected or any("order" in r for r in rows):
        raise ValueError("main judgments must contain the complete, unique declared matrix")
    choices = json.loads((out / "pilot_choices.json").read_text(encoding="utf-8"))
    if choices["best_single"] not in JUDGES or len(choices["jury3"]) != 3 or \
            len(set(choices["jury3"])) != 3 or not set(choices["jury3"]) <= set(JUDGES):
        raise ValueError("pilot choices must identify known, distinct judges")
    return items, rows, choices


def generate(root=ROOT):
    items, rows, choices = load_main(root)
    ps = pipelines(items, A.index_judgments(rows), choices["best_single"], choices["jury3"])
    original_tests = json.loads((root / "results/e1/tests.json").read_text(encoding="utf-8"))
    original_comparisons = {"H1": original_tests["primary_H1"], **original_tests["family_A"]}
    comparisons = {label: (test["a"], test["b"], None if test["subset"] == "all" else test["subset"])
                   for label, test in original_comparisons.items()}
    result = build_report(items, ps, comparisons, b=B, seed=SEED)
    original_metrics = json.loads((root / "results/e1/main_metrics.json").read_text(encoding="utf-8"))
    if set(original_metrics) != set(ps):
        raise ValueError("original metrics must describe the same pipelines")
    for key, metrics in result["pipelines"].items():
        for metric, legacy_ci in (("accuracy", "accuracy_ci"), ("false_accept_rate", "false_accept_ci"),
                                   ("false_reject_rate", "false_reject_ci")):
            if abs(metrics[metric]["estimate"] - original_metrics[key][metric]) > 1e-9:
                raise ValueError(f"original point estimate differs: {key}, {metric}")
            ci = metrics[metric]["item_bootstrap_ci"] if metric == "accuracy" else \
                metrics[metric]["original_wilson_ci"]
            if not np.allclose(ci, original_metrics[key][legacy_ci], atol=1e-9, rtol=0):
                raise ValueError(f"original interval differs: {key}, {metric}")
    for label, comparison in result["comparisons"].items():
        original = original_comparisons[label]
        if comparison["n_items"] != original["n"] or \
                abs(comparison["estimate"] - original["diff"]) > 1e-9 or \
                not np.allclose(comparison["item_bootstrap_ci"], original["diff_ci"], atol=1e-9, rtol=0):
            raise ValueError(f"original comparison differs: {label}")
    h1 = original_comparisons["H1"]
    h1_items = [it for it in items if h1["subset"] == "all" or it["gold"] == h1["subset"]]
    diff = np.asarray(A.right(ps[h1["a"]], h1_items), dtype=float) - \
        np.asarray(A.right(ps[h1["b"]], h1_items), dtype=float)
    diagnostic = endpoint_stability(diff, h1_items, replicate_counts=STABILITY_REPLICATES,
                                    seeds=STABILITY_SEEDS)
    for design, canonical in diagnostic["canonical"]["intervals"].items():
        if canonical != result["comparisons"]["H1"][design + "_ci"]:
            raise ValueError("Monte Carlo diagnostic must retain the canonical H1 interval")
    result["monte_carlo_stability"] = {"contrast": "H1", "a": h1["a"], "b": h1["b"],
                                       "subset": h1["subset"], **diagnostic}
    result["provenance"] = {"numpy": np.__version__, "random_generator": "numpy PCG64",
                            "inputs_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in INPUTS},
                            "code_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in CODE},
                            "legacy_consistency": "all 34 pipeline points/intervals and H1/family A match originals",
                            "model_calls": 0}
    return stable_round(result)


def interval(ci):
    return f"[{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]"


def render_stability(report):
    d = report["monte_carlo_stability"]
    lines = ["## H1 bootstrap Monte Carlo endpoint stability", "",
             "This diagnostic holds the same paired replies and question clusters fixed. It repeats only H1",
             f"at {', '.join(format(count, ',') for count in d['replicate_counts'])} draws with each of seeds "
             + ", ".join(map(str, d["seeds"])) + ". Both question-resampling designs are included.",
             "The original 4,000-draw, seed-0 intervals above are retained exactly; no seed is selected or discarded.",
             "A larger draw budget is an additional diagnostic, not an exact reference distribution or convergence proof.",
             "", "Endpoint ranges below are minimum and maximum across this finite seed grid, in percentage points.",
             "They are not new confidence intervals or bounds on Monte Carlo error. Interval endpoints remain",
             "quantiles of the same bootstrap distribution; seed variation does not create new statistical evidence.",
             "No significance decision, p-value or recommendation is derived from whether an endpoint crosses zero.",
             "", "| Resampling | Draws per seed | Lower endpoint range | Upper endpoint range | Largest endpoint spread | Largest absolute shift from canonical endpoint |",
             "|---|---:|---|---|---:|---:|"]
    for design, budgets in d["designs"].items():
        for budget in budgets:
            lines.append(f"| {design} | {budget['bootstrap_replicates']:,} | "
                         f"{interval(budget['lower_endpoint_range'])} | {interval(budget['upper_endpoint_range'])} | "
                         f"{100*budget['max_endpoint_spread']:.3f} | "
                         f"{100*budget['max_abs_endpoint_shift_from_canonical']:.3f} |")
    lines += ["", "All individual intervals (points; rounding is for display only):", "",
              "| Resampling | Draws | Seed | H1 bootstrap interval |", "|---|---:|---:|---|"]
    for design, budgets in d["designs"].items():
        for budget in budgets:
            for run in budget["runs"]:
                lines.append(f"| {design} | {budget['bootstrap_replicates']:,} | {run['seed']} | {interval(run['ci'])} |")
    lines += ["", "The JSON records endpoints at canonical ten-significant-digit precision; tables round to two decimals.",
              "The reported spread is empirical over five seeds; another seed can fall outside this range.",
              "This checks simulation variability conditional on the sample, not robustness to new questions,",
              "model replies, model selection, labels or between-question dependence.", ""]
    return lines


def render(report):
    sample = report["sample"]
    lines = ["# E1 question-cluster sensitivity (2026-10-09)", "",
             "Post-hoc analysis of the committed replies; no model calls or changes to frozen labels, replies,",
             "pre-registration, original metrics, McNemar tests, Holm corrections or recommendation.", "",
             f"The {sample['n_items']} main items contain {sample['n_questions']} distinct questions: "
             f"{sample['singleton_questions']} with one candidate and {sample['repeated_questions']} with multiple candidates.",
             f"There are {sample['extra_candidates']} additional candidates beyond one per question; the largest cluster "
             f"has {sample['max_candidates_per_question']} candidates. Repeated questions are different candidate answers, "
             "not duplicate judgments.", "", "| Source | Items | Questions | Questions with multiple candidates |",
             "|---|---:|---:|---:|"]
    for source, profile in report["by_source"].items():
        lines.append(f"| {source} | {profile['n_items']} | {profile['n_questions']} | {profile['repeated_questions']} |")
    lines += ["", "Candidate count per question: " + ", ".join(
        f"{count} candidates: {questions} questions" for count, questions in sample["candidate_count_histogram"].items()) + ".",
        "", "## Method and assumptions", "",
        "The canonical intervals use 4,000 percentile-bootstrap draws, seed 0, alpha 0.05. The item column reproduces",
        "the original item-resampled interval. The pooled question column samples 188 questions with replacement,",
        "taking every candidate of each selected question together. The stratified column samples the original",
        "number of questions within each source. Both retain the item-weighted statistic: total successes",
        "/ total sampled items (the denominator varies with cluster size). Paired differences resample both",
        "pipelines together. Conditional false-accept/reject rates and gold-subset comparisons resample the",
        "questions represented in that subset, keeping all its candidate answers together.", "",
        "The question-weighted column is a separate descriptive estimand: average each question's candidate-level",
        "accuracy, then give every question equal weight. It shows sensitivity to repeat-question weighting; it",
        "does not replace the original accuracy or arbitrarily select one candidate per question.", "",
        "Question resampling allows dependence within a question and assumes independent questions (within source",
        "for the stratified design). Similar templates or shared entities across questions can still violate that",
        "assumption. These synthetic/purposive questions are not a random sample of all Singapore questions.",
        "Clustered intervals may widen or narrow; neither proves the original item-independence assumption false.",
        "The original McNemar p-values remain item-level results; no new p-values, significance claims or power",
        "claims are derived here. The analysis holds judge replies, model selection and labels fixed.", "",
        "Method background: [Deen & de Rooij, ClusterBootstrap](https://doi.org/10.3758/s13428-019-01252-y).", "",
        "## Accuracy sensitivity", "",
        "Percentages and 95% intervals. Item-weighted accuracy is unchanged in every row.", "",
        "| Pipeline | Item-weighted % | Original item CI | Pooled question CI | Source-stratified question CI | Equal-question % |",
        "|---|---:|---|---|---|---:|"]
    for key, metrics in report["pipelines"].items():
        a = metrics["accuracy"]
        lines.append(f"| {key.replace('|', ' / ')} | {100*a['estimate']:.2f} | {interval(a['item_bootstrap_ci'])} | "
                     f"{interval(a['question_cluster_ci'])} | {interval(a['source_stratified_question_ci'])} | "
                     f"{100*a['question_weighted_estimate']:.2f} |")
    lines += ["", "## Paired accuracy differences", "",
              "Points, a minus b, with 95% intervals; for gold subsets this is classification correctness",
              "(A2 is specificity gain, A3 is sensitivity change). Existing test names and contrasts are retained.", "",
              "| Contrast | Items / questions | Original difference | Original item CI | Pooled question CI | Source-stratified question CI | Equal-question difference |",
              "|---|---:|---:|---|---|---|---:|"]
    for label, d in report["comparisons"].items():
        lines.append(f"| {label} | {d['n_items']} / {d['n_questions']} | {100*d['estimate']:+.2f} | "
                     f"{interval(d['item_bootstrap_ci'])} | {interval(d['question_cluster_ci'])} | "
                     f"{interval(d['source_stratified_question_ci'])} | {100*d['question_weighted_estimate']:+.2f} |")
    h1 = report["comparisons"]["H1"]
    lines += ["", f"H1 remains an item-weighted difference of {100*h1['estimate']:+.2f} points. Its pooled question CI is "
              f"{interval(h1['question_cluster_ci'])}; the source-stratified question CI is "
              f"{interval(h1['source_stratified_question_ci'])}. This is sensitivity evidence, not a replacement "
              "for the pre-registered test.", "", "## Conditional error rates", "",
              "The original error-rate intervals were Wilson intervals, not bootstrap intervals. They are preserved",
              "and labelled accordingly; newly computed item and question bootstrap intervals are in the JSON.", "",
              "| Pipeline | Rate | Items / questions | Rate % | Original Wilson CI | Pooled question CI | Source-stratified question CI |",
              "|---|---|---:|---:|---|---|---|"]
    for key, metrics in report["pipelines"].items():
        for rate in ("false_accept_rate", "false_reject_rate"):
            d = metrics[rate]
            lines.append(f"| {key.replace('|', ' / ')} | {rate} | {d['n_items']} / {d['n_questions']} | {100*d['estimate']:.2f} | "
                         f"{interval(d['original_wilson_ci'])} | {interval(d['question_cluster_ci'])} | "
                         f"{interval(d['source_stratified_question_ci'])} |")
    lines += [""] + render_stability(report)
    lines += ["## Reproduce", "", "```bash", "python scripts/e1_cluster_sensitivity.py",
              "python scripts/e1_cluster_sensitivity.py --check", "```", "",
              "The JSON records input and implementation SHA-256 hashes, numpy/PRNG versions, cluster sizes and",
              "all numerical results. Before reporting it checks the frozen sample, complete judgment matrix,",
              "and agreement with all 34 original pipeline metrics and the original H1/family A intervals.",
              "CI and tests independently regenerate both new artifacts. Canonical results are written with",
              "ten significant digits and LF newlines, as in the existing analysis.", ""]
    return "\n".join(lines)


def render_readme(report):
    sample, h1 = report["sample"], report["comparisons"]["H1"]
    return "\n".join([
        README_START, "### Question-cluster sensitivity (post-hoc, 2026-10-09)", "",
        f"The {sample['n_items']} main items represent {sample['n_questions']} questions; "
        f"{sample['repeated_questions']} questions have multiple candidate answers (up to "
        f"{sample['max_candidates_per_question']} each). A new offline analysis keeps those answers together",
        "when resampling a question, with a second variant stratified by source. Both retain the original",
        "item-weighted point estimate, and use 4,000 percentile-bootstrap draws with seed 0.", "",
        "| Measure | Original estimate | Original item CI | Pooled question CI | Source-stratified question CI |",
        "|---|---:|---|---|---|"] + [
        f"| {label} | {100*d['estimate']:.2f}% | {interval(d['item_bootstrap_ci'])} | "
        f"{interval(d['question_cluster_ci'])} | {interval(d['source_stratified_question_ci'])} |"
        for label, d in [("qwen3.5:4b accuracy", report["pipelines"]["qwen3.5:4b|bin_ref"]["accuracy"]),
                         ("J5 majority accuracy", report["pipelines"]["J5-majority|bin_ref"]["accuracy"])]
    ] + [f"| H1 accuracy difference (points) | {100*h1['estimate']:+.2f} | {interval(h1['item_bootstrap_ci'])} | "
         f"{interval(h1['question_cluster_ci'])} | {interval(h1['source_stratified_question_ci'])} |", "",
         "The clustered intervals allow within-question dependence and assume independent questions; repeated",
         "questions alone do not establish that dependence. These are sensitivity intervals, with no new p-values",
         "or significance/power claims. Original item-level results, McNemar/Holm tests and recommendations remain",
         "unchanged. Equal-question point estimates are reported separately to show weighting sensitivity.", "",
         "H1 also has a bootstrap Monte Carlo diagnostic at 4,000 and 16,000 draws with seeds 0–4.",
         "Every seed's interval is reported; cross-seed endpoint ranges describe simulation variability, not new",
         "confidence intervals or significance evidence. The original 4,000-draw, seed-0 intervals are retained.", "",
         "[Full tables and assumptions](results/e1/cluster_sensitivity.md), "
         "[machine-readable results and hashes](results/e1/cluster_sensitivity.json). Reproduce with",
         "`python scripts/e1_cluster_sensitivity.py`; CI uses `--check` to verify the JSON, Markdown and this block.",
         README_END])


def update_readme(report):
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    if text.count(README_START) != 1 or text.count(README_END) != 1 or text.index(README_START) > text.index(README_END):
        raise ValueError("README must have exactly one ordered cluster sensitivity marker pair")
    return text[:text.index(README_START)] + render_readme(report) + text[text.index(README_END) + len(README_END):]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = generate()
    outputs = {REPORT_JSON: json.dumps(report, indent=1, allow_nan=False) + "\n", REPORT_MD: render(report),
               ROOT / "README.md": update_readme(report)}
    if args.check:
        for path, content in outputs.items():
            if not path.exists() or path.read_bytes() != content.encode("utf-8"):
                raise SystemExit(f"stale cluster sensitivity artifact: {path.relative_to(ROOT)}")
        print("Question-cluster sensitivity JSON and Markdown reproduce exactly; original results match.")
    else:
        for path, content in outputs.items():
            path.write_text(content, encoding="utf-8", newline="\n")
        print(f"Wrote sensitivity for {report['sample']['n_items']} items / {report['sample']['n_questions']} questions.")


if __name__ == "__main__":
    main()
