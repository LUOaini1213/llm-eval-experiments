"""E1 amendment 1, analysis B (exploratory): reliability-weighted jury aggregation.

    python scripts/e1_aggregation.py

Compares, on the 400 main items and the cached bin_ref verdicts of the five judges: the fixed best single judge
(qwen3.5:4b), a cross-validated best single judge, J5 majority and unanimity, one-coin and two-coin naive-Bayes
weighted votes, two-coin weights per source, and Dawid-Skene EM without gold. Every fitted method is evaluated out
of fold (10 folds grouped by question, stratified by source). Design: docs/PREREGISTRATION.md, amendment 1.

Reads data/e1/items_main_order.jsonl and results/e1/judgments_main.jsonl only; no model is needed.
Writes results/e1/aggregation.json and results/e1/aggregation_predictions.csv.
"""
import csv
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from e1_analyze import JUDGES  # noqa: E402

from evalab import aggregate as G  # noqa: E402
from evalab import analysis as A  # noqa: E402
from evalab import judge as J  # noqa: E402
from evalab import stats as S  # noqa: E402
from evalab.items import load_jsonl  # noqa: E402
from evalab.experiment import results_dir  # noqa: E402

OUT = ROOT / "results" / "e1"
K, CV_SEED = 10, 20260928
SENSITIVITY_SEEDS = range(1, 51)
B, BOOT_SEED = 4000, 0
BEST = "qwen3.5:4b"
CV_METHODS = ["best_single_cv", "one_coin", "two_coin", "two_coin_per_source", "dawid_skene"]
FAMILY_C = ["best_single_cv", "J5_majority", "J5_unanimity", "one_coin", "two_coin", "two_coin_per_source",
            "dawid_skene"]
LABELS = {"best_single_fixed": f"best single, fixed ({BEST})", "best_single_cv": "best single, cross-validated",
          "J5_majority": "J5 majority", "J5_unanimity": "J5 unanimity", "J3_majority": "J3 majority (pilot-chosen)",
          "one_coin": "weighted vote, one-coin NB", "two_coin": "weighted vote, two-coin NB",
          "two_coin_per_source": "two-coin NB per source", "dawid_skene": "Dawid-Skene (no gold)",
          "dawid_skene_transductive": "Dawid-Skene, transductive (no gold)"}


def load():
    choices = json.loads((OUT / "pilot_choices.json").read_text(encoding="utf-8"))
    rows = load_jsonl(OUT / "judgments_main.jsonl")
    items = A.load_main_items(ROOT, OUT)
    idx = A.validate_matrix(rows, items, JUDGES, J.POINTWISE)
    votes = np.array([[int(A._pass(idx[(j, "bin_ref", it["item_id"])], "bin_ref")) for j in JUDGES]
                      for it in items])
    y = np.array([int(it["gold"] == "correct") for it in items])
    src = np.array([it["source"] for it in items])
    qid = [it["qid"] for it in items]
    return items, votes, y, src, qid, choices["jury3"]


def predictions(votes, y, src, qid, jury3, seed):
    folds = G.grouped_stratified_folds(qid, list(src), K, seed)
    pred = {"best_single_fixed": votes[:, JUDGES.index(BEST)].astype(bool),
            "J5_majority": np.array([J.majority([bool(v) for v in r]) for r in votes]),
            "J5_unanimity": np.array([J.unanimity([bool(v) for v in r]) for r in votes]),
            "J3_majority": np.array([J.majority([bool(r[JUDGES.index(j)]) for j in jury3]) for r in votes])}
    for m in CV_METHODS:
        fit, pr = G.METHODS[m]
        pred[m] = G.cross_val_predict(votes, y, folds, fit, pr, meta=src)
    return folds, pred


def rates(p, y):
    right = p == (y == 1)
    return {"accuracy": right.mean(), "false_accept_rate": p[y == 0].mean(), "false_reject_rate": (~p[y == 1]).mean()}


def paired_diff(a, b):
    d, lo, hi = S.paired_bootstrap_diff(a.astype(float), b.astype(float), b=B, seed=BOOT_SEED)
    return {"diff": d, "diff_ci": [lo, hi]}


def rnd(x):
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(f"{float(x):.10g}")
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return rnd(x.tolist())
    return x


def main():
    items, votes, y, src, qid, jury3 = load()
    folds, pred = predictions(votes, y, src, qid, jury3, CV_SEED)
    ds_all = G.dawid_skene(votes)
    pred["dawid_skene_transductive"] = ds_all["posterior"] > 0.5
    base = pred["best_single_fixed"]
    base_right = base == (y == 1)
    neg, pos = y == 0, y == 1
    methods = {}
    for m, p in pred.items():
        r = rates(p, y)
        right = p == (y == 1)
        lo, hi = S.bootstrap_ci(right.astype(float), b=B, seed=BOOT_SEED)
        fa, fr = int(p[neg].sum()), int((~p[pos]).sum())
        bb, cc = S.discordant(list(right), list(base_right))
        methods[m] = {
            "label": LABELS[m], **r, "accuracy_ci": [lo, hi],
            "false_accepts": fa, "false_accept_ci": list(S.wilson(fa, int(neg.sum()))),
            "false_rejects": fr, "false_reject_ci": list(S.wilson(fr, int(pos.sum()))),
            "vs_best_single": {
                "accuracy": paired_diff(right, base_right),
                "false_accept_rate": paired_diff(p[neg], base[neg]),
                "false_reject_rate": paired_diff(~p[pos], ~base[pos]),
                "only_this_right": bb, "only_best_single_right": cc, "p_mcnemar": S.mcnemar_exact(bb, cc)},
            "by_source": {s: {"n": int((src == s).sum()), **rates(p[src == s], y[src == s])}
                          for s in ("fm", "sgbus", "sql")}}
    for m, ph in zip(FAMILY_C, S.holm([methods[m]["vs_best_single"]["p_mcnemar"] for m in FAMILY_C])):
        methods[m]["vs_best_single"]["p_holm"] = ph
        methods[m]["vs_best_single"]["beats_best_single"] = bool(
            ph < 0.05 and methods[m]["vs_best_single"]["accuracy"]["diff"] > 0)

    # which judge the cross-validated best single picked in each fold
    cv_single = [JUDGES[G.METHODS["best_single_cv"][0](votes[folds != f], y[folds != f], None)] for f in range(K)]
    # parameters for interpretation only (fitted on all 400 items; never used to evaluate anything)
    two = G.fit_two_coin(votes, y)
    one = G.fit_one_coin(votes, y)
    per_src = {s: G.fit_two_coin(votes[src == s], y[src == s]) for s in ("fm", "sgbus", "sql")}
    params = {"note": "fitted on all 400 items for display only; the evaluated predictions are out of fold",
              "one_coin_weight": dict(zip(JUDGES, one["weight"])),
              "two_coin": {j: {"sensitivity": two["sensitivity"][k], "specificity": two["specificity"][k]}
                           for k, j in enumerate(JUDGES)},
              "two_coin_per_source": {s: {j: {"sensitivity": v["sensitivity"][k], "specificity": v["specificity"][k]}
                                          for k, j in enumerate(JUDGES)} for s, v in per_src.items()},
              "dawid_skene_transductive": {"prior": ds_all["prior"], "iterations": ds_all["iterations"],
                                           **{j: {"sensitivity": ds_all["sensitivity"][k],
                                                  "specificity": ds_all["specificity"][k]}
                                              for k, j in enumerate(JUDGES)}},
              "gold_prior": y.mean()}
    # split sensitivity: the same CV with 50 other seeds (no tests)
    sens = {m: [] for m in CV_METHODS}
    for sd in SENSITIVITY_SEEDS:
        _, p2 = predictions(votes, y, src, qid, jury3, sd)
        for m in CV_METHODS:
            sens[m].append(float((p2[m] == (y == 1)).mean()))
    split = {m: {"mean": float(np.mean(v)), "min": min(v), "max": max(v), "seeds": len(v)} for m, v in sens.items()}

    out = {"note": "Exploratory (docs/PREREGISTRATION.md, amendment 1). Family C: exact McNemar on accuracy against "
                   f"the fixed best single judge ({BEST}), Holm-adjusted over {len(FAMILY_C)} tests.",
           "design": {"n_items": len(items), "folds": K, "cv_seed": CV_SEED, "grouped_by": "question",
                      "stratified_by": "source", "fold_sizes": np.bincount(folds).tolist(),
                      "bootstrap": {"replicates": B, "seed": BOOT_SEED}, "family_C": FAMILY_C,
                      "cv_best_single_by_fold": cv_single},
           "methods": methods, "parameters": params, "split_sensitivity": split}
    (OUT / "aggregation.json").write_text(json.dumps(rnd(out), indent=1), encoding="utf-8", newline="\n")
    with (OUT / "aggregation_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["item_id", "source", "gold", "fold"] + list(pred))
        for i, it in enumerate(items):
            w.writerow([it["item_id"], it["source"], it["gold"], int(folds[i])] + [int(pred[m][i]) for m in pred])
    print(json.dumps({m: {"acc": round(v["accuracy"], 4), "fa": round(v["false_accept_rate"], 4),
                          "fr": round(v["false_reject_rate"], 4),
                          "diff": round(v["vs_best_single"]["accuracy"]["diff"], 4),
                          "p_holm": rnd(v["vs_best_single"].get("p_holm"))} for m, v in methods.items()}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="read and write results/e1/runs/NAME")
    args = parser.parse_args()
    OUT = results_dir(ROOT, args.run_id)
    main()
