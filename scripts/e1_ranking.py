"""E1 amendment 1, analysis A (exploratory): do judge errors change benchmark conclusions?

    python scripts/e1_ranking.py

Each candidate system (generator + configuration, e.g. phi4-mini:3.8b with RAG top-4) gets a benchmark score from
gold and from each judge pipeline, on the same main-set items. The script compares the rankings: score error per
system, Kendall's tau-b within each source, and pairwise flips, with bootstrap intervals (items resampled within each
system). A simulation with virtual candidates built by resampling real answers gives the flip rate against the gold
gap; it is labelled as a simulation in every output. Design: docs/PREREGISTRATION.md, amendment 1.

Reads data/e1/*.jsonl and results/e1/judgments_main.jsonl only; no model is needed.
Writes results/e1/ranking.json and results/e1/ranking_systems.csv.
"""
import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from e1_analyze import JUDGES, pipelines  # noqa: E402

from evalab import analysis as A  # noqa: E402
from evalab import ranking as R  # noqa: E402
from evalab import stats as S  # noqa: E402
from evalab.items import load_jsonl  # noqa: E402

OUT = ROOT / "results" / "e1"
PIPES = [f"{j}|bin_ref" for j in JUDGES] + ["J3-majority|bin_ref", "J5-majority|bin_ref", "J5-unanimity|bin_ref",
                                            "J5-majority|bin_free", "qwen3.5:4b|bin_free"]
SOURCES = ("fm", "sgbus", "sql")
B, BOOT_SEED = 4000, 0
REAL_EDGES = [0, 5, 10, 20, 50, 100]                  # gold gap bins, points
SIM_SEED, SIM_PAIRS, SIM_N, SIM_MIN_POOL = 20260928, 20000, 100, 5
SIM_MAX_GAP = 20
# Post-hoc sensitivity (added after the first run of this script, see amendment 1a): the three building-code
# questions whose gold is a phrase, not a number; the same set as the E1 sensitivity check in e1_analyze.py
PHRASE_Q = {"fm-q05", "fm-q33", "fm-q37"}


def system_name(it):
    return f"{it['source']}|{it['generator']}|{it['config']}"


def load():
    choices = json.loads((OUT / "pilot_choices.json").read_text(encoding="utf-8"))
    order = load_jsonl(ROOT / "data" / "e1" / "items_main_order.jsonl")
    rows = load_jsonl(OUT / "judgments_main.jsonl")
    n = len({r["item_id"] for r in rows})
    items = order[:n]
    ps = pipelines(items, A.index_judgments(rows), choices["best_single"], choices["jury3"])
    systems = sorted({system_name(it) for it in items}, key=lambda s: (SOURCES.index(s.split("|")[0]), s))
    sys_idx = np.array([systems.index(system_name(it)) for it in items])
    # inverse sampling fraction of each (source, gold) stratum: the main set is a stratified prefix of the order file
    n_order = Counter((it["source"], it["gold"]) for it in order)
    n_main = Counter((it["source"], it["gold"]) for it in items)
    weight = np.array([n_order[(it["source"], it["gold"])] / n_main[(it["source"], it["gold"])] for it in items])
    dec = np.array([[float(ps[p].decisions[it["item_id"]]) for p in PIPES] for it in items])
    gold = np.array([float(it["gold"] == "correct") for it in items])
    src_of = [s.split("|")[0] for s in systems]
    pairs = {s: [(int(a), int(b)) for a, b in zip(*R.all_pairs(len(systems)))
                 if src_of[a] == s and src_of[b] == s] for s in SOURCES}
    return {"items": items, "systems": systems, "sys_idx": sys_idx, "weight": weight, "dec": dec, "gold": gold,
            "pairs": pairs, "strata_weights": {f"{k[0]}|{k[1]}": n_order[k] / n_main[k] for k in sorted(n_main)}}


def evaluate(d, idx, weight):
    """All measures for one sample of items (idx indexes d['items'])."""
    ns = len(d["systems"])
    g = R.system_scores(d["gold"][idx], d["sys_idx"][idx], weight[idx], ns)
    j = R.system_scores(d["dec"][idx], d["sys_idx"][idx], weight[idx], ns)       # pipelines x systems
    out = {"gold": g, "judge": j, "err": j - g[None, :], "mae": np.abs(j - g[None, :]).mean(axis=1)}
    flips = np.zeros(len(PIPES), dtype=int)
    untied = 0
    ties = np.zeros(len(PIPES), dtype=int)
    for s in SOURCES:
        a = np.array([p[0] for p in d["pairs"][s]])
        b = np.array([p[1] for p in d["pairs"][s]])
        sg = R.pair_signs(g, a, b)
        sj = R.pair_signs(j, a, b)                                                  # pipelines x pairs
        out[f"tau|{s}"] = R.tau_b_from_signs(np.broadcast_to(sg, sj.shape), sj)
        fl = (sg[None, :] * sj) < 0
        out[f"flips|{s}"] = fl.sum(axis=1)
        out[f"pair_flip|{s}"] = fl
        out[f"gap|{s}"] = np.abs(g[a] - g[b])
        flips += fl.sum(axis=1)
        ties += ((sg[None, :] != 0) & (sj == 0)).sum(axis=1)
        untied += int((sg != 0).sum())
    out["flips"], out["judge_ties"], out["untied"] = flips, ties, untied
    out["flip_rate"] = flips / untied
    return out


def point_estimates(d, weighted=True, drop_qids=()):
    w = d["weight"] if weighted else np.ones_like(d["weight"])
    keep = np.array([it["qid"] not in drop_qids for it in d["items"]])
    return evaluate(d, np.flatnonzero(keep), w)


def bootstrap(d):
    rng = np.random.default_rng(BOOT_SEED)
    members = [np.flatnonzero(d["sys_idx"] == s) for s in range(len(d["systems"]))]
    keys = ["mae", "err", "flips", "flip_rate", "judge_ties"] + [f"{k}|{s}" for k in ("tau", "flips", "pair_flip")
                                                                for s in SOURCES]
    acc = defaultdict(list)
    for _ in range(B):
        idx = np.concatenate([m[rng.integers(0, len(m), size=len(m))] for m in members])
        e = evaluate(d, idx, d["weight"])
        for k in keys:
            acc[k].append(e[k])
    return {k: np.array(v) for k, v in acc.items()}


def ci(arr, axis=0):
    with warnings.catch_warnings():          # tau is undefined (NaN) when a pipeline ties every system of a source
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanquantile(arr, [0.025, 0.975], axis=axis)
    return lo, hi


def simulate(d):
    """Virtual candidates (SIMULATION): right answers from one real system, wrong answers from one real system."""
    rng = np.random.default_rng(SIM_SEED)
    gold_gap, judge_gap, src_lab, pools = [], [], [], {}
    for s in SOURCES:
        right_pools, wrong_pools = [], []
        for k, name in enumerate(d["systems"]):
            if not name.startswith(s + "|"):
                continue
            m = d["sys_idx"] == k
            r, w = np.flatnonzero(m & (d["gold"] == 1)), np.flatnonzero(m & (d["gold"] == 0))
            if len(r) >= SIM_MIN_POOL:
                right_pools.append(r)
                pools.setdefault(s, {"right_from": [], "wrong_from": []})["right_from"].append(name)
            if len(w) >= SIM_MIN_POOL:
                wrong_pools.append(w)
                pools.setdefault(s, {"right_from": [], "wrong_from": []})["wrong_from"].append(name)
        sim = R.virtual_pairs(rng, d["dec"], right_pools, wrong_pools, SIM_PAIRS, SIM_N)
        gold_gap.append(sim["gold_gap"])
        judge_gap.append(sim["judge_gap"])
        src_lab += [s] * SIM_PAIRS
    gg = np.concatenate(gold_gap)
    jg = np.concatenate(judge_gap, axis=1)
    src_lab = np.array(src_lab)
    gap_pts = np.rint(np.abs(gg) * 100).astype(int)
    flip = (np.sign(gg)[None, :] * np.sign(jg)) < 0
    untied = gap_pts > 0
    curves = {}
    for q, p in enumerate(PIPES):
        rows = []
        for gpt in range(1, SIM_MAX_GAP + 1):
            m = gap_pts == gpt
            k, n = int(flip[q, m].sum()), int(m.sum())
            rows.append({"gap_points": gpt, "pairs": n, "flips": k, "flip_rate": k / n, "wilson": list(S.wilson(k, n))})
        by_band = {}
        for lo, hi in ((1, 4), (5, 9), (10, 20), (21, 100)):
            m = (gap_pts >= lo) & (gap_pts <= hi)
            by_band[f"{lo}-{hi}"] = {s: float(flip[q, m & (src_lab == s)].mean()) for s in SOURCES} | {
                "all": float(flip[q, m].mean()), "pairs": int(m.sum())}
        tie = untied & (np.abs(jg[q]) < R.TOL)
        curves[p] = {"by_gap": rows, "by_band": by_band, "flip_rate_all_untied": float(flip[q, untied].mean()),
                     "judge_tie_rate_all_untied": float(tie[untied].mean())}
    return {"label": "SIMULATION: virtual candidates built by resampling real answers; not real systems",
            "seed": SIM_SEED, "pairs_per_source": SIM_PAIRS, "answers_per_candidate": SIM_N,
            "accuracy_range": [0.05, 0.95], "min_items_per_pool": SIM_MIN_POOL, "pools": pools,
            "gold_tied_pairs": int((~untied).sum()), "curves": curves}


def rnd(x):
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(f"{float(x):.10g}")
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return rnd(x.tolist())
    return x


def main():
    d = load()
    pt = point_estimates(d)
    unw = point_estimates(d, weighted=False)
    nophrase = point_estimates(d, drop_qids=PHRASE_Q)
    bs = bootstrap(d)
    systems = d["systems"]
    qsets = defaultdict(set)
    for it in d["items"]:
        qsets[system_name(it)].add(it["qid"])
    common = [len(qsets[systems[a]] & qsets[systems[b]]) for s in SOURCES for a, b in d["pairs"][s]]
    design = {
        "n_items": len(d["items"]), "n_systems": len(systems),
        "systems_per_source": {s: sum(x.startswith(s + "|") for x in systems) for s in SOURCES},
        "pairs_per_source": {s: len(d["pairs"][s]) for s in SOURCES},
        "common_questions_per_pair": {"median": float(np.median(common)), "min": min(common), "max": max(common),
                                      "pairs_with_at_least_10": sum(c >= 10 for c in common)},
        "strata_weights": d["strata_weights"], "bootstrap": {"replicates": B, "seed": BOOT_SEED,
                                                             "resampling": "items within each system"}}
    sys_rows = []
    for k, name in enumerate(systems):
        m = d["sys_idx"] == k
        sys_rows.append({"system": name, "source": name.split("|")[0], "n_items": int(m.sum()),
                         "n_right": int(d["gold"][m].sum()), "gold_score": pt["gold"][k],
                         "gold_score_unweighted": unw["gold"][k]})
    res = {}
    for q, p in enumerate(PIPES):
        lo, hi = ci(bs["err"][:, q, :])
        mlo, mhi = ci(bs["mae"][:, q])
        flo, fhi = ci(bs["flips"][:, q])
        rlo, rhi = ci(bs["flip_rate"][:, q])
        r = {"mae": pt["mae"][q], "mae_ci": [mlo, mhi],
             "error_by_system": {systems[k]: {"judge_score": pt["judge"][q, k], "error": pt["err"][q, k],
                                              "error_ci": [lo[k], hi[k]]} for k in range(len(systems))},
             "flips": int(pt["flips"][q]), "flips_ci": [flo, fhi], "flips_bootstrap_mean": bs["flips"][:, q].mean(),
             "gold_untied_pairs": int(pt["untied"]),
             "flip_rate": pt["flip_rate"][q], "flip_rate_ci": [rlo, rhi], "judge_ties": int(pt["judge_ties"][q]),
             "by_source": {}, "unweighted": {"mae": unw["mae"][q], "flips": int(unw["flips"][q]),
                                              "tau": {s: unw[f"tau|{s}"][q] for s in SOURCES}},
             "without_phrase_gold_questions": {"mae": nophrase["mae"][q], "flips": int(nophrase["flips"][q]),
                                               "gold_untied_pairs": int(nophrase["untied"]),
                                               "tau": {s: nophrase[f"tau|{s}"][q] for s in SOURCES}}}
        for s in SOURCES:
            tlo, thi = ci(bs[f"tau|{s}"][:, q])
            slo, shi = ci(bs[f"flips|{s}"][:, q])
            pair_p = bs[f"pair_flip|{s}"][:, q, :].mean(axis=0)
            r["by_source"][s] = {
                "tau_b": pt[f"tau|{s}"][q], "tau_b_ci": [tlo, thi], "flips": int(pt[f"flips|{s}"][q]),
                "flips_ci": [slo, shi], "pairs": len(d["pairs"][s]),
                "flipped_pairs": [{"a": systems[a], "b": systems[b], "gold_gap": pt[f"gap|{s}"][i],
                                   "judge_gap": pt["judge"][q, a] - pt["judge"][q, b],
                                   "bootstrap_flip_share": pair_p[i]}
                                  for i, (a, b) in enumerate(d["pairs"][s]) if pt[f"pair_flip|{s}"][q, i]]}
        gaps = np.concatenate([pt[f"gap|{s}"] for s in SOURCES]) * 100
        fl = np.concatenate([pt[f"pair_flip|{s}"][q] for s in SOURCES])
        untied = gaps > 100 * R.TOL
        bins = R.gap_bins(gaps[untied], fl[untied], REAL_EDGES)
        for x in bins:
            x["wilson_too_narrow"] = list(S.wilson(x["flips"], x["pairs"]))
        r["real_pairs_by_gap"] = bins
        res[p] = r
    out = {"note": "Exploratory (docs/PREREGISTRATION.md, amendment 1). Scores are inverse-sampling-weighted pass "
                   "rates on each system's main-set items.",
           "design": design, "systems": sys_rows, "pipelines": res, "simulation": simulate(d)}
    (OUT / "ranking.json").write_text(json.dumps(rnd(out), indent=1), encoding="utf-8", newline="\n")
    with (OUT / "ranking_systems.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["system", "n_items", "n_right", "gold"] + PIPES)
        for k, row in enumerate(sys_rows):
            w.writerow([row["system"], row["n_items"], row["n_right"], f"{pt['gold'][k]:.4f}"] +
                       [f"{pt['judge'][q, k]:.4f}" for q in range(len(PIPES))])
    print(json.dumps({p: {"mae": round(res[p]["mae"], 4), "flips": res[p]["flips"],
                          "untied": res[p]["gold_untied_pairs"],
                          "tau": {s: rnd(res[p]["by_source"][s]["tau_b"]) for s in SOURCES}} for p in PIPES},
                     indent=1))


if __name__ == "__main__":
    main()
