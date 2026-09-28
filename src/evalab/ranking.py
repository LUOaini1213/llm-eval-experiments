"""Benchmark-level consequences of judge errors (E1 amendment 1, analysis A; exploratory).

A benchmark user reads a judge-scored benchmark to decide which system is better. This module turns per-item
pass/fail decisions into per-system scores and compares the judge's ranking of systems with the gold ranking:

- `system_scores`: weighted pass rate per system (weights undo the main set's stratified sampling);
- `pair_signs`, `kendall_tau_b`, `tau_b_from_signs`: rank agreement, with ties handled as in tau-b;
- `flip_counts`: pairs whose judge order is the reverse of the gold order;
- `gap_bins`: flip rate against the size of the gold gap;
- `virtual_pairs`: the simulation of labelled virtual candidates built by resampling real answers.

Everything is numpy, so the bootstrap in scripts/e1_ranking.py can call the same functions thousands of times.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

TOL = 1e-12


def _sign(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    return np.where(np.abs(d) < TOL, 0, np.sign(d)).astype(int)


def system_scores(passed: np.ndarray, system: np.ndarray, weight: np.ndarray, n_systems: int) -> np.ndarray:
    """Weighted pass rate of each system. `passed` is (items,) or (items, pipelines); returns (systems,) or
    (pipelines, systems). A system with no items gets NaN."""
    w_sum = np.bincount(system, weights=weight, minlength=n_systems)
    with np.errstate(invalid="ignore", divide="ignore"):
        if passed.ndim == 1:
            return np.bincount(system, weights=weight * passed, minlength=n_systems) / w_sum
        return np.stack([np.bincount(system, weights=weight * passed[:, k], minlength=n_systems) / w_sum
                         for k in range(passed.shape[1])])


def all_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.triu_indices(n, k=1)
    return a, b


def pair_signs(scores: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sign of scores[a] - scores[b] for every pair (0 when tied). Works along the last axis."""
    s = np.asarray(scores, dtype=float)
    return _sign(s[..., a] - s[..., b])


def tau_b_from_signs(sg: np.ndarray, sj: np.ndarray) -> float:
    """Kendall's tau-b from the pairwise signs of two rankings over all pairs of a set of systems."""
    sg, sj = np.asarray(sg), np.asarray(sj)
    n0 = sg.shape[-1]
    n1 = (sg == 0).sum(axis=-1)
    n2 = (sj == 0).sum(axis=-1)
    den = np.sqrt((n0 - n1) * (n0 - n2).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = (sg * sj).sum(axis=-1) / den
    return np.where(den > 0, tau, np.nan)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = all_pairs(len(x))
    return float(tau_b_from_signs(pair_signs(np.asarray(x), a, b), pair_signs(np.asarray(y), a, b)))


def flip_counts(sg: np.ndarray, sj: np.ndarray) -> dict:
    """sg, sj: gold and judge pair signs. A flip is a pair ordered strictly by gold and strictly the other way by the
    judge; a judge tie is a pair ordered by gold that the judge ties. Pairs tied by gold are not counted."""
    sg, sj = np.asarray(sg), np.asarray(sj)
    untied = sg != 0
    flips = int(((sg * sj) < 0).sum())
    ties = int((untied & (sj == 0)).sum())
    n = int(untied.sum())
    return {"flips": flips, "judge_ties": ties, "gold_untied_pairs": n, "gold_tied_pairs": int((~untied).sum()),
            "flip_rate": flips / n if n else float("nan")}


def gap_bins(gaps: np.ndarray, flips: np.ndarray, edges: Sequence[float]) -> list[dict]:
    """Flip count per bin of absolute gold gap (same units as `edges`; the last edge is inclusive)."""
    gaps, flips = np.asarray(gaps, dtype=float), np.asarray(flips, dtype=bool)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        m = (gaps >= lo) & ((gaps <= hi) if last else (gaps < hi))
        out.append({"lo": lo, "hi": hi, "pairs": int(m.sum()), "flips": int(flips[m].sum())})
    return out


def virtual_pairs(rng: np.random.Generator, decisions: np.ndarray, right_pools: list[np.ndarray],
                  wrong_pools: list[np.ndarray], n_pairs: int, n_answers: int = 100,
                  acc_range: tuple[float, float] = (0.05, 0.95)) -> dict:
    """Simulated pairs of virtual candidates (clearly a simulation, not real systems).

    A virtual candidate draws its right answers from one real system's right answers (a pool in `right_pools`) and
    its wrong answers from one real system's wrong answers (`wrong_pools`), with replacement. Its gold accuracy is
    k / n_answers with k = round(p * n_answers), p uniform on `acc_range`. Its judge score under pipeline q is the
    mean of decisions[drawn items, q], so each pipeline's error rates on those systems' answers carry over.

    Returns the gold gap (a - b) and the judge gaps (pipelines x pairs) for n_pairs pairs.
    """
    q = decisions.shape[1]
    gold = np.empty((2, n_pairs))
    judge = np.empty((2, q, n_pairs))
    for side in range(2):
        p = rng.uniform(*acc_range, size=n_pairs)
        k = np.rint(p * n_answers).astype(int)
        r_sys = rng.integers(0, len(right_pools), size=n_pairs)
        w_sys = rng.integers(0, len(wrong_pools), size=n_pairs)
        gold[side] = k / n_answers
        for i in range(n_pairs):
            rp, wp = right_pools[r_sys[i]], wrong_pools[w_sys[i]]
            idx = np.concatenate([rp[rng.integers(0, len(rp), size=k[i])],
                                  wp[rng.integers(0, len(wp), size=n_answers - k[i])]])
            judge[side, :, i] = decisions[idx].mean(axis=0)
    return {"gold_gap": gold[0] - gold[1], "judge_gap": judge[0] - judge[1]}
