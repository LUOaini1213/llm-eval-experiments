"""Reliability-weighted jury aggregation (E1 amendment 1, analysis B; exploratory).

Votes are a (items x judges) 0/1 matrix, 1 = the judge passed the answer. Gold `y` is 1 for a right answer.

- `fit_one_coin` / `predict_one_coin`: naive-Bayes weighted vote with one accuracy per judge (weight = log-odds of
  the judge's accuracy);
- `fit_two_coin` / `predict_two_coin`: naive Bayes with a sensitivity and a specificity per judge;
- `dawid_skene`: the same two-coin model fitted by EM from the votes alone, without gold;
- `grouped_stratified_folds` and `cross_val_predict`: k-fold cross-validation grouped by question and stratified by
  source, so no parameter is fitted on an item it is then used to predict.

Smoothing is Laplace, (count + 1) / (n + 2), throughout.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np


def _logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


# ------------------------------------------------------------------------------------------ weighted voting
def fit_one_coin(votes: np.ndarray, y: np.ndarray) -> dict:
    v, y = np.asarray(votes, dtype=int), np.asarray(y, dtype=int)
    n = len(y)
    acc = ((v == y[:, None]).sum(axis=0) + 1) / (n + 2)
    prior = (y.sum() + 1) / (n + 2)
    return {"accuracy": acc, "weight": _logit(acc), "prior": prior}


def predict_one_coin(votes: np.ndarray, params: dict) -> np.ndarray:
    v = np.asarray(votes, dtype=int)
    score = (params["weight"] * (2 * v - 1)).sum(axis=1) + _logit(params["prior"])
    return score > 0


def fit_two_coin(votes: np.ndarray, y: np.ndarray) -> dict:
    v, y = np.asarray(votes, dtype=int), np.asarray(y, dtype=int)
    pos, neg = y == 1, y == 0
    se = (v[pos].sum(axis=0) + 1) / (pos.sum() + 2)
    sp = ((1 - v[neg]).sum(axis=0) + 1) / (neg.sum() + 2)
    prior = (y.sum() + 1) / (len(y) + 2)
    return {"sensitivity": se, "specificity": sp, "prior": prior}


def two_coin_llr(votes: np.ndarray, params: dict) -> np.ndarray:
    """Log posterior odds that each item is right, under the two-coin model."""
    v = np.asarray(votes, dtype=float)
    se, sp = np.asarray(params["sensitivity"]), np.asarray(params["specificity"])
    w_pass = np.log(se) - np.log1p(-sp)          # a pass vote: P(pass | right) / P(pass | wrong)
    w_fail = np.log1p(-se) - np.log(sp)          # a fail vote: P(fail | right) / P(fail | wrong)
    return (v * w_pass + (1 - v) * w_fail).sum(axis=1) + _logit(params["prior"])


def predict_two_coin(votes: np.ndarray, params: dict) -> np.ndarray:
    return two_coin_llr(votes, params) > 0


# ------------------------------------------------------------------------------------------ Dawid-Skene
def dawid_skene(votes: np.ndarray, max_iter: int = 500, tol: float = 1e-9) -> dict:
    """Two-class Dawid-Skene EM on binary votes, without gold. Started from the majority vote (a tie counts one
    half). The M-step uses the same Laplace smoothing as the supervised fit, so the two are comparable."""
    v = np.asarray(votes, dtype=float)
    n, j = v.shape
    maj = v.mean(axis=1)
    t = np.where(maj > 0.5, 1.0, np.where(maj < 0.5, 0.0, 0.5))
    prev = None
    for it in range(1, max_iter + 1):
        # M-step from the current soft labels
        se = ((t[:, None] * v).sum(axis=0) + 1) / (t.sum() + 2)
        sp = (((1 - t)[:, None] * (1 - v)).sum(axis=0) + 1) / ((1 - t).sum() + 2)
        prior = (t.sum() + 1) / (n + 2)
        params = {"sensitivity": se, "specificity": sp, "prior": prior}
        # E-step
        t = 1 / (1 + np.exp(-two_coin_llr(v, params)))
        if prev is not None and max(np.abs(se - prev["sensitivity"]).max(), np.abs(sp - prev["specificity"]).max(),
                                    abs(prior - prev["prior"])) < tol:
            break
        prev = params
    return {**params, "posterior": t, "iterations": it, "n_judges": j}


# ------------------------------------------------------------------------------------------ cross-validation
def grouped_stratified_folds(groups: Sequence, strata: Sequence, k: int, seed: int) -> np.ndarray:
    """Fold number per item. All items of a group (question) share a fold. Within each stratum (source), groups are
    shuffled with `seed` and assigned, largest first, to the fold with the fewest items of that stratum so far."""
    by_stratum = defaultdict(lambda: defaultdict(list))
    for i, (g, s) in enumerate(zip(groups, strata)):
        by_stratum[s][g].append(i)
    for s in by_stratum:
        gs = {g for g, ss in zip(groups, strata) if ss == s}
        if len(gs) < k:
            raise ValueError(f"stratum {s!r} has {len(gs)} groups, fewer than {k} folds")
    fold = np.full(len(groups), -1, dtype=int)
    rng = random.Random(seed)
    for s in sorted(by_stratum, key=str):
        gl = sorted(by_stratum[s], key=str)
        rng.shuffle(gl)
        gl.sort(key=lambda g: -len(by_stratum[s][g]))       # stable: ties keep the shuffled order
        load = [0] * k
        for g in gl:
            f = min(range(k), key=lambda x: (load[x], x))
            for i in by_stratum[s][g]:
                fold[i] = f
            load[f] += len(by_stratum[s][g])
    return fold


def cross_val_predict(votes: np.ndarray, y: np.ndarray, folds: np.ndarray,
                      fit: Callable, predict: Callable, meta: np.ndarray | None = None,
                      log: list | None = None) -> np.ndarray:
    """Out-of-fold predictions. `fit(votes, y, meta)` sees only the training folds; `predict(votes, params, meta)`
    is applied to the held-out fold. If `log` is given, (train indices, test indices) is appended per fold."""
    v, y = np.asarray(votes), np.asarray(y)
    meta = np.asarray(meta) if meta is not None else np.zeros(len(y))
    out = np.zeros(len(y), dtype=bool)
    for f in sorted(set(folds.tolist())):
        te = np.flatnonzero(folds == f)
        tr = np.flatnonzero(folds != f)
        params = fit(v[tr], y[tr], meta[tr])
        out[te] = predict(v[te], params, meta[te])
        if log is not None:
            log.append((tr, te))
    return out


# ------------------------------------------------------------------------------------------ the methods
def _single(v, y, meta):
    acc = (np.asarray(v) == np.asarray(y)[:, None]).mean(axis=0)
    return int(np.argmax(acc))           # ties: the first judge in column order


METHODS: dict[str, tuple[Callable, Callable]] = {
    "best_single_cv": (_single, lambda v, p, m: np.asarray(v)[:, p].astype(bool)),
    "one_coin": (lambda v, y, m: fit_one_coin(v, y), lambda v, p, m: predict_one_coin(v, p)),
    "two_coin": (lambda v, y, m: fit_two_coin(v, y), lambda v, p, m: predict_two_coin(v, p)),
    "two_coin_per_source": (
        lambda v, y, m: {s: fit_two_coin(v[m == s], y[m == s]) for s in set(m.tolist())},
        lambda v, p, m: np.array([predict_two_coin(v[i:i + 1], p[m[i]])[0] for i in range(len(m))], dtype=bool)),
    "dawid_skene": (lambda v, y, m: dawid_skene(v), lambda v, p, m: predict_two_coin(v, p)),
}
