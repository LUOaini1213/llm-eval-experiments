"""Statistics for paired comparisons of judges. Pure Python + numpy, so every function is easy to test by hand.

- agreement: Cohen's kappa (two raters), Fleiss' kappa (many raters);
- paired tests: exact McNemar (two-sided binomial on the discordant pairs), exact binomial test;
- intervals: Wilson and Clopper-Pearson for proportions, percentile bootstrap (paired, item-level resampling);
- multiplicity: Holm step-down adjustment;
- effect sizes: paired difference in proportions and the discordant-pair odds ratio b/c with an exact interval;
- power: McNemar sample size (Connor 1987) and a simulation check;
- calibration: expected calibration error and Brier score for scores mapped to [0, 1].
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np


# ------------------------------------------------------------------------------------------------ agreement
def cohen_kappa(a: Sequence, b: Sequence) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(counts: Sequence[Sequence[int]]) -> float:
    """counts[i][j] = number of raters who put item i in category j; every row sums to the same number of raters."""
    m = np.asarray(counts, dtype=float)
    n_items, _ = m.shape
    r = m.sum(axis=1)
    if not np.all(r == r[0]) or r[0] < 2:
        raise ValueError("every item needs the same number (>= 2) of ratings")
    r = r[0]
    p_j = m.sum(axis=0) / (n_items * r)
    p_i = ((m * m).sum(axis=1) - r) / (r * (r - 1))
    p_bar, pe = p_i.mean(), (p_j ** 2).sum()
    if pe == 1:
        return 1.0 if p_bar == 1 else 0.0
    return float((p_bar - pe) / (1 - pe))


# ------------------------------------------------------------------------------------------------ exact tests
def binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binom_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test (sum of probabilities no larger than that of the observed count)."""
    if n == 0:
        return 1.0
    obs = binom_pmf(k, n, p)
    tot = sum(binom_pmf(i, n, p) for i in range(n + 1) if binom_pmf(i, n, p) <= obs * (1 + 1e-7))
    return min(1.0, tot)


def mcnemar_exact(b: int, c: int) -> float:
    """b = pairs where only the first method is right, c = only the second. Two-sided exact p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(binom_pmf(i, n) for i in range(k + 1))
    return min(1.0, p)


def discordant(x: Sequence[bool], y: Sequence[bool]) -> tuple[int, int]:
    b = sum(1 for p, q in zip(x, y) if p and not q)
    c = sum(1 for p, q in zip(x, y) if q and not p)
    return b, c


# ------------------------------------------------------------------------------------------------ intervals
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _beta_ppf(q: float, a: float, b: float) -> float:
    from scipy.stats import beta  # only needed for the exact interval
    return float(beta.ppf(q, a, b))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def bootstrap_ci(values: Sequence[float], b: int = 10000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(b, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(x: Sequence[float], y: Sequence[float], b: int = 10000, seed: int = 0,
                          alpha: float = 0.05):
    """Mean of x minus mean of y with a percentile interval; items are resampled, keeping pairs together."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(b, len(d)))
    diffs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def _cluster_bootstrap_means(values, clusters, strata, b, seed, alpha):
    """Resample whole clusters; unequal sizes use a ratio of sums, not a mean of cluster means."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all():
        raise ValueError("values must be a finite, non-empty one-dimensional sample")
    if not isinstance(b, (int, np.integer)) or isinstance(b, bool) or b <= 0:
        raise ValueError("b must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    clusters = list(clusters)
    strata = [None] * len(v) if strata is None else list(strata)
    if len(clusters) != len(v) or len(strata) != len(v):
        raise ValueError("values, clusters and strata must have the same length")
    groups, group_strata = {}, {}
    for i, (cluster, stratum) in enumerate(zip(clusters, strata)):
        try:
            if cluster in group_strata and group_strata[cluster] != stratum:
                raise ValueError("a cluster cannot span multiple strata")
            group_strata[cluster] = stratum
            groups.setdefault(cluster, []).append(i)
        except TypeError:
            raise ValueError("cluster and stratum labels must be hashable") from None
    sums = np.array([v[indices].sum() for indices in groups.values()])
    sizes = np.array([len(indices) for indices in groups.values()])
    by_stratum = {}
    try:
        for i, cluster in enumerate(groups):
            by_stratum.setdefault(group_strata[cluster], []).append(i)
    except TypeError:
        raise ValueError("cluster and stratum labels must be hashable") from None
    rng = np.random.default_rng(seed)
    totals, counts = np.zeros(b), np.zeros(b, dtype=int)
    for group_indices in by_stratum.values():
        pool = np.asarray(group_indices)
        draws = pool[rng.integers(0, len(pool), size=(b, len(pool)))]
        totals += sums[draws].sum(axis=1)
        counts += sizes[draws].sum(axis=1)
    return v, totals / counts


def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence, strata: Sequence | None = None,
                         b: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI for the item-weighted mean, resampling whole clusters with replacement.

    Every selected cluster contributes all its items, including repeated selections. With strata, resample the
    original number of clusters within each stratum. Cluster sizes and thus replicate item counts may differ.
    This assumes clusters are independent; it does not establish that assumption from repeated observations.
    """
    _, means = _cluster_bootstrap_means(values, clusters, strata, b, seed, alpha)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_cluster_bootstrap_diff(x: Sequence[float], y: Sequence[float], clusters: Sequence,
                                  strata: Sequence | None = None, b: int = 10000, seed: int = 0,
                                  alpha: float = 0.05):
    """Item-weighted x-minus-y difference; pairs and all items of a cluster travel together."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    d, diffs = _cluster_bootstrap_means(x - y, clusters, strata, b, seed, alpha)
    return float(d.mean()), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


# ------------------------------------------------------------------------------------------------ multiplicity
def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------------------------------------ effect sizes
def discordant_odds_ratio(b: int, c: int, alpha: float = 0.05):
    """Conditional (McNemar) odds ratio b/c with an exact interval from the binomial interval of b/(b+c)."""
    n = b + c
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = clopper_pearson(b, n, alpha)
    f = lambda p: p / (1 - p) if p < 1 else float("inf")  # noqa: E731
    return (b / c if c else float("inf"), f(lo), f(hi))


# ------------------------------------------------------------------------------------------------ power
def mcnemar_sample_size(p_disc: float, diff: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Pairs needed to detect a difference `diff` in paired proportions when a fraction `p_disc` of pairs are
    discordant (Connor 1987, normal approximation, two-sided)."""
    from statistics import NormalDist
    za = NormalDist().inv_cdf(1 - alpha / 2)
    zb = NormalDist().inv_cdf(power)
    if not 0 < abs(diff) < p_disc <= 1:
        raise ValueError("need 0 < |diff| < p_disc <= 1")
    n = (za * math.sqrt(p_disc) + zb * math.sqrt(p_disc - diff ** 2)) ** 2 / diff ** 2
    return math.ceil(n)


def mcnemar_power_sim(n: int, p_disc: float, diff: float, alpha: float = 0.05, sims: int = 4000, seed: int = 0) -> float:
    """Simulated power of the exact McNemar test with n pairs."""
    rng = np.random.default_rng(seed)
    pb, pc = (p_disc + diff) / 2, (p_disc - diff) / 2
    hits = 0
    for _ in range(sims):
        draws = rng.multinomial(n, [pb, pc, 1 - pb - pc])
        if mcnemar_exact(int(draws[0]), int(draws[1])) < alpha:
            hits += 1
    return hits / sims


# ------------------------------------------------------------------------------------------------ calibration
def ece(probs: Sequence[float], labels: Sequence[bool], bins: int = 5) -> float:
    """Expected calibration error with equal-width bins over [0, 1]."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier(probs: Sequence[float], labels: Sequence[bool]) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Probability that a random positive outscores a random negative (ties count one half)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
