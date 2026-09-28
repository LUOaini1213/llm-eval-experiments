"""Statistics checked against hand calculations, textbook examples and scipy."""
import math

import pytest
from scipy.stats import binomtest

from evalab import stats as S


def test_cohen_kappa_textbook():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> po 0.70, pe 0.50, kappa 0.40
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert S.cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_perfect_and_chance():
    assert S.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert S.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        S.cohen_kappa([1], [1, 0])


def test_fleiss_kappa_wikipedia_example():
    counts = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
              [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert S.fleiss_kappa(counts) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_kappa_two_raters_matches_scott_pi():
    # with two raters Fleiss' kappa equals Scott's pi: po 0.75, pooled p(yes) 5/8, pe 34/64
    counts = [[2, 0], [2, 0], [0, 2], [1, 1]]
    assert S.fleiss_kappa(counts) == pytest.approx((0.75 - 34 / 64) / (1 - 34 / 64))
    with pytest.raises(ValueError):
        S.fleiss_kappa([[2, 0], [1, 0]])


def test_mcnemar_exact_known_values():
    # b=1, c=9: 2 * (C(10,0)+C(10,1)) / 2^10
    assert S.mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert S.mcnemar_exact(9, 1) == pytest.approx(22 / 1024)
    assert S.mcnemar_exact(5, 5) == 1.0
    assert S.mcnemar_exact(0, 0) == 1.0
    for b, c in [(3, 12), (0, 6), (7, 2), (20, 35)]:
        assert S.mcnemar_exact(b, c) == pytest.approx(binomtest(b, b + c, 0.5).pvalue)


def test_binom_test_matches_scipy():
    for k, n, p in [(3, 20, 0.5), (15, 20, 0.5), (2, 30, 0.2), (0, 5, 0.5)]:
        assert S.binom_test_two_sided(k, n, p) == pytest.approx(binomtest(k, n, p).pvalue)


def test_discordant_counts():
    x = [True, True, False, False, True]
    y = [True, False, True, False, False]
    assert S.discordant(x, y) == (2, 1)


def test_wilson_and_clopper_pearson():
    lo, hi = S.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = S.wilson(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))
    lo, hi = S.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-4)
    lo, hi = S.clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.6915, abs=1e-4)


def test_holm_by_hand():
    # sorted 0.01, 0.03, 0.04 -> 0.03, 0.06, max(0.06, 0.04) = 0.06
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert S.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert S.holm([0.001]) == [0.001]


def test_paired_bootstrap():
    d, lo, hi = S.paired_bootstrap_diff([1, 1, 1, 1], [0, 0, 0, 0])
    assert (d, lo, hi) == (1.0, 1.0, 1.0)
    d, lo, hi = S.paired_bootstrap_diff([1, 0, 1, 0], [1, 0, 1, 0])
    assert (d, lo, hi) == (0.0, 0.0, 0.0)
    x = [1] * 60 + [0] * 40
    y = [1] * 50 + [0] * 50
    d, lo, hi = S.paired_bootstrap_diff(x, y, b=4000, seed=1)
    assert d == pytest.approx(0.10)
    # only 10 discordant pairs, all favouring x: the interval stays above zero and is roughly +-0.06 wide
    assert 0.03 < lo < 0.08 and 0.12 < hi < 0.17
    with pytest.raises(ValueError):
        S.paired_bootstrap_diff([1, 0], [1])


def test_bootstrap_ci_brackets_mean():
    lo, hi = S.bootstrap_ci([0, 1] * 50, b=4000, seed=0)
    assert lo < 0.5 < hi
    assert hi - lo == pytest.approx(2 * 1.96 * math.sqrt(0.25 / 100), abs=0.03)


def test_discordant_odds_ratio():
    orr, lo, hi = S.discordant_odds_ratio(8, 2)
    assert orr == 4.0 and lo < 4.0 < hi
    lo_p, hi_p = S.clopper_pearson(8, 10)
    assert lo == pytest.approx(lo_p / (1 - lo_p)) and hi == pytest.approx(hi_p / (1 - hi_p))


def test_mcnemar_sample_size_by_hand():
    # (1.96*sqrt(.2) + 0.8416*sqrt(.19))^2 / .01 = 154.6
    assert S.mcnemar_sample_size(0.2, 0.1) == 155
    with pytest.raises(ValueError):
        S.mcnemar_sample_size(0.05, 0.1)


def test_mcnemar_power_simulation_near_nominal():
    n = S.mcnemar_sample_size(0.2, 0.1)
    p = S.mcnemar_power_sim(n, 0.2, 0.1, sims=2000, seed=3)
    assert 0.70 < p < 0.88  # the exact test is slightly conservative
    assert S.mcnemar_power_sim(n, 0.2, 0.0, sims=2000, seed=3) < 0.07


def test_calibration():
    assert S.ece([0.0, 0.0, 1.0, 1.0], [False, False, True, True]) == 0.0
    assert S.ece([1.0, 1.0], [False, False]) == 1.0
    assert S.brier([1.0, 0.0], [True, True]) == 0.5
    assert S.auroc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert S.auroc([0.5, 0.5], [True, False]) == 0.5
