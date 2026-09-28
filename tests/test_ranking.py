"""Benchmark rankings under judge errors (amendment 1, analysis A)."""
import numpy as np
import pytest
from scipy.stats import kendalltau

from evalab import ranking as R


def test_kendall_tau_b_hand_examples():
    assert R.kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert R.kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert R.kendall_tau_b([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)
    # a tie in x only: 2 concordant pairs, n0 = 3, one x-tied pair
    assert R.kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / np.sqrt(2 * 3))
    # a tie in y only
    assert R.kendall_tau_b([1, 2, 3], [1, 1, 3]) == pytest.approx(2 / np.sqrt(3 * 2))
    assert np.isnan(R.kendall_tau_b([1, 1, 1], [1, 2, 3]))


def test_kendall_tau_b_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(3, 12))
        x, y = rng.integers(0, 4, n).astype(float), rng.integers(0, 4, n).astype(float)
        want = kendalltau(x, y, variant="b").statistic
        got = R.kendall_tau_b(x, y)
        assert (np.isnan(want) and np.isnan(got)) or got == pytest.approx(want)


def test_flip_counting_hand_example():
    gold = np.array([0.9, 0.8, 0.5, 0.5, 0.3])
    judge = np.array([0.7, 0.85, 0.6, 0.4, 0.6])
    a, b = R.all_pairs(5)
    f = R.flip_counts(R.pair_signs(gold, a, b), R.pair_signs(judge, a, b))
    # (0,1) and (3,4) are reversed by the judge; (2,4) is tied by the judge only; (2,3) is tied by gold
    assert f == {"flips": 2, "judge_ties": 1, "gold_untied_pairs": 9, "gold_tied_pairs": 1,
                 "flip_rate": pytest.approx(2 / 9)}


def test_flip_counting_judge_ties_are_not_flips():
    a, b = R.all_pairs(3)
    f = R.flip_counts(R.pair_signs(np.array([0.9, 0.5, 0.1]), a, b), R.pair_signs(np.array([0.5, 0.5, 0.5]), a, b))
    assert f["flips"] == 0 and f["judge_ties"] == 3 and f["gold_untied_pairs"] == 3


def test_a_pair_tied_by_both_gold_and_judge_is_not_a_judge_tie():
    a, b = R.all_pairs(3)
    f = R.flip_counts(R.pair_signs(np.array([0.5, 0.5, 0.1]), a, b), R.pair_signs(np.array([0.4, 0.4, 0.4]), a, b))
    assert f == {"flips": 0, "judge_ties": 2, "gold_untied_pairs": 2, "gold_tied_pairs": 1, "flip_rate": 0.0}


def test_pair_signs_treat_float_noise_as_a_tie():
    a, b = R.all_pairs(2)
    assert R.pair_signs(np.array([0.3, 0.1 + 0.2]), a, b).tolist() == [0]


def test_flip_counts_and_tau_agree_on_untied_rankings():
    rng = np.random.default_rng(5)
    a, b = R.all_pairs(8)
    for _ in range(50):
        g, j = rng.random(8), rng.random(8)
        sg, sj = R.pair_signs(g, a, b), R.pair_signs(j, a, b)
        f = R.flip_counts(sg, sj)
        assert R.tau_b_from_signs(sg, sj) == pytest.approx(1 - 2 * f["flip_rate"])


def test_system_scores_are_weighted_pass_rates():
    passed = np.array([1, 0, 1, 1, 0], dtype=float)
    system = np.array([0, 0, 0, 1, 1])
    weight = np.array([1.0, 3.0, 1.0, 2.0, 2.0])
    s = R.system_scores(passed, system, weight, 3)
    assert s[0] == pytest.approx(2 / 5) and s[1] == pytest.approx(0.5) and np.isnan(s[2])
    both = R.system_scores(np.stack([passed, 1 - passed], axis=1), system, weight, 2)
    assert both[1] == pytest.approx([3 / 5, 0.5])


def test_gap_bins_count_pairs_and_flips():
    out = R.gap_bins(np.array([1, 4, 5, 10, 100]), np.array([1, 0, 1, 0, 1]), [0, 5, 10, 100])
    assert [(x["pairs"], x["flips"]) for x in out] == [(2, 1), (1, 1), (2, 1)]


def test_virtual_pairs_with_a_perfect_judge_never_flip():
    labels = np.array([1] * 20 + [0] * 20)
    dec = np.stack([labels, 1 - labels], axis=1).astype(float)     # a perfect judge and an inverted one
    rp, wp = [np.arange(0, 10), np.arange(10, 20)], [np.arange(20, 30), np.arange(30, 40)]
    sim = R.virtual_pairs(np.random.default_rng(0), dec, rp, wp, n_pairs=300, n_answers=50)
    assert np.allclose(sim["judge_gap"][0], sim["gold_gap"])
    assert np.allclose(sim["judge_gap"][1], -sim["gold_gap"])
    g = np.abs(sim["gold_gap"])
    assert g.max() <= 0.9 + 1e-9 and (g > 0).mean() > 0.9
