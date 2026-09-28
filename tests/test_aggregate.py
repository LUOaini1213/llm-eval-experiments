"""Reliability-weighted aggregation, Dawid-Skene and the cross-validation split (amendment 1, analysis B)."""
import numpy as np
import pytest

from evalab import aggregate as G
from evalab import judge as J


# ----------------------------------------------------------------------------------------------- weighted voting
def test_one_coin_fit_is_smoothed_log_odds_of_accuracy():
    v = np.array([[1, 1], [1, 0], [0, 0], [1, 1]])
    y = np.array([1, 1, 0, 0])
    p = G.fit_one_coin(v, y)
    # judge 0 right on items 0, 1, 2 -> (3 + 1) / (4 + 2); judge 1 right on items 0, 2 -> (2 + 1) / 6
    assert p["accuracy"] == pytest.approx([4 / 6, 3 / 6])
    assert p["weight"] == pytest.approx([np.log(2), 0.0])
    assert p["prior"] == pytest.approx(3 / 6)


def test_one_coin_strong_judge_outvotes_two_weak_ones():
    # accuracies 0.9, 2/3, 2/3 -> weights log 9, log 2, log 2; log 9 > 2 log 2
    p = {"weight": np.log([9.0, 2.0, 2.0]), "prior": 0.5}
    v = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0]])
    assert G.predict_one_coin(v, p).tolist() == [True, False, True, False]
    # the plain majority decides the first two the other way
    assert [J.majority([bool(x) for x in row]) for row in v] == [False, True, True, False]


def test_one_coin_with_equal_weights_is_the_majority_vote():
    rng = np.random.default_rng(3)
    v = rng.integers(0, 2, size=(200, 5))
    p = {"weight": np.full(5, 1.3), "prior": 0.5}
    assert G.predict_one_coin(v, p).tolist() == [J.majority([bool(x) for x in row]) for row in v]


def test_one_coin_prior_breaks_a_balanced_vote():
    p = {"weight": np.array([1.0, 1.0]), "prior": 0.7}
    assert G.predict_one_coin(np.array([[1, 0]]), p).tolist() == [True]
    p["prior"] = 0.3
    assert G.predict_one_coin(np.array([[1, 0]]), p).tolist() == [False]


def test_two_coin_fit_counts_sensitivity_and_specificity():
    v = np.array([[1, 1], [1, 0], [0, 0], [1, 0], [0, 1]])
    y = np.array([1, 1, 1, 0, 0])
    p = G.fit_two_coin(v, y)
    assert p["sensitivity"] == pytest.approx([(2 + 1) / 5, (1 + 1) / 5])
    assert p["specificity"] == pytest.approx([(1 + 1) / 4, (1 + 1) / 4])
    assert p["prior"] == pytest.approx(4 / 7)


def test_two_coin_weighs_pass_and_fail_votes_differently():
    # judge 0 passes nearly everything: its pass says little, its fail says a lot
    p = {"sensitivity": np.array([0.99, 0.8]), "specificity": np.array([0.5, 0.8]), "prior": 0.5}
    llr = G.two_coin_llr(np.array([[1, 0], [0, 1]]), p)
    assert llr[0] == pytest.approx(np.log(0.99 / 0.5) + np.log(0.2 / 0.8))
    assert llr[1] == pytest.approx(np.log(0.01 / 0.5) + np.log(0.8 / 0.2))
    assert G.predict_two_coin(np.array([[1, 0], [0, 1]]), p).tolist() == [False, False]


def test_per_source_fit_uses_each_sources_items_only():
    # judge 0 is perfect on source "a" and inverted on "b"; judge 1 is the reverse
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    src = np.array(["a"] * 200 + ["b"] * 200)
    v = np.stack([np.where(src == "a", y, 1 - y), np.where(src == "a", 1 - y, y)], axis=1)
    fit, pred = G.METHODS["two_coin_per_source"]
    p = fit(v, y, src)
    assert p["a"]["sensitivity"][0] > 0.95 and p["b"]["sensitivity"][0] < 0.05
    assert pred(v, p, src).tolist() == (y == 1).tolist()
    fit2, pred2 = G.METHODS["two_coin"]
    assert (pred2(v, fit2(v, y, src), src) == (y == 1)).mean() < 0.8


# ----------------------------------------------------------------------------------------------- Dawid-Skene
def _synthetic(seed=1, n=6000):
    se = np.array([0.95, 0.80, 0.70, 0.90, 0.60])
    sp = np.array([0.85, 0.70, 0.90, 0.60, 0.75])
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.6).astype(int)
    u = rng.random((n, len(se)))
    v = np.where(y[:, None] == 1, u < se, u >= sp).astype(int)
    return v, y, se, sp


def test_dawid_skene_recovers_known_confusion_matrices_without_labels():
    v, y, se, sp = _synthetic()
    p = G.dawid_skene(v)
    assert p["sensitivity"] == pytest.approx(se, abs=0.03)
    assert p["specificity"] == pytest.approx(sp, abs=0.03)
    assert p["prior"] == pytest.approx(0.6, abs=0.03)
    # and it predicts about as well as the two-coin model with the true parameters
    oracle = G.predict_two_coin(v, {"sensitivity": se, "specificity": sp, "prior": 0.6})
    ds = p["posterior"] > 0.5
    assert abs((ds == y).mean() - (oracle == y).mean()) < 0.01
    assert (ds == y).mean() > (np.array([J.majority(list(r == 1)) for r in v]) == y).mean()


def test_dawid_skene_takes_no_labels_and_is_deterministic():
    v, _, _, _ = _synthetic(seed=2, n=500)
    a, b = G.dawid_skene(v), G.dawid_skene(v.copy())
    assert np.array_equal(a["sensitivity"], b["sensitivity"]) and a["iterations"] == b["iterations"]
    fit, _ = G.METHODS["dawid_skene"]
    rng = np.random.default_rng(0)
    p1 = fit(v, rng.integers(0, 2, len(v)), None)
    p2 = fit(v, np.zeros(len(v), dtype=int), None)
    assert np.array_equal(p1["sensitivity"], p2["sensitivity"])


# ----------------------------------------------------------------------------------------------- cross-validation
def _toy_design():
    groups, strata = [], []
    for s, nq in (("fm", 30), ("sgbus", 40), ("sql", 25)):
        for q in range(nq):
            for _ in range(1 + q % 4):
                groups.append(f"{s}-{q}")
                strata.append(s)
    return groups, strata


def test_folds_keep_each_question_in_one_fold_and_balance_sources():
    groups, strata = _toy_design()
    folds = G.grouped_stratified_folds(groups, strata, k=10, seed=7)
    assert (folds >= 0).all() and set(folds.tolist()) == set(range(10))
    seen = {}
    for g, f in zip(groups, folds):
        assert seen.setdefault(g, f) == f, f"question {g} is in two folds"
    for s in set(strata):
        loads = np.bincount(folds[np.array(strata) == s], minlength=10)
        assert loads.max() - loads.min() <= 4        # the largest question has 4 items
    assert np.array_equal(folds, G.grouped_stratified_folds(groups, strata, k=10, seed=7))
    assert not np.array_equal(folds, G.grouped_stratified_folds(groups, strata, k=10, seed=8))


def test_folds_refuse_more_folds_than_questions():
    with pytest.raises(ValueError):
        G.grouped_stratified_folds(["a", "b"], ["x", "x"], k=3, seed=0)


def test_cross_validation_never_fits_on_the_item_it_predicts():
    groups, strata = _toy_design()
    n = len(groups)
    folds = G.grouped_stratified_folds(groups, strata, k=10, seed=0)
    y = np.random.default_rng(1).integers(0, 2, size=n)
    v = np.zeros((n, 1), dtype=int)
    log = []
    # a "memorizer": remembers the gold of every training item (by index) and replays it; it can only beat chance
    # if a test item was in its training data
    fit = lambda vv, yy, m: dict(zip(m.tolist(), yy.tolist()))  # noqa: E731
    pred = lambda vv, p, m: np.array([p.get(i, 0) == 1 for i in m.tolist()], dtype=bool)  # noqa: E731
    out = G.cross_val_predict(v, y, folds, fit, pred, meta=np.arange(n), log=log)
    assert not out.any()
    tested = np.concatenate([te for _, te in log])
    assert sorted(tested.tolist()) == list(range(n))
    for tr, te in log:
        assert not set(tr.tolist()) & set(te.tolist())
        assert not {groups[i] for i in tr} & {groups[i] for i in te}
