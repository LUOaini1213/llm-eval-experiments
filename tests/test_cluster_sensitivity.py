"""Hand-computable clustered samples and frozen-reply report reproduction (no model endpoint)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from evalab import stats as S
from evalab.analysis import Pipeline
from evalab.cluster_sensitivity import build_report, mean_sensitivity, question_profile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cluster_script", ROOT / "scripts/e1_cluster_sensitivity.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def test_one_observation_per_question_equals_original_bootstrap():
    values = [0, 1, 1, 0, 1, 1, 0]
    clusters = list(range(len(values)))
    assert S.cluster_bootstrap_ci(values, clusters, b=4000, seed=7) == \
        pytest.approx(S.bootstrap_ci(values, b=4000, seed=7))
    other = [1, 1, 0, 0, 1, 0, 0]
    assert S.paired_cluster_bootstrap_diff(values, other, clusters, b=4000, seed=7) == \
        pytest.approx(S.paired_bootstrap_diff(values, other, b=4000, seed=7))


def test_whole_question_travels_together():
    # One question has one successful and one unsuccessful answer: every whole-question draw has mean 1/2.
    assert S.cluster_bootstrap_ci([1, 0], ["q", "q"], b=1000) == (0.5, 0.5)
    assert S.bootstrap_ci([1, 0], b=1000) == (0.0, 1.0)
    # Two independent questions, each perfectly correlated internally; copies must not pretend to be 200 clusters.
    values, questions = [1] * 100 + [0] * 100, ["a"] * 100 + ["b"] * 100
    assert S.cluster_bootstrap_ci(values, questions, b=4000) == (0.0, 1.0)
    lo, hi = S.bootstrap_ci(values, b=4000)
    assert 0.4 < lo < 0.5 < hi < 0.6


def test_unequal_questions_keep_item_weighting_in_every_draw():
    # Enumerate the four equiprobable draws of two questions: AA -> 1, AB/BA -> 3/4, BB -> 0.
    # The central 20% interval must be 3/4, not 1/2 (the unweighted mean of the two question means).
    values, questions = [1, 1, 1, 0], ["a", "a", "a", "b"]
    assert S.cluster_bootstrap_ci(values, questions, b=4000, alpha=0.8) == (0.75, 0.75)
    assert S.paired_cluster_bootstrap_diff(values, [0]*4, questions, b=4000, alpha=0.8) == (0.75, 0.75, 0.75)


def test_strata_keep_question_counts_and_paired_sign():
    # One question per source means every stratified resample includes both: exactly 3/4, despite unequal sizes.
    x, y, q, src = [1, 1, 1, 0], [0]*4, ["a"]*3 + ["b"], ["fm"]*3 + ["sql"]
    assert S.cluster_bootstrap_ci(x, q, strata=src, b=100) == (0.75, 0.75)
    assert S.paired_cluster_bootstrap_diff(x, y, q, strata=src, b=100) == (0.75, 0.75, 0.75)
    assert S.paired_cluster_bootstrap_diff(y, x, q, strata=src, b=100) == (-0.75, -0.75, -0.75)
    assert S.paired_cluster_bootstrap_diff(x, x, q, b=100) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("values,clusters,kwargs", [
    ([], [], {}), ([np.nan], ["q"], {}), ([np.inf], ["q"], {}), ([[1, 0]], ["q"], {}),
    ([1], [], {}), ([1, 0], ["q", "q"], {"strata": ["fm", "sql"]}),
    ([1], [["unhashable"]], {}), ([1], ["q"], {"strata": [["unhashable"]]}),
    ([1], ["q"], {"strata": []}), ([1], ["q"], {"b": 0}), ([1], ["q"], {"b": 1.5}),
    ([1], ["q"], {"b": True}), ([1], ["q"], {"alpha": 0}), ([1], ["q"], {"alpha": 1}),
])
def test_invalid_cluster_samples_refused(values, clusters, kwargs):
    with pytest.raises(ValueError):
        S.cluster_bootstrap_ci(values, clusters, **kwargs)


def test_paired_shape_and_reproducibility():
    with pytest.raises(ValueError):
        S.paired_cluster_bootstrap_diff([1, 0], [1], ["a", "b"])
    args = ([1, 0, 1, 1, 0, 0], ["a", "a", "b", "c", "c", "c"])
    assert S.cluster_bootstrap_ci(*args, b=987, seed=13) == S.cluster_bootstrap_ci(*args, b=987, seed=13)


def tiny_items():
    return [{"item_id": str(i), "qid": "a" if i < 3 else "b", "question": "A?" if i < 3 else "B?",
             "source": "fm" if i < 3 else "sql", "gold": "correct" if i % 2 == 0 else "incorrect"}
            for i in range(4)]


def test_repeat_profile_and_equal_question_weighting_are_distinct():
    items = tiny_items()
    profile = question_profile(items)
    assert (profile["n_items"], profile["n_questions"], profile["repeated_questions"],
            profile["extra_candidates"]) == (4, 2, 1, 2)
    assert profile["candidate_count_histogram"] == {"1": 1, "3": 1}
    metrics = mean_sensitivity([1, 1, 1, 0], items, b=100)
    assert metrics["estimate"] == 0.75
    assert metrics["question_weighted_estimate"] == 0.5
    assert metrics["source_stratified_question_ci"] == [0.75, 0.75]


@pytest.mark.parametrize("field,value", [("qid", ""), ("question", "Different?"), ("source", "sgbus"),
                                           ("item_id", "0")])
def test_ambiguous_question_or_item_identity_refused(field, value):
    items = tiny_items()
    items[1][field] = value
    with pytest.raises(ValueError):
        question_profile(items)


def test_pipeline_pair_and_gold_subset_preserve_hand_computed_outcomes():
    items = tiny_items()
    a = Pipeline("a", (), "bin_ref", decisions={"0": True, "1": False, "2": True, "3": True})
    b = Pipeline("b", (), "bin_ref", decisions={"0": True, "1": True, "2": False, "3": False})
    report = build_report(items, {"a": a, "b": b}, {"all": ("a", "b", None),
                                                  "specificity": ("a", "b", "incorrect")}, b=1000)
    assert report["pipelines"]["a"]["accuracy"]["estimate"] == 0.75
    assert report["pipelines"]["a"]["false_accept_rate"]["estimate"] == 0.5
    assert report["comparisons"]["all"]["estimate"] == 0.25
    assert report["comparisons"]["specificity"]["estimate"] == 0.0
    assert report["comparisons"]["specificity"]["n_questions"] == 2
    a.decisions.pop("3")
    with pytest.raises(ValueError, match="cover exactly"):
        build_report(items, {"a": a}, {})


def test_identical_question_text_cannot_hide_under_multiple_ids():
    items = tiny_items()
    items[1]["qid"] = "new-id"
    with pytest.raises(ValueError, match="multiple question IDs"):
        question_profile(items)


def write_loader_fixture(root):
    (root / "data/e1").mkdir(parents=True)
    (root / "results/e1").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "docs/PREREGISTRATION.md").write_text("Main-set size: **4**", encoding="utf-8")
    items = tiny_items()
    (root / "data/e1/items_main_order.jsonl").write_text("\n".join(json.dumps(it) for it in items), encoding="utf-8")
    (root / "results/e1/run_manifest.json").write_text(json.dumps({"n_items": 4}), encoding="utf-8")
    choices = {"best_single": E.JUDGES[0], "jury3": E.JUDGES[:3]}
    (root / "results/e1/pilot_choices.json").write_text(json.dumps(choices), encoding="utf-8")
    rows = [{"judge": j, "condition": c, "item_id": it["item_id"]}
            for j in E.JUDGES for c in E.J.POINTWISE for it in items]
    path = root / "results/e1/judgments_main.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return rows, path


@pytest.mark.parametrize("defect", ["missing_item", "missing_axis", "duplicate", "unknown_axis", "pairwise"])
def test_report_loader_rejects_incomplete_or_ambiguous_matrix(tmp_path, defect):
    rows, path = write_loader_fixture(tmp_path)
    assert len(E.load_main(tmp_path)[0]) == 4
    if defect == "missing_item":
        rows = [r for r in rows if r["item_id"] != "0"]
    elif defect == "missing_axis":
        rows = [r for r in rows if r["judge"] != E.JUDGES[0]]
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "unknown_axis":
        rows[0]["judge"] = "unknown"
    else:
        rows[0]["order"] = "AB"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="complete, unique"):
        E.load_main(tmp_path)


def test_new_report_reproduces_from_frozen_judgments_and_tracks_provenance():
    report = E.generate()
    assert report["sample"]["n_items"] == 400
    assert report["sample"]["n_questions"] == 188
    assert report["sample"]["repeated_questions"] == 105
    assert report["provenance"]["model_calls"] == 0
    assert len(report["pipelines"]) == 34
    assert report["comparisons"]["H1"]["estimate"] == 0.0225
    committed = json.loads(E.REPORT_JSON.read_text(encoding="utf-8"))
    # A source fingerprint alone must not kill a behavioral mutant. The CLI/CI --check covers fingerprints;
    # here all computed outcomes, input identities and rendered tables must still match independently.
    report["provenance"].pop("code_sha256")
    committed["provenance"].pop("code_sha256")
    assert report == committed
    assert E.render(report).encode("utf-8") == E.REPORT_MD.read_bytes()
    assert E.update_readme(report) == (ROOT / "README.md").read_text(encoding="utf-8")
