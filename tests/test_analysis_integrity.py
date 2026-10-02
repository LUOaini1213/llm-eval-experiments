"""A missing experiment cell must not become a zero-cost rejection."""
import copy
import json
import sys
from pathlib import Path

import pytest

from evalab import analysis as A
from evalab.items import load_jsonl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def record(judge="a", parsed=True):
    return dict(judge=judge, condition="bin_ref", item_id="x", parsed=parsed,
                prompt_tokens=10, output_tokens=1, seconds=0.2)


def test_duplicate_judgment_cannot_silently_replace_a_verdict():
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        A.index_judgments([record(), record(parsed=False)])


@pytest.mark.parametrize("entry", ["pipeline", "agreement"])
def test_missing_judge_is_not_a_false_vote(entry):
    items = [dict(item_id="x", gold="correct")]
    idx = A.index_judgments([record()])
    with pytest.raises(ValueError, match="[Mm]issing"):
        if entry == "pipeline":
            A.build_pipeline("b", ["b"], "bin_ref", "single", items, idx)
        else:
            A.agreement(items, idx, ["a", "b"], "bin_ref")


def test_present_unparsed_reply_remains_a_costed_rejection():
    items = [dict(item_id="x", gold="correct")]
    p = A.build_pipeline("a", ["a"], "bin_ref", "single", items,
                         A.index_judgments([record(parsed=None)]))
    assert p.decisions == {"x": False}
    assert p.cost == dict(calls=1, prompt_tokens=10, output_tokens=1, seconds=0.2)


def test_subset_metrics_charge_only_selected_responses():
    items = [dict(item_id="x", gold="correct"), dict(item_id="y", gold="incorrect")]
    rows = [record(), dict(record(parsed=False), item_id="y", prompt_tokens=30, seconds=0.8)]
    p = A.build_pipeline("a", ["a"], "bin_ref", "single", items, A.index_judgments(rows))
    cheap, expensive = A.metrics(p, items[:1]), A.metrics(p, items[1:])
    assert cheap["calls_per_item"] == expensive["calls_per_item"] == 1
    assert cheap["tokens_per_item"] == 11
    assert expensive["tokens_per_item"] == 31
    assert cheap["seconds_per_item"] == pytest.approx(0.2)
    assert expensive["seconds_per_item"] == pytest.approx(0.8)


@pytest.mark.parametrize("change", [dict(parsed=1), dict(parsed="True"), dict(seconds=float("nan")),
                                    dict(seconds=-1), dict(prompt_tokens=float("inf")),
                                    dict(condition="score_ref", parsed=True),
                                    dict(condition="score_ref", parsed=6)])
def test_invalid_stored_verdict_or_cost_is_not_reported(change):
    with pytest.raises(ValueError):
        A.index_judgments([dict(record(), **change)])


def test_pairwise_requires_both_orders_and_unique_cells():
    pair = dict(pair_id="x")
    ab = dict(record(), condition="pair_ref", order="AB", parsed="A")
    ba = dict(ab, order="BA", parsed="B")
    assert len(A.validate_matrix([ab, ba], [pair], ["a"], ["pair_ref"], pairwise=True)) == 2
    for rows in ([ab], [ab, ba, ba]):
        with pytest.raises(ValueError):
            A.validate_matrix(rows, [pair], ["a"], ["pair_ref"], pairwise=True)


def test_main_prefix_cannot_shrink_with_a_truncated_item_file(monkeypatch):
    import evalab.items as I
    original = I.load_jsonl
    monkeypatch.setattr(I, "load_jsonl", lambda path: original(path)[:399])
    with pytest.raises(ValueError, match="truncated"):
        A.load_main_items(ROOT)


def test_named_main_uses_its_own_manifest_and_checks_identity(tmp_path):
    from evalab.experiment import run_identity
    items = A.load_main_items(ROOT)
    manifest = json.loads((ROOT / "results/e1/run_manifest.json").read_text(encoding="utf-8"))
    manifest.update(schema_version=1, identity=run_identity("main", items))
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert A.load_main_items(ROOT, tmp_path) == items
    manifest["identity"]["prompts_sha256"] = "changed"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        A.load_main_items(ROOT, tmp_path)


def test_named_auxiliary_data_requires_matching_provenance(tmp_path):
    from evalab.experiment import run_identity
    items = load_jsonl(ROOT / "data/e1/items_robust.jsonl")
    with pytest.raises(ValueError, match="Missing"):
        A.validate_aux_manifest(ROOT, tmp_path, "robust", items)
    path = tmp_path / "run_manifest_robust.json"
    path.write_text(json.dumps(dict(schema_version=1, identity=run_identity("robust", items))), encoding="utf-8")
    A.validate_aux_manifest(ROOT, tmp_path, "robust", items)
    changed = [dict(items[0], candidate="different answer")] + items[1:]
    with pytest.raises(ValueError, match="identity"):
        A.validate_aux_manifest(ROOT, tmp_path, "robust", changed)


@pytest.mark.parametrize("module", ["e1_ranking", "e1_aggregation"])
@pytest.mark.parametrize("axis", ["judge", "condition", "item_id", "duplicate", "unexpected"])
def test_followups_reject_incomplete_or_conflicting_main_data(monkeypatch, module, axis):
    import importlib
    m = importlib.import_module(module)
    original = m.load_jsonl

    def damaged(path):
        rows = original(path)
        if path.name != "judgments_main.jsonl":
            return rows
        if axis == "duplicate":
            return rows + [copy.deepcopy(rows[0])]
        if axis == "unexpected":
            extra = copy.deepcopy(rows[0])
            extra["item_id"] = "not-in-the-frozen-main-set"
            return rows + [extra]
        removed = rows[0][axis]
        return [r for r in rows if r[axis] != removed]

    monkeypatch.setattr(m, "load_jsonl", damaged)
    with pytest.raises(ValueError):
        m.load()


@pytest.mark.parametrize("filename", ["judgments_main.jsonl", "judgments_robust.jsonl", "judgments_pairs.jsonl"])
def test_main_analysis_preflights_all_sets_before_writing(monkeypatch, filename):
    import e1_analyze as E
    original = E.load_jsonl
    writes = []

    def damaged(path):
        rows = original(path)
        return rows[1:] if path.name == filename else rows

    def no_metrics_before_validation(*args, **kwargs):
        pytest.fail("analysis started before validating every input set")

    monkeypatch.setattr(E, "load_jsonl", damaged)
    monkeypatch.setattr(E, "dump", lambda *args: writes.append(args))
    monkeypatch.setattr(A, "metrics", no_metrics_before_validation)
    with pytest.raises(ValueError, match="[Mm]issing"):
        E.main_analysis()
    assert not writes
