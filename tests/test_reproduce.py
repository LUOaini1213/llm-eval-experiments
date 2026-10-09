"""Every committed E1/E2 number must be recomputable from the committed raw replies (no model needed)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "results" / "e1"
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.mark.skipif(not (R1 / "main_metrics.json").exists(), reason="E1 main run not present")
def test_main_metrics_recompute_from_judgments():
    from e1_analyze import pipelines

    from evalab import analysis as A
    from evalab.items import load_jsonl

    choices = json.loads((R1 / "pilot_choices.json").read_text(encoding="utf-8"))
    rows = load_jsonl(R1 / "judgments_main.jsonl")
    items = A.load_main_items(ROOT)
    ps = pipelines(items, A.index_judgments(rows), choices["best_single"], choices["jury3"])
    committed = json.loads((R1 / "main_metrics.json").read_text(encoding="utf-8"))
    assert set(committed) == set(ps)
    for k, p in ps.items():
        m = A.metrics(p, items)
        for f in ("accuracy", "false_accept_rate", "false_reject_rate", "calls_per_item", "seconds_per_item"):
            assert m[f] == pytest.approx(committed[k][f]), (k, f)


@pytest.mark.skipif(not (R1 / "run_manifest.json").exists(), reason="E1 main run not present")
def test_main_set_size_matches_preregistration():
    import re
    prereg = (ROOT / "docs" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    n = int(re.search(r"Main-set size:\s*\**\s*(\d+)", prereg).group(1))
    man = json.loads((R1 / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["n_items"] == n


@pytest.mark.skipif(not (R1 / "judgments_main.jsonl").exists(), reason="E1 main run not present")
def test_every_main_item_has_every_judgment():
    from evalab import analysis as A, judge as J
    from evalab.experiment import JUDGES
    from evalab.items import load_jsonl
    rows = load_jsonl(R1 / "judgments_main.jsonl")
    items = A.load_main_items(ROOT)
    idx = A.validate_matrix(rows, items, JUDGES, J.POINTWISE)
    assert len(idx) == len(JUDGES) * len(J.POINTWISE) * len(items) == len(rows)


def test_committed_subset_costs_match_only_the_selected_raw_calls():
    from evalab import analysis as A
    from evalab.items import load_jsonl
    items = A.load_main_items(ROOT)
    rows = load_jsonl(R1 / "judgments_main.jsonl")
    per_source = json.loads((R1 / "per_source.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((R1 / "sensitivity_phrase_gold.json").read_text(encoding="utf-8"))
    for source, report in [(s, [v for k, v in per_source.items() if k.startswith(s + "|")])
                            for s in {it["source"] for it in items}] + [(None, list(sensitivity["metrics"].values()))]:
        ids = {it["item_id"] for it in items if (it["source"] == source if source else
                                                it["qid"] not in sensitivity["excluded_questions"])}
        for m in report:
            calls = [r for r in rows if r["item_id"] in ids and r["judge"] in m["judges"]
                     and r["condition"] == m["condition"]]
            assert m["calls_per_item"] == pytest.approx(len(calls) / len(ids))
            assert m["tokens_per_item"] == pytest.approx(
                sum(r["prompt_tokens"] + r["output_tokens"] for r in calls) / len(ids))
            assert m["seconds_per_item"] == pytest.approx(sum(r["seconds"] for r in calls) / len(ids))


def test_time_gold_audit_is_reproducible_and_preserves_frozen_labels():
    from e1_audit_time_gold import audit
    committed = json.loads((R1 / "time_gold_audit.json").read_text(encoding="utf-8"))
    assert audit() == committed
    assert all(not s["changed_item_ids"] for s in committed["frozen_set_impact"].values())


@pytest.mark.skipif(not (R1 / "run_manifest.json").exists(), reason="E1 main run not present")
def test_preregistration_unchanged_since_main_run_started():
    """The original plan (everything before the '## Amendments' heading) must still have the hash recorded when the
    main run started. Dated, exploratory amendments may only be appended after it."""
    import hashlib
    man = json.loads((R1 / "run_manifest.json").read_text(encoding="utf-8"))
    raw = (ROOT / "docs" / "PREREGISTRATION.md").read_bytes()
    marker = b"\n\n## Amendments\n"
    assert raw.count(marker) <= 1
    original = raw[:raw.index(marker) + 1] if marker in raw else raw
    assert hashlib.sha256(original).hexdigest() == man["preregistration_sha256"]


# ---------------------------------------------------------------------------------- amendment 1 (exploratory)
@pytest.mark.skipif(not (R1 / "ranking.json").exists(), reason="analysis A not run")
def test_ranking_point_estimates_recompute_from_judgments():
    import e1_ranking as E

    d = E.load()
    pt = E.point_estimates(d)
    committed = json.loads((R1 / "ranking.json").read_text(encoding="utf-8"))
    assert [s["system"] for s in committed["systems"]] == d["systems"]
    for q, p in enumerate(E.PIPES):
        c = committed["pipelines"][p]
        assert c["mae"] == pytest.approx(pt["mae"][q], abs=1e-9), p
        assert c["flips"] == pt["flips"][q] and c["gold_untied_pairs"] == pt["untied"], p
        for s in E.SOURCES:
            want = c["by_source"][s]["tau_b"]
            got = pt[f"tau|{s}"][q]
            assert (want is None and got != got) or want == pytest.approx(got, abs=1e-9), (p, s)


@pytest.mark.skipif(not (R1 / "aggregation.json").exists(), reason="analysis B not run")
def test_aggregation_recomputes_out_of_fold_from_judgments():
    import e1_aggregation as E

    items, votes, y, src, qid, jury3 = E.load()
    folds, pred = E.predictions(votes, y, src, qid, jury3, E.CV_SEED)
    committed = json.loads((R1 / "aggregation.json").read_text(encoding="utf-8"))
    for m, p in pred.items():
        assert committed["methods"][m]["accuracy"] == pytest.approx(float((p == (y == 1)).mean()), abs=1e-9), m
    # the committed split keeps every question in one fold, and every item was predicted out of fold
    import csv
    with (R1 / "aggregation_predictions.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["fold"]) for r in rows] == folds.tolist()
    for m, p in pred.items():
        assert [r[m] == "1" for r in rows] == p.tolist(), m
    fold_of = {}
    for it, r in zip(items, rows):
        assert r["item_id"] == it["item_id"]
        assert fold_of.setdefault(it["qid"], r["fold"]) == r["fold"], it["qid"]
