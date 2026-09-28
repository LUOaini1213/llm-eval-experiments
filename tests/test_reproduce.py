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
    n = len({r["item_id"] for r in rows})
    items = load_jsonl(ROOT / "data" / "e1" / "items_main_order.jsonl")[:n]
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
    from evalab.items import load_jsonl
    rows = load_jsonl(R1 / "judgments_main.jsonl")
    keys = {(r["judge"], r["condition"], r["item_id"]) for r in rows}
    judges = {r["judge"] for r in rows}
    conds = {r["condition"] for r in rows}
    items = {r["item_id"] for r in rows}
    assert len(keys) == len(judges) * len(conds) * len(items) == len(rows)


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
