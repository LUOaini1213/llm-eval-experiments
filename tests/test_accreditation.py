"""E5: the rubric checks, totals and quotation matching, and the committed assessments."""
import copy
import json
from pathlib import Path

from evalab import accreditation as AC

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = json.loads((ROOT / "data" / "accreditation" / "rubric.json").read_text(encoding="utf-8"))
ASSESS = json.loads((ROOT / "data" / "accreditation" / "assessments.json").read_text(encoding="utf-8"))
IDS = [i["id"] for i in AC.indicators(RUBRIC)]


def toy(scores=None):
    s = {i: {"score": 1, "page": 1, "quote": "short quote", "why": "reason"} for i in IDS}
    s.update(scores or {})
    return {"id": "T", "scores": s}


def test_rubric_shape():
    assert len(RUBRIC["criteria"]) == 5 and len(IDS) == 20 and len(set(IDS)) == 20
    assert sum(i["critical"] for i in AC.indicators(RUBRIC)) == 6


def test_validate_accepts_complete_assessment():
    assert AC.validate(RUBRIC, toy()) == []


def test_validate_catches_each_defect():
    r = toy()
    del r["scores"]["C3.2"]
    assert AC.validate(RUBRIC, r) == ["T: C3.2 not scored"]
    assert AC.validate(RUBRIC, toy({"C9.9": {"score": 0, "why": "x"}})) == ["T: C9.9 is not in the rubric"]
    assert "off the 0-2 scale" in AC.validate(RUBRIC, toy({"C1.1": {"score": 3, "page": 1, "quote": "q", "why": "x"}}))[0]
    assert "needs a page and a quotation" in AC.validate(RUBRIC, toy({"C1.1": {"score": 1, "page": None, "quote": None, "why": "x"}}))[0]
    assert "no justification" in AC.validate(RUBRIC, toy({"C1.1": {"score": 0, "why": " "}}))[0]
    long_q = " ".join(["w"] * (AC.MAX_QUOTE_WORDS + 1))
    assert "longer than" in AC.validate(RUBRIC, toy({"C1.1": {"score": 2, "page": 1, "quote": long_q, "why": "x"}}))[0]
    ok_q = " ".join(["w"] * AC.MAX_QUOTE_WORDS)
    assert AC.validate(RUBRIC, toy({"C1.1": {"score": 2, "page": 1, "quote": ok_q, "why": "x"}})) == []
    assert AC.validate(RUBRIC, toy({"C1.1": {"score": 0, "page": None, "quote": None, "why": "looked for X"}})) == []


def test_totals_and_critical_gaps():
    r = toy({"C1.1": {"score": 0, "why": "x"}, "C1.2": {"score": 2, "page": 1, "quote": "q", "why": "x"},
             "C5.4": {"score": 0, "why": "x"}})
    t = AC.totals(RUBRIC, r)
    assert t["C1"] == {"name": "Coverage", "score": 0 + 2 + 1 + 1, "max": 8}
    assert t["C5"]["score"] == 3
    assert t["total"] == {"name": "Total", "score": 20 - 1 + 1 - 1, "max": 40}
    assert AC.critical_gaps(RUBRIC, r) == ["C1.1"]  # C5.4 is not critical


def test_quote_found_tolerates_extraction_noise_but_not_other_words():
    page = "Tests were executed using Resaro's LLM eval-\nuation platform: For narration T ools, alerts"
    assert AC.quote_found("Resaro's LLM evaluation platform", page)
    assert AC.quote_found("Tests were executed ... narration tools", page)
    assert not AC.quote_found("Resaro's LLM judging platform", page)
    assert not AC.quote_found("narration tools ... Tests were executed using a judge", page)


def test_committed_assessments_are_valid_and_match_results():
    for rep in ASSESS["reports"]:
        assert AC.validate(RUBRIC, rep) == [], rep["id"]
    scores = json.loads((ROOT / "results" / "accreditation" / "scores.json").read_text(encoding="utf-8"))
    for rep in ASSESS["reports"]:
        assert scores["totals"][rep["id"]] == AC.totals(RUBRIC, rep)
        assert scores["critical_gaps"][rep["id"]] == AC.critical_gaps(RUBRIC, rep)
