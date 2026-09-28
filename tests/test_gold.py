"""Deterministic gold checks, the E3 generator's gold correctness, and the controlled perturbations."""
import csv
from collections import Counter
from pathlib import Path

import pytest

from evalab import gold as G
from evalab.items import (fact_card, fm_wrong_reference, label_fm, label_sgbus, load_jsonl, perturbations,
                          sgbus_reference, sgbus_wrong_reference)
from evalab.sgfacts import FACTS, TIME_ANCHOR, generate, verify_items

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------- gold checks
@pytest.mark.parametrize("answer,want", [
    ("The minimum clear width is 850mm [1].", "correct"),
    ("Not less than 850 mm; exit access doors may be 750 mm [1].", "ambiguous"),   # a competing measurement
    ("It must be 0.85 m.", "ambiguous"),                                           # unit-converted
    ("It must be 8500 mm.", "incorrect"),                                          # x10 is not a unit change
    ("I don't know.", "incorrect"),
    ("1,850 mm", "incorrect"),
])
def test_check_expected_single(answer, want):
    assert G.check_expected(answer, [["850"]]) == want


def test_check_expected_groups_percent_and_text():
    assert G.check_expected("Buildings over four storeys need a 0.3 sq m chute.", [["4", "four"], ["0.3"]]) == "correct"
    assert G.check_expected("Buildings over 4 storeys need a 0.5 sq m chute.", [["4", "four"], ["0.3"]]) == "incorrect"
    assert G.check_expected("20% of units in buildings over 30 years", [["30%"]]) == "incorrect"
    assert G.check_expected("at least 30 per cent of the columns", [["30%"]]) == "correct"
    assert G.check_expected("Meters of IEC Class 1 or better", [["Class 1"]]) == "correct"
    assert G.check_expected("Meters of IEC Class 10", [["Class 1"]]) == "incorrect"
    # numbers that appear in the question do not count as competing measurements
    assert G.check_expected("A 1:12 ramp may run 6000 mm.", [["6000"]], question="ramp of 1:12") == "correct"


def test_check_number_ignores_question_numbers():
    assert G.check_number("Service 36 stops at 42 stops.", 42, exclude={36}) == "correct"
    assert G.check_number("Service 36 stops at 42 or 43 stops.", 42, exclude={36}) == "ambiguous"
    assert G.check_number("It has 41 stops.", 42) == "incorrect"
    assert G.check_number("About 24.6 km long", 24.6, tol=0.05) == "correct"
    assert G.check_number("About 24.7 km long", 24.6, tol=0.05 + 1e-9) == "incorrect"
    assert G.check_number("Stop 01012 has 11 services", 11, exclude={1012}) == "correct"


def test_check_time_formats():
    for a in ("0530", "05:30", "5:30 am", "5.30am", "The first bus leaves at 0530 hrs."):
        assert G.check_time(a, "0530") == "correct", a
    assert G.check_time("5:30 pm", "0530") == "incorrect"
    assert G.check_time("12:05 am", "0005") == "correct"
    assert G.check_time("0530 or 0600", "0530") == "ambiguous"


def test_check_choice_with_aliases_and_competitors():
    areas = {"KALLANG": ["Kallang"], "GEYLANG": ["Geylang"], "DOWNTOWN CORE": ["Downtown Core"]}
    assert G.check_choice("It is in Kallang.", "KALLANG", areas) == "correct"
    assert G.check_choice("Kallang or Geylang", "KALLANG", areas) == "ambiguous"
    assert G.check_choice("Geylang", "KALLANG", areas) == "incorrect"
    assert G.check_choice("Operated by SBS Transit", "SBST", G.OPERATORS) == "correct"
    assert G.check_choice("SMRT Buses runs it", "SBST", G.OPERATORS) == "incorrect"
    assert G.check_choice("Go-Ahead Singapore", "GAS", G.OPERATORS) == "correct"
    roads = {"Victoria St": ["Victoria St"], "Victoria Ave": ["Victoria Ave"]}
    assert G.check_choice("It is on Victoria Street.", "Victoria St", roads, normaliser=G.norm_road) == "correct"


def test_check_service_set():
    assert G.check_service_set("Services 43 and 43A", {"43", "43A"}) == "correct"
    assert G.check_service_set("Only 43", {"43", "43A"}) == "incorrect"
    assert G.check_service_set("43, 43A and 2", {"43", "43A"}) == "ambiguous"


def test_refusals():
    assert G.is_refusal("NOT_FOUND") and G.is_refusal("I'm sorry, I cannot answer") and G.is_refusal("  ")
    assert not G.is_refusal("It is 850 mm.")


# ------------------------------------------------------------------------------------------- generator
@pytest.fixture(scope="module")
def items():
    return generate(180)


def test_generator_gold_verifies_independently(items):
    assert len(items) == 180
    assert verify_items(items) == []


def test_verify_catches_a_planted_wrong_gold(items):
    bad = [type(items[0])(**{**items[0].__dict__})]
    bad[0].gold = bad[0].gold + "X"
    assert verify_items(bad)


def test_generator_is_deterministic_and_seeded(items):
    again = generate(180)
    assert [i.question for i in again] == [i.question for i in items]
    other = generate(180, seed=1)
    assert [i.question for i in other] != [i.question for i in items]


def test_generator_controls(items):
    lv = Counter(i.difficulty for i in items)
    assert (lv["easy"], lv["medium"], lv["hard"]) == (72, 63, 45)
    for t in {i.template for i in items}:
        ents = [i.entity for i in items if i.template == t]
        assert len(ents) == len(set(ents)), t                    # no entity reused within a template
        golds = Counter(i.gold for i in items if i.template == t)
        n = sum(golds.values())
        cap = 0.25 if t == "operator" else 0.2
        assert max(golds.values()) <= max(1, int(cap * n)), t    # answer-share cap
    stops = {r["stop_code"]: r for r in csv.DictReader((FACTS / "stops.csv").open(encoding="utf-8"))}
    for i in items:
        assert i.question.endswith(TIME_ANCHOR)
        if i.template == "area":
            assert float(stops[i.entity]["boundary_dist_m"]) >= 50   # no stop near a planning-area boundary


def test_mix_changes_counts():
    small = generate(40, mix=(1.0, 0.0, 0.0))
    assert {i.difficulty for i in small} == {"easy"}


def test_committed_items_match_generator(items):
    committed = load_jsonl(ROOT / "data" / "e3" / "sgbus_items.jsonl")
    assert [c["question"] for c in committed] == [i.question for i in items]
    assert [c["gold"] for c in committed] == [i.gold for i in items]


# ------------------------------------------------------------------------------------------- perturbations
def test_sgbus_references_and_perturbations_label_as_intended():
    for it in load_jsonl(ROOT / "data" / "e3" / "sgbus_items.jsonl"):
        assert label_sgbus(it, sgbus_reference(it)) == "correct", it["id"]
        assert label_sgbus(it, sgbus_wrong_reference(it)) == "incorrect", it["id"]
        vs = perturbations({"id": it["id"], "reference": sgbus_reference(it), "wrong": sgbus_wrong_reference(it)})
        assert Counter((v["style"], v["correct"]) for v in vs) == Counter(
            {("terse", True): 1, ("verbose", True): 1, ("terse", False): 1, ("verbose", False): 1})
        for v in vs:
            assert label_sgbus(it, v["answer"]) == ("correct" if v["correct"] else "incorrect"), (it["id"], v)


def test_sgbus_counts_ignore_mangled_stop_codes():
    it = {"kind": "int", "gold": "31", "question": "How many stops does service 172 call at from Choa Chu Kang Int "
                                                    "(44009) to Boon Lay Int (22009)?"}
    assert label_sgbus(it, "Service 172 calls at 31 stops from Choa Chu Kang Int (4409).") == "correct"
    assert label_sgbus(it, "Service 172 calls at 31 or 32 stops.") == "ambiguous"
    assert label_sgbus(it, "It calls at 30 stops.") == "incorrect"


def test_fm_perturbations_label_as_intended():
    for spec in load_jsonl(ROOT / "data" / "e1" / "fm_gold.jsonl"):
        assert label_fm(spec, spec["reference"]) == "correct", spec["id"]
        assert label_fm(spec, fm_wrong_reference(spec)) == "incorrect", spec["id"]


def test_verbose_variant_keeps_content():
    vs = perturbations({"id": "x", "reference": "13 m", "wrong": "10 m"})
    terse = {v["correct"]: v["answer"] for v in vs if v["style"] == "terse"}
    verbose = {v["correct"]: v["answer"] for v in vs if v["style"] == "verbose"}
    for c in (True, False):
        assert terse[c] in verbose[c] and len(verbose[c]) > 3 * len(terse[c])
    # the same wrapper for the right and the wrong answer, so only the value differs
    assert verbose[True].replace("13 m", "?") == verbose[False].replace("10 m", "?")


def test_fact_card_contains_the_answer_row():
    for it in load_jsonl(ROOT / "data" / "e3" / "sgbus_items.jsonl")[::7]:
        card = fact_card(it)
        assert len(card.splitlines()) >= 4
        if it["template"] in ("road", "area", "operator"):
            assert sgbus_reference(it).lower() in card.lower(), it["id"]
