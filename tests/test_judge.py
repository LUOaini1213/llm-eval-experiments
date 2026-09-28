"""Reply parsing, jury aggregation, pairwise bookkeeping and the pipeline analysis on toy data."""
import asyncio

import pytest

from evalab import analysis as A
from evalab import judge as J
from evalab.jury import run_jury


@pytest.mark.parametrize("text,want", [
    ("CORRECT", True), ("Correct.", True), ("INCORRECT", False), ("incorrect\n", False),
    ("The answer is NOT CORRECT.", False), ("**INCORRECT**", False), ("I cannot tell", None), ("", None),
])
def test_parse_verdict(text, want):
    assert J.parse_verdict(text) is want


@pytest.mark.parametrize("text,want", [("5", 5), ("Score: 2", 2), ("I would rate this a 4.", 4), ("0", None),
                                       ("10", None), ("4.5", None), ("", None)])
def test_parse_score(text, want):
    assert J.parse_score(text) == want


@pytest.mark.parametrize("text,want", [("A", "A"), ("B.", "B"), ("**B**", "B"), ("(A)", "A"), ("Answer B", "B"),
                                       ("answer: a", None), ("Both are wrong", None), ("A reasonable choice", None),
                                       ("Option A is right", "A")])
def test_parse_choice(text, want):
    assert J.parse_choice(text) == want


def test_majority_unanimity_mean():
    assert J.majority([True, True, False]) is True
    assert J.majority([True, False, None]) is False
    assert J.majority([True, True, False, False]) is False   # a tie rejects
    assert J.majority([True, True, True, False, None]) is True
    assert J.unanimity([True, True, True]) is True
    assert J.unanimity([True, True, None]) is False
    assert J.unanimity([]) is False
    assert J.score_mean([5, 4, 3]) is True
    assert J.score_mean([5, 4, None]) is False              # a missing score counts as 1: mean 3.33
    assert J.score_mean([4, 4, 3.99]) is False
    assert J.score_pass(4) and not J.score_pass(3) and not J.score_pass(None)


def test_pair_outcome():
    o = J.pair_outcome("A", "B")
    assert o["both_right"] and o["consistent"] and o["first_slot"] == 1
    o = J.pair_outcome("A", "A")          # always picks the first slot: inconsistent, right once
    assert not o["consistent"] and o["right_ab"] and not o["right_ba"] and o["first_slot"] == 2
    o = J.pair_outcome(None, "B")
    assert not o["consistent"] and o["right_ba"]


def test_prompts_put_instructions_first_and_hide_reference_when_free():
    item = {"question": "Q?", "reference": "REF", "candidate": "CAND"}
    for cond in J.POINTWISE:
        m = J.pointwise_messages(item, cond)
        assert m[0]["role"] == "system" and "CAND" in m[1]["content"]
        assert ("REF" in m[1]["content"]) == (cond != "bin_free")
    pair = {"question": "Q?", "reference": "REF", "first": "RIGHT", "second": "WRONG"}
    ab = J.pairwise_messages(pair, "pair_free", "AB")[1]["content"]
    ba = J.pairwise_messages(pair, "pair_free", "BA")[1]["content"]
    assert ab.index("RIGHT") < ab.index("WRONG") and ba.index("WRONG") < ba.index("RIGHT")
    assert "REF" not in ab and "REF" in J.pairwise_messages(pair, "pair_ref", "AB")[1]["content"]
    sql = {"task": "sql", "schema": "t(a)", "question": "Q?", "reference": "SELECT 1", "candidate": "SELECT 2"}
    assert "t(a)" in J.pointwise_messages(sql, "bin_ref")[1]["content"]
    with pytest.raises(ValueError):
        J.pointwise_messages(item, "pair_free")


def _toy():
    items = [{"item_id": f"i{k}", "gold": g} for k, g in enumerate(["correct"] * 4 + ["incorrect"] * 4)]
    votes = {  # per judge: pass/fail for i0..i7
        "a": [1, 1, 1, 1, 1, 1, 0, 0],
        "b": [1, 1, 1, 0, 0, 0, 0, 0],
        "c": [1, 1, 0, 1, 1, 0, 1, 0],
    }
    rows = [{"judge": j, "condition": "bin_ref", "item_id": f"i{k}", "parsed": bool(v[k]), "prompt_tokens": 10,
             "output_tokens": 1, "seconds": 0.5} for j, v in votes.items() for k in range(8)]
    return items, A.index_judgments(rows)


def test_pipeline_metrics_on_toy_data():
    items, idx = _toy()
    a = A.build_pipeline("a", ["a"], "bin_ref", "single", items, idx)
    m = A.metrics(a, items)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["false_accept_rate"] == pytest.approx(2 / 4) and m["false_reject_rate"] == 0
    maj = A.build_pipeline("maj", ["a", "b", "c"], "bin_ref", "majority", items, idx)
    assert [maj.decisions[f"i{k}"] for k in range(8)] == [True, True, True, True, True, False, False, False]
    una = A.build_pipeline("una", ["a", "b", "c"], "bin_ref", "unanimity", items, idx)
    assert [una.decisions[f"i{k}"] for k in range(8)] == [True, True, False, False, False, False, False, False]
    mu = A.metrics(una, items)
    assert mu["false_accept_rate"] == 0 and mu["false_reject_rate"] == pytest.approx(0.5)
    assert mu["calls_per_item"] == 3 and mu["seconds_per_item"] == pytest.approx(1.5)
    c = A.compare(maj, a, items)
    assert (c["only_a_right"], c["only_b_right"]) == (1, 0)
    assert A.compare(una, maj, items, subset="incorrect")["n"] == 4


def test_choose_best_and_jury_respects_families():
    pm = {"qwen2.5:3b": {"accuracy": 0.9, "seconds_per_item": 1}, "qwen3.5:4b": {"accuracy": 0.85, "seconds_per_item": 2},
          "llama3.2:3b": {"accuracy": 0.8, "seconds_per_item": 1}, "gemma3:4b": {"accuracy": 0.7, "seconds_per_item": 2},
          "phi4-mini:3.8b": {"accuracy": 0.9, "seconds_per_item": 0.5}}
    best, jury = A.choose_best_and_jury(pm, list(pm))
    assert best == "phi4-mini:3.8b"
    assert jury == ["phi4-mini:3.8b", "qwen2.5:3b", "llama3.2:3b"]


def test_pareto():
    pts = [{"pipeline": "x", "accuracy": 0.9, "seconds_per_item": 3}, {"pipeline": "y", "accuracy": 0.8,
           "seconds_per_item": 1}, {"pipeline": "z", "accuracy": 0.8, "seconds_per_item": 2}]
    assert A.pareto(pts) == ["x", "y"]


def test_run_jury_with_fake_judges():
    async def yes(system, prompt):
        return "CORRECT"

    async def no(system, prompt):
        return "INCORRECT"

    async def garbled(system, prompt):
        return "maybe"

    out = asyncio.run(run_jury(["q1", "q2"], ["a1", "a2"], ["r1", ["r2", "r2b"]],
                               {"j1": yes, "j2": yes, "j3": no}))
    assert out["verdicts"] == [True, True] and out["pass_rate"] == 1.0
    assert "r2 / r2b" in out["prompts"][1]
    out = asyncio.run(run_jury(["q1"], ["a1"], ["r1"], {"j1": yes, "j2": garbled, "j3": no}))
    assert out["verdicts"] == [False] and out["unparsed"]["j2"] == 1
    out = asyncio.run(run_jury(["q1"], ["a1"], ["r1"], {"j1": yes, "j2": yes, "j3": no}, aggregation="unanimity"))
    assert out["verdicts"] == [False]
    with pytest.raises(ValueError):
        asyncio.run(run_jury(["q"], ["a"], ["r"], {"j": yes}, condition="score_ref"))
