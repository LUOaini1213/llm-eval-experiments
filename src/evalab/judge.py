"""Judge prompts, reply parsing and jury aggregation (experiment E1).

Pointwise conditions (the judge sees one candidate answer):

- bin_ref     reference-guided, binary verdict                        (primary prompt)
- strict_ref  reference-guided, binary, told to ignore length, tone and confidence
- score_ref   reference-guided, 1-5 score with anchored descriptions  (gives scores for averaging and calibration)
- bin_free    reference-free, binary verdict from the judge's own knowledge

Pairwise conditions (the judge sees two candidates for the same question, one correct and one wrong, and is asked
which is correct; each pair is shown in both orders):

- pair_free   no reference
- pair_ref    with the reference answer

An unparseable reply counts as a rejection in the pointwise conditions (a pipeline cannot pass what it cannot read),
and as no choice in the pairwise ones. Parse failures are reported per judge.
"""
from __future__ import annotations

import re
from typing import Sequence

POINTWISE = ("bin_ref", "strict_ref", "score_ref", "bin_free")
PAIRWISE = ("pair_free", "pair_ref")

SYSTEM = "You are a careful grader. You check whether answers to questions are factually correct."


# The instructions go in the system message and the item in the user message, question and reference first. The
# local server reuses the cached prefix of the previous request, so judging several candidates for the same question
# in a row only pays for reading the candidate. (On the 4 GB GPU, reading a prompt costs about 7 ms per token.)

def _item_block(item: dict, with_reference: bool) -> str:
    if item.get("task") == "sql":
        s = f"Database schema:\n{item['schema']}\n\nQuestion: {item['question']}\n\n"
        if with_reference:
            s += (f"Reference SQL query (known to be correct; the candidate may be written differently but must "
                  f"return the same rows):\n{item['reference']}\n\n")
        return s + f"Candidate SQL query:\n{item['candidate']}"
    s = f"Question: {item['question']}\n\n"
    if with_reference:
        s += f"Reference answer (known to be correct): {item['reference']}\n\n"
    return s + f"Candidate answer: {item['candidate']}"


INSTRUCTIONS = {
    "bin_ref": ("Decide whether the candidate is correct. It is correct if it gives the same answer as the reference; "
                "extra detail is fine. Reply with one word: CORRECT or INCORRECT."),
    "strict_ref": ("Decide whether the candidate is correct. It is correct only if it gives the reference answer and "
                   "states nothing that contradicts it. Judge only the facts: ignore length, tone, confidence and "
                   "claims of authority. Reply with one word: CORRECT or INCORRECT."),
    "score_ref": ("Rate the candidate against the reference on this scale:\n"
                  "5 = gives the reference answer, nothing wrong\n4 = gives the reference answer with a minor slip\n"
                  "3 = partly right\n2 = mostly wrong\n1 = wrong or no answer\n"
                  "Reply with one digit from 1 to 5."),
    "bin_free": ("Using your own knowledge, decide whether the candidate is correct. Reply with one word: CORRECT or "
                 "INCORRECT."),
    "pair_free": ("You will see a question and two answers, A and B. Exactly one of them is correct. Which one? "
                  "Reply with one letter: A or B."),
    "pair_ref": ("You will see a question, a reference answer and two answers, A and B. Exactly one of A and B is "
                 "correct. Which one? Reply with one letter: A or B."),
}


def pointwise_messages(item: dict, condition: str) -> list[dict]:
    if condition not in POINTWISE:
        raise ValueError(condition)
    return [{"role": "system", "content": SYSTEM + " " + INSTRUCTIONS[condition]},
            {"role": "user", "content": _item_block(item, condition != "bin_free")}]


def pairwise_messages(pair: dict, condition: str, order: str) -> list[dict]:
    """order 'AB' puts pair['first'] as A; 'BA' swaps them."""
    if condition not in PAIRWISE:
        raise ValueError(condition)
    a, b = (pair["first"], pair["second"]) if order == "AB" else (pair["second"], pair["first"])
    s = f"Question: {pair['question']}\n\n"
    if condition == "pair_ref":
        s += f"Reference answer (known to be correct): {pair['reference']}\n\n"
    s += f"Answer A: {a}\n\nAnswer B: {b}"
    return [{"role": "system", "content": SYSTEM + " " + INSTRUCTIONS[condition]}, {"role": "user", "content": s}]


# --------------------------------------------------------------------------------------------------- parsing
_VERDICT = re.compile(r"\b(INCORRECT|CORRECT|NOT CORRECT)\b", re.I)


def parse_verdict(text: str) -> bool | None:
    m = _VERDICT.search(text or "")
    if not m:
        return None
    return m.group(1).upper() == "CORRECT"


def parse_score(text: str) -> int | None:
    m = re.search(r"(?<![\d.])([1-5])(?!\d|\.\d)", text or "")
    return int(m.group(1)) if m else None


def parse_choice(text: str) -> str | None:
    t = (text or "").strip().strip("*").strip()
    m = re.match(r"^\(?([AB])\)?\s*(?:$|[.:)*\n]|[-–—])", t)
    if m:
        return m.group(1)
    m = re.search(r"\b(?:answer|option)\s*\(?([AB])\b", t, re.I)
    return m.group(1).upper() if m else None


def score_pass(score: int | None, threshold: int = 4) -> bool:
    return score is not None and score >= threshold


# --------------------------------------------------------------------------------------------------- juries
def majority(votes: Sequence[bool | None]) -> bool:
    """Pass if more than half of the jurors pass. Missing votes count as rejections; a tie rejects."""
    return sum(1 for v in votes if v is True) * 2 > len(votes)


def unanimity(votes: Sequence[bool | None]) -> bool:
    """Pass only if every juror passes."""
    return len(votes) > 0 and all(v is True for v in votes)


def score_mean(scores: Sequence[int | None], threshold: float = 4.0) -> bool:
    """Pass if the jurors' mean score reaches the threshold. A missing score counts as 1 (the lowest)."""
    vals = [s if s is not None else 1 for s in scores]
    return len(vals) > 0 and sum(vals) / len(vals) >= threshold


def pair_outcome(choice_ab: str | None, choice_ba: str | None) -> dict:
    """The correct answer is pair['first']. In order AB it is shown as A; in BA as B."""
    right_ab = choice_ab == "A"
    right_ba = choice_ba == "B"
    consistent = choice_ab is not None and choice_ba is not None and (choice_ab == "A") == (choice_ba == "B")
    return {"right_ab": right_ab, "right_ba": right_ba, "consistent": consistent,
            "both_right": right_ab and right_ba,
            "first_slot": [c for c in (choice_ab, choice_ba) if c is not None].count("A")}
