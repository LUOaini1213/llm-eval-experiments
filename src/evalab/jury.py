"""LLM-as-jury scoring, independent of Moonshot, so it can be tested with fake judges.

`run_jury` takes async judge callables (system prompt, user prompt -> reply text). Judges are called one
after another, each over all items, because a 4 GB GPU holds one model at a time and switching models per item
would reload them.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Sequence

from . import judge as J
from .stats import fleiss_kappa

AGGREGATIONS = {"majority": J.majority, "unanimity": J.unanimity}


def judge_messages(question: str, candidate: str, reference, condition: str = "bin_ref") -> list[dict]:
    ref = " / ".join(map(str, reference)) if isinstance(reference, (list, tuple)) else str(reference)
    return J.pointwise_messages({"question": question, "candidate": candidate, "reference": ref}, condition)


async def run_jury(questions: Sequence[str], candidates: Sequence[str], references: Sequence,
                   judges: dict[str, Callable[[str, str], Awaitable[str]]], condition: str = "bin_ref",
                   aggregation: str = "majority") -> dict:
    if condition not in ("bin_ref", "strict_ref", "bin_free"):
        raise ValueError("the jury metric needs a binary pointwise condition")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {sorted(AGGREGATIONS)}")
    msgs = [judge_messages(q, c, r, condition) for q, c, r in zip(questions, candidates, references)]
    system = msgs[0][0]["content"] if msgs else ""
    prompts = [m[1]["content"] for m in msgs]
    votes: dict[str, list] = {}
    replies: dict[str, list] = {}
    for name, call in judges.items():
        replies[name] = [await call(system, p) for p in prompts]
        votes[name] = [J.parse_verdict(r) for r in replies[name]]
    agg = AGGREGATIONS[aggregation]
    names = list(judges)
    verdicts = [agg([votes[n][i] for n in names]) for i in range(len(prompts))]
    counts = [[sum(1 for n in names if votes[n][i] is True), sum(1 for n in names if votes[n][i] is not True)]
              for i in range(len(prompts))]
    kappa = fleiss_kappa(counts) if len(names) >= 2 and prompts else float("nan")
    n = max(1, len(prompts))
    return {"verdicts": verdicts, "votes": votes, "replies": replies, "prompts": prompts,
            "pass_rate": sum(verdicts) / n,
            "judge_pass_rate": {k: sum(1 for v in votes[k] if v is True) / n for k in names},
            "unparsed": {k: sum(1 for v in votes[k] if v is None) for k in names},
            "fleiss_kappa": kappa}
