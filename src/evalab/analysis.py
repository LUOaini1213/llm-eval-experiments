"""E1 analysis: turn judgments into pipeline decisions, metrics, paired tests and a cost table.

A *pipeline* is a rule that maps an item to pass/fail: one judge under one prompt condition, or a jury of judges
with an aggregation rule. All pipelines are evaluated on the same items (paired design).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import judge as J
from . import stats as S

FAMILY = {"qwen2.5:3b": "qwen", "qwen3.5:4b": "qwen", "llama3.2:3b": "llama", "gemma3:4b": "gemma",
          "phi4-mini:3.8b": "phi"}


@dataclass
class Pipeline:
    name: str
    judges: tuple
    condition: str
    rule: str = "single"          # single | majority | unanimity | mean_score
    decisions: dict = field(default_factory=dict)   # item_id -> bool (pass)
    cost: dict = field(default_factory=dict)        # calls, prompt_tokens, output_tokens, seconds


def index_judgments(rows: list[dict]) -> dict:
    """(judge, condition, item_id) -> row, pointwise only."""
    return {(r["judge"], r["condition"], r["item_id"]): r for r in rows if "order" not in r}


def _pass(row: dict | None, condition: str) -> bool:
    if row is None:
        return False
    if condition == "score_ref":
        return J.score_pass(row["parsed"])
    return row["parsed"] is True


def build_pipeline(name, judges, condition, rule, items, idx) -> Pipeline:
    p = Pipeline(name, tuple(judges), condition, rule)
    cost = defaultdict(float)
    for it in items:
        rows = [idx.get((j, condition, it["item_id"])) for j in judges]
        for r in rows:
            if r is not None:
                cost["calls"] += 1
                cost["prompt_tokens"] += r["prompt_tokens"]
                cost["output_tokens"] += r["output_tokens"]
                cost["seconds"] += r["seconds"]
        if rule == "single":
            p.decisions[it["item_id"]] = _pass(rows[0], condition)
        elif rule == "majority":
            p.decisions[it["item_id"]] = J.majority([_pass(r, condition) for r in rows])
        elif rule == "unanimity":
            p.decisions[it["item_id"]] = J.unanimity([_pass(r, condition) for r in rows])
        elif rule == "mean_score":
            p.decisions[it["item_id"]] = J.score_mean([r["parsed"] if r else None for r in rows])
        else:
            raise ValueError(rule)
    p.cost = dict(cost)
    return p


def right(p: Pipeline, items) -> list[bool]:
    """Per item: did the pipeline agree with the gold label?"""
    return [p.decisions[it["item_id"]] == (it["gold"] == "correct") for it in items]


def metrics(p: Pipeline, items, boot_seed: int = 0) -> dict:
    ok = right(p, items)
    pos = [it for it in items if it["gold"] == "correct"]
    neg = [it for it in items if it["gold"] == "incorrect"]
    fa = sum(p.decisions[it["item_id"]] for it in neg)
    fr = sum(not p.decisions[it["item_id"]] for it in pos)
    n = len(items)
    acc = sum(ok) / n
    lo, hi = S.bootstrap_ci([float(x) for x in ok], b=4000, seed=boot_seed)
    far = fa / len(neg) if neg else float("nan")
    frr = fr / len(pos) if pos else float("nan")
    kappa = S.cohen_kappa([p.decisions[it["item_id"]] for it in items], [it["gold"] == "correct" for it in items])
    return {"pipeline": p.name, "condition": p.condition, "rule": p.rule, "judges": list(p.judges), "n": n,
            "accuracy": acc, "accuracy_ci": [lo, hi],
            "balanced_accuracy": 1 - (far + frr) / 2,
            "false_accept_rate": far, "false_accept_ci": list(S.wilson(fa, len(neg))), "n_wrong_answers": len(neg),
            "false_reject_rate": frr, "false_reject_ci": list(S.wilson(fr, len(pos))), "n_right_answers": len(pos),
            "kappa_vs_gold": kappa,
            "calls_per_item": p.cost.get("calls", 0) / n, "tokens_per_item":
                (p.cost.get("prompt_tokens", 0) + p.cost.get("output_tokens", 0)) / n,
            "seconds_per_item": p.cost.get("seconds", 0) / n}


def compare(a: Pipeline, b: Pipeline, items, subset=None, seed: int = 0) -> dict:
    """Paired comparison of a against b on the items (optionally a subset: 'correct' or 'incorrect' gold)."""
    its = [it for it in items if subset is None or it["gold"] == subset]
    xa, xb = right(a, its), right(b, its)
    bb, cc = S.discordant(xa, xb)
    d, lo, hi = S.paired_bootstrap_diff([float(x) for x in xa], [float(x) for x in xb], b=4000, seed=seed)
    orr = S.discordant_odds_ratio(bb, cc)
    return {"a": a.name, "b": b.name, "subset": subset or "all", "n": len(its), "a_right": sum(xa),
            "b_right": sum(xb), "only_a_right": bb, "only_b_right": cc, "diff": d, "diff_ci": [lo, hi],
            "discordant_odds_ratio": orr[0], "odds_ratio_ci": [orr[1], orr[2]], "p_mcnemar": S.mcnemar_exact(bb, cc)}


def agreement(items, idx, judges, condition) -> dict:
    counts, votes = [], {j: [] for j in judges}
    for it in items:
        v = [_pass(idx.get((j, condition, it["item_id"])), condition) for j in judges]
        for j, x in zip(judges, v):
            votes[j].append(x)
        counts.append([sum(v), len(v) - sum(v)])
    pair = {f"{a}~{b}": S.cohen_kappa(votes[a], votes[b]) for i, a in enumerate(judges) for b in judges[i + 1:]}
    return {"condition": condition, "fleiss_kappa": S.fleiss_kappa(counts), "cohen_kappa_pairs": pair}


def choose_best_and_jury(pilot_metrics: dict[str, dict], judges) -> tuple[str, list[str]]:
    """Pre-registered rule: best single judge = highest pilot accuracy under bin_ref (ties: fewer seconds per item);
    three-judge jury = the three most accurate pilot judges with at most one per model family."""
    ranked = sorted(judges, key=lambda j: (-pilot_metrics[j]["accuracy"], pilot_metrics[j]["seconds_per_item"]))
    jury, fams = [], set()
    for j in ranked:
        if FAMILY[j] not in fams:
            jury.append(j)
            fams.add(FAMILY[j])
        if len(jury) == 3:
            break
    return ranked[0], jury


def pareto(points: list[dict], x="seconds_per_item", y="accuracy") -> list[str]:
    """Names of pipelines not dominated (no other is at least as accurate and at most as costly, one strictly)."""
    front = []
    for p in points:
        dominated = any((q[y] >= p[y] and q[x] <= p[x]) and (q[y] > p[y] or q[x] < p[x]) for q in points)
        if not dominated:
            front.append(p["pipeline"])
    return front
