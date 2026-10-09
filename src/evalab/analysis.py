"""E1 analysis: turn judgments into pipeline decisions, metrics, paired tests and a cost table.

A *pipeline* is a rule that maps an item to pass/fail: one judge under one prompt condition, or a jury of judges
with an aggregation rule. All pipelines are evaluated on the same items (paired design).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math

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
    item_cost: dict = field(default_factory=dict)   # item_id -> same cost fields, for subset reports


def index_judgments(rows: list[dict]) -> dict:
    """(judge, condition, item_id) -> row, pointwise only."""
    idx = {}
    for r in rows:
        if "order" in r:
            continue
        key = (r["judge"], r["condition"], r["item_id"])
        if key in idx:
            raise ValueError(f"Duplicate judgment: {key}")
        _validate_record(r)
        idx[key] = r
    return idx


def _validate_record(row):
    condition, parsed = row["condition"], row["parsed"]
    if condition in J.POINTWISE:
        valid = (type(parsed) is int and 1 <= parsed <= 5) if condition == "score_ref" else type(parsed) is bool
    elif condition in J.PAIRWISE:
        valid = type(parsed) is str and parsed in ("A", "B")
        if row.get("order") not in ("AB", "BA"):
            raise ValueError(f"Invalid pairwise order: {row.get('order')}")
    else:
        raise ValueError(f"Unknown judgment condition: {condition}")
    if parsed is not None and not valid:
        raise ValueError(f"Invalid parsed judgment for {condition}: {parsed!r}")
    for name in ("prompt_tokens", "output_tokens", "seconds"):
        value = row[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid judgment cost {name}: {value!r}")


def validate_matrix(rows, items, judges, conditions, *, pairwise=False):
    """Require the declared experiment, not axes inferred from whatever rows survived.

    A present unparseable response (parsed=None) is valid; an absent record is not.
    Return a unique index with a fourth order component for pairwise judgments.
    """
    ids = [it["pair_id" if pairwise else "item_id"] for it in items]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicate experiment items")
    if not judges or len(set(judges)) != len(judges) or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Empty or duplicate experiment axes")
    expected = {(j, c, i) + ((o,) if pairwise else ())
                for j in judges for c in conditions for i in ids
                for o in (("AB", "BA") if pairwise else (None,))}
    idx = {}
    for r in rows:
        key = (r["judge"], r["condition"], r["item_id"]) + ((r.get("order"),) if pairwise else ())
        if key in idx:
            raise ValueError(f"Duplicate judgment: {key}")
        if key not in expected or (not pairwise and "order" in r):
            raise ValueError(f"Unexpected judgment: {key}")
        _validate_record(r)
        idx[key] = r
    missing = expected - idx.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} judgments; first: {sorted(missing)[0]}")
    return idx


def load_main_items(root, results_dir=None):
    """Select the fixed, pre-registered prefix; never infer its size from replies."""
    from .experiment import main_size, validate_main_manifest, run_identity
    from .items import load_jsonl

    prereg = root / "docs" / "PREREGISTRATION.md"
    n = main_size(prereg)
    results_dir = results_dir if results_dir is not None else root / "results" / "e1"
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    validate_main_manifest(manifest, prereg, n)
    order = load_jsonl(root / "data" / "e1" / "items_main_order.jsonl")
    if len(order) < n:
        raise ValueError(f"Main item order is truncated: expected at least {n}, found {len(order)}")
    items = order[:n]
    if "schema_version" in manifest:
        if manifest["schema_version"] != 1 or manifest.get("identity") != run_identity("main", items):
            raise ValueError("Main experiment identity/configuration differs from its manifest")
    return items


def validate_aux_manifest(root, out, which, items):
    """Check new-run provenance; historical pilot/robust/pair sets have no manifest."""
    from .experiment import run_identity
    path = out / f"run_manifest_{which}.json"
    if not path.exists():
        if out != root / "results" / "e1":
            raise ValueError(f"Missing {which} run manifest: {path}")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("identity") != run_identity(which, items):
        raise ValueError(f"{which} experiment identity/configuration differs from its manifest")


def _required(idx, judge, condition, item_id):
    key = (judge, condition, item_id)
    if key not in idx:
        raise ValueError(f"Missing judgment: {key}")
    return idx[key]


def _pass(row: dict | None, condition: str) -> bool:
    if row is None:
        raise ValueError("Missing judgment is not a parsed rejection")
    if condition == "score_ref":
        return J.score_pass(row["parsed"])
    return row["parsed"] is True


def build_pipeline(name, judges, condition, rule, items, idx) -> Pipeline:
    p = Pipeline(name, tuple(judges), condition, rule)
    cost = defaultdict(float)
    for it in items:
        rows = [_required(idx, j, condition, it["item_id"]) for j in judges]
        item_cost = defaultdict(float)
        for r in rows:
            item_cost["calls"] += 1
            item_cost["prompt_tokens"] += r["prompt_tokens"]
            item_cost["output_tokens"] += r["output_tokens"]
            item_cost["seconds"] += r["seconds"]
        p.item_cost[it["item_id"]] = dict(item_cost)
        for k, value in item_cost.items():
            cost[k] += value
        if rule == "single":
            p.decisions[it["item_id"]] = _pass(rows[0], condition)
        elif rule == "majority":
            p.decisions[it["item_id"]] = J.majority([_pass(r, condition) for r in rows])
        elif rule == "unanimity":
            p.decisions[it["item_id"]] = J.unanimity([_pass(r, condition) for r in rows])
        elif rule == "mean_score":
            p.decisions[it["item_id"]] = J.score_mean([r["parsed"] for r in rows])
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
    cost = {k: sum(p.item_cost[it["item_id"]][k] for it in items)
            for k in ("calls", "prompt_tokens", "output_tokens", "seconds")}
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
            "calls_per_item": cost["calls"] / n, "tokens_per_item":
                (cost["prompt_tokens"] + cost["output_tokens"]) / n,
            "seconds_per_item": cost["seconds"] / n}


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
        v = [_pass(_required(idx, j, condition, it["item_id"]), condition) for j in judges]
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
