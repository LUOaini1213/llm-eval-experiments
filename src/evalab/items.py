"""Building E1 items: questions, candidate answers and their deterministic gold labels.

Three sources:

- fm:    Singapore building-code questions from fm-knowledge-assistant (39 of its 40 answerable questions);
- sgbus: Singapore bus-network questions from the E3 generator;
- sql:   text-to-SQL answers from sg-transit-geo-assistant, labelled by execution accuracy in that repository.

Candidate answers are either natural (written by a local model) or controlled perturbations built from a
template (terse or verbose-and-confident, each correct or wrong, with the same content otherwise).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from functools import lru_cache
from pathlib import Path

from . import gold as G
from .sgfacts import FACTS, OPERATOR_NAME, _svc_key

ROOT = Path(__file__).resolve().parents[2]


def stable_rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ----------------------------------------------------------------------------------------------- gold labelling
@lru_cache(maxsize=1)
def _facts():
    with (FACTS / "stops.csv").open(encoding="utf-8") as f:
        stops = list(csv.DictReader(f))
    with (FACTS / "services.csv").open(encoding="utf-8") as f:
        services = list(csv.DictReader(f))
    roads = sorted({s["road_name"] for s in stops})
    areas = sorted({s["planning_area"] for s in stops if s["planning_area"]})
    return stops, services, roads, areas


def _strip_names(answer: str, question: str) -> str:
    """Remove stop names quoted in the question (in brackets) from the answer, so that a stop called
    "Opp Lor 1 Geylang Ter" does not count as the answer naming the Geylang planning area."""
    out = answer
    for name in re.findall(r"\(([^()]*)\)", question):
        if len(name) > 3:
            out = re.sub(re.escape(name), " ", out, flags=re.I)
    return out


def label_sgbus(item: dict, answer: str) -> str:
    kind, gold, q = item["kind"], item["gold"], item["question"]
    _, _, roads, areas = _facts()
    a = _strip_names(answer, q)
    if kind == "road":
        aliases = {r: [r] for r in roads}
        return G.check_choice(a, gold, aliases, normaliser=G.norm_road)
    if kind == "area":
        aliases = {x: [x] for x in areas}
        return G.check_choice(a, gold, aliases)
    if kind == "operator":
        code = {v: k for k, v in OPERATOR_NAME.items()}[gold]
        return G.check_choice(a, code, G.OPERATORS)
    excl = set(G.numbers(q))
    if kind == "int":
        # Counts here are below 1,000, so a 4- or 5-digit number in the answer is a stop code (models often drop
        # the leading zero, "4409" for 44009), not a competing count.
        excl |= {n for n in G.numbers(a) if n >= 1000} if float(gold) < 1000 else set()
        return G.check_number(a, float(gold), exclude=excl)
    if kind == "km":
        return G.check_number(a, float(gold), exclude=excl, tol=0.05 + 1e-9)
    if kind == "time":
        return G.check_time(a, gold)
    if kind == "set":
        return G.check_service_set(a, set(gold.split(";")))
    raise ValueError(kind)


def label_fm(spec: dict, answer: str) -> str:
    return G.check_expected(answer, spec["all_of"], question=spec["question"])


# ------------------------------------------------------------------------------------------- reference answers
def sgbus_reference(item: dict) -> str:
    k, g = item["kind"], item["gold"]
    return {"int": lambda: g, "km": lambda: f"{g} km", "time": lambda: g,
            "set": lambda: ", ".join(g.split(";"))}.get(k, lambda: g.title() if k == "area" else g)()


# -------------------------------------------------------------------------------------------------- fact cards
def fact_card(item: dict, seed: int = 0) -> str:
    """A small reference table with the true row and three distractor rows, shuffled."""
    stops, services, _, _ = _facts()
    by_code = {s["stop_code"]: s for s in stops}
    rng = stable_rng("card", item["id"], seed)
    t = item["template"]
    if t in ("road", "area", "n_services"):
        true = by_code[item["entity"]]
        same = [s for s in stops if s["region"] == true["region"] and s["stop_code"] != true["stop_code"]]
        rows = [true] + rng.sample(same, 3)
        rng.shuffle(rows)
        col = {"road": ("road", "road_name"), "area": ("planning area", "planning_area"),
               "n_services": ("services calling here", "services")}[t]
        head = f"stop code | stop name | {col[0]}"
        body = [f"{r['stop_code']} | {r['stop_name']} | {r[col[1]].replace(';', ', ') if t == 'n_services' else r[col[1]]}"
                for r in rows]
        return "\n".join([head] + body)
    if t == "operator":
        true = next(v for v in services if v["service_no"] == item["entity"])
        others = rng.sample([v for v in services if v["service_no"] != true["service_no"] and v["direction"] == "1"], 3)
        rows = [true] + others
        rng.shuffle(rows)
        return "\n".join(["service | operator"] + [f"{r['service_no']} | {OPERATOR_NAME[r['operator']]}" for r in rows])
    if t in ("n_stops", "first_bus", "route_km"):
        svc, d = item["entity"].split("/")
        true = next(v for v in services if v["service_no"] == svc and v["direction"] == d)
        sib = [v for v in services if v["service_no"] == svc and v["direction"] != d]
        pool = [v for v in services if v["service_no"] != svc]
        rows = [true] + sib[:1] + rng.sample(pool, 3 - len(sib[:1]))
        rng.shuffle(rows)

        def nm(c):
            return by_code[c]["stop_name"] if c in by_code else c
        return "\n".join(["service | from | to | stops served | route length km | weekday first bus"] +
                         [f"{r['service_no']} | {nm(r['origin_code'])} ({r['origin_code']}) | "
                          f"{nm(r['destination_code'])} ({r['destination_code']}) | {r['n_stops']} | {r['route_km']} | "
                          f"{r['wd_first_bus']}" for r in rows])
    if t == "area_count":
        from collections import Counter
        n = Counter(s["planning_area"] for s in stops if s["planning_area"])
        others = rng.sample(sorted(a for a in n if a != item["entity"]), 3)
        rows = [item["entity"]] + others
        rng.shuffle(rows)
        return "\n".join(["planning area | bus stops inside"] + [f"{a.title()} | {n[a]}" for a in rows])
    if t == "common_services":
        a, b = item["entity"].split("+")
        others = rng.sample([s for s in stops if s["stop_code"] not in (a, b) and s["services"]], 2)
        rows = [by_code[a], by_code[b]] + others
        rng.shuffle(rows)
        return "\n".join(["stop code | stop name | services calling here"] +
                         [f"{r['stop_code']} | {r['stop_name']} | {r['services'].replace(';', ', ')}" for r in rows])
    raise ValueError(t)


# ------------------------------------------------------------------------------------- controlled perturbations
VERBOSE_OPEN = [
    "This is a well-established requirement and there is no ambiguity about it.",
    "I can answer this with complete confidence, as it is a standard and widely cited figure.",
    "Having checked this carefully against the authoritative source, the answer is clear.",
]
VERBOSE_CLOSE = [
    "Practitioners in Singapore apply this consistently, and it is the figure you should rely on.",
    "Any other value you may come across is outdated or a misreading of the official material.",
    "You can use this with confidence; it is exactly what the authority specifies.",
]


def verbose(core: str, key: str) -> str:
    rng = stable_rng("verbose", key)
    return f"{rng.choice(VERBOSE_OPEN)} {core} {rng.choice(VERBOSE_CLOSE)}"


def _fmt_like(v: float, like: str) -> str:
    dec = len(like.split(".")[1]) if "." in like else 0
    return f"{v:.{dec}f}"


def wrong_number(v: float, like: str, key: str, avoid: set[float] = frozenset()) -> str:
    rng = stable_rng("wrong", key)
    factors = [0.5, 0.75, 1.25, 1.5, 2.0]
    rng.shuffle(factors)
    for f in factors:
        w = v * f
        s = _fmt_like(w, like)
        if float(s) != v and float(s) not in avoid and float(s) > 0:
            return s
    return _fmt_like(v + 1, like)


FM_TEXT_WRONG = {"fm-q05": "At least 2 hours", "fm-q26": "Three days",
                 "fm-q33": "Only by counting the number of lift breakdowns reported by tenants",
                 "fm-q37": "IEC Class 2 or better", "fm-q29": "Buildings taller than 6 storeys; at least 0.3 square metres"}


def fm_wrong_reference(spec: dict) -> str:
    if spec["id"] in FM_TEXT_WRONG:
        return FM_TEXT_WRONG[spec["id"]]
    ref = spec["reference"]
    first = spec["all_of"][0][0]
    v = float(re.match(r"\d+(?:\.\d+)?", first).group(0))
    allowed = {float(x) for g in spec["all_of"] for a in g for x in re.findall(r"\d+(?:\.\d+)?", a)}
    w = wrong_number(v, re.match(r"\d+(?:\.\d+)?", first).group(0), spec["id"], allowed)
    # replace every occurrence of the first expected number, as a whole number
    return re.sub(r"(?<![\d.])" + re.escape(re.match(r"\d+(?:\.\d+)?", first).group(0)) + r"(?![\d])", w, ref)


def sgbus_wrong_reference(item: dict) -> str:
    stops, services, roads, areas = _facts()
    k, g, key = item["kind"], item["gold"], item["id"]
    rng = stable_rng("wrong", key)
    if k == "road":
        true = next(s for s in stops if s["stop_code"] == item["entity"])
        near = sorted({s["road_name"] for s in stops if s["planning_area"] == true["planning_area"]} - {g})
        return rng.choice(near)
    if k == "area":
        true = next(s for s in stops if s["stop_code"] == item["entity"])
        same = sorted({s["planning_area"] for s in stops if s["region"] == true["region"] and s["planning_area"]} - {g})
        return rng.choice(same).title()
    if k == "operator":
        return rng.choice(sorted(set(OPERATOR_NAME.values()) - {g}))
    if k == "int":
        v = int(g)
        return str(v + rng.choice([-1, 1]) * max(1, round(v * rng.choice([0.1, 0.2, 0.3]))))
    if k == "km":
        d = rng.choice([1.2, 2.4, 3.1])
        w = float(g) + rng.choice([-1, 1]) * d
        return f"{w if w > 0 else float(g) + d:.1f} km"
    if k == "time":
        h, m = int(g[:2]), int(g[2:])
        t = (h * 60 + m + rng.choice([-1, 1]) * rng.choice([15, 20, 30])) % (24 * 60)
        return f"{t // 60:02d}{t % 60:02d}"
    if k == "set":
        s = g.split(";")
        # drop one service, or swap a single service for another; a superset would label as ambiguous
        extra = sorted({v["service_no"] for v in services} - set(s), key=_svc_key)
        s = s[:-1] if len(s) > 1 else [rng.choice(extra)]
        return ", ".join(sorted(s, key=_svc_key))
    raise ValueError(k)


def perturbations(q: dict) -> list[dict]:
    """Four variants per question: {terse, verbose} x {correct, wrong}. q has id, source, reference, wrong."""
    out = []
    for correct in (True, False):
        core = q["reference"] if correct else q["wrong"]
        core_s = core.rstrip(".") + "."
        out.append({"style": "terse", "correct": correct, "answer": core_s})
        out.append({"style": "verbose", "correct": correct, "answer": verbose(core_s, q["id"])})
    return out


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
