"""Singapore bus-network test cases with exact gold answers (experiment E3).

Input: the derived tables in data/sg_facts/ (see scripts/e3_derive_facts.py for provenance and licence).

Nine templates in three difficulty levels:

- easy (one lookup): the road of a stop, the operator of a service, the planning area of a stop;
- medium (a count or a time): stops served by a service direction, services calling at a stop, the weekday first bus;
- hard (aggregation or a set): route length, stops inside a planning area, services common to two stops.

Controls:

- difficulty mix (share of easy / medium / hard);
- entities are sampled stratified by region (stops) or operator (services), and no entity is reused within a
  template;
- the share of any single gold answer within a template is capped (so "SBS Transit" cannot dominate);
- ambiguity filters: stops within `min_boundary_m` of a planning-area boundary are not used for area questions,
  stops whose name is shared by another stop are always identified by code as well, loop services and
  services with a missing route are skipped;
- near-duplicate questions (token Jaccard >= 0.9) are dropped;
- every item is re-derived by `verify_items`, an independent pandas implementation, and must agree.
"""
from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .gold import OPERATOR_NAME, OPERATORS

FACTS = Path(__file__).resolve().parents[2] / "data" / "sg_facts"
DIFFICULTY = {"road": "easy", "operator": "easy", "area": "easy", "n_stops": "medium", "n_services": "medium",
              "first_bus": "medium", "route_km": "hard", "area_count": "hard", "common_services": "hard"}
# Bus routes, timings and operators change; every question names the data snapshot it is true for.
TIME_ANCHOR = "Use LTA DataMall data for September 2026."
KIND = {"road": "road", "operator": "operator", "area": "area", "n_stops": "int", "n_services": "int",
        "first_bus": "time", "route_km": "km", "area_count": "int", "common_services": "set"}


@dataclass
class Item:
    id: str
    template: str
    difficulty: str
    kind: str
    question: str
    gold: str                    # canonical answer string
    accept: list = field(default_factory=list)   # all acceptable surface forms (for exact-match scoring)
    exclude: list = field(default_factory=list)  # numbers/tokens that come from the question itself
    entity: str = ""
    stratum: str = ""


def load_facts(folder: Path = FACTS):
    with (folder / "stops.csv").open(encoding="utf-8") as f:
        stops = list(csv.DictReader(f))
    with (folder / "services.csv").open(encoding="utf-8") as f:
        services = list(csv.DictReader(f))
    return stops, services


def _jaccard(a: str, b: str) -> float:
    x, y = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(x & y) / max(1, len(x | y))


def _stop_label(s: dict) -> str:
    return f"{s['stop_code']} ({s['stop_name']}, {s['road_name']})"


def _stratified(rows, key, n, rng):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups)
    out = []
    while len(out) < n and any(groups.values()):
        for g in order:
            if groups[g] and len(out) < n:
                out.append(groups[g].pop())
    return out


def _capped(cands, n, cap_share, gold_of):
    """Take candidates in order, skipping any whose gold answer already fills its share of the template."""
    cap = max(1, int(cap_share * n))
    seen = Counter()
    out = []
    for c in cands:
        g = gold_of(c)
        if seen[g] >= cap:
            continue
        seen[g] += 1
        out.append(c)
        if len(out) == n:
            break
    return out


def generate(n: int = 180, seed: int = 20260928, mix=(0.4, 0.35, 0.25), min_boundary_m: float = 50.0,
             cap_share: float = 0.2, folder: Path = FACTS) -> list[Item]:
    rng = random.Random(seed)
    stops, services = load_facts(folder)
    by_code = {s["stop_code"]: s for s in stops}
    sg_stops = [s for s in stops if s["planning_area"]]
    name_count = Counter(s["stop_name"] for s in stops)
    svc_ok = [v for v in services if not v["loop_desc"] and v["origin_code"] != v["destination_code"]
              and v["origin_code"] in by_code and v["destination_code"] in by_code]
    region = lambda s: s["region"] or "OUTSIDE"  # noqa: E731

    per_level = {"easy": round(n * mix[0]), "medium": round(n * mix[1])}
    per_level["hard"] = n - per_level["easy"] - per_level["medium"]
    templates = defaultdict(list)
    for t, d in DIFFICULTY.items():
        templates[d].append(t)
    quota = {}
    for d, ts in templates.items():
        k, r = divmod(per_level[d], len(ts))
        for i, t in enumerate(ts):
            quota[t] = k + (1 if i < r else 0)

    items: list[Item] = []

    def add(t, q, gold, accept, exclude, entity, stratum):
        items.append(Item(id="", template=t, difficulty=DIFFICULTY[t], kind=KIND[t], question=q, gold=str(gold),
                          accept=[str(a) for a in accept], exclude=[str(e) for e in exclude], entity=entity,
                          stratum=stratum))

    # --- easy
    pool = _stratified(sg_stops, region, 4 * quota["road"], rng)
    for s in _capped(pool, quota["road"], cap_share, lambda s: s["road_name"]):
        add("road", f"On which road is Singapore bus stop {s['stop_code']} ({s['stop_name']})?", s["road_name"],
            [s["road_name"]], [s["stop_code"]], s["stop_code"], region(s))
    svc_dir1 = {}
    for v in svc_ok:
        svc_dir1.setdefault(v["service_no"], v)
    pool = _stratified(list(svc_dir1.values()), lambda v: v["operator"], 4 * quota["operator"], rng)
    for v in _capped(pool, quota["operator"], max(cap_share, 0.25), lambda v: v["operator"]):
        add("operator", f"Which company operates Singapore bus service {v['service_no']}?", OPERATOR_NAME[v["operator"]],
            OPERATORS[v["operator"]], [v["service_no"]], v["service_no"], v["operator"])
    pool = [s for s in sg_stops if s["boundary_dist_m"] and float(s["boundary_dist_m"]) >= min_boundary_m]
    pool = _stratified(pool, region, 4 * quota["area"], rng)
    for s in _capped(pool, quota["area"], cap_share, lambda s: s["planning_area"]):
        add("area", f"In which URA Master Plan 2019 planning area is Singapore bus stop {s['stop_code']} "
                    f"({s['stop_name']}) located?", s["planning_area"], [s["planning_area"]], [s["stop_code"]],
            s["stop_code"], region(s))

    # --- medium
    pool = _stratified(svc_ok, lambda v: v["operator"], 4 * quota["n_stops"], rng)
    used = set()
    picked = []
    for v in pool:
        if v["service_no"] in used:
            continue
        used.add(v["service_no"])
        picked.append(v)
    for v in _capped(picked, quota["n_stops"], cap_share, lambda v: v["n_stops"]):
        o, d = by_code[v["origin_code"]], by_code[v["destination_code"]]
        add("n_stops", f"How many bus stops does Singapore bus service {v['service_no']} call at when travelling from "
                       f"{o['stop_name']} ({o['stop_code']}) to {d['stop_name']} ({d['stop_code']}), counting both "
                       f"terminals?", int(v["n_stops"]), [v["n_stops"]],
            [v["service_no"], o["stop_code"], d["stop_code"]], f"{v['service_no']}/{v['direction']}", v["operator"])
    pool = _stratified([s for s in sg_stops if s["services"]], region, 4 * quota["n_services"], rng)
    for s in _capped(pool, quota["n_services"], cap_share, lambda s: len(s["services"].split(";"))):
        k = len(s["services"].split(";"))
        add("n_services", f"How many different bus services call at Singapore bus stop {s['stop_code']} "
                          f"({s['stop_name']})?", k, [k], [s["stop_code"]], s["stop_code"], region(s))
    pool = _stratified(svc_ok, lambda v: v["operator"], 4 * quota["first_bus"], rng)
    used = set()
    picked = []
    for v in pool:
        if v["service_no"] in used or not re.fullmatch(r"\d{4}", v["wd_first_bus"]):
            continue
        used.add(v["service_no"])
        picked.append(v)
    for v in _capped(picked, quota["first_bus"], cap_share, lambda v: v["wd_first_bus"]):
        o = by_code[v["origin_code"]]
        add("first_bus", f"On weekdays, at what time does the first bus of Singapore bus service {v['service_no']} "
                         f"leave its starting stop {o['stop_name']} ({o['stop_code']})? Give the time as HHMM "
                         f"(24-hour clock).", v["wd_first_bus"], [v["wd_first_bus"]],
            [v["service_no"], o["stop_code"]], f"{v['service_no']}/{v['direction']}", v["operator"])

    # --- hard
    pool = _stratified(svc_ok, lambda v: v["operator"], 4 * quota["route_km"], rng)
    used = set()
    picked = []
    for v in pool:
        if v["service_no"] in used:
            continue
        used.add(v["service_no"])
        picked.append(v)
    for v in _capped(picked, quota["route_km"], cap_share, lambda v: v["route_km"]):
        o, d = by_code[v["origin_code"]], by_code[v["destination_code"]]
        add("route_km", f"According to LTA's bus route data, how long is the route of Singapore bus service "
                        f"{v['service_no']} from {o['stop_name']} ({o['stop_code']}) to {d['stop_name']} "
                        f"({d['stop_code']}), in kilometres to one decimal place?", v["route_km"], [v["route_km"]],
            [v["service_no"], o["stop_code"], d["stop_code"]], f"{v['service_no']}/{v['direction']}", v["operator"])
    area_n = Counter(s["planning_area"] for s in sg_stops)
    areas = sorted(a for a, k in area_n.items() if k >= 5)
    rng.shuffle(areas)
    region_of = {s["planning_area"]: s["region"] for s in sg_stops}
    areas = _stratified([{"a": a, "r": region_of[a]} for a in areas], lambda x: x["r"], quota["area_count"], rng)
    for x in areas[:quota["area_count"]]:
        a = x["a"]
        add("area_count", f"How many of LTA's bus stops lie inside the {a.title()} planning area (URA Master Plan "
                          f"2019 boundaries)?", area_n[a], [area_n[a]], [], a, x["r"])
    # pairs of consecutive stops on one route that share 1-4 services
    route_pairs = []
    svc_sets = {s["stop_code"]: set(s["services"].split(";")) - {""} for s in stops}
    rp_pool = _stratified(svc_ok, lambda v: v["operator"], 6 * quota["common_services"], rng)
    for v in rp_pool:
        o, d = v["origin_code"], v["destination_code"]
        common = svc_sets.get(o, set()) & svc_sets.get(d, set())
        if 1 <= len(common) <= 4 and o in by_code and d in by_code:
            route_pairs.append((o, d, common, v))
    seen_pairs = set()
    for o, d, common, v in route_pairs:
        key = tuple(sorted((o, d)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        so, sd = by_code[o], by_code[d]
        gold = ";".join(sorted(common, key=_svc_key))
        add("common_services", f"Which bus services call at both Singapore bus stop {so['stop_code']} "
                               f"({so['stop_name']}) and bus stop {sd['stop_code']} ({sd['stop_name']})? List all of "
                               f"them.", gold, [gold], [o, d], f"{o}+{d}", v["operator"])
        if sum(1 for i in items if i.template == "common_services") >= quota["common_services"]:
            break

    # stops whose name is shared are already identified by code in every template; drop near-duplicates
    kept: list[Item] = []
    for it in items:
        if any(_jaccard(it.question, k.question) >= 0.9 and it.gold == k.gold for k in kept):
            continue
        kept.append(it)
    for i, it in enumerate(kept, start=1):
        it.id = f"sgbus-{i:04d}"
        it.question += " " + TIME_ANCHOR
    _ = name_count
    return kept


def _svc_key(s: str):
    digits = "".join(c for c in s if c.isdigit())
    return (int(digits) if digits else 10**6, s)


def verify_items(items: list[Item], folder: Path = FACTS) -> list[str]:
    """Independent re-derivation with pandas. Returns a list of disagreements (empty means every gold holds)."""
    import pandas as pd

    st = pd.read_csv(folder / "stops.csv", dtype=str, keep_default_na=False)
    sv = pd.read_csv(folder / "services.csv", dtype=str, keep_default_na=False)
    st = st.set_index("stop_code")
    problems = []
    for it in items:
        q = it.question
        if it.template == "road":
            want = st.loc[it.entity, "road_name"]
        elif it.template == "area":
            want = st.loc[it.entity, "planning_area"]
        elif it.template == "operator":
            ops = set(sv.loc[sv.service_no == it.entity, "operator"])
            want = OPERATOR_NAME[ops.pop()] if len(ops) == 1 else f"multiple operators {ops}"
        elif it.template in ("n_stops", "first_bus", "route_km"):
            svc, d = it.entity.split("/")
            row = sv[(sv.service_no == svc) & (sv.direction == d)].iloc[0]
            want = {"n_stops": row.n_stops, "first_bus": row.wd_first_bus, "route_km": row.route_km}[it.template]
            if row.origin_code not in q:
                problems.append(f"{it.id}: origin stop missing from the question")
        elif it.template == "n_services":
            want = str(len([x for x in st.loc[it.entity, "services"].split(";") if x]))
        elif it.template == "area_count":
            want = str(int((st.planning_area == it.entity).sum()))
        elif it.template == "common_services":
            a, b = it.entity.split("+")
            sa = set(filter(None, st.loc[a, "services"].split(";")))
            sb = set(filter(None, st.loc[b, "services"].split(";")))
            want = ";".join(sorted(sa & sb, key=_svc_key))
        else:
            problems.append(f"{it.id}: unknown template {it.template}")
            continue
        if str(want) != it.gold:
            problems.append(f"{it.id}: gold {it.gold!r} but re-derived {want!r}")
    return problems


def to_dicts(items):
    return [asdict(i) for i in items]
