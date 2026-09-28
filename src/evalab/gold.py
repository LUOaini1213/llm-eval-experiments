"""Deterministic correctness checks for free-text answers.

Each check returns one of three labels:

- "correct":   the answer states the gold value and nothing that competes with it;
- "incorrect": the gold value is absent (or a refusal was given);
- "ambiguous": the gold value is present but so is a competing value of the same kind (another planning area,
  another count, a unit-converted variant of the expected number ...), so a string rule cannot decide.

Ambiguous items are not labelled by hand. They are excluded from the primary analysis and listed in
results/e1/ambiguous_items.csv, so the reader can see exactly what was left out.
"""
from __future__ import annotations

import re
import unicodedata

REFUSAL = re.compile(r"\b(i (do not|don't) know|not sure|cannot (determine|answer|find)|unable to|no information|"
                     r"not (available|provided|specified|stated) in|NOT_FOUND|i'm sorry|i am sorry)\b", re.I)
NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])-?\d+(?:\.\d+)?(?![\d])")

ROAD_ABBR = {"st": "street", "rd": "road", "ave": "avenue", "avenue": "avenue", "dr": "drive", "cres": "crescent",
             "ctrl": "central", "ctr": "centre", "pk": "park", "cl": "close", "hts": "heights", "gdns": "gardens",
             "pl": "place", "lor": "lorong", "jln": "jalan", "bt": "bukit", "upp": "upper", "nth": "north",
             "sth": "south", "c'wealth": "commonwealth", "tg": "tanjong", "kg": "kampong", "blvd": "boulevard",
             "expwy": "expressway", "ter": "terrace", "tce": "terrace", "lk": "link", "sq": "square", "mt": "mount",
             "ln": "lane", "gr": "grove", "hwy": "highway", "cct": "circuit", "ind": "industrial", "est": "estate",
             "stn": "station", "int": "interchange", "sch": "school", "blk": "block", "opp": "opposite",
             "aft": "after", "bef": "before", "vw": "view", "wk": "walk", "e": "east", "w": "west"}

OPERATORS = {
    "SBST": ["sbs transit", "sbs", "sbst"],
    "SMRT": ["smrt buses", "smrt"],
    "TTS": ["tower transit singapore", "tower transit", "tts"],
    "GAS": ["go-ahead singapore", "go ahead singapore", "go-ahead", "go ahead", "gas"],
}
OPERATOR_NAME = {"SBST": "SBS Transit", "SMRT": "SMRT Buses", "TTS": "Tower Transit Singapore",
                 "GAS": "Go-Ahead Singapore"}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"[^\w' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_road(text: str) -> str:
    return " ".join(ROAD_ABBR.get(w, w) for w in norm(text).split())


def numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in NUM.findall(text)]


def is_refusal(answer: str) -> bool:
    return not answer.strip() or bool(REFUSAL.search(answer))


def _contains_phrase(hay: str, needle: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", hay) is not None


def check_number(answer: str, gold: float, exclude: set[float] = frozenset(), tol: float = 1e-9) -> str:
    """Numbers that already occur in the question (a service number, a stop code) are ignored."""
    if is_refusal(answer):
        return "incorrect"
    nums = [n for n in numbers(answer) if not any(abs(n - e) <= 1e-9 for e in exclude)]
    hit = [n for n in nums if abs(n - gold) <= tol]
    if not hit:
        return "incorrect"
    others = {round(n, 6) for n in nums if abs(n - gold) > tol}
    return "ambiguous" if others else "correct"


def hhmm(text: str) -> list[str]:
    out = []
    for h, m, ap in re.findall(r"(?<!\d)(\d{1,2})[:.]?(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?(?!\d)", text, re.I):
        h = int(h)
        ap = ap.lower().replace(".", "")
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if 0 <= h <= 24 and 0 <= int(m) < 60:
            out.append(f"{h:02d}{m}")
    return out


def check_time(answer: str, gold: str, exclude: set[str] = frozenset()) -> str:
    if is_refusal(answer):
        return "incorrect"
    times = [t for t in hhmm(answer) if t not in exclude]
    if gold not in times:
        return "incorrect"
    return "ambiguous" if set(times) - {gold} else "correct"


def check_choice(answer: str, gold: str, aliases: dict[str, list[str]], normaliser=norm) -> str:
    """gold is a key of `aliases`; every other key is a competitor. Longest alias wins where aliases overlap."""
    if is_refusal(answer):
        return "incorrect"
    text = normaliser(answer)
    found = set()
    spans = []
    for key, names in aliases.items():
        for name in sorted(names, key=len, reverse=True):
            n = normaliser(name)
            for m in re.finditer(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text):
                spans.append((m.start(), m.end(), key))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    end = -1
    for s, e, key in spans:
        if s >= end:
            found.add(key)
            end = e
        elif e > end:  # partial overlap: keep the longer, earlier match
            continue
    if gold not in found:
        return "incorrect"
    return "ambiguous" if found - {gold} else "correct"


SERVICE_TOKEN = re.compile(r"(?<![\w.])(\d{1,3}[A-Za-z]?)(?!\w|\.\d)")


def check_service_set(answer: str, gold: set[str], exclude: set[str] = frozenset()) -> str:
    if is_refusal(answer):
        return "incorrect"
    toks = {t.upper() for t in SERVICE_TOKEN.findall(answer)} - {e.upper() for e in exclude}
    g = {x.upper() for x in gold}
    if toks == g:
        return "correct"
    if g <= toks:
        return "ambiguous"
    return "incorrect"


PCT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|per ?cent)", re.I)


def _alt_hit(answer: str, text: str, nums: list[float], alt: str) -> str:
    """'hit', 'scaled' (only a x100 or x1000 variant of a number is present) or 'miss'."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(%?)\s*", alt)
    if not m:
        return "hit" if _contains_phrase(text, norm(alt)) else "miss"
    v = float(m.group(1))
    pool = [float(x) for x in PCT.findall(answer)] if m.group(2) else nums
    if any(abs(n - v) < 1e-9 for n in pool):
        return "hit"
    scaled = [v * k for k in (1000, 100, 0.01, 0.001)]  # mm<->m, cm<->m or fraction<->percent; not x10
    if any(abs(n - s) < 1e-9 * max(1, abs(s)) for n in nums for s in scaled):
        return "scaled"
    return "miss"


MEAS = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(mm|m|cm|km|kpa|pa|l/min|litres?|liters?|l|%|per ?cent|hours?|hrs?|"
                  r"days?|years?|months?|persons?|storeys?|sq ?m|m2|lux|kw/rt)(?![\w/])", re.I)


def check_expected(answer: str, all_of: list[list[str]], question: str = "") -> str:
    """fm-knowledge-assistant items: every group in `all_of` must be matched by one of its alternatives.

    Numbers are matched as whole numbers ("13" does not match "130" or "1.3"); a percentage must be written with
    "%" or "per cent". If a group is missed but a x100 or x1000 variant of its number is present (mm against m), the unit
    may simply differ, so the item is "ambiguous" rather than "incorrect". An answer that matches every group but also
    states another measurement (a number with a unit that is neither expected nor in the question) is "ambiguous" too,
    because a string rule cannot tell which of the two values the answer gives as its answer.
    """
    if is_refusal(answer):
        return "incorrect"
    nums = numbers(answer)
    text = norm(answer)
    status = "correct"
    for group in all_of:
        hits = [_alt_hit(answer, text, nums, a) for a in group]
        if "hit" in hits:
            continue
        if "scaled" in hits:
            status = "ambiguous"
            continue
        return "incorrect"
    if status == "correct":
        allowed = {float(x) for g in all_of for a in g for x in re.findall(r"\d+(?:\.\d+)?", a)}
        allowed |= set(numbers(question))
        extra = {float(n) for n, _ in MEAS.findall(answer)} - allowed
        if extra:
            return "ambiguous"
    return status
