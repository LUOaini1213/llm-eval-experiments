"""Automated quality checks for benchmark datasets in Moonshot's format (experiment E2).

Each check takes the list of examples ({"input", "target"}) and returns flags (index, check, detail).
Items are identified as "<dataset>#<index>" (0-based position in the dataset's examples at the pinned
moonshot-data commit), so findings can be published without copying the text of unlicensed datasets.

Checks:
- exact_duplicate / near_duplicate  (normalised text; token Jaccard >= 0.8), within and across datasets
- mcq_key_mismatch                  the target's letter and the target's text point to different options
- mcq_key_missing                   the target letter is not among the options, or the target text is not an option
- mcq_duplicate_option              two options with the same text
- tnf_bad_target                    a true/false item whose target is not TRUE or FALSE
- encoding                          replacement characters or mojibake
- header_row                        the example looks like a CSV header ("Term" / "Explanation")
- whitespace                        runs of 3+ spaces or leading/trailing whitespace in the target
- ambiguity                         hedges, absolute quantifiers, or compound claims joined by "and" in a
                                    true/false item, and "all/none of the above" options (a heuristic, not an error)
- time_sensitive                    wording whose truth can change: "as of 20xx", "currently", "latest", office
                                    holders, amounts of money, years from 2015 on (a heuristic, not an error)
- metadata_name                     the dataset's name does not match its file name's topic
"""
from __future__ import annotations

import re
import unicodedata
from itertools import combinations

OPTION = re.compile(r"(?m)^\s*([A-H])[).:]\s*(.+?)\s*$")
TARGET_MCQ = re.compile(r"^\s*([A-H])[).:]\s*(.*?)\s*$", re.S)


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"^answer in true or false:\s*", "", t)
    t = re.sub(r"^is the statement in single quotes true or false in the singapore context\? respond with only "
               r"true or false\.\s*", "", t)
    t = re.sub(r"[^\w ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


INSTRUCTION = re.compile(r"^\s*(answer in true or false:|is the statement in single quotes true or false in the "
                         r"singapore context\? respond with only true or false\.)\s*", re.I)


def statement(text: str) -> str:
    """The claim itself, without the answer-format instruction in front of it."""
    return INSTRUCTION.sub("", text).strip().strip("'")


def tokens(text: str) -> set[str]:
    return set(norm(text).split())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def options(text: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in OPTION.finditer(text)}


def is_mcq(ex: dict) -> bool:
    return len(options(ex["input"])) >= 2 and bool(TARGET_MCQ.match(str(ex["target"])))


def is_tnf(ex: dict) -> bool:
    return bool(re.search(r"\btrue or false\b", ex["input"], re.I))


def _same(a: str, b: str) -> bool:
    return norm(a) == norm(b)


def check_mcq(ex: dict) -> list[tuple[str, str]]:
    out = []
    opts = options(ex["input"])
    m = TARGET_MCQ.match(str(ex["target"]))
    if not m:
        return out
    letter, text = m.group(1), m.group(2)
    if letter not in opts:
        out.append(("mcq_key_missing", f"target letter {letter} is not an option"))
    elif text and not _same(opts[letter], text):
        other = [k for k, v in opts.items() if _same(v, text)]
        if other:
            out.append(("mcq_key_mismatch", f"target says {letter}) but its text is option {other[0]}"))
        else:
            out.append(("mcq_key_missing", f"target text does not match option {letter}"))
    vals = [norm(v) for v in opts.values()]
    if len(vals) != len(set(vals)):
        out.append(("mcq_duplicate_option", "two options have the same text"))
    if any(re.search(r"\b(all|none) of the above\b", v, re.I) for v in opts.values()):
        out.append(("ambiguity", "all/none of the above option"))
    return out


HEDGE = re.compile(r"\b(usually|typically|often|generally|mostly|some|many|most|largely|commonly|may|might|"
                   r"particularly|widely)\b", re.I)
ABSOLUTE = re.compile(r"\b(always|never|only|all|none|every|entirely|unilaterally)\b", re.I)
STALE = re.compile(r"\b(as of (?:19|20)\d\d|current(?:ly)?|latest|recent(?:ly)?|now|today|incumbent|"
                   r"prime minister|president|minister|population|\$\s?\d|s\$|price|cost[s]?)\b", re.I)
YEAR = re.compile(r"\b(20(1[5-9]|2\d))\b")


def check_tnf(ex: dict) -> list[tuple[str, str]]:
    out = []
    tgt = str(ex["target"]).strip().upper()
    if tgt not in ("TRUE", "FALSE"):
        out.append(("tnf_bad_target", f"target {ex['target']!r}"))
    body = statement(ex["input"])
    reasons = []
    if HEDGE.search(body):
        reasons.append("hedge: " + HEDGE.search(body).group(0))
    if ABSOLUTE.search(body):
        reasons.append("absolute: " + ABSOLUTE.search(body).group(0))
    stmt = norm(body)
    if tgt == "TRUE" and len(re.findall(r"\band\b|,", body)) >= 2 and len(stmt.split()) > 25:
        reasons.append("long compound claim keyed TRUE: every part must hold")
    if reasons:
        out.append(("ambiguity", "; ".join(reasons)))
    return out


def check_common(ex: dict) -> list[tuple[str, str]]:
    out = []
    text = ex["input"] + " " + str(ex["target"])
    if "�" in text or re.search(r"Ã.|â€|ï¿½", text):
        out.append(("encoding", "replacement character or mojibake"))
    if re.search(r" {3,}", ex["input"]) or str(ex["target"]) != str(ex["target"]).strip():
        out.append(("whitespace", "runs of spaces or padded target"))
    if norm(ex["input"]) in ("term", "question", "input", "prompt") and len(str(ex["target"]).split()) <= 2:
        out.append(("header_row", "looks like a CSV header row"))
    claim = statement(ex["input"])
    stale = [m.group(0) for m in STALE.finditer(claim)] + [m.group(0) for m in YEAR.finditer(claim)]
    if stale:
        out.append(("time_sensitive", ", ".join(sorted(set(s.lower() for s in stale)))))
    return out


def check_examples(examples: list[dict]) -> list[tuple[int, str, str]]:
    flags = []
    for i, ex in enumerate(examples):
        for c, d in check_common(ex):
            flags.append((i, c, d))
        if is_mcq(ex):
            for c, d in check_mcq(ex):
                flags.append((i, c, d))
        elif is_tnf(ex):
            for c, d in check_tnf(ex):
                flags.append((i, c, d))
    return flags


def duplicates(named: dict[str, list[dict]], near: float = 0.8) -> list[tuple[str, str, str, float]]:
    """Across all datasets: (id_a, id_b, 'exact_duplicate' | 'near_duplicate', jaccard). Both input and target
    are compared for exact duplicates; near duplicates compare the input only."""
    rows = [(f"{n}#{i}", ex) for n, exs in named.items() for i, ex in enumerate(exs)]
    toks = {rid: tokens(ex["input"]) for rid, ex in rows}
    keys = {rid: (norm(ex["input"]), norm(str(ex["target"]))) for rid, ex in rows}
    out = []
    for (a, _), (b, _) in combinations(rows, 2):
        if keys[a][0] == keys[b][0]:
            out.append((a, b, "exact_duplicate" if keys[a][1] == keys[b][1] else "same_input_different_target", 1.0))
            continue
        j = jaccard(toks[a], toks[b])
        if j >= near:
            out.append((a, b, "near_duplicate", round(j, 3)))
    return out


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def metadata_check(file_stem: str, meta: dict) -> str | None:
    """Flag a dataset whose name shares no topic word with its file name (e.g. a housing file named 'Transport'),
    and name words that are one or two edits away from a file-name word (a likely misspelling)."""
    stop = {"singapore", "sg", "in", "the", "of", "and", "tnf", "about", "questions", "facts", "statements", "true"}
    fw = {w for w in re.split(r"[-_ ]+", file_stem.lower()) if w not in stop}
    nw = {w for w in re.split(r"[^a-z]+", str(meta.get("name", "")).lower()) if w and w not in stop}
    close = [(a, b) for a in fw for b in nw if a != b and len(a) > 3 and edit_distance(a, b) <= 2]
    if close:
        return f"name {meta.get('name')!r}: {close[0][1]!r} looks like a misspelling of {close[0][0]!r}"
    if fw and nw and not (fw & nw):
        return f"name {meta.get('name')!r} does not match file {file_stem!r}"
    return None
