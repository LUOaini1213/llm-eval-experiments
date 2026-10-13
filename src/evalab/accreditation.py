"""E5: scoring third-party AI test reports against the rubric in data/accreditation/rubric.json.

The functions here validate an assessment (every indicator scored on the 0-2 scale, evidence for every positive
score, short quotations), total it per criterion, list the critical indicators that block reliance, and check the
quotations against the text of the source document when that text is available locally.
"""
from __future__ import annotations

import re

MAX_QUOTE_WORDS = 25


def indicators(rubric: dict) -> list[dict]:
    return [dict(ind, criterion=c["id"]) for c in rubric["criteria"] for ind in c["indicators"]]


def validate(rubric: dict, report: dict) -> list[str]:
    """Problems with one report's assessment; an empty list means it is complete and well formed."""
    problems = []
    ids = [i["id"] for i in indicators(rubric)]
    scores = report["scores"]
    for missing in sorted(set(ids) - set(scores)):
        problems.append(f"{report['id']}: {missing} not scored")
    for extra in sorted(set(scores) - set(ids)):
        problems.append(f"{report['id']}: {extra} is not in the rubric")
    for k, s in scores.items():
        if s["score"] not in (0, 1, 2):
            problems.append(f"{report['id']} {k}: score {s['score']} is off the 0-2 scale")
        if not s.get("why", "").strip():
            problems.append(f"{report['id']} {k}: no justification")
        if s["score"] > 0 and (not s.get("quote") or not s.get("page")):
            problems.append(f"{report['id']} {k}: a positive score needs a page and a quotation")
        if s.get("quote") and len(s["quote"].split()) > MAX_QUOTE_WORDS:
            problems.append(f"{report['id']} {k}: quotation longer than {MAX_QUOTE_WORDS} words")
    return problems


def totals(rubric: dict, report: dict) -> dict:
    out = {}
    for c in rubric["criteria"]:
        got = sum(report["scores"][i["id"]]["score"] for i in c["indicators"])
        out[c["id"]] = {"name": c["name"], "score": got, "max": 2 * len(c["indicators"])}
    out["total"] = {"name": "Total", "score": sum(v["score"] for v in out.values()),
                    "max": sum(v["max"] for v in out.values())}
    return out


def critical_gaps(rubric: dict, report: dict) -> list[str]:
    """Critical indicators scored 0: the report is referred back for evidence, whatever its total."""
    return [i["id"] for i in indicators(rubric) if i["critical"] and report["scores"][i["id"]]["score"] == 0]


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def quote_found(quote: str, page_text: str) -> bool:
    """A quotation may elide words with '...'; each fragment must appear on the page. Spacing, punctuation and case
    are ignored, because PDF text extraction breaks lines and drops or splits characters."""
    hay = _squash(page_text)
    return all(_squash(frag) in hay for frag in quote.split("...") if _squash(frag))
