"""Shared E1 design constants and the immutable original pre-registration boundary."""
import hashlib
import inspect
import json
import re
from pathlib import Path

from . import judge as J
from .llm import DEFAULT_OPTIONS

JUDGES = ["qwen2.5:3b", "llama3.2:3b", "phi4-mini:3.8b", "gemma3:4b", "qwen3.5:4b"]
CONDITIONS = {"pilot": list(J.POINTWISE), "main": list(J.POINTWISE),
              "robust": ["bin_ref", "strict_ref"], "pairs": list(J.PAIRWISE)}
NUM_PREDICT = {"bin_ref": 8, "strict_ref": 8, "bin_free": 8, "score_ref": 16,
               "pair_free": 8, "pair_ref": 8}


def results_dir(root: Path, run_id: str | None = None) -> Path:
    out = root / "results" / "e1"
    if run_id is None:
        return out
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id):
        raise ValueError("run-id must be 1-64 letters, digits, dots, hyphens or underscores, starting with a letter/digit")
    return out / "runs" / run_id


def sha256_json(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def judgment_cells(which: str, items: list[dict], conditions=None):
    """Yield the requested prompts with their full condition/item/order identities."""
    for cond in CONDITIONS[which] if conditions is None else conditions:
        for it in items:
            if which == "pairs":
                for order in ("AB", "BA"):
                    yield cond, it["pair_id"], order, J.pairwise_messages(it, cond, order)
            else:
                yield cond, it["item_id"], None, J.pointwise_messages(it, cond)


def run_identity(which: str, items: list[dict], *, judges=None, conditions=None, num_predict=None) -> dict:
    """Recompute the pinned input/configuration identity for running or analysing a set."""
    if which != "pairs":
        items = sorted(items, key=lambda it: it.get("qid", ""))
    judges = JUDGES if judges is None else judges
    conditions = CONDITIONS[which] if conditions is None else conditions
    num_predict = NUM_PREDICT if num_predict is None else num_predict
    return {"set": which, "n_items": len(items), "judge_names": judges, "conditions": conditions,
            "items_sha256": sha256_json(items),
            "prompts_sha256": sha256_json(list(judgment_cells(which, items, conditions))),
            "options": {cond: {**DEFAULT_OPTIONS, "num_predict": num_predict[cond]} for cond in conditions},
            "think": False,
            "parser_sha256": sha256_json([inspect.getsource(f) for f in
                                          (J.parse_verdict, J.parse_score, J.parse_choice)])}


def original_preregistration(prereg: Path) -> bytes:
    """Keep the exact bytes hashed at the original run, before appended amendments."""
    if not prereg.exists():
        raise ValueError("docs/PREREGISTRATION.md is missing: write it before judging the main set")
    raw = prereg.read_bytes()
    marker = b"\n\n## Amendments\n"
    if raw.count(marker) > 1:
        raise ValueError("pre-registration has more than one Amendments boundary")
    return raw[:raw.index(marker) + 1] if marker in raw else raw


def preregistration_sha256(prereg: Path) -> str:
    return hashlib.sha256(original_preregistration(prereg)).hexdigest()


def main_size(prereg: Path) -> int:
    text = original_preregistration(prereg).decode("utf-8")
    sizes = re.findall(r"^\s*(?:\*\*)?Main-set size:\s*(?:\*\*)?(\d+)(?:\*\*)?\s*$", text, re.M)
    if len(sizes) != 1 or int(sizes[0]) <= 0:
        raise ValueError("PREREGISTRATION.md must state exactly one positive integer 'Main-set size: N'")
    return int(sizes[0])


def validate_main_manifest(manifest: dict, prereg: Path, n_items: int) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("main run manifest must be an object")
    expected = main_size(prereg)
    if type(manifest.get("n_items")) is not int or manifest["n_items"] != expected or n_items != expected:
        raise ValueError("main run manifest/item count disagrees with the pre-registration main-set size")
    if manifest.get("preregistration_sha256") != preregistration_sha256(prereg):
        raise ValueError("pre-registration original bytes do not match the main run manifest")
