"""Run or safely resume one E1 item set.

    python scripts/e1_run_judges.py pilot
    python scripts/e1_run_judges.py main qwen2.5:3b
    python scripts/e1_run_judges.py main --run-id replication-1

Existing replies are never regenerated or discarded. New runs pin the items, prompts,
options and model digests in an immutable manifest, then atomically checkpoint after
each condition (and on an ordinary interruption). A per-set lock prevents competing
writers. A hard crash can leave the lock: remove it only after the old process stops.

Historical outputs without this provenance can be read unchanged if the requested
slice is complete; filling historical gaps requires a new --run-id. Named runs write
under results/e1/runs/<id>; their provenance-keyed caches cannot reuse old model tags.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab import judge as J  # noqa: E402
from evalab.experiment import (CONDITIONS, JUDGES, NUM_PREDICT, main_size as declared_main_size,
                               judgment_cells, preregistration_sha256, results_dir, run_identity,
                               sha256_json, validate_main_manifest)  # noqa: E402
from evalab.items import load_jsonl  # noqa: E402
from evalab.llm import Cache, chat, model_digest, unload  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION.md"


def main_size() -> int:
    return declared_main_size(PREREG)


def load(which: str):
    names = {"pilot": "items_pilot", "main": "items_main_order", "robust": "items_robust", "pairs": "pairs"}
    if which not in names:
        raise ValueError(f"unknown set {which}")
    items = load_jsonl(ROOT / "data" / "e1" / f"{names[which]}.jsonl")
    if which == "main":
        n = main_size()
        if len(items) < n:
            raise ValueError(f"main-set size is {n}, but only {len(items)} source items are available")
        items = items[:n]
    field = "pair_id" if which == "pairs" else "item_id"
    ids = [it.get(field) for it in items]
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{which} requires nonempty, unique {field} values")
    return items


def _parsed(reply, cond):
    if cond in J.PAIRWISE:
        return J.parse_choice(reply)
    return J.parse_score(reply) if cond == "score_ref" else J.parse_verdict(reply)


def _index(rows, which, cells):
    allowed = {(judge, cond, item_id, order) for judge in JUDGES for cond, item_id, order, _ in cells}
    seen = set()
    for n, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"judgment row {n} must be an object")
        try:
            key = (row["judge"], row["condition"], row["item_id"], row.get("order"))
            valid = key in allowed and key not in seen and row["set"] == which
        except (KeyError, TypeError):
            valid = False
        if not valid or (which != "pairs" and "order" in row):
            raise ValueError(f"judgment row {n} has a duplicate, foreign or invalid identity")
        if not isinstance(row.get("reply"), str) or "parsed" not in row:
            raise ValueError(f"judgment row {n} is missing its reply or parsed result")
        parsed = _parsed(row["reply"], row["condition"])
        if type(row["parsed"]) is not type(parsed) or row["parsed"] != parsed:
            raise ValueError(f"judgment row {n} disagrees with the current reply parser")
        for field in ("prompt_tokens", "output_tokens", "seconds"):
            value = row.get(field)
            kinds = (int, float) if field == "seconds" else (int,)
            if type(value) not in kinds or not math.isfinite(value) or value < 0:
                raise ValueError(f"judgment row {n} has invalid {field}")
        seen.add(key)
    return seen


def _atomic_write(path, text):
    """Replace a complete file on the same filesystem; failed writes leave the old file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _checkpoint(path, rows):
    _atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


@contextmanager
def _lock(path):
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError(f"run is locked by {path}; remove the lock only after the other process has stopped") from None
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def _run(which, judges, out):
    items = load(which)
    if which != "pairs":
        items = sorted(items, key=lambda it: it.get("qid", ""))
    cells = list(judgment_cells(which, items, CONDITIONS[which]))
    identity = run_identity(which, items, judges=JUDGES, conditions=CONDITIONS[which], num_predict=NUM_PREDICT)
    path = out / f"judgments_{which}.jsonl"
    manifest_path = out / ("run_manifest.json" if which == "main" else f"run_manifest_{which}.json")
    rows = load_jsonl(path) if path.exists() else []
    seen = _index(rows, which, cells)
    missing = [judge for judge in judges if any((judge, c, i, o) not in seen for c, i, o, _ in cells)]
    manifest_exists = manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_exists else None
    if manifest_exists:
        if not isinstance(manifest, dict):
            raise ValueError("run manifest must be an object")
        if which == "main":
            validate_main_manifest(manifest, PREREG, len(items))
        if type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1:
            if manifest.get("identity") != identity:
                raise ValueError("experiment identity/configuration changed; use a new --run-id")
            if type(manifest.get("n_items")) is not int or manifest["n_items"] != len(items):
                raise ValueError("run manifest item count disagrees with its pinned identity")
            digests = manifest.get("judges")
            if not isinstance(digests, dict) or set(digests) != set(JUDGES) or any(
                    not isinstance(d, str) or not d for d in digests.values()):
                raise ValueError("run manifest has invalid judge digest identities")
        elif "schema_version" in manifest:
            raise ValueError("unsupported run manifest schema_version")
    if which == "main" and path.exists() and manifest is None:
        raise ValueError("legacy main results have no run manifest; use a new --run-id")
    legacy = (manifest is not None and "schema_version" not in manifest) or (path.exists() and manifest is None)
    if legacy and missing:
        raise ValueError("legacy run lacks item/prompt identity provenance for filling gaps; use a new --run-id")
    if not missing:
        print(f"{which}: requested judgments already complete; {len(rows)} existing rows unchanged")
        return
    if manifest is None:
        digests = {judge: model_digest(judge) for judge in JUDGES}
        if any(not isinstance(d, str) or not d for d in digests.values()):
            raise ValueError("cannot start without every configured judge model digest")
        manifest = {"schema_version": 1, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "run_directory": out.relative_to(ROOT / "results" / "e1").as_posix(),
                    "n_items": len(items), "judges": digests, "identity": identity}
        if which == "main":
            manifest["preregistration_sha256"] = preregistration_sha256(PREREG)
        _atomic_write(manifest_path, json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    else:
        for judge in missing:
            if model_digest(judge) != manifest["judges"][judge]:
                raise ValueError(f"model digest changed or unavailable for {judge}; use a new --run-id")
    cache = Cache(ROOT / "cache" / "judge" / sha256_json(manifest) / f"{which}.jsonl")
    dirty = False
    try:
        for judge in missing:
            t0 = time.time()
            try:
                for cond in CONDITIONS[which]:
                    for c, item_id, order, messages in cells:
                        key = judge, c, item_id, order
                        if c != cond or key in seen:
                            continue
                        reply = chat(judge, messages, cache=cache, num_predict=NUM_PREDICT[cond])
                        row = {"set": which, "item_id": item_id, "judge": judge, "condition": cond,
                               "reply": reply.text, "parsed": _parsed(reply.text, cond),
                               "prompt_tokens": reply.prompt_tokens, "output_tokens": reply.output_tokens,
                               "seconds": reply.seconds}
                        if order is not None:
                            row["order"] = order
                        rows.append(row)
                        seen.add(key)
                        dirty = True
                    if dirty:
                        _checkpoint(path, rows)
                        dirty = False
            finally:
                unload(judge)
            print(f"{which} {judge} done in {time.time() - t0:.0f}s", flush=True)
    finally:
        if dirty:
            _checkpoint(path, rows)
    print("rows", len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("set", choices=CONDITIONS)
    parser.add_argument("judges", nargs="?", help="comma-separated subset of the configured judges")
    parser.add_argument("--run-id", help="write an independent run under results/e1/runs/NAME")
    args = parser.parse_args()
    judges = args.judges.split(",") if args.judges is not None else list(JUDGES)
    if len(set(judges)) != len(judges) or any(judge not in JUDGES for judge in judges):
        raise ValueError("judges must be a nonempty, unique subset of the configured judges")
    out = results_dir(ROOT, args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    with _lock(out / f".judgments_{args.set}.lock"):
        _run(args.set, judges, out)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
