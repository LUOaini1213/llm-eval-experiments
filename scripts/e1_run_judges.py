"""Run the judges over one item set (E1 step 3).

    python scripts/e1_run_judges.py pilot
    python scripts/e1_run_judges.py main      # refuses to run unless docs/PREREGISTRATION.md fixes the main-set size
    python scripts/e1_run_judges.py robust
    python scripts/e1_run_judges.py pairs

Each judge model runs over every item and condition before the next model is loaded (one model fits on the GPU).
Replies are cached (cache/judge.jsonl), so a re-run costs nothing; outputs go to results/e1/judgments_<set>.jsonl.
For the main set, the SHA-256 of the pre-registration and the start time are written to
results/e1/run_manifest.json before the first call.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab import judge as J  # noqa: E402
from evalab.items import load_jsonl, write_jsonl  # noqa: E402
from evalab.llm import Cache, chat, model_digest, unload  # noqa: E402

JUDGES = ["qwen2.5:3b", "llama3.2:3b", "phi4-mini:3.8b", "gemma3:4b", "qwen3.5:4b"]
CONDITIONS = {"pilot": list(J.POINTWISE), "main": list(J.POINTWISE), "robust": ["bin_ref", "strict_ref"],
              "pairs": list(J.PAIRWISE)}
NUM_PREDICT = {"bin_ref": 8, "strict_ref": 8, "bin_free": 8, "score_ref": 16, "pair_free": 8, "pair_ref": 8}
PREREG = ROOT / "docs" / "PREREGISTRATION.md"


def main_size() -> int:
    if not PREREG.exists():
        raise SystemExit("docs/PREREGISTRATION.md is missing: the main set may not be judged before it is written")
    m = re.search(r"Main-set size:\s*\**\s*(\d+)", PREREG.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit("PREREGISTRATION.md does not state 'Main-set size: N'")
    return int(m.group(1))


def load(which: str):
    d = ROOT / "data" / "e1"
    if which == "pilot":
        return load_jsonl(d / "items_pilot.jsonl")
    if which == "main":
        return load_jsonl(d / "items_main_order.jsonl")[:main_size()]
    if which == "robust":
        return load_jsonl(d / "items_robust.jsonl")
    if which == "pairs":
        return load_jsonl(d / "pairs.jsonl")
    raise SystemExit(f"unknown set {which}")


def main():
    which = sys.argv[1]
    judges = sys.argv[2].split(",") if len(sys.argv) > 2 else JUDGES
    items = load(which)
    if which != "pairs":  # same-question items in a row, so the server reuses the cached question prefix
        items = sorted(items, key=lambda it: it.get("qid", ""))
    if which == "main":
        man = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_items": len(items),
               "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
               "judges": {m: model_digest(m) for m in judges}}
        (ROOT / "results" / "e1" / "run_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")
    cache = Cache(ROOT / "cache" / "judge.jsonl")
    rows = []
    for m in judges:
        t0 = time.time()
        for cond in CONDITIONS[which]:
            for it in items:
                if which == "pairs":
                    for order in ("AB", "BA"):
                        r = chat(m, J.pairwise_messages(it, cond, order), cache=cache, num_predict=NUM_PREDICT[cond])
                        rows.append({"set": which, "item_id": it["pair_id"], "judge": m, "condition": cond,
                                     "order": order, "reply": r.text, "parsed": J.parse_choice(r.text),
                                     "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens,
                                     "seconds": r.seconds})
                    continue
                r = chat(m, J.pointwise_messages(it, cond), cache=cache, num_predict=NUM_PREDICT[cond])
                parsed = J.parse_score(r.text) if cond == "score_ref" else J.parse_verdict(r.text)
                rows.append({"set": which, "item_id": it["item_id"], "judge": m, "condition": cond, "reply": r.text,
                             "parsed": parsed, "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens,
                             "seconds": r.seconds})
        unload(m)
        print(f"{which} {m} done in {time.time() - t0:.0f}s", flush=True)
    write_jsonl(ROOT / "results" / "e1" / f"judgments_{which}.jsonl", rows)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
