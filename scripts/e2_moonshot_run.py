"""E2 step 1: run Moonshot's own Singapore-facts recipes against the five local models, through Moonshot.

    python scripts/e2_moonshot_run.py

Recipes singapore-facts-tf (3 true/false datasets) and singapore-facts-mcq (4 multiple-choice datasets) from
moonshot-data, with their stock prompt templates and the stock exactstrmatch metric, via this repository's
ollama-connector. The per-item predictions are written to results/e2/moonshot_run.json with each item's dataset and
index, together with exactstrmatch's accuracy and the accuracy after a lenient parse of the same replies (first
TRUE/FALSE word, or leading option letter), which shows how much of the score is answer formatting.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import moonshot_workspace as W  # noqa: E402
from e2_audit import DATA, key_answer, parse_model_answer  # noqa: E402
from evalab import audit as A  # noqa: E402
from evalab.moonshot_io import endpoint_id  # noqa: E402

MODELS = ["qwen2.5:3b", "llama3.2:3b", "phi4-mini:3.8b", "gemma3:4b", "qwen3.5:4b"]
RECIPES = ["singapore-facts-tf", "singapore-facts-mcq"]


def main():
    os.environ.setdefault("OLLAMA_CONNECTOR_USAGE_LOG", str(ROOT / "local" / "e2_usage.jsonl"))
    # one runner per model: Moonshot sends a runner's endpoints their prompts concurrently, and on a 4 GB GPU that
    # makes Ollama swap models on every call
    runs = []
    for m in MODELS:
        res, path = W.run(RECIPES, [endpoint_id(m)], 100, 0, runner_id=f"e2-sg-facts-{endpoint_id(m)}")
        runs.append((res, path))
    preds, acc = [], defaultdict(lambda: {"n": 0, "exactstrmatch": 0.0, "lenient": 0})
    examples = {}
    for r in [r for res, _ in runs for r in res["results"]["recipes"]]:
        for d in r["details"]:
            ds = d["dataset_id"]
            if ds not in examples:
                examples[ds] = json.loads((DATA / "datasets" / f"{ds}.json").read_text(encoding="utf-8"))["examples"]
            exs = examples[ds]
            used = defaultdict(int)
            for row in d["data"]:
                # map the templated prompt back to the dataset item (identical inputs are taken in order)
                cands = [i for i, e in enumerate(exs) if e["input"] in row["prompt"]]
                if not cands:
                    continue
                i = cands[min(used[row["prompt"]], len(cands) - 1)]
                used[row["prompt"]] += 1
                resp = row["predicted_result"]["response"] if isinstance(row["predicted_result"], dict) \
                    else str(row["predicted_result"])
                preds.append({"dataset": ds, "index": i, "model": d["model_id"], "response": resp})
                a = acc[(r["id"], d["model_id"])]
                a["n"] += 1
                a["lenient"] += parse_model_answer(resp, A.is_mcq(exs[i])) == key_answer(exs[i])
            for m in d["metrics"]:
                if "exactstrmatch" in m:
                    acc[(r["id"], d["model_id"])]["exactstrmatch_by_dataset"] = \
                        acc[(r["id"], d["model_id"])].get("exactstrmatch_by_dataset", {}) | \
                        {ds: m["exactstrmatch"]["accuracy"]}
    cmp = {}
    for (rec, model), a in acc.items():
        by = a.get("exactstrmatch_by_dataset", {})
        cmp[f"{rec}|{model}"] = {"n": a["n"], "lenient_accuracy": a["lenient"] / max(1, a["n"]),
                                 "exactstrmatch_by_dataset": by}
    out = {"moonshot_result_files": [p.name for _, p in runs], "run_ids": [r["metadata"]["id"] for r, _ in runs],
           "duration_s": sum(r["metadata"]["duration"] for r, _ in runs), "predictions": preds,
           "metric_comparison": cmp}
    (ROOT / "results" / "e2").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "e2" / "moonshot_run.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print("predictions", len(preds), json.dumps(cmp, indent=1))


if __name__ == "__main__":
    main()
