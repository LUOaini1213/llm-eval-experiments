"""E3: a small benchmark run of the Singapore bus-network test set through Moonshot's library.

    python scripts/e3_moonshot_run.py [percentage]      # default 34 (about 60 of the 180 questions)

Recipe sgbus-network-facts, the five local models as targets through ollama-connector, scored by two metrics in
the same run: sgfacts-match (rules, the gold) and llm-jury (the pre-registered three-judge jury, majority vote).
Writes results/e3/moonshot_run.json: each model's score under both metrics, and how often the jury passed an
answer the rules mark wrong (a false accept) or failed one they mark right.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import moonshot_workspace as W  # noqa: E402
from evalab.moonshot_io import RECIPE_ID, endpoint_id  # noqa: E402
from evalab.stats import wilson  # noqa: E402

MODELS = ["qwen2.5:3b", "llama3.2:3b", "phi4-mini:3.8b", "gemma3:4b", "qwen3.5:4b"]


def main():
    pct = int(sys.argv[1]) if len(sys.argv) > 1 else 34
    usage = ROOT / "local" / "e3_usage.jsonl"
    usage.parent.mkdir(exist_ok=True)
    usage.unlink(missing_ok=True)
    os.environ["OLLAMA_CONNECTOR_USAGE_LOG"] = str(usage)
    # one runner per target model, so that only one model is on the GPU at a time (see e2_moonshot_run.py)
    runs = [W.run([RECIPE_ID], [endpoint_id(m)], pct, 0, runner_id=f"e3-sgbus-{endpoint_id(m)}") for m in MODELS]
    out = {"moonshot_result_files": [p.name for _, p in runs], "run_ids": [r["metadata"]["id"] for r, _ in runs],
           "percentage": pct, "duration_s": sum(r["metadata"]["duration"] for r, _ in runs), "models": {}}
    for r in [r for res, _ in runs for r in res["results"]["recipes"]]:
        for d in r["details"]:
            mets = {k: v for m in d["metrics"] for k, v in m.items() if k in ("sgfacts-match", "llm-jury")}
            rule = {x["prompt"]: x["eval"] == "correct" for bucket in ("successful", "unsuccessful")
                    for x in mets["sgfacts-match"]["individual_scores"][bucket]}
            jury = {x["prompt"]: x["eval"] == "correct" for bucket in ("successful", "unsuccessful")
                    for x in mets["llm-jury"]["individual_scores"][bucket]}
            n = len(rule)
            fa = sum(1 for p in rule if jury[p] and not rule[p])
            fr = sum(1 for p in rule if rule[p] and not jury[p])
            wrong, right = sum(1 for p in rule if not rule[p]), sum(1 for p in rule if rule[p])
            out["models"][d["model_id"]] = {
                "n": n, "rule_accuracy": sum(rule.values()) / n,
                "rule_accuracy_ci95": wilson(sum(rule.values()), n),
                "jury_accuracy": sum(jury.values()) / n,
                "jury_false_accepts": fa, "rule_wrong": wrong, "jury_false_rejects": fr, "rule_right": right,
                "jury_fleiss_kappa": mets["llm-jury"]["fleiss_kappa"],
                "judge_pass_rate": mets["llm-jury"]["judge_pass_rate"],
                "examples": [{"prompt": p.split("\n\n", 1)[-1], "rule": rule[p], "jury": jury[p]}
                             for p in list(rule)[:3]]}
    calls = [json.loads(x) for x in usage.read_text(encoding="utf-8").splitlines()] if usage.exists() else []
    out["usage"] = {"calls": len(calls), "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                    "output_tokens": sum(c["output_tokens"] for c in calls),
                    "seconds": round(sum(c["seconds"] for c in calls), 1)}
    (ROOT / "results" / "e3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "e3" / "moonshot_run.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in out["models"].items()},
                     indent=1), out["usage"])


if __name__ == "__main__":
    main()
