"""Write the E3 test set and its Moonshot configuration into moonshot_ext/.

- datasets/sgbus-network-facts.json       Moonshot dataset format (name, description, license, reference, examples)
- prompt-templates/sgbus-short-answer.json
- recipes/sgbus-network-facts.json        metrics: sgfacts-match (rule) and llm-jury
- connectors-endpoints/ollama-*.json      one endpoint per local model (ollama-connector)
- metrics/metrics_config.json             the llm-jury judges and aggregation

The jury is the three-judge jury fixed by the E1 pre-registration (docs/PREREGISTRATION.md, section "Juries"),
read from results/e1/pilot_choices.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab import moonshot_io as M  # noqa: E402
from evalab.items import load_jsonl  # noqa: E402

MODELS = ["qwen2.5:3b", "llama3.2:3b", "phi4-mini:3.8b", "gemma3:4b", "qwen3.5:4b"]
EXT = ROOT / "moonshot_ext"


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main():
    items = load_jsonl(ROOT / "data" / "e3" / "sgbus_items.jsonl")
    dump(EXT / "datasets" / f"{M.DATASET_ID}.json", M.dataset(items))
    dump(EXT / "prompt-templates" / f"{M.TEMPLATE_ID}.json", M.prompt_template())
    dump(EXT / "recipes" / f"{M.RECIPE_ID}.json", M.recipe())
    for m in MODELS:
        ep = M.endpoint(m)
        dump(EXT / "connectors-endpoints" / f"{ep['name']}.json", ep)
    choices = json.loads((ROOT / "results" / "e1" / "pilot_choices.json").read_text(encoding="utf-8"))
    jury = [M.endpoint_id(m) for m in choices["jury3"]]
    dump(EXT / "metrics" / "metrics_config.json",
         {"llm-jury": {"endpoints": jury, "configurations": {"condition": "bin_ref", "aggregation": "majority"}},
          "sgfacts-match": {"endpoints": [], "configurations": {}}})
    print("exported", len(items), "items; jury:", jury)


if __name__ == "__main__":
    main()
