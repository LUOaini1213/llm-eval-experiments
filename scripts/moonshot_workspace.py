"""Build a Moonshot workspace (git-ignored, under local/moonshot-ws) and run recipes through Moonshot's library.

Moonshot (aiverify-moonshot 0.7.6) reads connectors, metrics, datasets, recipes and prompt templates from one folder
per kind. The workspace combines:
- from moonshot-data (cloned to external/moonshot-data, Apache-2.0): the runner, result, database and I/O modules,
  the stock exactstrmatch/relaxstrmatch metrics, and the Singapore datasets, recipes and prompt templates;
- from this repository (moonshot_ext/): the Ollama connector, the llm-jury and sgfacts-match metrics, the endpoint
  files for the local models, and the E3 dataset, prompt template and recipe.

    python scripts/moonshot_workspace.py run <recipe>[,<recipe>] <endpoint>[,<endpoint>] [percentage] [seed]
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MOONSHOT_DATA", ROOT / "external" / "moonshot-data"))
EXT = ROOT / "moonshot_ext"
WS = ROOT / "local" / "moonshot-ws"
KINDS = {"CONNECTORS": "connectors", "CONNECTORS_ENDPOINTS": "connectors-endpoints", "METRICS": "metrics",
         "DATASETS": "datasets", "RECIPES": "recipes", "COOKBOOKS": "cookbooks", "PROMPT_TEMPLATES": "prompt-templates",
         "RUNNERS_MODULES": "runners-modules", "RESULTS_MODULES": "results-modules",
         "DATABASES_MODULES": "databases-modules", "IO_MODULES": "io-modules", "ATTACK_MODULES": "attack-modules",
         "CONTEXT_STRATEGY": "context-strategy", "DATABASES": "generated-outputs/databases",
         "RESULTS": "generated-outputs/results", "RUNNERS": "generated-outputs/runners",
         "BOOKMARKS": "generated-outputs/bookmarks"}
FROM_DATA = {
    "runners-modules": ["benchmarking.py"], "results-modules": ["benchmarking-result.py"],
    "databases-modules": ["sqlite.py"], "io-modules": ["jsonio.py"],
    "metrics": ["exactstrmatch.py", "relaxstrmatch.py"],
    "datasets": ["singapore-facts-tnf.json", "singapore-places-tnf.json", "singapore-food-tnf.json",
                 "singapore-transport-system.json", "singapore-iconic-places.json",
                 "singapore-political-history.json", "singapore-public-housing.json"],
    "recipes": ["singapore-facts-tf.json", "singapore-facts-mcq.json"],
    "prompt-templates": ["singapore-facts-tf.json", "singapore-facts-mcq.json"],
}


def build() -> dict:
    for sub in KINDS.values():
        (WS / sub).mkdir(parents=True, exist_ok=True)
    for sub, files in FROM_DATA.items():
        for f in files:
            shutil.copy2(DATA / sub / f, WS / sub / f)
    for sub in ("connectors", "connectors-endpoints", "metrics", "datasets", "recipes", "prompt-templates"):
        for f in (EXT / sub).glob("*"):
            if f.is_file() and f.name != "metrics_config.json":
                shutil.copy2(f, WS / sub / f.name)
    cfg = json.loads((EXT / "metrics" / "metrics_config.json").read_text(encoding="utf-8"))
    (WS / "metrics" / "metrics_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8", newline="\n")
    return {k: str(WS / v) for k, v in KINDS.items()}


def run(recipes: list[str], endpoints: list[str], percentage: int = 100, seed: int = 0, runner_id: str | None = None):
    sys.path.insert(0, str(ROOT / "src"))
    env = build()
    from moonshot.api import api_create_runner, api_delete_runner, api_get_all_runner_name, api_set_environment_variables
    api_set_environment_variables(env)
    name = runner_id or f"run-{'-'.join(recipes)}-{percentage}pct"
    rid = name.lower().replace("_", "-")
    if rid in api_get_all_runner_name():
        api_delete_runner(rid)
    runner = api_create_runner(name, endpoints, description="llm-eval-experiments")

    async def go():
        await runner.run_recipes(recipes, prompt_selection_percentage=percentage, random_seed=seed)
        await runner.close()

    asyncio.run(go())
    res = WS / "generated-outputs" / "results" / f"{runner.id}.json"
    return json.loads(res.read_text(encoding="utf-8")), res


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        print(json.dumps(build(), indent=1))
    elif len(sys.argv) >= 4 and sys.argv[1] == "run":
        pct = int(sys.argv[4]) if len(sys.argv) > 4 else 100
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        out, path = run(sys.argv[2].split(","), sys.argv[3].split(","), pct, seed)
        print("results:", path)
    else:
        print(__doc__)
