"""Moonshot metric: an LLM jury that grades each response against the dataset target.

Every judge endpoint (set in metrics_config.json under "llm-jury") receives the reference-guided binary prompt from
experiment E1 and answers CORRECT or INCORRECT; the jury's verdict is the majority (default) or unanimity of the
votes. Unparseable replies count as INCORRECT. The metric reports the jury's pass rate, each judge's pass rate and
Fleiss' kappa between the judges.

The prompt and aggregation code live in this repository's src/evalab (jury.py, judge.py), which must be importable.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors.connector_prompt_arguments import ConnectorPromptArguments
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit

try:
    import evalab  # noqa: F401
except ImportError:  # the metric file is usually copied into a Moonshot workspace next to the repository
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "evalab").is_dir():
            sys.path.insert(0, str(parent / "src"))
            break
from evalab.jury import run_jury  # noqa: E402


class LLMJury(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "LLM Jury"
        self.description = ("Grades each response against the target with a jury of LLM judges (reference-guided, "
                            "binary verdicts, majority or unanimity).")
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        return {"id": self.id, "name": self.name, "description": self.description, "endpoints": self.endpoints,
                "configurations": self.configurations}

    def _judge(self, connector):
        async def call(system: str, prompt: str) -> str:
            connector.system_prompt = system
            args = ConnectorPromptArguments(prompt_index=0, prompt=prompt, target="CORRECT")
            await Connector.get_prediction(args, connector)
            return args.predicted_results.response

        return call

    @timeit
    async def get_results(self, prompts: Any, predicted_results: Any, targets: Any, *args, **kwargs) -> dict:
        if not self.endpoints:
            raise RuntimeError("llm-jury needs judge endpoints in metrics_config.json")
        judges = {ep: self._judge(Connector.create(ConnectorEndpoint.read(ep))) for ep in self.endpoints}
        responses = [r.response for r in predicted_results]
        out = await run_jury(list(prompts), responses, list(targets), judges,
                             condition=self.configurations.get("condition", "bin_ref"),
                             aggregation=self.configurations.get("aggregation", "majority"))
        successful, unsuccessful = [], []
        for i, (p, r, t) in enumerate(zip(prompts, responses, targets)):
            rec = {"prompt": p, "predicted_value": r, "target": t,
                   "votes": {k: out["votes"][k][i] for k in out["votes"]},
                   "eval": "correct" if out["verdicts"][i] else "wrong"}
            (successful if out["verdicts"][i] else unsuccessful).append(rec)
        accuracy = 100 * out["pass_rate"]
        return {
            "llm-jury": {
                "accuracy": accuracy,
                "aggregation": self.configurations.get("aggregation", "majority"),
                "judge_pass_rate": {k: 100 * v for k, v in out["judge_pass_rate"].items()},
                "unparsed": out["unparsed"],
                "fleiss_kappa": out["fleiss_kappa"],
                "individual_scores": {"successful": successful, "unsuccessful": unsuccessful},
            },
            "grading_criteria": {"accuracy": accuracy},
        }
