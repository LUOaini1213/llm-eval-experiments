"""Moonshot metric: deterministic scoring of the Singapore bus-network test set (E3).

Moonshot's exactstrmatch compares the whole response with the target, so "Marine Parade Road." fails against
"Marine Parade Rd". This metric applies the same rules the E1 gold labels use (src/evalab/gold.py): road-name
abbreviations, operator aliases, whole-number matching that ignores numbers copied from the question, HHMM times,
and service sets. An answer that also names a competing value (a second planning area, another count) is not
counted as correct.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit

try:
    import evalab  # noqa: F401
except ImportError:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "evalab").is_dir():
            sys.path.insert(0, str(parent / "src"))
            break
from evalab.moonshot_io import score_sgbus_response  # noqa: E402


class SGFactsMatch(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "SG facts match"
        self.description = "Rule-based scoring of short answers about Singapore's bus network against exact gold."
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        return {"id": self.id, "name": self.name, "description": self.description, "endpoints": self.endpoints,
                "configurations": self.configurations}

    @timeit
    async def get_results(self, prompts: Any, predicted_results: Any, targets: Any, *args, **kwargs) -> dict:
        successful, unsuccessful = [], []
        for p, r, t in zip(prompts, predicted_results, targets):
            label = score_sgbus_response(p, r.response, t)
            rec = {"prompt": p, "predicted_value": r.response, "target": t, "eval": label}
            (successful if label == "correct" else unsuccessful).append(rec)
        n = max(1, len(successful) + len(unsuccessful))
        accuracy = 100 * len(successful) / n
        return {"sgfacts-match": {"accuracy": accuracy,
                                  "ambiguous": sum(1 for x in unsuccessful if x["eval"] == "ambiguous"),
                                  "individual_scores": {"successful": successful, "unsuccessful": unsuccessful}},
                "grading_criteria": {"accuracy": accuracy}}
