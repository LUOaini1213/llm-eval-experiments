"""Moonshot file formats for the E3 test set: dataset, prompt template, recipe, endpoints, metric config."""
from __future__ import annotations

from .items import label_sgbus

DATASET_ID = "sgbus-network-facts"
RECIPE_ID = "sgbus-network-facts"
TEMPLATE_ID = "sgbus-short-answer"
SHORT_ANSWER = ("Answer the question about Singapore's public bus network. Reply with the answer only, "
                "without explanation.\n\n{{ prompt }}")

KIND_BY_PHRASE = [
    ("On which road", "road"), ("planning area is", "area"), ("Which company operates", "operator"),
    ("at what time", "time"), ("in kilometres", "km"), ("call at both", "set"),
]

ATTRIBUTION = ("Contains information from LTA DataMall (bus stops, services and routes, snapshot of 26 September "
               "2026) and URA Master Plan 2019 Planning Area Boundary (No Sea) from data.gov.sg, made available "
               "under the terms of the Singapore Open Data Licence version 1.0 "
               "(https://beta.data.gov.sg/open-data-license).")


def infer_kind(prompt: str) -> str:
    for phrase, kind in KIND_BY_PHRASE:
        if phrase in prompt:
            return kind
    return "int"


def score_sgbus_response(prompt: str, response: str, target) -> str:
    gold = target[0] if isinstance(target, list) else str(target)
    kind = infer_kind(prompt)
    if kind == "set":
        gold = gold.replace(", ", ";")
    if kind == "km":
        gold = gold.replace(" km", "")
    if kind == "area":
        gold = gold.upper()
    return label_sgbus({"kind": kind, "gold": gold, "question": prompt}, response)


def dataset(items: list[dict]) -> dict:
    return {
        "name": "Singapore bus network facts",
        "description": ("Short-answer questions with exact answers about Singapore's public bus network (stops, "
                        "services, routes, first-bus times, planning areas), generated from LTA DataMall and URA "
                        "open data by a seeded generator with an independent verification step. "
                        "Every question names the data snapshot it is true for."),
        "license": "Derived from data under the Singapore Open Data Licence v1.0 (see reference)",
        "reference": ATTRIBUTION,
        "examples": [{"input": it["question"], "target": _target(it)} for it in items],
    }


def _target(it: dict) -> str:
    if it["kind"] == "set":
        return ", ".join(it["gold"].split(";"))
    if it["kind"] == "km":
        return f"{it['gold']} km"
    if it["kind"] == "area":
        return it["gold"].title()
    return it["gold"]


def prompt_template() -> dict:
    return {"name": TEMPLATE_ID, "description": "Ask for the answer only, so that short-answer metrics can score it.",
            "template": SHORT_ANSWER}


def recipe(metrics=("sgfacts-match", "llm-jury")) -> dict:
    return {"id": RECIPE_ID, "name": "Singapore bus network facts",
            "description": ("Exact-answer questions about Singapore's bus network in three difficulty levels. "
                            "Scored by rule (sgfacts-match) and by an LLM jury (llm-jury) for comparison."),
            "tags": ["Singapore", "Transport"], "categories": ["Capability"], "datasets": [DATASET_ID],
            "prompt_templates": [TEMPLATE_ID], "metrics": list(metrics),
            "grading_scale": {"A": [80, 100], "B": [60, 79], "C": [40, 59], "D": [20, 39], "E": [0, 19]}}


def endpoint(model: str, num_predict: int = 64) -> dict:
    return {"name": f"ollama-{model.replace(':', '-').replace('.', '-')}", "connector_type": "ollama-connector",
            "uri": "http://127.0.0.1:11434", "token": "", "max_calls_per_second": 50, "max_concurrency": 1,
            "model": model,
            "params": {"timeout": 300, "max_attempts": 2, "temperature": 0, "seed": 0, "num_ctx": 4096,
                       "num_predict": num_predict, "think": False}}


def endpoint_id(model: str) -> str:
    return endpoint(model)["name"]
