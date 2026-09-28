"""The Moonshot connector and metrics, with the model mocked (no server needed)."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("moonshot")
from moonshot.src.connectors_endpoints.connector_endpoint_arguments import ConnectorEndpointArguments  # noqa: E402
from moonshot.src.metrics.metric_interface import MetricInterface  # noqa: E402

from evalab import moonshot_io as M  # noqa: E402
from evalab.items import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "moonshot_ext"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def connector_mod():
    return load(EXT / "connectors" / "ollama-connector.py", "ollama_connector")


def _ep(**params):
    return ConnectorEndpointArguments(id="ollama-test", name="ollama-test", connector_type="ollama-connector",
                                      uri="http://127.0.0.1:11434/v1", token="", max_calls_per_second=100,
                                      max_concurrency=1, model="qwen2.5:3b", params=params, created_date="")


def test_payload_passes_options_and_disables_thinking(connector_mod):
    p = connector_mod.build_payload("m", "hello", "sys", {"temperature": 0, "seed": 0, "num_ctx": 4096,
                                                          "num_predict": 8, "timeout": 5, "junk": 1})
    assert p["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert p["think"] is False and p["stream"] is False
    assert p["options"] == {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": 8}
    assert connector_mod.build_payload("m", "x", "", {"think": True})["think"] is True
    assert connector_mod.build_payload("m", "x", "", {})["messages"] == [{"role": "user", "content": "x"}]


def test_connector_calls_native_api_and_records_usage(connector_mod, monkeypatch, tmp_path):
    seen = {}

    def fake_post(url, payload, timeout):
        seen.update(url=url, payload=payload, timeout=timeout)
        return {"message": {"content": "CORRECT"}, "prompt_eval_count": 12, "eval_count": 2}

    monkeypatch.setattr(connector_mod, "post_json", fake_post)
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OLLAMA_CONNECTOR_USAGE_LOG", str(log))
    c = connector_mod.OllamaConnector(_ep(timeout=7, max_attempts=1, temperature=0, num_predict=8,
                                          system_prompt="grader", pre_prompt="<", post_prompt=">"))
    r = asyncio.run(c.get_response("q"))
    assert r.response == "CORRECT"
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"         # the /v1 suffix is stripped
    assert seen["timeout"] == 7
    assert seen["payload"]["messages"][-1]["content"] == "<q>"
    assert seen["payload"]["messages"][0]["content"] == "grader"
    assert seen["payload"]["options"] == {"temperature": 0, "num_predict": 8}
    assert c.usage[-1]["prompt_tokens"] == 12 and c.usage[-1]["output_tokens"] == 2
    assert json.loads(log.read_text().splitlines()[-1])["output_tokens"] == 2


def _metric(monkeypatch, fname, cls, cfg):
    monkeypatch.setattr(MetricInterface, "get_metrics_configuration", lambda self, mid: cfg)
    mod = load(EXT / "metrics" / fname, fname.replace("-", "_").replace(".py", ""))
    return mod, getattr(mod, cls)()


def test_llm_jury_metric_majority_with_mocked_judges(monkeypatch):
    mod, metric = _metric(monkeypatch, "llm-jury.py", "LLMJury",
                          {"endpoints": ["j1", "j2", "j3"], "configurations": {"aggregation": "majority"}})
    replies = {"j1": ["CORRECT", "INCORRECT"], "j2": ["CORRECT", "CORRECT"], "j3": ["INCORRECT", "INCORRECT"]}
    systems = []

    class FakeConn:
        def __init__(self, ep):
            self.ep, self.n, self.system_prompt = ep, 0, ""

    monkeypatch.setattr(mod.ConnectorEndpoint, "read", staticmethod(lambda ep: ep))
    monkeypatch.setattr(mod.Connector, "create", staticmethod(lambda ep: FakeConn(ep)))

    async def fake_predict(args, conn, prompt_callback=None):
        systems.append(conn.system_prompt)
        args.predicted_results = SimpleNamespace(response=replies[conn.ep][conn.n])
        conn.n += 1
        return args

    monkeypatch.setattr(mod.Connector, "get_prediction", staticmethod(fake_predict))
    preds = [SimpleNamespace(response="21"), SimpleNamespace(response="19")]
    out = asyncio.run(metric.get_results(["q1", "q2"], preds, ["21", "21"]))
    res = out["llm-jury"]
    assert res["accuracy"] == 50.0 and out["grading_criteria"]["accuracy"] == 50.0
    assert [x["votes"] for x in res["individual_scores"]["successful"]] == [{"j1": True, "j2": True, "j3": False}]
    assert res["judge_pass_rate"] == {"j1": 50.0, "j2": 100.0, "j3": 0.0}
    assert all("CORRECT or INCORRECT" in s for s in systems)


def test_llm_jury_metric_unanimity(monkeypatch):
    mod, metric = _metric(monkeypatch, "llm-jury.py", "LLMJury",
                          {"endpoints": ["j1", "j2"], "configurations": {"aggregation": "unanimity"}})

    class FakeConn:
        def __init__(self, ep):
            self.ep, self.system_prompt = ep, ""

    monkeypatch.setattr(mod.ConnectorEndpoint, "read", staticmethod(lambda ep: ep))
    monkeypatch.setattr(mod.Connector, "create", staticmethod(lambda ep: FakeConn(ep)))

    async def fake_predict(args, conn, prompt_callback=None):
        args.predicted_results = SimpleNamespace(response="CORRECT" if conn.ep == "j1" else "INCORRECT")
        return args

    monkeypatch.setattr(mod.Connector, "get_prediction", staticmethod(fake_predict))
    out = asyncio.run(metric.get_results(["q"], [SimpleNamespace(response="x")], ["y"]))
    assert out["llm-jury"]["accuracy"] == 0.0


def test_sgfacts_match_metric(monkeypatch):
    _, metric = _metric(monkeypatch, "sgfacts-match.py", "SGFactsMatch", {})
    items = load_jsonl(ROOT / "data" / "e3" / "sgbus_items.jsonl")
    ds = M.dataset(items)
    prompts = [M.SHORT_ANSWER.replace("{{ prompt }}", e["input"]) for e in ds["examples"]]
    right = [SimpleNamespace(response=e["target"] + ".") for e in ds["examples"]]
    out = asyncio.run(metric.get_results(prompts, right, [e["target"] for e in ds["examples"]]))
    assert out["sgfacts-match"]["accuracy"] == 100.0
    wrong = [SimpleNamespace(response="I don't know") for _ in ds["examples"]]
    out = asyncio.run(metric.get_results(prompts, wrong, [e["target"] for e in ds["examples"]]))
    assert out["sgfacts-match"]["accuracy"] == 0.0


def test_dataset_export_format():
    items = load_jsonl(ROOT / "data" / "e3" / "sgbus_items.jsonl")
    ds = M.dataset(items)
    assert set(ds) == {"name", "description", "license", "reference", "examples"}
    assert len(ds["examples"]) == len(items)
    assert all(set(e) == {"input", "target"} and e["target"] for e in ds["examples"])
    assert "Singapore Open Data Licence" in ds["reference"]
    committed = json.loads((EXT / "datasets" / f"{M.DATASET_ID}.json").read_text(encoding="utf-8"))
    assert committed == ds
    r = M.recipe()
    assert r["datasets"] == [M.DATASET_ID] and "llm-jury" in r["metrics"] and "sgfacts-match" in r["metrics"]
    ep = M.endpoint("qwen3.5:4b")
    assert ep["connector_type"] == "ollama-connector" and ep["params"]["think"] is False
    assert "{{ prompt }}" in M.prompt_template()["template"]


def test_kind_inference_covers_every_template():
    items = load_jsonl(ROOT / "data" / "e3" / "sgbus_items.jsonl")
    for it in items:
        assert M.infer_kind(it["question"]) == it["kind"], it["id"]
