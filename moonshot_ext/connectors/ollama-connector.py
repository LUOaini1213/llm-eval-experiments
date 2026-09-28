"""Moonshot connector for a local Ollama server (native /api/chat).

Moonshot's stock openai-connector can reach Ollama through its OpenAI-compatible endpoint, and moonshot-data ships
an endpoint file for that. This connector talks to Ollama's native API instead, because the OpenAI-compatible
endpoint cannot set:

- num_ctx (the context window; Ollama's default silently truncates long prompts),
- keep_alive (how long the model stays on the GPU), and
- think=false for reasoning models such as qwen3.5 (the compatible endpoint takes reasoning_effort, which not every
  Moonshot component passes through).

Endpoint params understood here (all optional): temperature, seed, num_ctx, num_predict, top_p, think, keep_alive,
plus Moonshot's own timeout, max_attempts, system_prompt, pre_prompt and post_prompt.
Token counts and latency of every call are kept in `self.usage` and, if the environment variable
OLLAMA_CONNECTOR_USAGE_LOG names a file, appended to it as JSON lines.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request

from moonshot.src.connectors.connector import Connector, perform_retry
from moonshot.src.connectors.connector_response import ConnectorResponse
from moonshot.src.connectors_endpoints.connector_endpoint_arguments import ConnectorEndpointArguments

OPTION_KEYS = ("temperature", "seed", "num_ctx", "num_predict", "top_p", "top_k", "repeat_penalty", "stop")


def build_payload(model: str, prompt: str, system_prompt: str, params: dict) -> dict:
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
               [{"role": "user", "content": prompt}]
    return {"model": model, "messages": messages, "stream": False,
            "think": bool(params.get("think", False)),
            "keep_alive": params.get("keep_alive", "30m"),
            "options": {k: params[k] for k in OPTION_KEYS if k in params}}


def post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OllamaConnector(Connector):
    def __init__(self, ep_arguments: ConnectorEndpointArguments):
        super().__init__(ep_arguments)
        self.base_url = (self.endpoint or "http://127.0.0.1:11434").rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.usage: list[dict] = []

    @Connector.rate_limited
    @perform_retry
    async def get_response(self, prompt: str) -> ConnectorResponse:
        payload = build_payload(self.model, f"{self.pre_prompt}{prompt}{self.post_prompt}", self.system_prompt,
                                self.optional_params)
        t0 = time.perf_counter()
        out = await asyncio.to_thread(post_json, self.base_url + "/api/chat", payload, self.timeout)
        rec = {"model": self.model, "prompt_tokens": int(out.get("prompt_eval_count") or 0),
               "output_tokens": int(out.get("eval_count") or 0), "seconds": round(time.perf_counter() - t0, 3)}
        self.usage.append(rec)
        log = os.environ.get("OLLAMA_CONNECTOR_USAGE_LOG")
        if log:
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return ConnectorResponse(response=(out.get("message") or {}).get("content", ""))
