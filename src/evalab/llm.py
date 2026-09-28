"""Minimal client for a local Ollama server, with an on-disk cache.

Every call is deterministic (temperature 0, fixed seed) and is cached by a hash of (model, messages, options), so a
re-run of any script reads the cache instead of calling the model. The cache files are JSON lines and are committed
for the experiment runs, which is what lets CI recompute every table without a model.

Reasoning is switched off with Ollama's native ``think: false``. This is the native-API equivalent of the
``reasoning_effort: none`` that the OpenAI-compatible endpoint takes (the setting used for qwen3.5 elsewhere).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 4096}


@dataclass
class Reply:
    text: str
    model: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    cached: bool = False


def cache_key(model: str, messages: list[dict], options: dict) -> str:
    blob = json.dumps({"model": model, "messages": messages, "options": options}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    """Append-only JSONL cache. One file per experiment keeps diffs readable."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, rec: dict) -> None:
        with self.lock:
            self.data[rec["key"]] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _post(path: str, payload: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(model: str, messages: list[dict], cache: Cache | None = None, num_predict: int = 64,
         options: dict | None = None, offline: bool = False) -> Reply:
    opts = {**DEFAULT_OPTIONS, "num_predict": num_predict, **(options or {})}
    key = cache_key(model, messages, opts)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return Reply(hit["text"], model, hit["prompt_tokens"], hit["output_tokens"], hit["seconds"], cached=True)
    if offline:
        raise LookupError(f"not in cache and offline mode is on: {model} {key[:12]}")
    t0 = time.perf_counter()
    out = _post("/api/chat", {"model": model, "messages": messages, "stream": False, "think": False,
                              "options": opts, "keep_alive": "30m"})
    secs = time.perf_counter() - t0
    text = out["message"]["content"]
    rep = Reply(text, model, int(out.get("prompt_eval_count") or 0), int(out.get("eval_count") or 0), round(secs, 3))
    if cache is not None:
        cache.put({"key": key, "model": model, "text": text, "prompt_tokens": rep.prompt_tokens,
                   "output_tokens": rep.output_tokens, "seconds": rep.seconds})
    return rep


def unload(model: str) -> None:
    """Free the GPU: the 4 GB card holds one model at a time."""
    try:
        _post("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
    except Exception:
        pass


def model_digest(model: str) -> str | None:
    try:
        with urllib.request.urlopen(BASE_URL + "/api/tags", timeout=10) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    for m in tags.get("models", []):
        if m["name"] == model or m["model"] == model:
            return m["digest"]
    return None
