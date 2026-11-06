"""Ollama HTTP backend (fallback for a local Ollama server)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .base import Backend, GenerateRequest, GenerateResult


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model_name: str, host: str = "http://localhost:11434", timeout: int = 300, **_) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = int(timeout)

    def generate(self, request: GenerateRequest) -> GenerateResult:
        start = time.perf_counter()
        payload = {
            "model": request.model or self.model_name,
            "messages": request.messages,
            "stream": False,
            "options": {
                "temperature": float(request.temperature),
                "top_p": float(request.top_p),
                "num_predict": int(request.max_new_tokens),
            },
        }
        if request.seed is not None:
            payload["options"]["seed"] = int(request.seed)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return GenerateResult(
                text="",
                model=self.model_name,
                latency_s=time.perf_counter() - start,
                error=f"ollama: {exc}",
            )
        text = str((body.get("message") or {}).get("content", ""))
        return GenerateResult(
            text=text.strip(),
            model=self.model_name,
            latency_s=time.perf_counter() - start,
        )
