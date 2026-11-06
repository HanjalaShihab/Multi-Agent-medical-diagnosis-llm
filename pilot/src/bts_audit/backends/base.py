"""Backend interface and factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerateRequest:
    messages: list[dict[str, str]]
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 320
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResult:
    text: str
    model: str
    latency_s: float = 0.0
    error: str | None = None


class Backend:
    name = "base"

    def generate(self, request: GenerateRequest) -> GenerateResult:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        pass


def build_backends(cfg: dict[str, Any], models: list[str]) -> dict[str, Backend]:
    """Create one backend object per model name."""
    btype = cfg["backend"]["type"]
    seed = int(cfg.get("seed", 13))
    if btype == "mock":
        from .mock import MockBackend

        mock_cfg = dict(cfg["backend"].get("mock", {}))
        return {m: MockBackend(m, seed=seed, **mock_cfg) for m in models}

    if btype == "ollama":
        from .ollama import OllamaBackend

        ocfg = dict(cfg["backend"].get("ollama", {}))
        return {m: OllamaBackend(m, **ocfg) for m in models}

    if btype == "hf":
        from .hf_local import HFBackend, HFModelPool

        hcfg = dict(cfg["backend"].get("hf", {}))
        max_loaded = int(hcfg.pop("max_loaded_models", 1))
        pool = HFModelPool(max_loaded=max_loaded, **hcfg)
        return {m: HFBackend(m, pool) for m in models}

    raise ValueError(f"unknown backend type: {btype}")
