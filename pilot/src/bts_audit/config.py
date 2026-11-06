"""Configuration loading and defaults."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "run_name": "bts_pilot",
    "seed": 13,
    "output_dir": None,
    "data": {
        "path": None,
        "dataset": "medqa",
        "limit": None,
        "max_options": 10,
    },
    "backend": {
        "type": "mock",
        "hf": {
            "device_map": "auto",
            "dtype": "auto",
            "load_in_4bit": False,
            "trust_remote_code": False,
            "max_loaded_models": 1,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "timeout": 300,
        },
        "mock": {
            "skill": 0.65,
            "signal_strength": 2.0,
            "forecast_own_weight": 0.0,
            "forecast_noise": 0.10,
            "decoy_strength": 2.4,
            "pressure_strength": 3.2,
            "prevalence_weight": 0.30,
        },
    },
    "team": {
        "agents": 5,
        "models": ["mock-a", "mock-b", "mock-c"],
        "roles": [
            "attending physician",
            "internist",
            "specialist",
            "radiologist",
            "resident",
        ],
        "temperature": 0.7,
        "forecast_temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 320,
        "rounds": 1,
        "forecast_mode": "separate",
        "forecast_include_own_answer": True,
        "citation_required": False,
        "deliberation": False,
    },
    "cells": ["H0", "H1", "H2", "H3", "H4", "H5", "SYC"],
    "scoring": {
        "eps": 1e-9,
        "variant": "geometric",
        # Models often report exact zeros. A 1% uniform prior keeps logs finite
        # and stops one zero from producing a 20-nat score.
        "forecast_smoothing": 0.01,
    },
    "analysis": {
        "n_bootstrap": 1000,
        "bootstrap_seed": 7,
        "ece_bins": 10,
        "alpha": 0.05,
        "rarity_premium_threshold": 0.05,
        "discrimination_gap_threshold": 0.10,
        "p5_delta_threshold": 0.15,
        "p2_delta_threshold": 0.10,
        "oracle_abs_threshold": 0.05,
    },
}


def _deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".json"}:
            raw = json.load(handle)
        else:
            raw = yaml.safe_load(handle) or {}
    cfg = _deep_merge(DEFAULTS, raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    team = cfg["team"]
    if int(team["agents"]) < 1:
        raise ValueError("team.agents must be >= 1")
    if not team["models"]:
        raise ValueError("team.models is empty")
    if not team["roles"]:
        raise ValueError("team.roles is empty")
    backend = cfg["backend"]["type"]
    if backend not in {"mock", "hf", "ollama"}:
        raise ValueError(f"unknown backend: {backend}")
    if not cfg["cells"]:
        raise ValueError("no cells requested")
    for cell in cfg["cells"]:
        if cell not in {"H0", "H1", "H2", "H3", "H4", "H5", "SYC"}:
            raise ValueError(f"unknown cell: {cell}")


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
