#!/usr/bin/env python3
"""Generate separate 6B/7B GPU configs for all five human-written datasets.

Each (model, dataset) pair gets its own config file and its own output
directory, so `--resume` can never mix records from a different run.
PubMedQA keeps the full H0-H5/SYC ladder; support datasets keep the
pre-registered support subset H1,H4,H5,SYC.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"

MODELS = {
    "6b": "01-ai/Yi-1.5-6B-Chat",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
}
DATASETS = {
    "pubmedqa": ("pubmedqa", "data/cases/pubmedqa_300.jsonl", ["H0", "H1", "H2", "H3", "H4", "H5", "SYC"]),
    "medqa": ("medqa", "data/cases/medqa_300.jsonl", ["H1", "H4", "H5", "SYC"]),
    "medmcqa": ("medmcqa", "data/cases/medmcqa_300.jsonl", ["H1", "H4", "H5", "SYC"]),
    "mmlu_med": ("mmlu_med", "data/cases/mmlu_med_300.jsonl", ["H1", "H4", "H5", "SYC"]),
    "medxpertqa": ("medxpertqa", "data/cases/medxpertqa_300.jsonl", ["H1", "H4", "H5", "SYC"]),
}


def build(model_tag: str, model_id: str, ds_name: str, dataset: str, path: str, cells: list[str]) -> dict:
    return {
        "run_name": f"gpu_{model_tag}_{ds_name}",
        "seed": 13,
        "data": {"dataset": dataset, "path": path, "limit": 300},
        "backend": {
            "type": "hf",
            "hf": {
                "device_map": "auto",
                "dtype": "auto",
                "load_in_4bit": True,
                "max_loaded_models": 1,
                "trust_remote_code": False,
            },
        },
        "team": {
            "agents": 5,
            "models": [model_id],
            "roles": ["attending physician", "internist", "specialist", "radiologist", "resident"],
            "temperature": 0.7,
            "forecast_temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 256,
            "rounds": 2,
        },
        "cells": cells,
        "scoring": {"variant": "geometric", "eps": 1.0e-9},
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
        "output_dir": f"results/gpu_{model_tag}_{ds_name}",
    }


def main() -> None:
    CFG.mkdir(parents=True, exist_ok=True)
    for model_tag, model_id in MODELS.items():
        for ds_name, (dataset, path, cells) in DATASETS.items():
            cfg = build(model_tag, model_id, ds_name, dataset, path, cells)
            out = CFG / f"gpu_{model_tag}_{ds_name}.yaml"
            out.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            print(f"wrote {out.relative_to(ROOT.parent)} -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
