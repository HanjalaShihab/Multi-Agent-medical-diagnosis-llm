"""Prefetch HF model files once, while online, so the audit can run offline.

Why: the 7B run stalled on repeated HTTP 429 responses while the tokenizer
looked up `additional_chat_templates`, and a fresh 6B model is not in the
local cache at all. Downloading the full snapshot first removes both problems.

Usage (GPU PC, venv active, internet available):
    python scripts/prefetch_gpu_models.py 01-ai/Yi-1.5-6B-Chat Qwen/Qwen2.5-7B-Instruct

After this succeeds, `scripts/run_gpu_quick_6b7b.ps1` sets
HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 and generation does not touch HF Hub.
"""
from __future__ import annotations

import sys

from huggingface_hub import snapshot_download


def main() -> int:
    models = sys.argv[1:]
    if not models:
        print(
            "Usage: python scripts/prefetch_gpu_models.py <model-id> [model-id ...]"
        )
        return 2

    for model in models:
        print(f"[prefetch] {model}", flush=True)
        path = snapshot_download(repo_id=model)
        print(f"[prefetch] cached at {path}", flush=True)

    print("[prefetch] done - the audit can now run with HF_HUB_OFFLINE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
