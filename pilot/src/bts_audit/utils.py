"""Small shared helpers. Keep dependencies to numpy/stdlib."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def letter(index: int) -> str:
    if index < 0 or index >= len(LETTERS):
        raise ValueError(f"option index out of range: {index}")
    return LETTERS[index]


def letter_to_index(value: str) -> int | None:
    value = value.strip().upper()
    if len(value) == 1 and value in LETTERS:
        return LETTERS.index(value)
    return None


def stable_seed(*parts: Any) -> int:
    """Deterministic 63-bit seed from arbitrary parts."""
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def normalize(vec: Any, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).ravel()
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= eps:
        return np.full(arr.shape, 1.0 / max(len(arr), 1))
    return arr / total


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if temperature <= 0:
        out = np.zeros_like(logits)
        out[int(np.argmax(logits))] = 1.0
        return out
    z = logits / max(temperature, 1e-6)
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / exp.sum()


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = normalize(p, eps)
    q = normalize(q, eps)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], eps, None))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(out):
        return "n/a"
    return f"{out:.{digits}f}"
