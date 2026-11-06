"""Bayesian Truth Serum scoring and decomposition.

The implementation follows ``theory_prop1_prop2.md`` §1.2 and §2.6 exactly:

    bar x(k) = (1/G) sum_j 1{x_j = k}
    bar y(k) = exp((1/G) sum_j log y_j(k))       (geometric, Prelec)
    I_i      = log(bar x(x_i) / bar y(x_i))
    P_i      = - sum_k bar x(k) log(bar x(k) / bar y(k))
    r_i      = I_i + P_i

The arithmetic variant replaces the geometric mean by the arithmetic mean.
The decomposition used in the report is

    I_i = log(bar x / bar y_arith) + log(bar y_arith / bar y_geo)
        = rarity component       + disagreement premium.
"""

from __future__ import annotations

import math
from math import comb, log
from typing import Any, Iterable, Sequence

import numpy as np

from .utils import normalize


def h_g(G: int, u: float) -> float:
    """Exact finite-team rarity premium h_G(u) from Proposition 1(iv)."""
    if not (0 < u <= 1):
        raise ValueError("u must be in (0, 1]")
    n = G - 1
    expectation = 0.0
    for k in range(n + 1):
        expectation += comb(n, k) * (u**k) * ((1 - u) ** (n - k)) * log((1 + k) / G)
    return expectation - log(u)


def h_g_table() -> list[dict[str, float]]:
    rows = []
    for G in (3, 5, 10, 64):
        rows.append({"G": G, **{f"{u:.2f}": h_g(G, u) for u in (0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.95)}})
    return rows


def _valid(answers: Sequence[int | None], n_options: int) -> list[int]:
    return [i for i, a in enumerate(answers) if a is not None and 0 <= int(a) < n_options]


def compute_bts(
    answers: Sequence[int],
    forecasts: Sequence[Sequence[float] | None],
    n_options: int,
    variant: str = "geometric",
    eps: float = 1e-9,
    smoothing: float = 0.0,
) -> dict[str, Any]:
    """Compute BTS scores for one case.

    ``answers`` and ``forecasts`` must be aligned. Invalid answers should be
    removed by the caller. Returns arrays aligned with the inputs.
    """
    if variant not in {"geometric", "arithmetic"}:
        raise ValueError(f"unknown variant: {variant}")
    G = len(answers)
    if G == 0:
        return {}
    m = int(n_options)
    ans = np.asarray([int(a) for a in answers], dtype=int)
    if np.any(ans < 0) or np.any(ans >= m):
        raise ValueError("answer index out of range")

    counts = np.bincount(ans, minlength=m).astype(float)
    bar_x = counts / G

    rows = []
    for f in forecasts:
        if f is None:
            rows.append(np.full(m, 1.0 / m))
        else:
            rows.append(normalize(f, eps))
    F = np.vstack(rows)
    # A forecast of exactly zero for an option that some agent chose would make
    # the log score infinite. Smooth every row so all logarithms are finite.
    if smoothing > 0:
        alpha = float(min(max(smoothing, 0.0), 0.5))
        F = (1.0 - alpha) * F + alpha / m
    F = np.clip(F, eps, None)
    F = F / F.sum(axis=1, keepdims=True)
    bar_y_arith = F.mean(axis=0)

    if variant == "geometric":
        bar_y = np.exp(np.mean(np.log(np.clip(F, eps, 1.0)), axis=0))
    else:
        bar_y = bar_y_arith

    present = bar_x > 0
    I = np.log(bar_x[ans] / bar_y[ans])
    P = -float(np.sum(bar_x[present] * np.log(bar_x[present] / bar_y[present])))
    r = I + P

    rarity = np.log(bar_x[ans] / bar_y_arith[ans])
    disagreement = np.log(bar_y_arith[ans] / bar_y[ans])

    return {
        "G": G,
        "answers": ans,
        "bar_x": bar_x,
        "bar_y": bar_y,
        "bar_y_arith": bar_y_arith,
        "information": I,
        "prediction": float(P),
        "score": r,
        "rarity_component": rarity,
        "disagreement_component": disagreement,
        "surprise": -np.log(bar_y[ans]),
        "log_rarity": -np.log(bar_x[ans]),
        "answer_count": counts[ans].astype(int),
        "answer_freq": bar_x[ans],
        "modal_count": int(counts.max()),
        "is_minority": counts[ans] < counts.max(),
    }


def oracle_forecasts(
    answers: Sequence[int],
    n_options: int,
    eps: float = 1e-9,
    smoothing: float = 0.0,
) -> list[np.ndarray]:
    """Leave-one-out empirical teammate distribution.

    Using ``bar x`` itself would make the BTS information score identically zero
    by construction. The leave-one-out distribution is the honest oracle for
    "what will a random *other* teammate answer?".
    """
    G = len(answers)
    m = int(n_options)
    if G == 0:
        return []
    ans = np.asarray([int(a) for a in answers], dtype=int)
    counts = np.bincount(ans, minlength=m).astype(float)
    if G == 1:
        return [np.full(m, 1.0 / m)]
    out = []
    for a in ans:
        onehot = np.zeros(m)
        onehot[int(a)] = 1.0
        loo = (counts - onehot) / (G - 1)
        if smoothing > 0:
            alpha = float(min(max(smoothing, 0.0), 0.5))
            loo = (1.0 - alpha) * loo + alpha / m
        loo = (loo + eps) / (1.0 + m * eps)
        out.append(loo / loo.sum())
    return out


def score_records(
    records: Sequence[dict[str, Any]],
    n_options: int,
    variant: str = "geometric",
    oracle: bool = False,
    eps: float = 1e-9,
    smoothing: float = 0.0,
) -> list[dict[str, Any]]:
    """Score a list of per-agent records for one case.

    Each record needs ``answer`` and optionally ``forecast``. Records with an
    invalid answer are dropped from the BTS calculation (they still exist in the
    raw log).
    """
    valid_idx = _valid([r.get("answer") for r in records], n_options)
    if not valid_idx:
        return []
    answers = [int(records[i]["answer"]) for i in valid_idx]
    forecasts = [records[i].get("forecast") for i in valid_idx]
    if oracle:
        forecasts = oracle_forecasts(answers, n_options, eps=eps, smoothing=smoothing)
    res = compute_bts(answers, forecasts, n_options, variant=variant, eps=eps, smoothing=smoothing)
    out = []
    for j, i in enumerate(valid_idx):
        out.append(
            {
                "record_index": i,
                "score": float(res["score"][j]),
                "information": float(res["information"][j]),
                "prediction": float(res["prediction"]),
                "rarity_component": float(res["rarity_component"][j]),
                "disagreement_component": float(res["disagreement_component"][j]),
                "surprise": float(res["surprise"][j]),
                "log_rarity": float(res["log_rarity"][j]),
                "answer_count": int(res["answer_count"][j]),
                "answer_freq": float(res["answer_freq"][j]),
                "is_minority": bool(res["is_minority"][j]),
                "modal_count": int(res["modal_count"]),
                "bar_x": res["bar_x"].tolist(),
                "bar_y": res["bar_y"].tolist(),
                "G": int(res["G"]),
            }
        )
    return out


def select_answer(
    records: Sequence[dict[str, Any]],
    n_options: int,
    eps: float = 1e-9,
) -> dict[str, int | None]:
    """Aggregate one case by four rules: majority, confidence, BTS, log score."""
    valid = [r for r in records if r.get("answer") is not None and 0 <= int(r["answer"]) < n_options]
    if not valid:
        return {"majority": None, "confidence": None, "bts": None, "proper": None}

    m = int(n_options)
    counts = np.zeros(m)
    conf = np.zeros(m)
    bts = np.zeros(m)
    bts_n = np.zeros(m)
    logscore = np.zeros(m)
    logscore_n = np.zeros(m)

    for r in valid:
        k = int(r["answer"])
        counts[k] += 1
        c = r.get("confidence")
        conf[k] += float(c) if c is not None else 1.0 / m
        s = r.get("score")
        if s is not None and math.isfinite(float(s)):
            bts[k] += float(s)
            bts_n[k] += 1
        dist = r.get("distribution")
        if dist is None or len(dist) != m:
            dist = [0.0] * m
            dist[k] = 1.0
        for j in range(m):
            logscore[j] += math.log(max(float(dist[j]), eps))
            logscore_n[j] += 1

    def _argmax(values: np.ndarray, tiebreak: np.ndarray | None = None) -> int | None:
        if not np.any(values > -np.inf):
            return None
        best = np.flatnonzero(values == values.max())
        if len(best) == 1:
            return int(best[0])
        if tiebreak is not None:
            sub = tiebreak[best]
            if np.any(sub == sub.max()):
                best = best[sub == sub.max()]
        return int(best[0])

    majority_best = counts.max() if counts.max() > 0 else None
    majority = None
    if majority_best is not None:
        candidates = np.flatnonzero(counts == majority_best)
        if len(candidates) == 1:
            majority = int(candidates[0])
        else:
            majority = int(candidates[np.argmax(conf[candidates])])

    bts_mean = np.divide(bts, bts_n, out=np.full(m, np.nan), where=bts_n > 0)
    if np.all(np.isnan(bts_mean)):
        bts_pick = majority
    else:
        bts_pick = _argmax(np.nan_to_num(bts_mean, nan=-np.inf), tiebreak=counts)

    proper_mean = np.divide(logscore, logscore_n, out=np.full(m, -np.inf), where=logscore_n > 0)
    proper_pick = _argmax(proper_mean, tiebreak=counts)

    return {
        "majority": majority,
        "confidence": _argmax(conf, tiebreak=counts),
        "bts": bts_pick,
        "proper": proper_pick,
    }
