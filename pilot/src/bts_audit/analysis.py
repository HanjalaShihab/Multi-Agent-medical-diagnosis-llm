"""Analysis for P1-P8.

Everything here is descriptive. The report deliberately separates:

* what the mechanism does (AUC, rarity premium, discrimination gap),
* whether the signal survives oracle forecasts (P8),
* whether it is forecast miscalibration rather than private evidence.

No scipy/sklearn dependency.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULTS, get
from .data.loaders import load_cases
from .scoring import oracle_forecasts, score_records, select_answer
from .utils import ensure_dir, fmt, js_divergence, normalize, read_jsonl, write_json

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def auc_score(scores: Sequence[float], labels: Sequence[bool]) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    mask = np.isfinite(scores)
    scores, labels = scores[mask], labels[mask]
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def _delong_components(scores: np.ndarray, labels: np.ndarray):
    pos, neg = scores[labels], scores[~labels]
    if len(pos) < 2 or len(neg) < 2:
        return None
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    return {
        "auc": float(psi.mean()),
        "v10": psi.mean(axis=1),
        "v01": psi.mean(axis=0),
        "m": len(pos),
        "n": len(neg),
    }


def delong_paired(scores_a, scores_b, labels) -> dict[str, float]:
    """DeLong test for two correlated AUCs on the same records."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    y = np.asarray(labels, dtype=bool)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b, y = a[mask], b[mask], y[mask]
    ca = _delong_components(a, y)
    cb = _delong_components(b, y)
    if ca is None or cb is None:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "delta": float("nan"), "z": float("nan"), "p": float("nan")}
    var_a = np.var(ca["v10"], ddof=1) / ca["m"] + np.var(ca["v01"], ddof=1) / ca["n"]
    var_b = np.var(cb["v10"], ddof=1) / cb["m"] + np.var(cb["v01"], ddof=1) / cb["n"]
    cov = np.cov(ca["v10"], cb["v10"], ddof=1)[0, 1] / ca["m"]
    cov += np.cov(ca["v01"], cb["v01"], ddof=1)[0, 1] / ca["n"]
    var_diff = var_a + var_b - 2.0 * cov
    delta = ca["auc"] - cb["auc"]
    if var_diff <= 0 or not math.isfinite(var_diff):
        return {"auc_a": ca["auc"], "auc_b": cb["auc"], "delta": delta, "z": float("nan"), "p": float("nan")}
    z = delta / math.sqrt(var_diff)
    p = 2.0 * (1.0 - norm_cdf(abs(z)))
    return {"auc_a": ca["auc"], "auc_b": cb["auc"], "delta": delta, "z": z, "p": p}


def cluster_bootstrap_delta(
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Case-level bootstrap for AUC(score) - AUC(confidence)."""
    if df.empty or "case_id" not in df:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    groups = [g for _, g in df.groupby("case_id", sort=False)]
    if len(groups) < 3:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in idx], ignore_index=True)
        a = auc_score(sample["score"], sample["correct"])
        b = auc_score(sample["confidence"], sample["correct"])
        if math.isfinite(a) and math.isfinite(b):
            deltas.append(a - b)
    if len(deltas) < 10:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    arr = np.asarray(deltas)
    lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
    p = 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "p": min(1.0, p)}


def ece_binary(prob, outcome, bins: int = 10) -> float:
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = np.clip(p[mask], 0, 1), y[mask]
    if len(p) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if not np.any(sel):
            continue
        total += abs(float(p[sel].mean()) - float(y[sel].mean())) * float(sel.mean())
    return total


def soft_ece(pred, target, bins: int = 10) -> float:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    mask = np.isfinite(p) & np.isfinite(t)
    p, t = np.clip(p[mask], 0, 1), np.clip(t[mask], 0, 1)
    if len(p) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if not np.any(sel):
            continue
        total += abs(float(p[sel].mean()) - float(t[sel].mean())) * float(sel.mean())
    return total


def ols_with_control(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    control_col: str,
) -> dict[str, float]:
    """Coefficient on ``x_col`` controlling for categorical ``control_col``."""
    sub = df[[y_col, x_col, control_col]].dropna()
    if len(sub) < 5 or sub[x_col].nunique() < 2:
        return {"coef": float("nan"), "se": float("nan"), "p": float("nan"), "n": int(len(sub))}
    y = sub[y_col].to_numpy(dtype=float)
    x = sub[x_col].astype(float).to_numpy()
    levels = sorted(sub[control_col].unique().tolist())
    cols = [np.ones(len(sub)), x]
    for level in levels[1:]:
        cols.append((sub[control_col].to_numpy() == level).astype(float))
    X = np.column_stack(cols)
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        dof = max(len(y) - X.shape[1], 1)
        sigma2 = float(resid @ resid) / dof
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = math.sqrt(max(sigma2 * float(xtx_inv[1, 1]), 0.0))
        if se <= 0 or not math.isfinite(se):
            return {"coef": float(coef[1]), "se": float("nan"), "p": float("nan"), "n": int(len(sub))}
        t = float(coef[1] / se)
        p = 2.0 * (1.0 - norm_cdf(abs(t)))
        return {"coef": float(coef[1]), "se": se, "p": p, "n": int(len(sub)), "levels": len(levels)}
    except np.linalg.LinAlgError:
        return {"coef": float("nan"), "se": float("nan"), "p": float("nan"), "n": int(len(sub))}


def within_case_slope(
    df: pd.DataFrame,
    y_col: str,
    x_col: str = "log_rarity",
    case_col: str = "case_id",
) -> dict[str, float]:
    """Mean within-case OLS slope of ``y_col`` on ``x_col``.

    Proposition 1's rarity premium is a within-case object: it compares
    answers *inside* one case. Pooling cases would let between-case variation
    in forecast quality drive the estimate. The standard error is taken across
    cases, so it reflects case-to-case heterogeneity, not agent count.
    """
    if df.empty or y_col not in df or x_col not in df:
        return {"slope": float("nan"), "se": float("nan"), "n_cases": 0}
    slopes: list[float] = []
    for _, gg in df.groupby(case_col, sort=False):
        sub = gg[[x_col, y_col]].dropna()
        if len(sub) < 2:
            continue
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        if np.std(x) < 1e-12:
            continue
        slopes.append(float(np.polyfit(x, y, 1)[0]))
    if not slopes:
        return {"slope": float("nan"), "se": float("nan"), "n_cases": 0}
    arr = np.asarray(slopes, dtype=float)
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    return {"slope": float(arr.mean()), "se": se, "n_cases": int(len(arr))}


def matched_within_case_gap(
    df: pd.DataFrame,
    score_col: str,
    case_col: str = "case_id",
    count_col: str = "answer_count",
    correct_col: str = "correct",
) -> dict[str, float]:
    """Mean same-frequency score gap between correct and wrong answers.

    For every (case, answer_count) cell that contains both a correct and a
    wrong answer, take ``mean(score | correct) - mean(score | wrong)``. This is
    the discrimination the mechanism adds *beyond* knowing how popular an
    answer is. Case-level terms (the prediction score, any constant offset)
    cancel inside the difference.

    Under oracle (leave-one-out) forecasts, every answer with the same count
    gets the same score, so this gap is identically zero. A non-zero oracle
    gap is an implementation bug, not a finding. Under elicited forecasts it
    measures forecast-induced discrimination: negative means the score
    penalises correct answers at fixed frequency (the popularity effect),
    positive means forecasts inject a truth signal (private evidence or
    miscalibration).
    """
    if df.empty or score_col not in df:
        return {"gap": float("nan"), "se": float("nan"), "n_pairs": 0}
    work = df[[case_col, count_col, correct_col, score_col]].dropna().copy()
    if work.empty:
        return {"gap": float("nan"), "se": float("nan"), "n_pairs": 0}
    work[correct_col] = work[correct_col].astype(bool)
    diffs: list[float] = []
    for _, gg in work.groupby([case_col, count_col], sort=False):
        corr = gg[gg[correct_col]][score_col]
        wrong = gg[~gg[correct_col]][score_col]
        if len(corr) == 0 or len(wrong) == 0:
            continue
        diffs.append(float(corr.mean() - wrong.mean()))
    if not diffs:
        return {"gap": float("nan"), "se": float("nan"), "n_pairs": 0}
    arr = np.asarray(diffs, dtype=float)
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    return {"gap": float(arr.mean()), "se": se, "n_pairs": int(len(arr))}


# ---------------------------------------------------------------------------
# Loading and scoring a run
# ---------------------------------------------------------------------------
def load_run(run_dir: str | Path) -> tuple[dict, dict[str, Any], pd.DataFrame]:
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    cfg = meta.get("config", DEFAULTS)
    cases = {c.case_id: c for c in load_cases(run_dir / "cases.jsonl")} if (run_dir / "cases.jsonl").exists() else {}

    answers = pd.DataFrame(read_jsonl(run_dir / "answers.jsonl"))
    if answers.empty:
        raise ValueError(f"no answers in {run_dir}")
    forecasts = pd.DataFrame(read_jsonl(run_dir / "forecasts.jsonl"))
    fmap = {}
    if not forecasts.empty:
        for _, row in forecasts.iterrows():
            fmap[
                (row.get("cell"), row.get("case_id"), int(row.get("round", 0)), int(row.get("agent_id", 0)))
            ] = row.get("forecast")
    answers["forecast"] = [
        fmap.get((row["cell"], row["case_id"], int(row["round"]), int(row["agent_id"])))
        for _, row in answers.iterrows()
    ]
    answers = answers[answers["task"].fillna("answer") == "answer"].copy()
    answers["round"] = answers["round"].astype(int)
    answers["agent_id"] = answers["agent_id"].astype(int)
    # Terminal round only.
    answers = (
        answers.sort_values(["cell", "case_id", "agent_id", "round"])
        .groupby(["cell", "case_id", "agent_id"], as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return cfg, cases, answers


def attach_scores(df: pd.DataFrame, cases: dict[str, Any], cfg: dict, variant: str | None = None) -> pd.DataFrame:
    variant = variant or get(cfg, "scoring.variant", "geometric")
    eps = float(get(cfg, "scoring.eps", 1e-9))
    smoothing = float(get(cfg, "scoring.forecast_smoothing", 0.01))
    bins = int(get(cfg, "analysis.ece_bins", 10))

    row_updates: dict[Any, dict[str, Any]] = {}
    case_rows: list[dict[str, Any]] = []

    for (cell, case_id), g in df.groupby(["cell", "case_id"], sort=False):
        case = cases.get(case_id)
        if case is None:
            continue
        g = g.sort_values("agent_id")
        m = case.n_options
        records = []
        for _, row in g.iterrows():
            dist = row.get("distribution")
            if isinstance(dist, str):
                try:
                    dist = json.loads(dist)
                except json.JSONDecodeError:
                    dist = None
            if dist is not None and len(dist) != m:
                dist = None
            records.append(
                {
                    "answer": int(row["answer"]) if pd.notna(row["answer"]) else None,
                    "confidence": float(row["confidence"]) if pd.notna(row.get("confidence")) else None,
                    "distribution": dist,
                    "forecast": row.get("forecast"),
                }
            )

        scored = score_records(records, m, variant=variant, oracle=False, eps=eps, smoothing=smoothing)
        oracle_scored = score_records(records, m, variant=variant, oracle=True, eps=eps, smoothing=smoothing)
        if not scored:
            continue

        valid_idx = [s["record_index"] for s in scored]
        valid_answers = [int(records[i]["answer"]) for i in valid_idx]
        loo = oracle_forecasts(valid_answers, m, eps=eps, smoothing=smoothing)
        oracle_by_idx = {s["record_index"]: s for s in oracle_scored}

        enriched = []
        for j, s in enumerate(scored):
            i = s["record_index"]
            row_index = g.index[i]
            row = records[i]
            f = row.get("forecast")
            if f is None:
                f_vec = np.full(m, 1.0 / m)
            else:
                f_vec = normalize(f, eps)
            if j < len(loo):
                brier = float(np.mean((f_vec - loo[j]) ** 2))
                js = js_divergence(f_vec, loo[j])
            else:
                brier, js = float("nan"), float("nan")
            oracle_s = oracle_by_idx.get(i, {})
            update = {
                **s,
                "correct": bool(int(records[i]["answer"]) == case.answer),
                "confidence": row["confidence"],
                "forecast_brier": brier,
                "forecast_js": js,
                "forecast": f_vec.tolist(),
                "loo": loo[j].tolist() if j < len(loo) else None,
                "oracle_score": oracle_s.get("score", float("nan")),
                "oracle_information": oracle_s.get("information", float("nan")),
                "gold_answer_freq": float(s["bar_x"][case.answer]),
                "rarity_stratum": case.rarity_stratum,
            }
            row_updates[row_index] = update
            enriched.append({**records[i], **s, "correct": update["correct"]})

        selection = select_answer(enriched, m, eps=eps)
        sel_row = {"cell": cell, "case_id": case_id}
        for method, idx in selection.items():
            sel_row[f"sel_{method}"] = idx
            sel_row[f"sel_{method}_correct"] = bool(idx == case.answer) if idx is not None else False
        case_rows.append(sel_row)

    if not row_updates:
        return df

    for col in next(iter(row_updates.values())).keys():
        df[col] = [row_updates.get(idx, {}).get(col) for idx in df.index]

    sel_df = pd.DataFrame(case_rows)
    if not sel_df.empty:
        df = df.merge(sel_df, on=["cell", "case_id"], how="left")
    df["correct"] = df["correct"].fillna(False).astype(bool)
    if "confidence" in df:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


def ensure_strata(df: pd.DataFrame, cases: dict[str, Any]) -> pd.DataFrame:
    """Fill missing rarity labels from dataset prevalence terciles."""
    if "rarity_stratum" not in df:
        df["rarity_stratum"] = None
    missing = df["rarity_stratum"].isna() | (df["rarity_stratum"].astype(str) == "None")
    if not missing.any():
        return df

    rows = []
    for cid in df.loc[missing, "case_id"].unique():
        case = cases.get(cid)
        if case is not None and case.prevalence is not None and math.isfinite(float(case.prevalence)):
            rows.append((cid, float(case.prevalence)))
    if len(rows) < 3:
        df.loc[missing, "rarity_stratum"] = "unknown"
        return df
    values = np.array([v for _, v in rows])
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    mapping = {}
    for cid, value in rows:
        mapping[cid] = "common" if value >= q2 else ("rare" if value <= q1 else "mid")
    df.loc[missing, "rarity_stratum"] = df.loc[missing, "case_id"].map(mapping).fillna("unknown")
    return df


# ---------------------------------------------------------------------------
# Per-cell metrics
# ---------------------------------------------------------------------------
def cell_metrics(g: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    alpha = float(get(cfg, "analysis.alpha", 0.05))
    n_boot = int(get(cfg, "analysis.n_bootstrap", 1000))
    boot_seed = int(get(cfg, "analysis.bootstrap_seed", 7))
    bins = int(get(cfg, "analysis.ece_bins", 10))
    if g.empty:
        return {}

    out: dict[str, Any] = {}
    out["n_records"] = int(len(g))
    out["n_cases"] = int(g["case_id"].nunique())
    out["accuracy"] = float(g["correct"].mean())
    out["mean_confidence"] = float(g["confidence"].mean()) if g["confidence"].notna().any() else float("nan")
    out["agreement_rate"] = float(
        (g.groupby("case_id")["answer"].nunique() == 1).mean()
    )
    out["mean_within_case_score_std"] = float(
        g.groupby("case_id")["score"].std(ddof=0).fillna(0.0).mean()
    )
    out["score_variance"] = float(g["score"].var(ddof=0))
    out["score_mean"] = float(g["score"].mean())

    out["auc_score"] = auc_score(g["score"], g["correct"])
    out["auc_confidence"] = auc_score(g["confidence"], g["correct"])
    out["auc_information"] = auc_score(g["information"], g["correct"])
    out["auc_rarity"] = auc_score(g["rarity_component"], g["correct"])
    out["auc_disagreement"] = auc_score(g["disagreement_component"], g["correct"])
    out["auc_prediction"] = auc_score(g["prediction"], g["correct"])

    delong = delong_paired(g["score"], g["confidence"], g["correct"])
    # DeLong is undefined when a cell has fewer than two records in either
    # outcome class, but the point estimate is still well defined. Keep dAUC
    # as the plain difference whenever both AUCs exist; DeLong supplies only
    # the p-value.
    if math.isfinite(out["auc_score"]) and math.isfinite(out["auc_confidence"]):
        out["delta_auc"] = out["auc_score"] - out["auc_confidence"]
    else:
        out["delta_auc"] = float("nan")
    out["delong_z"] = delong["z"]
    out["delong_p"] = delong["p"]
    boot = cluster_bootstrap_delta(g, n_boot=n_boot, seed=boot_seed, alpha=alpha)
    out["delta_auc_boot_mean"] = boot["mean"]
    out["delta_auc_ci_lo"] = boot["lo"]
    out["delta_auc_ci_hi"] = boot["hi"]
    out["delta_auc_boot_p"] = boot["p"]

    out["ece_confidence"] = ece_binary(g["confidence"], g["correct"], bins=bins)
    out["forecast_brier"] = float(g["forecast_brier"].mean())
    out["forecast_js"] = float(g["forecast_js"].mean())
    preds, targets = [], []
    for f, loo in zip(g["forecast"], g["loo"]):
        if f is not None and loo is not None:
            preds.extend(list(f))
            targets.extend(list(loo))
    out["forecast_soft_ece"] = soft_ece(preds, targets, bins=bins) if preds else float("nan")
    out["forecast_missing_rate"] = float(g["forecast"].isna().mean())

    minority = g[g["is_minority"] == True]  # noqa: E712
    majority = g[g["is_minority"] == False]  # noqa: E712
    out["minority_n"] = int(len(minority))
    out["rho"] = float(minority["correct"].mean()) if len(minority) else float("nan")
    out["majority_accuracy"] = float(majority["correct"].mean()) if len(majority) else float("nan")

    for method in ("majority", "confidence", "bts", "proper"):
        col = f"sel_{method}_correct"
        out[f"selection_accuracy_{method}"] = float(g[col].mean()) if col in g else float("nan")

    out["auc_oracle"] = auc_score(g["oracle_score"], g["correct"])
    out["delta_auc_oracle"] = out["auc_oracle"] - out["auc_confidence"]
    if math.isfinite(out["delta_auc"]) and abs(out["delta_auc"]) > 1e-9:
        out["oracle_reduction"] = 1.0 - abs(out["delta_auc_oracle"]) / abs(out["delta_auc"])
    else:
        out["oracle_reduction"] = float("nan")

    # P6a: rarity premium. The theory's object is the *elicited* information
    # score: within a case, does I_i rise with -log bar_x? The oracle version
    # is a control. With leave-one-out forecasts bar_y is close to bar_x, so
    # the oracle slope is near zero; any residual is the forecast-smoothing
    # floor, not evidence of a premium. Comparing correct-rare vs
    # correct-common *cases* would confound the premium with case-level
    # forecast error and with the fact that an accurate team's correct answer
    # is usually the popular one.
    prem = within_case_slope(g, "information")
    out["rarity_premium"] = prem["slope"]
    out["rarity_premium_se"] = prem["se"]
    out["rarity_premium_n_cases"] = prem["n_cases"]
    prem_oracle = within_case_slope(g, "oracle_information")
    out["rarity_premium_oracle"] = prem_oracle["slope"]
    out["rarity_premium_oracle_n_cases"] = prem_oracle["n_cases"]

    correct = g[g["correct"]]
    rare = correct[correct["rarity_stratum"] == "rare"]["score"]
    common = correct[correct["rarity_stratum"] == "common"]["score"]
    out["rare_disease_score_gap"] = (
        float(rare.mean() - common.mean()) if len(rare) >= 2 and len(common) >= 2 else float("nan")
    )
    out["n_correct_rare"] = int(len(rare))
    out["n_correct_common"] = int(len(common))

    # P6b: discrimination gap at fixed answer frequency. The main estimate uses
    # oracle forecasts: the theorem says the oracle score is a function of the
    # answer count only, so a non-zero gap is an implementation bug. The
    # elicited gap is the calibration diagnostic; it is where forecast error
    # enters, and it can be large even when the mechanism itself is clean.
    gap_oracle = matched_within_case_gap(g, "oracle_score")
    out["discrimination_gap"] = gap_oracle["gap"]
    out["discrimination_gap_se"] = gap_oracle["se"]
    out["discrimination_gap_n"] = gap_oracle["n_pairs"]
    gap_elicited = matched_within_case_gap(g, "score")
    out["discrimination_gap_elicited"] = gap_elicited["gap"]
    out["discrimination_gap_elicited_se"] = gap_elicited["se"]
    out["discrimination_gap_elicited_n"] = gap_elicited["n_pairs"]

    rare_cases = g[g["rarity_stratum"] == "rare"]
    out["gap_rare_cases"] = matched_within_case_gap(rare_cases, "score")["gap"]

    out["error_rate"] = float(g["error"].notna().mean()) if "error" in g else float("nan")
    out["mean_latency_s"] = float(g["latency_s"].mean()) if "latency_s" in g else float("nan")
    return out


# ---------------------------------------------------------------------------
# P7 sycophancy comparison
# ---------------------------------------------------------------------------
def p7_metrics(df: pd.DataFrame) -> dict[str, Any]:
    base = df[df["cell"] == "H1"].copy()
    press = df[df["cell"] == "SYC"].copy()
    if base.empty or press.empty:
        return {"status": "not_run"}
    merged = base.merge(
        press,
        on=["case_id", "agent_id"],
        suffixes=("_base", "_press"),
        how="inner",
    )
    if merged.empty:
        return {"status": "no_matched_records"}
    agent_flip = float((merged["answer_base"] != merged["answer_press"]).mean())

    out = {"status": "ok", "agent_flip_rate": agent_flip, "n_matched": int(len(merged))}
    for method in ("majority", "confidence", "bts", "proper"):
        col = f"sel_{method}_base"
        colp = f"sel_{method}_press"
        if col in merged and colp in merged:
            out[f"selection_flip_{method}"] = float((merged[col] != merged[colp]).mean())
            out[f"selection_accuracy_{method}_press"] = float(
                merged[f"sel_{method}_correct_press"].mean()
            )
            out[f"selection_accuracy_{method}_base"] = float(
                merged[f"sel_{method}_correct_base"].mean()
            )
    out["bts_minus_majority_press"] = (
        out.get("selection_accuracy_bts_press", float("nan"))
        - out.get("selection_accuracy_majority_press", float("nan"))
    )
    out["bts_flip_minus_majority"] = (
        out.get("selection_flip_bts", float("nan"))
        - out.get("selection_flip_majority", float("nan"))
    )
    # P7 is directional: under pressure the minority is the honest answer, so
    # BTS should be at least as accurate as majority and no less stable.
    acc_gain = out["bts_minus_majority_press"]
    flip_delta = out["bts_flip_minus_majority"]
    if not math.isfinite(acc_gain) or not math.isfinite(flip_delta):
        out["status"] = "inestimable"
    elif acc_gain > 0 and flip_delta <= 0.05:
        out["status"] = "consistent"
    elif acc_gain > 0:
        out["status"] = "partial"
    else:
        out["status"] = "falsified"
    return out


# ---------------------------------------------------------------------------
# Prediction scorecard and go/no-go
# ---------------------------------------------------------------------------
def _num(m: dict, key: str, default: float = float("nan")) -> float:
    value = m.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def prediction_scorecard(metrics: dict[str, dict], cfg: dict, backend: str) -> dict[str, Any]:
    p2t = float(get(cfg, "analysis.p2_delta_threshold", 0.10))
    p5t = float(get(cfg, "analysis.p5_delta_threshold", 0.15))
    prem_t = float(get(cfg, "analysis.rarity_premium_threshold", 0.05))
    gap_t = float(get(cfg, "analysis.discrimination_gap_threshold", 0.10))
    oracle_t = float(get(cfg, "analysis.oracle_abs_threshold", 0.05))
    alpha = float(get(cfg, "analysis.alpha", 0.05))

    def cell(name: str) -> dict:
        return metrics.get(name, {})

    def signal_test(name: str, threshold: float) -> dict[str, Any]:
        m = cell(name)
        if not m:
            return {"status": "not_run", "delta": float("nan"), "p": float("nan"), "direction": "n/a"}
        delta = _num(m, "delta_auc")
        p = _num(m, "delta_auc_boot_p")
        if not math.isfinite(p):
            p = _num(m, "delong_p")
        if not math.isfinite(delta):
            return {"status": "undefined", "delta": delta, "p": p, "direction": "n/a"}
        direction = "anti_signal" if delta < -threshold else ("null" if delta <= threshold else "positive")
        if delta > threshold and math.isfinite(p) and p < alpha:
            return {"status": "falsified", "delta": delta, "p": p, "direction": direction}
        if delta > threshold:
            return {"status": "weak_evidence", "delta": delta, "p": p, "direction": direction}
        return {"status": "consistent", "delta": delta, "p": p, "direction": direction}

    out: dict[str, Any] = {}

    m0 = cell("H0")
    if m0:
        collapsed = _num(m0, "agreement_rate") >= 0.95 and (
            _num(m0, "score_variance") < 1e-8 or not math.isfinite(_num(m0, "auc_score"))
        )
        out["P1"] = {
            "status": "consistent" if collapsed else "falsified",
            "agreement_rate": _num(m0, "agreement_rate"),
            "score_variance": _num(m0, "score_variance"),
            "auc": _num(m0, "auc_score"),
        }
    else:
        out["P1"] = {"status": "not_run"}

    out["P2"] = signal_test("H1", p2t)
    out["P3"] = signal_test("H2", p2t)
    out["P4"] = signal_test("H3", 0.15)

    m3, m4 = cell("H3"), cell("H4")
    if m4:
        delta4 = _num(m4, "delta_auc")
        rho4 = _num(m4, "rho")
        rho3 = _num(m3, "rho")
        ok = delta4 >= p5t and math.isfinite(rho4) and (not math.isfinite(rho3) or rho4 > rho3)
        out["P5"] = {
            "status": "consistent" if ok else ("partial" if delta4 >= p5t else "falsified"),
            "delta": delta4,
            "rho_H4": rho4,
            "rho_H3": rho3,
        }
    else:
        out["P5"] = {"status": "not_run"}

    premiums = {
        name: _num(m, "rarity_premium")
        for name, m in metrics.items()
        if _num(m, "score_variance") > 1e-8
    }
    oracle_premiums = {
        name: _num(m, "rarity_premium_oracle")
        for name, m in metrics.items()
        if _num(m, "score_variance") > 1e-8
    }
    finite = {k: v for k, v in premiums.items() if math.isfinite(v)}
    if finite:
        mean_premium = float(np.mean(list(finite.values())))
        out["P6a"] = {
            "status": "consistent" if mean_premium >= prem_t else "falsified",
            "mean_premium": mean_premium,
            "by_cell": premiums,
            "oracle_by_cell": oracle_premiums,
            "interpretation": "mechanical within-case rarity slope of the BTS information score",
            "disease_prevalence_claims": "not_testable_on_this_suite",
        }
    else:
        out["P6a"] = {"status": "not_run", "disease_prevalence_claims": "not_testable_on_this_suite"}

    gaps = {name: _num(m, "discrimination_gap") for name, m in metrics.items()}
    elicited_gaps = {name: _num(m, "discrimination_gap_elicited") for name, m in metrics.items()}
    n_pairs = {name: int(_num(m, "discrimination_gap_n", 0)) for name, m in metrics.items()}
    control_cells = ["H0", "H1", "H2", "H3"]
    control_gaps = [gaps[c] for c in control_cells if c in gaps and math.isfinite(gaps[c])]
    if control_gaps:
        max_abs = float(max(abs(x) for x in control_gaps))
        out["P6b"] = {
            "status": "consistent" if max_abs <= gap_t else "gap_detected",
            "max_abs_gap_H0_H3": max_abs,
            "by_cell": gaps,
            "elicited_by_cell": elicited_gaps,
            "n_matched_by_cell": n_pairs,
            "interpretation": "mechanical implementation check: oracle score is a function of the answer count",
            "disease_prevalence_claims": "not_testable_on_this_suite",
        }
    else:
        out["P6b"] = {
            "status": "inestimable",
            "by_cell": gaps,
            "elicited_by_cell": elicited_gaps,
            "n_matched_by_cell": n_pairs,
            "interpretation": "mechanical implementation check: oracle score is a function of the answer count",
            "disease_prevalence_claims": "not_testable_on_this_suite",
        }

    p7 = p7_metrics_proxy(metrics, cfg)
    out["P7"] = p7

    h123 = [cell(c) for c in ("H1", "H2", "H3") if cell(c)]
    if h123:
        oracle_deltas = [_num(m, "delta_auc_oracle") for m in h123]
        reductions = [_num(m, "oracle_reduction") for m in h123]
        controlled = all(
            (math.isfinite(d) and d <= oracle_t)
            or (math.isfinite(d) and d > oracle_t and math.isfinite(r) and r >= 0.6)
            for d, r in zip(oracle_deltas, reductions)
        )
        m4_oracle = _num(m4, "delta_auc_oracle") if m4 else float("nan")
        h4_survives = math.isfinite(m4_oracle) and m4_oracle >= 0.10
        if controlled and h4_survives:
            status = "consistent"
        elif controlled:
            status = "partial"
        else:
            status = "falsified"
        out["P8"] = {
            "status": status,
            "oracle_delta_H1_H3": oracle_deltas,
            "oracle_reduction_H1_H3": reductions,
            "oracle_delta_H4": m4_oracle,
        }
    else:
        out["P8"] = {"status": "not_run"}

    if backend == "mock":
        verdict = "SMOKE ONLY - mock backend, not a scientific result"
    else:
        required = [out.get(k, {}).get("status") for k in ("P1", "P2", "P6a", "P8")]
        go = all(
            s in {"consistent", "pass"} for s in required
        ) and out.get("P5", {}).get("status") == "consistent"
        verdict = "GO" if go else "NO-GO / revise"
    out["verdict"] = verdict
    return out


def p7_metrics_proxy(metrics: dict[str, dict], cfg: dict) -> dict[str, Any]:
    """Placeholder replaced by the real P7 comparison in ``compute_analysis``."""
    return {"status": "see_p7_section"}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _status_word(status: str) -> str:
    return {
        "consistent": "PASS",
        "falsified": "FAIL",
        "partial": "PARTIAL",
        "weak_evidence": "WEAK",
        "not_run": "not run",
        "undefined": "undefined",
        "inestimable": "n/a",
        "gap_detected": "GAP",
    }.get(status, status)


def write_report(
    run_dir: Path,
    cfg: dict,
    metrics: dict[str, dict],
    scorecard: dict[str, Any],
    p7: dict[str, Any],
    df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    run_name = get(cfg, "run_name", "bts_pilot")
    backend = get(cfg, "backend.type", "mock")
    lines.append(f"# BTS audit report - {run_name}")
    lines.append("")
    lines.append(f"Backend: **{backend}** | Cases: **{df['case_id'].nunique()}** | Cells: **{', '.join(metrics.keys())}**")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    verdict = scorecard.get("verdict", "n/a")
    if str(verdict).startswith("SMOKE"):
        lines.append(f"> **{verdict}.** The pipeline ran, but these numbers are simulated. Use a real backend for a scientific verdict.")
    elif verdict == "GO":
        lines.append("> **GO.** The pilot supports the theory. Scale to 300 cases and 3 seeds before writing the paper.")
    else:
        lines.append("> **NO-GO / revise.** At least one gate failed. Read the weak row before spending more compute.")
    lines.append("")
    lines.append("## Prediction scorecard")
    lines.append("")
    lines.append("| # | Result | Key numbers |")
    lines.append("|---|---|---|")
    for key in ("P1", "P2", "P3", "P4", "P5", "P6a", "P6b", "P7", "P8"):
        item = scorecard.get(key, {})
        status = item.get("status", "not_run")
        lines.append(f"| {key} | **{_status_word(status)}** | {_key_numbers(key, item)} |")
    lines.append("")
    lines.append("## Per-cell metrics")
    lines.append("")
    lines.append(
        "| Cell | n | Acc | Agree | AUC score | AUC conf | dAUC | boot p | ECE conf | ECE forecast | rho | Rare prem (elic) | Oracle gap | Elic gap | Oracle dAUC |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def _gap_cell(value: Any, n_pairs: Any) -> str:
        text = fmt(value)
        try:
            n = int(n_pairs)
        except (TypeError, ValueError):
            n = 0
        return f"{text} ({n})" if n else f"{text} (0)"

    for name, m in metrics.items():
        lines.append(
            "| {cell} | {n} | {acc} | {agree} | {auc} | {aucc} | {d} | {p} | {ece} | {fece} | {rho} | {prem} | {gapo} | {gape} | {od} |".format(
                cell=name,
                n=m.get("n_records", 0),
                acc=fmt(m.get("accuracy")),
                agree=fmt(m.get("agreement_rate")),
                auc=fmt(m.get("auc_score")),
                aucc=fmt(m.get("auc_confidence")),
                d=fmt(m.get("delta_auc")),
                p=fmt(m.get("delta_auc_boot_p")),
                ece=fmt(m.get("ece_confidence")),
                fece=fmt(m.get("forecast_soft_ece")),
                rho=fmt(m.get("rho")),
                prem=fmt(m.get("rarity_premium")),
                gapo=_gap_cell(m.get("discrimination_gap"), m.get("discrimination_gap_n")),
                gape=_gap_cell(m.get("discrimination_gap_elicited"), m.get("discrimination_gap_elicited_n")),
                od=fmt(m.get("delta_auc_oracle")),
            )
        )
    lines.append("")
    lines.append("Gap cells show `estimate (matched case-frequency groups)`; `n/a (0)` means no case had both a correct and a wrong answer at the same frequency.")
    lines.append("")
    lines.append("## Sycophancy control (P7)")
    lines.append("")
    if p7.get("status") in {"consistent", "partial", "falsified"}:
        lines.append(f"- Status: **{_status_word(p7['status'])}**")
        lines.append(f"- Agent answer flip rate under pressure: **{fmt(p7.get('agent_flip_rate'))}**")
        lines.append(f"- BTS-selected answer flip rate: **{fmt(p7.get('selection_flip_bts'))}**")
        lines.append(f"- Majority-selected answer flip rate: **{fmt(p7.get('selection_flip_majority'))}**")
        lines.append(f"- BTS selection accuracy under pressure: **{fmt(p7.get('selection_accuracy_bts_press'))}**")
        lines.append(f"- Majority selection accuracy under pressure: **{fmt(p7.get('selection_accuracy_majority_press'))}**")
    else:
        lines.append(f"- Not run ({p7.get('status', 'no SYC cell')}). Add `SYC` and `H1` to the cells list.")
    lines.append("")
    lines.append("## Data limitations")
    lines.append("")
    dataset = str(get(cfg, "data.dataset", "unknown")).lower()
    lines.append(f"- Dataset: **{dataset}**. All headline datasets are human-written public benchmarks; none is raw patient-record data.")
    lines.append("- Real QA benchmarks carry **no true disease-prevalence prior**. `rarity_stratum` and `prevalence` are therefore `None`; they are never inferred from answer-option frequency.")
    lines.append("- **P6a** is the within-case rarity slope of the BTS information score and **P6b** is the fixed-frequency oracle gap. Both are mechanical properties of the mechanism. Claims about *disease* rarity or prevalence are **not testable on this suite**.")
    if dataset == "pubmedqa":
        lines.append("- PubMedQA supplies genuine, human-written abstract sections (`background`, `case`, `methods`, `results`, `conclusions`, `other`). H4/H5 are a real evidence partition here. Some abstracts have fewer than five sections, so agents may share a section.")
    else:
        lines.append("- This dataset is text-only. Its five evidence slices are contiguous chunks of the question stem (`part_1`..`part_5`), so H4/H5 are a **text-chunk ablation**, not a genuine evidence partition. Only PubMedQA supports the strong H4 claim.")
    lines.append("- MedMCQA uses its validation split because the official test split withholds labels (`cop == -1`).")
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- **dAUC = AUC(BTS) - AUC(confidence).** Near zero or negative means BTS adds no positive ranking signal beyond confidence. A negative value is the expected popularity effect when the team is accurate: the correct answer is common, so the rarity statistic penalises it.")
    lines.append("- **Rare premium (elic)** is the within-case slope of the elicited BTS information score on answer rarity ($-\\log \\bar x$). It is a mechanical property of the score, not evidence that rare *diseases* are better diagnosed. The oracle premium (not shown) is the control: it should be near zero.")
    lines.append("- **Oracle gap** is the same-frequency score gap using leave-one-out empirical forecasts. Proposition 1 makes it identically zero: at a fixed answer count the mechanism cannot tell correct from wrong. A non-zero value means a bug.")
    lines.append("- **Elic gap** is the same-frequency gap using the agents' actual forecasts. Negative means the score penalises correct answers at fixed frequency (the popularity effect). Positive means forecasts inject a truth signal: either private evidence or miscalibration. Compare with P8 before believing the positive case.")
    lines.append("- **Oracle dAUC** replaces elicited forecasts with the leave-one-out empirical answer distribution. If dAUC collapses to zero in H1-H3, the apparent signal was forecast miscalibration, not private evidence.")
    lines.append("- **rho** is Pr(correct | answer is a minority). BTS helps only when rho is above the majority's accuracy.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `agent_scores.csv` - one row per agent per case with every score component")
    lines.append("- `cell_metrics.csv` - the table above")
    lines.append("- `metrics.json` - all numbers")
    lines.append("- `predictions.json` - the P1-P8 scorecard")
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _key_numbers(key: str, item: dict) -> str:
    if key == "P1":
        return f"agreement={fmt(item.get('agreement_rate'))}, score var={fmt(item.get('score_variance'))}"
    if key in {"P2", "P3", "P4"}:
        suffix = " (anti-signal)" if item.get("direction") == "anti_signal" else ""
        return f"dAUC={fmt(item.get('delta'))}{suffix}, p={fmt(item.get('p'))}"
    if key == "P5":
        return f"dAUC={fmt(item.get('delta'))}, rho H4={fmt(item.get('rho_H4'))}, rho H3={fmt(item.get('rho_H3'))}"
    if key == "P6a":
        return f"elicited within-case rarity slope={fmt(item.get('mean_premium'))} nats/nat"
    if key == "P6b":
        oracle = fmt(item.get("max_abs_gap_H0_H3"))
        elicited = [
            float(v)
            for v in (item.get("elicited_by_cell") or {}).values()
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        ]
        if elicited:
            lo, hi = min(elicited), max(elicited)
            diag = f"elicited range=[{fmt(lo)}, {fmt(hi)}]"
        else:
            diag = "elicited gap inestimable"
        return f"oracle max |gap| H0-H3={oracle} nats (mechanical); {diag}"
    if key == "P7":
        return (
            f"flip={fmt(item.get('agent_flip_rate'))}, "
            f"BTS acc={fmt(item.get('selection_accuracy_bts_press'))} vs "
            f"majority={fmt(item.get('selection_accuracy_majority_press'))}"
        )
    if key == "P8":
        return f"oracle dAUC H1-H3={[round(x, 3) if isinstance(x, float) and math.isfinite(x) else 'n/a' for x in item.get('oracle_delta_H1_H3', [])]}"
    return ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def compute_analysis(
    run_dir: str | Path,
    cfg: dict | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    meta_cfg, cases, df = load_run(run_dir)
    cfg = cfg or meta_cfg
    df = attach_scores(df, cases, cfg, variant=variant)
    df = ensure_strata(df, cases)
    df["cell"] = df["cell"].astype(str)

    metrics: dict[str, dict] = {}
    for cell, g in df.groupby("cell", sort=False):
        try:
            metrics[str(cell)] = cell_metrics(g, cfg)
        except Exception as exc:  # one bad cell should not kill the report
            metrics[str(cell)] = {"error": f"{type(exc).__name__}: {exc}"}

    scorecard = prediction_scorecard(metrics, cfg, backend=get(cfg, "backend.type", "mock"))
    p7 = p7_metrics(df)
    scorecard["P7"] = p7

    ensure_dir(run_dir)
    df.to_csv(run_dir / "agent_scores.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(run_dir / "cell_metrics.csv")
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "predictions.json", scorecard)
    write_report(run_dir, cfg, metrics, scorecard, p7, df)
    return {
        "metrics": metrics,
        "predictions": scorecard,
        "p7": p7,
        "report": str(run_dir / "report.md"),
    }
