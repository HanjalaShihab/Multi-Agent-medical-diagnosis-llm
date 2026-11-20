#!/usr/bin/env python3
"""Generate basic figures from a BTS audit run directory.

Reads cell_metrics.csv / agent_scores.csv produced by `bts_audit.cli run|analyze`
and saves PNG figures into <results_dir>/figures/.

This helper is intentionally simple (matplotlib only). The pilot's scientific
outputs are the CSV/JSON/markdown artifacts; plots are convenience views.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def plot_cell_bars(df: pd.DataFrame, out: Path, plt) -> None:
    if df.empty or "cell" not in df:
        return
    cells = df["cell"].astype(str).tolist()
    x = range(len(cells))

    def bar(cols, title, ylabel, fname, ylim=None):
        present = [c for c in cols if c in df]
        if not present:
            return
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(cells)), 4))
        width = 0.8 / max(len(present), 1)
        for i, c in enumerate(present):
            vals = _numeric(df[c]).to_numpy(dtype=float)
            offsets = [xi - 0.4 + width * (i + 0.5) for xi in x]
            ax.bar(offsets, vals, width=width, label=c)
        ax.set_xticks(list(x))
        ax.set_xticklabels(cells)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        _save(fig, out / fname)
        plt.close(fig)

    bar(["accuracy", "selection_accuracy_majority", "selection_accuracy_confidence",
         "selection_accuracy_bts", "selection_accuracy_proper"],
        "Accuracy by cell/method", "accuracy", "accuracy.png", ylim=(0, 1))

    bar(["auc_score", "auc_confidence", "auc_oracle"],
        "AUC by score type", "AUC", "auc.png", ylim=(0, 1))

    bar(["delta_auc", "delta_auc_oracle"],
        "ΔAUC (BTS - confidence)", "ΔAUC", "delta_auc.png")

    bar(["agreement_rate", "rho", "majority_accuracy"],
        "Agreement and minority-correct (rho)", "rate", "agreement_rho.png", ylim=(0, 1))

    bar(["rarity_premium", "rarity_premium_oracle"],
        "Within-case rarity premium (P6a, mechanical)", "slope (nats/nat)", "rarity_premium.png")

    bar(["discrimination_gap", "discrimination_gap_elicited"],
        "Same-frequency discrimination gap (P6b diagnostic)", "gap", "discrimination_gap.png")

    bar(["ece_confidence", "forecast_soft_ece"],
        "Calibration (lower is better)", "ECE", "calibration.png", ylim=(0, 1))

    bar(["mean_within_case_score_std", "score_variance"],
        "Score dispersion", "value", "dispersion.png")


def plot_agent_scores(df: pd.DataFrame, out: Path, plt) -> None:
    if df.empty:
        return
    # Per-cell mean agent confidence/score by correctness (if columns exist)
    needed = {"cell", "correct"}
    if not needed.issubset(df.columns):
        return
    work = df.copy()
    work["correct"] = work["correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col, fname, title in [
        ("score", "agent_score_by_correctness.png", "Mean agent score by correctness"),
        ("confidence", "agent_confidence_by_correctness.png", "Mean agent confidence by correctness"),
        ("information", "agent_information_by_correctness.png", "Mean information by correctness"),
    ]:
        if col not in work:
            continue
        agg = work.groupby(["cell", "correct"], dropna=False)[col].mean().unstack(fill_value=float("nan"))
        if agg.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(agg)), 4))
        agg.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="correct", fontsize=8)
        _save(fig, out / fname)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Run/results directory (contains cell_metrics.csv)")
    args = ap.parse_args()
    run_dir = Path(args.results)
    if not run_dir.exists():
        raise SystemExit(f"results dir not found: {run_dir}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it (e.g. `pip install matplotlib`) and retry.\n"
            f"Import error: {e}"
        )

    out = run_dir / "figures"
    metrics_path = run_dir / "cell_metrics.csv"
    if metrics_path.exists():
        cm = pd.read_csv(metrics_path)
        print(f"loaded {metrics_path} ({len(cm)} rows)")
        plot_cell_bars(cm, out, plt)
    else:
        print(f"no {metrics_path}; skipping cell plots")

    agent_path = run_dir / "agent_scores.csv"
    if agent_path.exists():
        ag = pd.read_csv(agent_path)
        print(f"loaded {agent_path} ({len(ag)} rows)")
        plot_agent_scores(ag, out, plt)
    else:
        print(f"no {agent_path}; skipping agent plots")

    figs = sorted(out.glob("*.png")) if out.exists() else []
    if not figs:
        print("No figures generated (insufficient data or missing columns).")
        return 1
    print(f"Done. {len(figs)} figure(s) in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
