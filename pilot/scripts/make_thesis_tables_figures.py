#!/usr/bin/env python3
"""Build Chapter 4 tables and figures from the completed PubMedQA GPU runs.

Reads only the shareable summaries in pilot/results_summary/. No numbers are
typed by hand. Writes markdown table fragments to pilot/thesis/table_data/ and
three publication-style PNG figures to pilot/thesis/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results_summary"
OUT = ROOT / "thesis"
TABLES = OUT / "table_data"
FIGS = OUT / "figures"
RUNS = {
    "Qwen2.5-1.5B": "gpu_pubmedqa_2b_full",
    "Qwen2.5-3B": "gpu_pubmedqa_3b_full",
}
CELLS = ["H0", "H1", "H2", "H3", "H4", "H5", "SYC"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 200,
})


def fmt(value, nd=3):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if pd.isna(v) else f"{v:.{nd}f}"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    frames = {}
    for label, run in RUNS.items():
        df = pd.read_csv(SRC / run / "cell_metrics.csv", index_col=0)
        df.index.name = "cell"
        frames[label] = df.reindex(CELLS)

    # Table 4.1: one row per model and cell, values taken from cell_metrics.csv.
    rows = []
    for label, df in frames.items():
        for cell in CELLS:
            r = df.loc[cell]
            rows.append(
                "| {model} | {cell} | 300 | {acc} | {maj} | {conf} | {bts} | {prop} | {auc_c} | {auc_b} | {d} | {agr} |".format(
                    model=label,
                    cell=cell,
                    acc=fmt(r["accuracy"]),
                    maj=fmt(r["selection_accuracy_majority"]),
                    conf=fmt(r["selection_accuracy_confidence"]),
                    bts=fmt(r["selection_accuracy_bts"]),
                    prop=fmt(r["selection_accuracy_proper"]),
                    auc_c=fmt(r["auc_confidence"]),
                    auc_b=fmt(r["auc_score"]),
                    d=fmt(r["delta_auc"]),
                    agr=fmt(r["agreement_rate"]),
                )
            )
    (TABLES / "table_4_1_rows.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Table 4.3: answer-parsing errors from agent_scores.csv (answer task records).
    errors = {}
    for label, run in RUNS.items():
        df = pd.read_csv(SRC / run / "agent_scores.csv", low_memory=False)
        err = df[df["error"].notna()]
        errors[label] = pd.crosstab(err["cell"], err["error"]).reindex(CELLS).fillna(0).astype(int)
    erows = []
    for cell in CELLS:
        for label in RUNS:
            t = errors[label].loc[cell]
            not_mapped = int(t.get("answer_not_mapped", 0))
            no_json = int(t.get("no_json", 0))
            erows.append(
                f"| {label} | {cell} | {not_mapped + no_json} | {not_mapped} | {no_json} |"
            )
    (TABLES / "table_4_3_rows.md").write_text("\n".join(erows) + "\n", encoding="utf-8")

    # Figure 4.1: selection accuracy by rule.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    rules = [
        ("selection_accuracy_majority", "Majority"),
        ("selection_accuracy_confidence", "Confidence"),
        ("selection_accuracy_bts", "BTS"),
        ("selection_accuracy_proper", "Proper score"),
    ]
    x = range(len(CELLS))
    width = 0.2
    for ax, (label, df) in zip(axes, frames.items()):
        for i, (col, name) in enumerate(rules):
            ax.bar([k - 1.5 * width + i * width for k in x], df[col], width, label=name)
        ax.set_title(label)
        ax.set_xticks(list(x))
        ax.set_xticklabels(CELLS)
        ax.set_xlabel("Team condition")
        ax.set_ylim(0, 0.7)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Selection accuracy")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "figure_4_1_selection_accuracy.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4.2: rarity premium (elicited and oracle) by cell.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = range(len(CELLS))
    width = 0.35
    for ax, (label, df) in zip(axes, frames.items()):
        ax.bar([k - width / 2 for k in x], df["rarity_premium"], width, label="Elicited")
        oracle = df["rarity_premium_oracle"].fillna(0)
        ax.bar([k + width / 2 for k in x], oracle, width, label="Oracle forecasts")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(label)
        ax.set_xticks(list(x))
        ax.set_xticklabels(CELLS)
        ax.set_xlabel("Team condition")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rarity premium (nats)")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "figure_4_2_rarity_premium.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4.3: oracle-forecast ablation on delta AUC.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (label, df) in zip(axes, frames.items()):
        ax.plot(CELLS, df["delta_auc"], "o-", label="Elicited forecasts")
        ax.plot(CELLS, df["delta_auc_oracle"], "s--", label="Oracle forecasts")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel("Team condition")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Delta AUC (BTS - confidence)")
    fig.tight_layout()
    fig.savefig(FIGS / "figure_4_3_oracle_ablation.png", bbox_inches="tight")
    plt.close(fig)

    payload = {
        model: {
            cell: {
                key: fmt(df.loc[cell, key])
                for key in (
                    "accuracy",
                    "selection_accuracy_majority",
                    "selection_accuracy_confidence",
                    "selection_accuracy_bts",
                    "selection_accuracy_proper",
                    "auc_confidence",
                    "auc_score",
                    "delta_auc",
                    "delta_auc_ci_lo",
                    "delta_auc_ci_hi",
                    "delta_auc_boot_p",
                    "rarity_premium",
                    "rarity_premium_oracle",
                    "delta_auc_oracle",
                    "agreement_rate",
                    "ece_confidence",
                    "discrimination_gap_elicited",
                )
            }
            for cell in CELLS
        }
        for model, df in frames.items()
    }
    (TABLES / "chapter4_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("Wrote table fragments, chapter4_metrics.json and three figures.")


if __name__ == "__main__":
    main()