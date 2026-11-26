#!/usr/bin/env python3
"""Copy only shareable run artefacts out of the git-ignored results/ folder.

Raw `answers.jsonl` / `forecasts.jsonl` are large and stay local. This script
copies report tables, figures and config metadata into `results_summary/`, which
is tracked by git, and writes a combined `summary_all.csv` so model/dataset
accuracy can be compared after `git pull`.

Run this on the GPU PC after the runs finish, commit `results_summary/`, push;
then `git pull` on the laptop.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

TEXT_FILES = [
    "report.md",
    "cell_metrics.csv",
    "metrics.json",
    "predictions.json",
    "agent_scores.csv",
    "run_meta.json",
    "config_used.yaml",
]
KEY_COLS = [
    "run",
    "cell",
    "n_cases",
    "accuracy",
    "selection_accuracy_majority",
    "selection_accuracy_confidence",
    "selection_accuracy_bts",
    "selection_accuracy_proper",
    "auc_score",
    "auc_confidence",
    "delta_auc",
    "delta_auc_ci_lo",
    "delta_auc_ci_hi",
    "ece_confidence",
    "agreement_rate",
    "rho",
    "rarity_premium",
    "auc_oracle",
    "delta_auc_oracle",
    "oracle_reduction",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results", help="directory containing run folders")
    ap.add_argument("--out", default="results_summary", help="tracked output directory")
    ap.add_argument(
        "--pattern",
        default="*",
        help="run-folder name glob, e.g. 'gpu_pubmedqa_*_full'",
    )
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    index: list[str] = [
        "# Run summary",
        "",
        "Raw `answers.jsonl` / `forecasts.jsonl` stay local (git-ignored). "
        "This folder is the shareable, pushable part of the runs.",
        "",
    ]

    run_dirs = sorted(p for p in src.glob(args.pattern) if p.is_dir())
    if not run_dirs:
        print(f"No run folders matched {src}/{args.pattern}")

    for run_dir in run_dirs:
        dest = out / run_dir.name
        dest.mkdir(parents=True, exist_ok=True)

        for name in TEXT_FILES:
            path = run_dir / name
            if path.is_file():
                shutil.copy2(path, dest / name)

        figs = sorted((run_dir / "figures").glob("*.png")) if (run_dir / "figures").is_dir() else []
        if figs:
            (dest / "figures").mkdir(exist_ok=True)
            for fig in figs:
                shutil.copy2(fig, dest / "figures" / fig.name)

        metrics_path = run_dir / "cell_metrics.csv"
        complete = metrics_path.is_file() and (run_dir / "report.md").is_file()
        status = "complete" if complete else "incomplete - run the same command with --resume"
        index.append(f"## {run_dir.name} - {status}")
        index.append("")
        index.append(f"- copied: {', '.join(p.name for p in sorted(dest.rglob('*')) if p.is_file())}")
        index.append("")

        if metrics_path.is_file():
            df = pd.read_csv(metrics_path)
            first = df.columns[0]
            if first.startswith("Unnamed") or first == "":
                df = df.rename(columns={first: "cell"})
            df.insert(0, "run", run_dir.name)
            keep = [c for c in KEY_COLS if c in df.columns]
            rows.append(df[keep])

    if rows:
        combined = pd.concat(rows, ignore_index=True, sort=False)
        combined.to_csv(out / "summary_all.csv", index=False)
        index.append("## Combined table")
        index.append("")
        index.append(f"`summary_all.csv`: {len(combined)} rows x {len(combined.columns)} columns.")
        index.append("")
        index.append("Key columns: " + ", ".join(KEY_COLS) + ".")
    else:
        (out / "summary_all.csv").write_text("run,cell,status\n", encoding="utf-8")
        index.append("No `cell_metrics.csv` found yet; re-run with --resume on the GPU PC.")

    (out / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'summary_all.csv'} and {out / 'INDEX.md'}")


if __name__ == "__main__":
    main()
