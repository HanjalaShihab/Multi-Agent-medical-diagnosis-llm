# Run summary

Raw `answers.jsonl` / `forecasts.jsonl` stay local (git-ignored). This folder is the shareable, pushable part of the runs.

## gpu_pubmedqa_2b_full - complete

- copied: agent_scores.csv, cell_metrics.csv, config_used.yaml, metrics.json, predictions.json, report.md, run_meta.json

## gpu_pubmedqa_3b_full - complete

- copied: agent_scores.csv, cell_metrics.csv, config_used.yaml, metrics.json, predictions.json, report.md, run_meta.json

## gpu_pubmedqa_4b_full - incomplete - run the same command with --resume

- copied: config_used.yaml, run_meta.json

## gpu_pubmedqa_7b_full - incomplete - run the same command with --resume

- copied: config_used.yaml, run_meta.json

## Combined table

`summary_all.csv`: 14 rows x 20 columns.

Key columns: run, cell, n_cases, accuracy, selection_accuracy_majority, selection_accuracy_confidence, selection_accuracy_bts, selection_accuracy_proper, auc_score, auc_confidence, delta_auc, delta_auc_ci_lo, delta_auc_ci_hi, ece_confidence, agreement_rate, rho, rarity_premium, auc_oracle, delta_auc_oracle, oracle_reduction.
