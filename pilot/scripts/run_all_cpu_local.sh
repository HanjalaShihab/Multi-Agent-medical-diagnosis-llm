#!/usr/bin/env bash
# Run the full BTS audit pilot on CPU using locally downloaded small models.
# Full = 5 human-written datasets x 300 cases x 7 cells (H0-H5,SYC).
# WARNING: This is extremely slow on CPU and results are NOT scientific
# (small 0.5B/1.5B models). For meaningful results use Kaggle GPU configs.
set -euo pipefail

ROOT="/home/hanjala-shihab/Documents/Multi-Agent medical diagnosis llm/pilot"
cd "$ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PY="${ROOT}/.venv/bin/python"
# Optional overrides, e.g.:
#   LIMIT=10 CELLS=H1,H4 bash scripts/run_all_cpu_local.sh
LIMIT="${LIMIT:-300}"
CELLS="${CELLS:-H0,H1,H2,H3,H4,H5,SYC}"
RESUME="${RESUME:-1}"

DATASETS=(pubmedqa medqa mmlu_med medmcqa medxpertqa)

echo "CPU full run: limit=$LIMIT cells=$CELLS resume=$RESUME"
echo "Datasets: ${DATASETS[*]}"
echo "Tip: to do a quicker trial first, run with LIMIT=10 CELLS=H1,H4"

for ds in "${DATASETS[@]}"; do
  cfg="configs/cpu_${ds}_local.yaml"
  out="results/cpu_${ds}_local_full"

  echo ""
  echo "============================================================"
  echo "DATASET: $ds"
  echo "CONFIG : $cfg"
  echo "OUT    : $out"
  echo "============================================================"

  args=( -m bts_audit.cli run --config "$cfg" --out "$out" --limit "$LIMIT" --cells "$CELLS" )
  if [[ "$RESUME" == "1" ]]; then args+=( --resume ); fi

  "$PY" "${args[@]}"

  echo "--- generating plots for $ds ---"
  "$PY" scripts/plot_results.py --results "$out" || echo "plot generation failed for $ds (continuing)"

  echo "--- summary for $ds ---"
  if [[ -f "$out/report.md" ]]; then
    sed -n '1,80p' "$out/report.md"
  fi
  if [[ -f "$out/cell_metrics.csv" ]]; then
    "$PY" - <<PYTHON
import pandas as pd
df=pd.read_csv("$out/cell_metrics.csv")
cols=[c for c in ['cell','n_cases','accuracy','selection_accuracy_majority','selection_accuracy_confidence','selection_accuracy_bts','selection_accuracy_proper','auc_score','auc_confidence','delta_auc'] if c in df.columns]
print(df[cols].to_string(index=False))
PYTHON
  fi
done

echo ""
echo "All runs finished. Results under: $ROOT/results/cpu_*_local_full"
echo "Open each: results/cpu_<dataset>_local_full/report.md and figures/*.png"
