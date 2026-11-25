#!/usr/bin/env bash
# Google Colab: run the small local-PC model team (Qwen2.5-0.5B + 1.5B)
# across all five human-written benchmarks. Each dataset has its own config
# and output dir, so results are correct and --resume is safe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"

DATASETS=(pubmedqa medqa mmlu_med medmcqa medxpertqa)
LIMIT="${LIMIT:-300}"
CELLS="${CELLS:-H0,H1,H2,H3,H4,H5,SYC}"
RESUME="${RESUME:-1}"
# Persist results in Google Drive to resume across Colab disconnect/reset.
OUT_ROOT="${OUT_ROOT:-results}"
mkdir -p "$OUT_ROOT"

echo "Colab small-model sweep"
echo "  datasets: ${DATASETS[*]}"
echo "  limit:    $LIMIT"
echo "  cells:    $CELLS"
echo "  resume:   $RESUME"
echo "  out_root: $OUT_ROOT"

for ds in "${DATASETS[@]}"; do
  cfg="configs/colab_${ds}_small.yaml"
  out="${OUT_ROOT}/colab_${ds}_small"
  if [ ! -f "$cfg" ]; then
    echo "[skip] missing config: $cfg"
    continue
  fi

  echo ""
  echo "============================================================"
  echo "DATASET: $ds"
  echo "CONFIG : $cfg"
  echo "OUT    : $out"
  echo "============================================================"

  args=(-m bts_audit.cli run --config "$cfg" --out "$out" --limit "$LIMIT" --cells "$CELLS")
  if [ "$RESUME" = "1" ]; then args+=(--resume); fi
  "$PY" "${args[@]}"

  echo "--- plots for $ds ---"
  "$PY" scripts/plot_results.py --results "$out" || echo "plot failed for $ds (continuing)"

  echo "--- report head for $ds ---"
  if [ -f "$out/report.md" ]; then
    sed -n '1,60p' "$out/report.md"
  fi
done

echo ""
echo "Done. Results: ${OUT_ROOT}/colab_*_small"
