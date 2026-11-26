#!/usr/bin/env bash
# Time-limited thesis scope (5 days):
#   PRIMARY   PubMedQA 300 cases x H0,H1,H2,H3,H4,H5,SYC
#   SUPPORT   MedQA + MedMCQA, 60 cases x H1,H4,H5,SYC
# Uses the small local-PC model team on a T4 GPU. Each run has its own config
# and output dir, so --resume is safe across Colab disconnects.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"

OUT_ROOT="${OUT_ROOT:-/content/drive/MyDrive/bts_audit_colab_results}"
RUNS=(
  "pubmedqa_small|colab_pubmedqa_small|300|H0,H1,H2,H3,H4,H5,SYC"
  "medqa_support|colab_medqa_support|60|H1,H4,H5,SYC"
  "medmcqa_support|colab_medmcqa_support|60|H1,H4,H5,SYC"
)

mkdir -p "$OUT_ROOT"
echo "Thesis-scope sweep | OUT_ROOT=$OUT_ROOT"

for spec in "${RUNS[@]}"; do
  IFS='|' read -r key cfg limit cells <<<"$spec"
  out="${OUT_ROOT}/${cfg}"

  echo ""
  echo "============================================================"
  echo "RUN: $cfg | limit=$limit | cells=$cells"
  echo "OUT: $out"
  echo "============================================================"

  "$PY" -m bts_audit.cli run \
    --config "configs/${cfg}.yaml" \
    --out "$out" \
    --limit "$limit" \
    --cells "$cells" \
    --resume

  echo "--- plots: $cfg ---"
  "$PY" scripts/plot_results.py --results "$out" || echo "plot failed for $cfg (continuing)"

  echo "--- summary: $cfg ---"
  if [ -f "$out/report.md" ]; then
    sed -n '1,80p' "$out/report.md"
  fi
  if [ -f "$out/cell_metrics.csv" ]; then
    "$PY" - "$out/cell_metrics.csv" <<'PY'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
cols = [c for c in [
    "cell", "n_cases", "accuracy", "selection_accuracy_majority",
    "selection_accuracy_confidence", "selection_accuracy_bts",
    "selection_accuracy_proper", "auc_score", "auc_confidence",
    "delta_auc", "agreement_rate", "rho", "delta_auc_oracle",
] if c in df.columns]
print(df[cols].to_string(index=False))
PY
  fi
done

echo ""
echo "Done. Results under: $OUT_ROOT/colab_{pubmedqa_small,medqa_support,medmcqa_support}"
