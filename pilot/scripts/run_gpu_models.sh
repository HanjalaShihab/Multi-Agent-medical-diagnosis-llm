#!/usr/bin/env bash
# Run one or more GPU model configs on the GPU PC.
# Independent runs: each config has its own run_name/output_dir.
# Resume-safe: if PC shuts down, rerun same command with --resume (default).
#
# Examples:
#   bash scripts/run_gpu_models.sh                       # all gpu_pubmedqa_{3..8}b
#   MODELS="7b 8b" bash scripts/run_gpu_models.sh        # only 7B,8B
#   LIMIT=60 CELLS=H1,H4,H5,SYC bash scripts/run_gpu_models.sh  # first pass
#   RESUME=0 bash scripts/run_gpu_models.sh              # fresh (careful: overwrites run logs)
#
# Before full runs:
#   - activate venv, install requirements-gpu.txt
#   - prepare data/cases/pubmedqa_300.jsonl
#   - check HF access for gated models (Llama 8B needs HF token/license)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; fi

MODELS="${MODELS:-3b 4b 5b 6b 7b 8b}"
LIMIT="${LIMIT:-}"
CELLS="${CELLS:-}"
RESUME="${RESUME:-1}"
DATA="${DATA:-data/cases/pubmedqa_300.jsonl}"

echo "GPU model runs"
echo "  models: $MODELS"
echo "  data:   $DATA"
echo "  limit:  ${LIMIT:-<config>}"
echo "  cells:  ${CELLS:-<config>}"
echo "  resume: $RESUME"

for m in $MODELS; do
  cfg="configs/gpu_pubmedqa_${m}.yaml"
  if [ ! -f "$cfg" ]; then
    echo "[skip] missing config: $cfg"
    continue
  fi
  # output dir from config (fallback to results/gpu_pubmedqa_<m>)
  out="$($PY - <<PYTHON
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path('$cfg').read_text())
print(cfg.get('output_dir', 'results/gpu_pubmedqa_${m}'))
PYTHON
)"
  echo ""
  echo "============================================================"
  echo "MODEL : $m"
  echo "CONFIG: $cfg"
  echo "OUT   : $out"
  echo "============================================================"

  args=(-m bts_audit.cli run --config "$cfg" --data "$DATA" --out "$out")
  if [ -n "$LIMIT" ]; then args+=(--limit "$LIMIT"); fi
  if [ -n "$CELLS" ]; then args+=(--cells "$CELLS"); fi
  if [ "$RESUME" = "1" ]; then args+=(--resume); fi

  "$PY" "${args[@]}"

  echo "--- plotting $m ---"
  "$PY" scripts/plot_results.py --results "$out" || echo "plot failed for $m (continuing)"
  echo "--- report head $m ---"
  [ -f "$out/report.md" ] && sed -n '1,60p' "$out/report.md" || true
done

echo ""
echo "Done. Check results/gpu_pubmedqa_*_full or configured output dirs."
