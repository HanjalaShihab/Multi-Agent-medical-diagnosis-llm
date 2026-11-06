#!/usr/bin/env bash
# Full pilot on a local GPU. No API key.
#
#   DATASET=medqa bash scripts/run_pilot.sh
#   DATASET=pubmedqa LIMIT=60 CELLS=H1,H4,H5,SYC bash scripts/run_pilot.sh
#
# DATASET: medqa | pubmedqa | mmlu_med | medmcqa | medxpertqa
# LIMIT:   optional first pass over the first N prepared cases
# CELLS:   optional comma-separated subset of H0,H1,H2,H3,H4,H5,SYC
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

DATASET="${DATASET:-medqa}"
LIMIT="${LIMIT:-}"
CELLS="${CELLS:-}"

# The shipped configs point at data/cases/<dataset>_300.jsonl. Prepare the
# full 300-case file once; use LIMIT for a cheaper first pass.
PREPARE_N=300

case "$DATASET" in
  medqa)       RAW="${RAW:-data/raw/medqa}";       DOWNLOAD="medqa" ;;
  pubmedqa)    RAW="${RAW:-data/raw/pubmedqa}";    DOWNLOAD="pubmedqa" ;;
  mmlu_med)    RAW="${RAW:-data/raw/mmlu}";        DOWNLOAD="mmlu" ;;
  medmcqa)     RAW="${RAW:-data/raw/medmcqa}";     DOWNLOAD="medmcqa" ;;
  medxpertqa)  RAW="${RAW:-data/raw/medxpertqa}";  DOWNLOAD="medxpertqa" ;;
  *) echo "Unknown DATASET=$DATASET"; exit 1 ;;
esac

CASES="data/cases/${DATASET}_${PREPARE_N}.jsonl"
CONFIG="configs/pilot_${DATASET}.yaml"

if [ ! -f "$CONFIG" ]; then
  echo "No config at $CONFIG"
  exit 1
fi
if [ ! -d "$RAW" ] && [ ! -f "$RAW" ]; then
  echo "No raw data in $RAW."
  echo "Try: python scripts/download_data.py ${DOWNLOAD} --out ${RAW}"
  exit 1
fi

"$PYTHON" -m bts_audit.cli doctor --config "$CONFIG"
"$PYTHON" -m bts_audit.cli prepare --dataset "$DATASET" --raw "$RAW" --out "$CASES" --n "$PREPARE_N"

RUN_ARGS=(run --config "$CONFIG" --resume)
if [ -n "$LIMIT" ]; then RUN_ARGS+=(--limit "$LIMIT"); fi
if [ -n "$CELLS" ]; then RUN_ARGS+=(--cells "$CELLS"); fi
"$PYTHON" -m bts_audit.cli "${RUN_ARGS[@]}"
