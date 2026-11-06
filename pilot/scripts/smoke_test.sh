#!/usr/bin/env bash
# One-command dry run: tests -> real MedQA subset -> all cells with mock -> report.
# The numbers are SMOKE ONLY; this checks the pipeline, not the hypothesis.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "== 1/4 unit tests =="
"$PYTHON" -m unittest discover -s tests -v

if [ ! -f data/raw/medqa/test.jsonl ]; then
  echo "== MedQA not found; downloading the public test split =="
  "$PYTHON" scripts/download_data.py medqa --out data/raw/medqa
fi

echo "== 2/4 prepare a real MedQA subset =="
"$PYTHON" -m bts_audit.cli prepare \
  --dataset medqa --raw data/raw/medqa \
  --out data/cases/medqa_60.jsonl --n 60

echo "== 3/4 run all cells with the mock backend =="
"$PYTHON" -m bts_audit.cli run --config configs/mock_medqa.yaml

echo "== 4/4 analyze =="
"$PYTHON" -m bts_audit.cli analyze --run results/mock_medqa

echo
echo "Open results/mock_medqa/report.md"
echo "This run uses a mock backend. Its numbers are SMOKE ONLY."
