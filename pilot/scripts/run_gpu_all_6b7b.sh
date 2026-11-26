#!/usr/bin/env bash
# One command, unattended: all ten full GPU audits (6B + 7B x 5 datasets).
# Usage: bash scripts/run_gpu_all_6b7b.sh [--fast]   # --fast = 128-token configs
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

prefix=gpu
pattern='gpu_[67]b_*'
if [[ "${1:-}" == "--fast" ]]; then
  prefix=gpu_fast
  pattern='gpu_fast_[67]b_*'
fi

models=(6b 7b)
datasets=(pubmedqa medqa medmcqa mmlu_med medxpertqa)

for d in "${datasets[@]}"; do
  [[ -f "data/cases/${d}_300.jsonl" ]] || { echo "Missing data/cases/${d}_300.jsonl; prepare data first." >&2; exit 1; }
  for m in "${models[@]}"; do
    [[ -f "configs/${prefix}_${m}_${d}.yaml" ]] || { echo "Missing configs/${prefix}_${m}_${d}.yaml; git pull and run make_gpu_all_configs.py first." >&2; exit 1; }
  done
done

mkdir -p "logs/${prefix}_all_6b7b"
echo "[START] $(date -Is) - 10 full runs, 300 cases each (prefix=$prefix)"
declare -a summary=()

for m in "${models[@]}"; do
  for d in "${datasets[@]}"; do
    name="${prefix}_${m}_${d}"
    log="logs/${prefix}_all_6b7b/${name}.log"
    echo "[RUN] $name $(date -Is) log=$log"
    python -m bts_audit.cli run --config "configs/${name}.yaml" --resume 2>&1 | tee "$log"
    code=${PIPESTATUS[0]}
    if [[ $code -eq 0 ]]; then
      echo "[OK]   $name $(date -Is)"
      summary+=("$name OK")
    else
      echo "[FAIL] $name exit=$code see $log" >&2
      summary+=("$name FAIL(exit=$code)")
    fi
  done
done

python scripts/collect_results.py --src results --out results_summary --pattern "$pattern"
echo "[SUMMARY] $(date -Is)"
printf '  %s\n' "${summary[@]}"
