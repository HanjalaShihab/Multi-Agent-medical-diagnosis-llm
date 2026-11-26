# Running new/bigger models on a GPU PC (resume-safe)

This is for the case where you **git clone/pull on another GPU PC** and run new models there.
It does **not** mix with your old CPU small-model runs.

## Why resume works

`bts_audit.cli run --resume` loads existing `answers.jsonl`/`forecasts.jsonl` in the **same output dir** and skips completed `(cell, case_id, round, agent_id, task)` records.
So if the PC shuts down, run the **same command again** (same config, same `--out`, same `--resume`) and it continues.

**Important:** do not resume into an old run that used different models/config. Use a new output dir per model/config.

## 1) On this laptop (before going to GPU PC)

```bash
cd "/home/hanjala-shihab/Documents/Multi-Agent medical diagnosis llm"
git add -A
git commit -m "Add GPU model configs and run script"
git push origin main
```

Note: git ignores `hf_models/`, `pilot/.venv/`, `pilot/results/`, `pilot/data/raw/`, `pilot/data/cases/*.jsonl`. That is expected.

## 2) On the GPU PC

```bash
git clone git@github.com:HanjalaShihab/Multi-Agent-medical-diagnosis-llm.git
# or: git pull
cd Multi-Agent-medical-diagnosis-llm/pilot

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
pip install -e . --no-deps

python -m bts_audit.cli doctor --config configs/pilot_pubmedqa.yaml
```

Check that `doctor` shows CUDA/torch GPU. Install a CUDA-enabled PyTorch build if needed.

## 3) Download/prepare data (not in git)

```bash
python scripts/download_data.py all --out data/raw

python -m bts_audit.cli prepare --dataset pubmedqa   --raw data/raw/pubmedqa   --out data/cases/pubmedqa_300.jsonl   --n 300
python -m bts_audit.cli prepare --dataset medqa      --raw data/raw/medqa      --out data/cases/medqa_300.jsonl      --n 300
python -m bts_audit.cli prepare --dataset mmlu_med   --raw data/raw/mmlu       --out data/cases/mmlu_med_300.jsonl   --n 300
python -m bts_audit.cli prepare --dataset medmcqa    --raw data/raw/medmcqa    --out data/cases/medmcqa_300.jsonl    --n 300
python -m bts_audit.cli prepare --dataset medxpertqa --raw data/raw/medxpertqa --out data/cases/medxpertqa_300.jsonl --n 300
```

## 4) New model configs (already added)

Independent configs, one model per run, separate output dirs:

| Config | Target model | Output dir | Notes |
|---|---|---|---|
| `configs/gpu_pubmedqa_2b.yaml` | `Qwen/Qwen2.5-1.5B-Instruct` | `results/gpu_pubmedqa_2b` | 1.5B, closest open 2B-class instruct |
| `configs/gpu_pubmedqa_3b.yaml` | `Qwen/Qwen2.5-3B-Instruct` | `results/gpu_pubmedqa_3b` | Open HF model |
| `configs/gpu_pubmedqa_4b.yaml` | `Qwen/Qwen3-4B-Instruct-2507` | `results/gpu_pubmedqa_4b` | Verify HF ID on GPU PC |
| `configs/gpu_pubmedqa_5b.yaml` | placeholder (replace) | `results/gpu_pubmedqa_5b` | **Edit model ID**; true 5B instruct is uncommon |
| `configs/gpu_pubmedqa_6b.yaml` | `01-ai/Yi-1.5-6B-Chat` | `results/gpu_pubmedqa_6b` | Verify chat template/parse rate |
| `configs/gpu_pubmedqa_7b.yaml` | `Qwen/Qwen2.5-7B-Instruct` | `results/gpu_pubmedqa_7b` | Good baseline |

All use `device_map: auto`, `dtype: auto`, `load_in_4bit: true`, `max_loaded_models: 1`.
Change `load_in_4bit` to `false` only if you have enough VRAM and want full precision.

## 5) Download/check models on GPU PC

Hugging Face models are downloaded automatically on first use (to HF cache). For gated models (if you later add one), run `huggingface-cli login` or set `HF_TOKEN` after accepting the HF license.

Optional pre-download (example):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
for m in [
  "Qwen/Qwen2.5-1.5B-Instruct",
  "Qwen/Qwen2.5-3B-Instruct",
  "Qwen/Qwen3-4B-Instruct-2507",
  "01-ai/Yi-1.5-6B-Chat",
  "Qwen/Qwen2.5-7B-Instruct",
]:
    print('downloading', m)
    snapshot_download(repo_id=m)
PY
```

If a model ID fails, check the exact HF repo name and replace it in that config.

## 6) Run (first pass recommended)

Single model, quick first pass:

```bash
python -m bts_audit.cli run \
  --config configs/gpu_pubmedqa_7b.yaml \
  --data data/cases/pubmedqa_300.jsonl \
  --out results/gpu_pubmedqa_7b_firstpass \
  --limit 60 --cells H1,H4,H5,SYC
```

Full run, resume-safe:

```bash
python -m bts_audit.cli run \
  --config configs/gpu_pubmedqa_7b.yaml \
  --data data/cases/pubmedqa_300.jsonl \
  --out results/gpu_pubmedqa_7b_full \
  --resume
```

If PC shuts down, rerun the **same** command. It resumes from existing `answers.jsonl/forecasts.jsonl`.

## 7) Run all selected models

```bash
# first pass for all 2b-7b configs
LIMIT=60 CELLS=H1,H4,H5,SYC bash scripts/run_gpu_models.sh

# full runs (resume-safe)
bash scripts/run_gpu_models.sh

# only selected models
MODELS="6b 7b" bash scripts/run_gpu_models.sh
```

## 8) Outputs

Each run dir has: `report.md`, `cell_metrics.csv`, `metrics.json`, `predictions.json`, `agent_scores.csv`, raw `answers.jsonl`/`forecasts.jsonl`, and `figures/*.png` after plotting.

## 9) Safety rules

- **Do not** point a new model config at old CPU output dirs (`results/cpu_*_local_full`).
- **Do not** resume with a changed model into the same out dir.
- Keep **one output dir per model/config**.
- Old 0.5B/1.5B CPU runs can be run again later only by using their original config+output dir; they are independent from these GPU runs.

## 10) Deadline rescue: fast 6B + 7B probes (today only)

If the 300-case full runs will not finish in time, **stop them** (`Ctrl+C` in each
PowerShell terminal; `--resume` keeps everything already written) and run the
small probes instead. They use **new configs and new output dirs**, so nothing
mixes with the full runs.

```powershell
cd pilot
.\\scripts\\run_gpu_quick_6b7b.ps1
```

One command runs **6B on GPU 0 and 7B on GPU 1 in parallel**. It first downloads
the full model snapshots once (needed for the un-cached 6B model and for the 7B
chat-template files that caused the HTTP 429 retries), then runs both audits
with `HF_HUB_OFFLINE=1`.

Each probe: **40 PubMedQA cases**, cells `H1,H4,H5,SYC`, `max_new_tokens: 96`,
4-bit. Outputs go to `results/gpu_quick_pubmedqa_6b/` and
`results/gpu_quick_pubmedqa_7b/`. If the download was already done, use
`-SkipPrefetch`. If you prefer two terminals, use the older
`scripts/run_gpu_quick.ps1` with `-Models @("6b")` and `-Models @("7b")`
after running the prefetch script yourself.

## 11) Share the results with the laptop after the runs finish

`pilot/results/` is git-ignored, so a plain `git push` from this PC will **not**
send `report.md` or `cell_metrics.csv`. Convert the finished runs into a small,
tracked summary first:

```bash
cd pilot
python scripts/collect_results.py --src results --out results_summary \
  --pattern 'gpu*'
```

This copies only the shareable files (`report.md`, `cell_metrics.csv`,
`metrics.json`, `predictions.json`, `agent_scores.csv`, `run_meta.json`,
`config_used.yaml`, `figures/*.png`) into `pilot/results_summary/` and writes a
combined `summary_all.csv` plus an `INDEX.md` that marks any incomplete run.
Raw `answers.jsonl` / `forecasts.jsonl` stay local.

Then, on this PC:

```bash
git add pilot/results_summary
git commit -m "Add GPU PubMedQA model comparison summary"
git push origin main
```

On the laptop:

```bash
cd "/home/hanjala-shihab/Documents/Multi-Agent medical diagnosis llm"
git pull origin main
```

You can then open `pilot/results_summary/summary_all.csv` for a side-by-side
comparison of `accuracy`, `selection_accuracy_*`, `auc_score`, `delta_auc`,
`agreement_rate`, `rarity_premium`, `delta_auc_oracle`, etc. per model and cell,
and `pilot/results_summary/<run>/report.md` for the P1-P8 verdict.
