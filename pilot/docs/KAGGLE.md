# Running on Kaggle (GPU) and later on CPU

This guide covers two environments:

1. **Kaggle** — run the pilot on a free/hosted GPU (T4/P100) with no local setup.
2. **CPU** — run a small, real-model check later on a machine with no GPU.

The mock backend (`configs/mock_medqa.yaml`) works anywhere and needs no GPU; it is for pipeline smoke tests only (`SMOKE ONLY`).

---

## 1. Kaggle (GPU)

### 1.1 What you need

- A Kaggle account.
- Notebook settings: **Accelerator = GPU T4 x2** (or P100) and **Internet = On**.
- This repository, either:
  - attached as a Kaggle dataset/input, or
  - cloned inside the notebook (`git clone https://github.com/HanjalaShihab/Multi-Agent-medical-diagnosis-llm.git`).

Optional but recommended: attach a dataset that already contains prepared case files (`data/cases/*.jsonl`) so you can skip download and preparation.

### 1.2 Quick start (notebook)

Open or upload [`pilot/notebooks/kaggle_run.ipynb`](../notebooks/kaggle_run.ipynb) in Kaggle and run the cells in order. The notebook:

1. locates/copies the project to `/kaggle/working` (writable),
2. installs `requirements-kaggle.txt`,
3. downloads raw benchmarks and prepares 300-case files under `/kaggle/working/data`,
4. runs `doctor`,
5. runs a 60-case first pass (recommended),
6. runs the full 300-case sweep (optional, long),
7. zips results for download.

### 1.3 Kaggle configs

Kaggle-ready configs write to `/kaggle/working` and use `float16` + 4-bit for T4/P100 compatibility:

| Config | Dataset | Output |
|---|---|---|
| `configs/kaggle_pubmedqa.yaml` | PubMedQA (genuine evidence sections) | `/kaggle/working/results/kaggle_pubmedqa` |
| `configs/kaggle_medqa.yaml` | MedQA-USMLE | `/kaggle/working/results/kaggle_medqa` |
| `configs/kaggle_mmlu_med.yaml` | MMLU-Med | `/kaggle/working/results/kaggle_mmlu_med` |
| `configs/kaggle_medmcqa.yaml` | MedMCQA (validation) | `/kaggle/working/results/kaggle_medmcqa` |
| `configs/kaggle_medxpertqa.yaml` | MedXpertQA (text, 10 options) | `/kaggle/working/results/kaggle_medxpertqa` |

All run cells H0–H5 + SYC with the same 3-model team as the GPU pilot configs.

### 1.4 Manual commands (Kaggle console / notebook cell)

```bash
cd /kaggle/working/Multi-Agent-medical-diagnosis-llm/pilot
pip install -q -r requirements-kaggle.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Download + prepare (once)
python scripts/download_data.py pubmedqa --out /kaggle/working/data/raw
python scripts/download_data.py medqa    --out /kaggle/working/data/raw

python -m bts_audit.cli prepare --dataset pubmedqa \
  --raw /kaggle/working/data/raw/pubmedqa \
  --out /kaggle/working/data/cases/pubmedqa_300.jsonl --n 300
python -m bts_audit.cli prepare --dataset medqa \
  --raw /kaggle/working/data/raw/medqa \
  --out /kaggle/working/data/cases/medqa_300.jsonl --n 300

# Check environment
python -m bts_audit.cli doctor --config configs/kaggle_pubmedqa.yaml

# 60-case first pass (recommended)
python -m bts_audit.cli run \
  --config configs/kaggle_pubmedqa.yaml \
  --data /kaggle/working/data/cases/pubmedqa_300.jsonl \
  --out  /kaggle/working/results/kaggle_pubmedqa_60 \
  --limit 60 --cells H1,H4,H5,SYC

# Full run (resume-safe)
python -m bts_audit.cli run \
  --config configs/kaggle_pubmedqa.yaml \
  --data /kaggle/working/data/cases/pubmedqa_300.jsonl \
  --out  /kaggle/working/results/kaggle_pubmedqa_300 \
  --resume
```

`--data` and `--out` override the config paths, so you can point at attached input datasets or custom locations without editing YAML.

### 1.5 Runtime, memory, and resume

- **Budget:** ~22,500 model calls per 300-case, 7-cell run. Expect many hours on a T4; the 60-case, 4-cell first pass is ~1–3 hours.
- **Memory:** configs load one 4-bit model at a time (`max_loaded_models: 1`). A 16 GB T4 should fit the 3B–7B team sequentially. If you hit OOM, lower `max_new_tokens`, reduce `--limit`, or use a smaller model set in a copy of the config.
- **Resume:** always use `--resume`. Answers and forecasts are written incrementally, so a session timeout does not lose completed work (keep the output directory).
- **Parse rate:** after the first pass, inspect `answers.jsonl` / `forecasts.jsonl`. Target >90% successful parses before scaling.
- **Model cache:** the notebook sets `HF_HOME=/kaggle/working/hf_cache` by default. Large model downloads count against session/disk limits; attach models as a Kaggle dataset if internet or disk is constrained.

### 1.6 Reading results

Each run directory contains `report.md` (verdict + P1–P8 scorecard + data limitations), `predictions.json`, `metrics.json`, `cell_metrics.csv`, and raw `answers.jsonl` / `forecasts.jsonl`.

Interpretation reminders:
- **ΔAUC is one-sided:** negative ΔAUC in H1–H3 is the expected popularity effect, not a failed run.
- **PubMedQA** is the only genuine H4 evidence partition; H4 on other datasets is a text-chunk ablation.
- **MedMCQA** uses the validation split (test labels are withheld).
- **No disease prevalence:** P6a/P6b are mechanical properties of the score, not disease-prevalence tests.
- Mock-backend runs are labelled `SMOKE ONLY`.

---

## 2. CPU (later)

CPU runs are supported for real-model checks, but they are **slow**. Use them to validate parsing and the pipeline on your machine, not for the full scientific sweep (use a GPU for that).

### 2.1 Install

```bash
cd pilot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-cpu.txt
pip install -e . --no-deps
```

`requirements-cpu.txt` omits `bitsandbytes` (4-bit is CUDA-oriented). The HF backend automatically disables 4-bit when CUDA is unavailable.

### 2.2 CPU configs

| Config | Default scope | Notes |
|---|---|---|
| `configs/cpu_pubmedqa.yaml` | 20 cases, H1,H4,H5,SYC | Genuine evidence sections; best CPU sanity check. |
| `configs/cpu_medqa.yaml` | 20 cases, H1,H4,H5,SYC | 4-option replication check. |

Both use small models (`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`), `device_map: cpu`, `dtype: float32`, `load_in_4bit: false`, and `max_new_tokens: 160`. H3 uses size heterogeneity, not the full GPU pilot team.

### 2.3 Commands

```bash
# Quickest parse/runtime check (a few cases)
python -m bts_audit.cli run \
  --config configs/cpu_pubmedqa.yaml \
  --limit 5 --cells H1

# Default CPU pass (20 cases, H1,H4,H5,SYC)
python -m bts_audit.cli run --config configs/cpu_pubmedqa.yaml --resume

# Larger (very slow): raise limit / add cells explicitly
python -m bts_audit.cli run \
  --config configs/cpu_medqa.yaml \
  --limit 60 --cells H0,H1,H2,H3,H4,H5,SYC --resume

python -m bts_audit.cli analyze --run results/cpu_pubmedqa
```

If the prepared case files are missing, download and prepare them first:

```bash
python scripts/download_data.py pubmedqa --out data/raw
python -m bts_audit.cli prepare --dataset pubmedqa \
  --raw data/raw/pubmedqa --out data/cases/pubmedqa_300.jsonl --n 300
```

### 2.4 Tips

- **Expect hours, not minutes.** Start with `--limit 5 --cells H1`.
- **RAM:** 0.5B/1.5B models in float32 fit in ~8 GB; close other heavy processes.
- **Mock first:** if anything is unclear, run `bash scripts/smoke_test.sh` (mock backend, ~2 minutes) to confirm the pipeline.
- **Scientific runs:** CPU configs are for feasibility and debugging. Use `configs/pilot_*.yaml` or `configs/kaggle_*.yaml` on a GPU for the actual study.

---

## 3. Choosing a path

| Goal | Use |
|---|---|
| Verify pipeline on any machine, no model | `bash scripts/smoke_test.sh` (mock) |
| Run the real study on hosted GPU | `configs/kaggle_*.yaml` + `notebooks/kaggle_run.ipynb` |
| Run the real study on your own GPU | `configs/pilot_*.yaml` + `requirements-gpu.txt` |
| Quick real-model check on CPU | `configs/cpu_*.yaml` + `requirements-cpu.txt` |
| Ollama (local, no API key) | `configs/pilot_medqa_ollama.yaml` |

For all paths: **start with PubMedQA**, do a small first pass, check parse rates, then scale with `--resume`.
