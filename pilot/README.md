# BTS audit pilot

**Truth Serum or Popularity Contest? Peer-Prediction Mechanisms in Clinical Multi-Agent Diagnosis**

This is a runnable, local-first pilot for auditing Bayesian Truth Serum (BTS) as a
consensus rule in multi-agent LLM diagnosis. It tests one claim:

> Under shared evidence, BTS is an **answer-rarity statistic**, not a truth
> statistic. It becomes a truth signal only when the correct answer is a
> minority, i.e. when the team has genuinely private evidence.

The pilot is designed so the claim can fail. It runs a six-cell heterogeneity
ladder plus a sycophancy control, then evaluates nine pre-registered checks
(P1–P8, with P6 split into P6a/P6b), including an oracle-forecast ablation that
separates real information from forecast miscalibration.

**No API key and no paid service are needed.** The harness runs a local
HuggingFace model, a local Ollama server, or a deterministic mock backend. The
mock backend is for smoke tests only; its numbers are never a scientific result.

---

## 1. Quick start (CPU, about 2 minutes)

Requirements: Python 3.10+, `numpy`, `pandas`, `pyarrow`, `PyYAML`. `tqdm` is
optional.

```bash
cd pilot
pip install -r requirements.txt
pip install -e . --no-deps        # makes `bts-audit` and `python -m bts_audit.cli` work
bash scripts/smoke_test.sh
```

That command:

1. runs the unit tests,
2. downloads a real MedQA subset if it is missing,
3. prepares 60 real MedQA cases,
4. runs H0–H5 + SYC with the mock backend,
5. writes `results/mock_medqa/report.md`.

Open `results/mock_medqa/report.md`. The verdict will say **SMOKE ONLY**. That is
intentional: the mock backend simulates agents, so its numbers test the pipeline,
not the hypothesis.

The editable install adds the `bts-audit` entry point. These are equivalent:

```bash
bts-audit doctor --config configs/mock_medqa.yaml
PYTHONPATH=src python3 -m bts_audit.cli doctor --config configs/mock_medqa.yaml
```

**Other environments:** for a Kaggle GPU notebook see
[`docs/KAGGLE.md`](docs/KAGGLE.md) and
[`notebooks/kaggle_run.ipynb`](notebooks/kaggle_run.ipynb). For a small real-model
run on CPU (slow, no 4-bit) use `configs/cpu_pubmedqa.yaml` or
`configs/cpu_medqa.yaml` with `requirements-cpu.txt`.

---

## 2. What the pilot runs

Each cell changes **one** source of heterogeneity. This is the whole design: if
BTS helps, which change made it help?

| Cell | Team | What changes | What it tests |
|---|---|---|---|
| **H0** | 1 model, 1 prompt, T=0 | Nothing | Consensus collapse control. BTS must have zero variance. |
| **H1** | 1 model, 1 prompt, T=0.7 | Sampling noise | Is disagreement alone enough? Theory: no. |
| **H2** | 1 model, 5 role prompts | Role heterogeneity | Do role prompts act as private evidence? Theory: no. |
| **H3** | 3 model families, 1 prompt | Model heterogeneity | Do different models create a truth channel? Theory: no. |
| **H4** | 3 model families, one disjoint evidence slice each | **Evidence heterogeneity** | The only cell where BTS should recover a truth signal. |
| **H5** | H4 + citations + one deliberation round | Evidence + communication | Does deliberation keep or destroy the private-evidence advantage? |
| **SYC** | H1 + wrong-answer user pressure | Shared bias | When the truth is a minority, BTS should look good. |

The task is always the same: choose one option from a short list. Each agent
also reports a probability distribution over the options and a forecast of how
its teammates will answer. BTS uses the answers and the forecasts; confidence is
the baseline.

Two properties of the benchmark suite matter for interpretation:

- **H4/H5 are a genuine evidence partition only on PubMedQA**, whose slices are
  real abstract sections. On MedQA, MMLU-Med, MedMCQA, and MedXpertQA the five
  slices are contiguous chunks of the question stem. There, H4/H5 are a
  **text-chunk ablation**, not evidence partitioning.
- **No benchmark carries disease prevalence.** `rarity_stratum` and `prevalence`
  are `None`; they are never inferred from answer-option frequency. P6a/P6b are
  mechanical properties of the score, not disease-prevalence tests.

---

## 3. Data

### 3.1 The five human-written benchmarks

| Benchmark | Public source | License | Rows | Role |
|---|---|---|---|---|
| **MedQA-USMLE** | [`GBaker/MedQA-USMLE-4-options`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options), `phrases_no_exclude_test.jsonl` | CC-BY-4.0 | 1,273 | 4-option USMLE-style QA. |
| **PubMedQA** | [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA), `pqa_labeled` | MIT | 1,000 | 3-way yes/no/maybe over real abstract sections. The only genuine H4 partition. |
| **MMLU-Med** | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu), six medical subjects | MIT | 1,111 | Anatomy, clinical knowledge, college medicine, medical genetics, professional medicine, virology. |
| **MedMCQA** | [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa), validation split | Apache-2.0 | 4,183 | Broad medical entrance-exam QA. The test split withholds labels. |
| **MedXpertQA** | [`TsinghuaC3I/MedXpertQA`](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA), `Text/test.jsonl` | MIT | 2,450 | Expert-level QA with up to 10 options. |

Every item is human-written. None is generated, simulated, or restructured by an
LLM.

**Excluded on purpose.** DDXPlus is a synthetic patient simulator; MedLA /
MedDDx is constructed from a biomedical knowledge base; geteff1 is an
LLM-restructured derivative of published case reports. They are not human-written
benchmark data and are not part of this pilot. BioASQ is gated with no reliable
public mirror; treat it as optional and manual.

### 3.2 Download and prepare

```bash
cd pilot
python scripts/download_data.py all --out data/raw
```

The downloader uses public `resolve` URLs and the Python standard library, so
`huggingface_hub` is not required. To fetch one benchmark:

```bash
python scripts/download_data.py medqa --out data/raw/medqa
python scripts/download_data.py pubmedqa --out data/raw/pubmedqa
python scripts/download_data.py mmlu --out data/raw/mmlu
python scripts/download_data.py medmcqa --out data/raw/medmcqa
python scripts/download_data.py medxpertqa --out data/raw/medxpertqa
```

Prepare 300-case files:

```bash
python -m bts_audit.cli prepare --dataset medqa \
  --raw data/raw/medqa --out data/cases/medqa_300.jsonl --n 300

python -m bts_audit.cli prepare --dataset pubmedqa \
  --raw data/raw/pubmedqa --out data/cases/pubmedqa_300.jsonl --n 300

python -m bts_audit.cli prepare --dataset mmlu_med \
  --raw data/raw/mmlu --out data/cases/mmlu_med_300.jsonl --n 300

python -m bts_audit.cli prepare --dataset medmcqa \
  --raw data/raw/medmcqa --out data/cases/medmcqa_300.jsonl --n 300

python -m bts_audit.cli prepare --dataset medxpertqa \
  --raw data/raw/medxpertqa --out data/cases/medxpertqa_300.jsonl --n 300
```

Use `--seed` to change the sample and `--max-options` (default 10) to cap the
option list. The gold option is always kept and the order is shuffled
deterministically.

### 3.3 Expected raw layouts

```
data/raw/medqa/
  test.jsonl
data/raw/pubmedqa/
  pqa_labeled.parquet
data/raw/mmlu/
  anatomy_test.parquet
  clinical_knowledge_test.parquet
  college_medicine_test.parquet
  medical_genetics_test.parquet
  professional_medicine_test.parquet
  virology_test.parquet
data/raw/medmcqa/
  validation.parquet
data/raw/medxpertqa/
  test.jsonl
```

For MMLU you can also prepare a subset:

```bash
python -m bts_audit.cli prepare --dataset mmlu_med \
  --raw data/raw/mmlu --out data/cases/mmlu_anatomy_100.jsonl \
  --subjects anatomy --n 100
```

### 3.4 Dataset-specific notes

**MedQA.** The loader reads `question`, the `options` dict (`A`–`D`), and
`answer_idx`. It strips the original question into five text chunks and asks
"Which single option is the most likely answer?".

**PubMedQA.** Only `pqa_labeled` is accepted. `pqa_artificial` is generated from
unlabeled abstracts and is rejected. The labels are bucketed into `background`,
`case`, `methods`, `results`, `conclusions`, and `other`; the original section
labels are stored in `metadata.section_labels`. Many abstracts have only 2–4
sections, so agents may share a section in H4. That is expected and should be
reported.

**MMLU-Med.** The six subjects are downloaded separately. Answers are the
official test-split indices; `choices` is a NumPy array and the loader handles
it. The raw MMLU test files are small, so the 300-case sample draws across all
six subjects.

**MedMCQA.** The official `test` split ships `cop == -1` for every row because
the labels are withheld. The loader therefore uses the **validation** split, the
public labelled split, and refuses a file whose labels are all missing. This is a
deliberate deviation from the paper's test-set protocol and must be stated.

**MedXpertQA.** The text split contains all ten options, and the question text
repeats them inline after "Answer Choices:". The loader strips that inline block
so the options are not duplicated in the prompt. It keeps all ten options by
default.

**No prevalence.** None of these files has a patient-population prevalence
prior. The loaders leave `rarity_stratum=None` and `prevalence=None`. Do not
derive either from answer-option frequency.

---

## 4. Full GPU run

### 4.1 Hardware and software

Recommended for the main run:

- Linux, Python 3.10–3.12
- NVIDIA GPU with **24 GB VRAM** (RTX 3090/4090, A5000, L4, A10, …)
- CUDA-enabled PyTorch (install the wheel that matches your driver)
- ~60 GB disk for models and datasets

Smaller setups work:

| VRAM | Suggested team | Notes |
|---|---|---|
| 24 GB | Qwen2.5-7B + Qwen2.5-3B + Phi-3.5-mini, 4-bit | Default `pilot_*.yaml` configs. |
| 16 GB (Kaggle T4/P100) | Qwen2.5-7B + Qwen2.5-3B + Phi-3.5-mini, 4-bit | Use `configs/kaggle_*.yaml`; see [`docs/KAGGLE.md`](docs/KAGGLE.md). |
| 12–16 GB | Qwen2.5-3B + Phi-3.5-mini + Qwen2.5-1.5B, 4-bit | Edit `team.models` in a copy of the config; keep `max_loaded_models: 1`. |
| 8 GB | one 3B model + two 1.5B models, 4-bit | Use H1/H4/H5/SYC first. |
| CPU only | `Qwen/Qwen2.5-0.5B-Instruct` / 1.5B | Very slow; use `configs/cpu_*.yaml` with `requirements-cpu.txt`. |

The runner loads **one model at a time** and evicts it before the next family.
That is why a 3-model cell fits in 16–24 GB. Do not raise
`backend.hf.max_loaded_models` unless you have spare VRAM.

### 4.2 Install

**Local GPU (own machine):**

```bash
cd pilot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
pip install -e . --no-deps
```

**Kaggle GPU:** use `requirements-kaggle.txt` and the `configs/kaggle_*.yaml`
configs, or run [`notebooks/kaggle_run.ipynb`](notebooks/kaggle_run.ipynb). See
[`docs/KAGGLE.md`](docs/KAGGLE.md).

**CPU later:** `pip install -r requirements-cpu.txt` and use
`configs/cpu_medqa.yaml` / `configs/cpu_pubmedqa.yaml` (no 4-bit, small models,
small limits by default).

If you already have a working CUDA PyTorch, install it first and then install
the remaining packages without letting pip replace it.

### 4.3 Run one benchmark

Each config points at `data/cases/<dataset>_300.jsonl` and runs H0–H5 + SYC.

```bash
# MedQA
python -m bts_audit.cli doctor --config configs/pilot_medqa.yaml
python -m bts_audit.cli run --config configs/pilot_medqa.yaml --resume

# PubMedQA (the only genuine evidence partition)
python -m bts_audit.cli doctor --config configs/pilot_pubmedqa.yaml
python -m bts_audit.cli run --config configs/pilot_pubmedqa.yaml --resume

# MMLU-Med
python -m bts_audit.cli doctor --config configs/pilot_mmlu_med.yaml
python -m bts_audit.cli run --config configs/pilot_mmlu_med.yaml --resume

# MedMCQA (validation split)
python -m bts_audit.cli doctor --config configs/pilot_medmcqa.yaml
python -m bts_audit.cli run --config configs/pilot_medmcqa.yaml --resume

# MedXpertQA (10 options)
python -m bts_audit.cli doctor --config configs/pilot_medxpertqa.yaml
python -m bts_audit.cli run --config configs/pilot_medxpertqa.yaml --resume
```

Or use the wrapper:

```bash
DATASET=medqa bash scripts/run_pilot.sh
DATASET=pubmedqa LIMIT=60 CELLS=H1,H4,H5,SYC bash scripts/run_pilot.sh
```

The wrapper prepares `data/cases/<dataset>_300.jsonl` (the path in the config)
and then runs it. `LIMIT` and `CELLS` are forwarded to the run command.

**Cheaper first pass.** Before a full sweep:

```bash
python -m bts_audit.cli run \
  --config configs/pilot_medqa.yaml \
  --limit 60 --cells H1,H4,H5,SYC
python -m bts_audit.cli analyze --run results/medqa_300
```

Use `--resume` to continue an interrupted run. Answers and forecasts are written
incrementally, so a crash does not lose completed cases.

### 4.4 Expected runtime

A full 300-case run:

- H0–H4 and SYC: 6 cells × 5 agents × 300 cases = 9,000 answer calls.
- H5: 2 rounds × 5 agents × 300 cases = 3,000 answer calls.
- Forecasts: 7 cells × 5 agents × 300 cases = 10,500 calls.
- Total: about **22,500 calls**.

On a 24 GB GPU with 4-bit 3B–7B models, expect roughly **6–20 hours**,
depending on model and generation length. The 60-case first pass is about
**1–3 hours**. Reductions that keep the design intact: fewer cases, fewer cells,
or `max_new_tokens: 160`.

### 4.5 Ollama alternative

If you already run Ollama locally:

```bash
ollama pull qwen2.5:3b
ollama pull llama3.1:8b
ollama pull phi3.5
python -m bts_audit.cli run --config configs/pilot_medqa_ollama.yaml
```

Ollama still runs locally; no API key is involved. The HF backend is the primary
path because it controls quantization and seeds more tightly.

---

## 5. Output files

For a run directory such as `results/medqa_300/`:

| File | Contents |
|---|---|
| `report.md` | Human-readable verdict, P1–P8 scorecard, per-cell table, data limitations. |
| `predictions.json` | Machine-readable P1–P8 scorecard. |
| `metrics.json` | All per-cell metrics. |
| `cell_metrics.csv` | The per-cell table. |
| `agent_scores.csv` | One row per agent per case, with every score component. |
| `answers.jsonl` | Raw answer calls (including failures). |
| `forecasts.jsonl` | Raw forecast calls. |
| `cases.jsonl` | The exact cases used in the run. |
| `run_meta.json` | Config, package versions, platform, cell plans. |
| `config_used.yaml` | The merged config, including CLI overrides. |

Resume reads `answers.jsonl` and `forecasts.jsonl`; keep them.

---

## 6. The scorecard

| # | What it checks | Pass condition |
|---|---|---|
| **P1** | H0 consensus collapse | Agreement ≥ 0.95 and score variance ≈ 0. |
| **P2** | H1: sampling noise only | ΔAUC = AUC(BTS) − AUC(confidence) ≤ 0.10. Negative is the expected popularity effect. |
| **P3** | H2: role prompts only | ΔAUC ≤ 0.10. |
| **P4** | H3: model families only | ΔAUC ≤ 0.15. |
| **P5** | H4: evidence slices | ΔAUC ≥ 0.15 and minority-truth rate ρ rises versus H3. |
| **P6a** | Rarity premium | Mean within-case slope of the information score on answer rarity ≥ 0.05 nats/nat. |
| **P6b** | Oracle discrimination gap | Oracle same-frequency gap on H0–H3 ≈ 0 (mechanical check). The elicited gap is a calibration diagnostic. |
| **P7** | Sycophancy control | Under pressure, BTS selection accuracy exceeds majority and its flip rate is not materially worse (≤ 0.05). |
| **P8** | Oracle-forecast ablation | Replacing elicited forecasts with leave-one-out empirical forecasts drives ΔAUC to ≈ 0 in H1–H3, but not in H4. |

**Decision rule.** GO if P1, P2, P6a and P8 hold and P5 shows recovery. If P5
fails, the paper is the negative result plus the monitoring tool, which is a
workshop paper. If P2 fails (BTS tracks truth under shared evidence), assumption
A1 is violated and the paper pivots to "BTS measures forecast miscalibration".

### Reading ΔAUC correctly

ΔAUC is **one-sided**. The theory predicts BTS will often be *worse* than
confidence when the team is accurate: the correct answer is common, so the
rarity statistic penalises it. A large negative ΔAUC is evidence *for* the
theory, not a failed run. What would falsify the theory is a large **positive**
ΔAUC in H1–H3 that survives the oracle ablation.

### The two rarity numbers

- **Rare premium (elicited):** within-case slope of the actual BTS information
  score on rarity. This is the mechanical premium. It says nothing about whether
  rare *diseases* are better diagnosed.
- **Oracle gap:** the same-frequency gap when forecasts are replaced by the
  empirical answer distribution. Proposition 1 makes this identically zero at a
  fixed answer count. A non-zero value is a bug, not a finding. The **elicited
  gap** is the diagnostic: negative means the score penalises correct answers at
  fixed frequency (popularity effect); positive means forecasts inject a truth
  signal that must be checked against P8.

P6a and P6b are mechanical properties of the score. **Disease-prevalence claims
are not testable on this suite**, and the report says so.

---

## 7. Configuration

Configs live in `configs/`:

| Config | Backend | Run directory / notes |
|---|---|---|
| `pilot_medqa.yaml` | HF | `results/medqa_300` (local GPU) |
| `pilot_pubmedqa.yaml` | HF | `results/pubmedqa_300` (local GPU) |
| `pilot_mmlu_med.yaml` | HF | `results/mmlu_med_300` |
| `pilot_medmcqa.yaml` | HF | `results/medmcqa_300` |
| `pilot_medxpertqa.yaml` | HF | `results/medxpertqa_300` |
| `kaggle_medqa.yaml` | HF | `/kaggle/working/results/kaggle_medqa` (T4/P100, 4-bit float16) |
| `kaggle_pubmedqa.yaml` | HF | `/kaggle/working/results/kaggle_pubmedqa` |
| `kaggle_mmlu_med.yaml` | HF | `/kaggle/working/results/kaggle_mmlu_med` |
| `kaggle_medmcqa.yaml` | HF | `/kaggle/working/results/kaggle_medmcqa` |
| `kaggle_medxpertqa.yaml` | HF | `/kaggle/working/results/kaggle_medxpertqa` |
| `cpu_medqa.yaml` | HF | `results/cpu_medqa` (CPU, small models, small default scope) |
| `cpu_pubmedqa.yaml` | HF | `results/cpu_pubmedqa` |
| `pilot_medqa_ollama.yaml` | Ollama | `results/medqa_ollama` |
| `mock_medqa.yaml` | mock | `results/mock_medqa` |

`python -m bts_audit.cli run` accepts `--data PATH` and `--out DIR` to override
`data.path` and `output_dir` without editing the config (useful on Kaggle, e.g.
`--data /kaggle/working/data/cases/pubmedqa_300.jsonl`).

The important fields:

```yaml
run_name: my_run
seed: 13

data:
  dataset: medqa
  path: data/cases/medqa_300.jsonl
  limit: 300

backend:
  type: hf                 # mock | hf | ollama
  hf:
    device_map: auto
    dtype: auto
    load_in_4bit: true
    max_loaded_models: 1

team:
  agents: 5
  models: [Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct, microsoft/Phi-3.5-mini-instruct]
  roles: [attending physician, internist, specialist, radiologist, resident]
  temperature: 0.7
  forecast_temperature: 0.0
  max_new_tokens: 384
  rounds: 2

cells: [H0, H1, H2, H3, H4, H5, SYC]

scoring:
  variant: geometric        # geometric (Prelec) or arithmetic
  forecast_smoothing: 0.01  # 1% uniform prior; keeps log scores finite

output_dir: results/my_run
```

Notes:

- `team.models` is cycled across agents for H3–H5. For H4 the evidence slices are
  also cycled; with 5 agents and 5 slices, each agent gets one slice. PubMedQA
  abstracts with fewer than 5 sections reuse sections; the runner prints a
  warning.
- `team.roles` is cycled for H2.
- `rounds` is used by H5 (minimum 2). Other cells are single-round.
- `forecast_smoothing` is applied to both elicited and oracle forecasts. It is a
  numerical floor, not a modelling choice. Do not set it to 0 with real LLMs:
  one exact-zero probability makes the log score infinite.
- `scoring.variant: arithmetic` measures pure rarity (the disagreement term
  vanishes). Use it as a robustness check, not as the main result.

---

## 8. Adding a model, dataset, or backend

### Add a model

Put the HuggingFace repo id in `team.models`. For gated models, log in first:

```bash
huggingface-cli login
```

### Add a dataset

The loaders are in `src/bts_audit/data/loaders.py`. A loader must return `Case`
objects with `case_id`, `question`, `options`, `answer`, and `evidence_slices`.
`rarity_stratum` and `prevalence` are optional; leave them `None` unless the
source supplies a real population prior. See `data/README.md` for the schemas.

`--dataset generic` is an optional escape hatch for a user-supplied
question/answer file. It is not part of the pilot suite.

### Add a backend

Implement `Backend.generate(GenerateRequest) -> GenerateResult` in
`src/bts_audit/backends/`, then register it in `backends/__init__.py`. The mock
backend is the reference for deterministic behaviour.

---

## 9. Troubleshooting

**CUDA out of memory.** Keep `max_loaded_models: 1`, lower `max_new_tokens`, use
`load_in_4bit: true`, or use smaller models. A 24 GB card should fit one 7B model
in 4-bit with room for the KV cache.

**`torch_dtype` deprecation warnings.** Harmless on current `transformers`. If
your version rejects `torch_dtype`, replace it with `dtype` in
`backend/hf_local.py`.

**The model does not follow JSON.** The parser is deliberately tolerant: it
extracts the first balanced JSON object, maps letters or option text, and falls
back to a one-hot answer when only a letter is present. Check `answers.jsonl` for
`error` fields. If the parse rate is low, shorten the prompt or use a more
instruction-tuned model.

**No chat template.** Some base models do not define one. Use an Instruct model,
or add a fallback in `HFBackend.generate`.

**`doctor` says `tqdm: MISSING`.** It is optional. Progress bars disappear; the
run continues.

**The report says `SMOKE ONLY`.** You ran the mock backend. That is correct.
Point `backend.type` at `hf` or `ollama` for a scientific run.

**A run was interrupted.** Re-run the same command with `--resume`.

**P6b says `n/a`.** No case had both a correct and a wrong answer at the same
answer frequency, so the gap is not estimable. That is normal in H1/H2 when the
team is nearly always right. The oracle check uses the cells that do have
matched pairs.

**The MMLU loader finds no files.** Confirm the files are named
`<subject>_test.parquet` under `data/raw/mmlu`, or pass `--subjects` with a
comma-separated list.

**MedMCQA raises "no labelled rows".** You pointed it at the test split. Use
`data/raw/medmcqa/validation.parquet`.

**PubMedQA raises "pqa_artificial is generated".** Use `pqa_labeled`.

---

## 10. What is deliberately not here

- **No paid API calls.** No OpenAI, Anthropic, or hosted inference.
- **No synthetic, constructed, or LLM-restructured datasets.** The pilot uses
  only the five human-written benchmarks above.
- **No fine-tuning.** The pilot audits a fixed team; it does not train one.
- **No equilibrium claim.** BTS is evaluated as an ex post statistic, not as a
  truthful-reporting mechanism for finite teams.
- **No clinical use.** Benchmark cases only. This is a measurement study, not a
  diagnostic device.
- **No severity weighting in the score.** The theory says an additive
  severity-weighted proper score does not exist. The fix is to elicit beliefs
  with a proper score and make the severity-weighted decision separately. The
  pilot measures the audit; it does not implement the clinical decision layer.

---

## 11. Project layout

```
pilot/
  src/bts_audit/
    cli.py            command line: prepare, run, analyze, doctor, list-cells
    config.py         defaults and validation
    schemas.py        Case and AgentRecord
    prompts.py        prompts + tolerant JSON/answer/forecast parsing
    team.py           H0–H5/SYC cell plans, evidence slices, pressure
    runner.py         execution loop, resume, metadata
    scoring.py        BTS geometric/arithmetic, h_G, oracle forecasts, selection
    analysis.py       P1–P8 metrics, bootstrap/DeLong, report generation
    backends/         mock, HuggingFace, Ollama
    data/             loaders for the five benchmarks plus the generic escape hatch
  configs/            pilot, Kaggle, CPU, Ollama, and mock configs
  notebooks/          kaggle_run.ipynb
  scripts/            smoke_test.sh, run_pilot.sh, download_data.py
  tests/              41 unittest tests (stdlib only)
  data/README.md      dataset schemas, layouts, licenses
  docs/PILOT_GUIDE.md 2-week run plan and go/no-go decision tree
  docs/KAGGLE.md      Kaggle (GPU) and CPU run guide
  results/            generated runs (git-ignored)
```

---

## 12. Tests

```bash
cd pilot
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests cover the BTS formulas (`h_G`, the decomposition, consensus collapse,
the numerical example from the theory note), parsing, selection rules, AUC and
DeLong, the bootstrap, calibration, the within-case rarity slope, the
same-frequency gap, all five dataset loaders, and an end-to-end mock run.

---

## 13. Citation and data

The theory note is `../theory_prop1_prop2.md`. The repository and provenance map
is `../repos_and_datasets.md`. Data licenses and raw layouts are in
`data/README.md`. Record each dataset's download date and the analysis code
version in your paper appendix.
