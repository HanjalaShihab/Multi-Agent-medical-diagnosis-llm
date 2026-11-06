# Two-week pilot guide

A practical plan for running the audit on one GPU PC, with a go/no-go decision at
the end of week 2. No API key, no paid service.

The scientific claim is in `../../theory_prop1_prop2.md`. This guide is about
execution.

---

## 0. Before you start: freeze the analysis

Write these down and do not change them after seeing results:

- **Primary evidence-partition dataset:** PubMedQA, 300 cases.
- **Primary replication dataset:** MedQA, 300 cases.
- **Secondary datasets:** MMLU-Med, MedMCQA, MedXpertQA, 300 cases each.
- **Primary cells:** H1, H2, H3, H4, H5, SYC; H0 is a control.
- **Primary endpoint:** ΔAUC = AUC(BTS) − AUC(confidence) in H1–H3.
- **Mechanism endpoints:** within-case rarity slope (P6a) and oracle gap (P6b).
- **Ablation:** P8, oracle forecasts.
- **Gate:** P1, P2, P6a, P8 plus **P5 recovery on PubMedQA**.
- **Seed:** 13 for the pilot; 3 seeds before the paper.
- **Thresholds:** P2/P3 ΔAUC ≤ 0.10; P4 ≤ 0.15; P5 ΔAUC ≥ 0.15 and ρ rises;
  rarity premium ≥ 0.05; oracle gap ≤ 0.10; oracle ΔAUC threshold 0.05.
- **The decision rule below.**

Changing the endpoint after seeing the data turns a test into a story.

### The five human-written benchmarks

| Dataset | Download | Why it is in the suite |
|---|---|---|
| **PubMedQA** (`pqa_labeled`) | `scripts/download_data.py pubmedqa` | 1,000 human-annotated yes/no/maybe items with **real abstract sections**. The only genuine H4/H5 evidence partition. |
| **MedQA-USMLE** | `scripts/download_data.py medqa` | 1,273 human-written 4-option USMLE items. Primary replication for H1–H3/SYC. |
| **MMLU-Med** | `scripts/download_data.py mmlu` | 1,111 human-written items across six medical subjects. Breadth check. |
| **MedMCQA** | `scripts/download_data.py medmcqa` | 4,183 human-written validation items. Uses validation because the test split withholds labels. |
| **MedXpertQA** | `scripts/download_data.py medxpertqa` | 2,450 expert-written text items with up to 10 options. Hardest set. |

**Excluded on purpose:** DDXPlus (synthetic simulator), MedLA/MedDDx
(constructed from a knowledge base), geteff1 (LLM-restructured derivative), and
all built-in toy data. None is human-written benchmark data. BioASQ is optional
and manual only.

State this provenance explicitly in the paper.

### Two facts that constrain the claims

1. **No disease prevalence.** `rarity_stratum` and `prevalence` are `None` for
   every benchmark. P6a/P6b are mechanical properties of the BTS score. Claims
   about disease rarity or prevalence are **not testable on this suite**.
2. **Only PubMedQA has real evidence sections.** MedQA, MMLU-Med, MedMCQA, and
   MedXpertQA are text-only. Their five slices are contiguous chunks of the
   question stem; H4/H5 there are a **text-chunk ablation**, not evidence
   partitioning. The strong P5 claim rests on PubMedQA.

---

## 1. Compute budget

A full 300-case run is about 22,500 model calls: 12,000 answer calls (H5 adds a
second round) and 10,500 forecast calls.

| Run | Cases | Cells | Calls | Time on 24 GB, 4-bit |
|---|---|---|---|---|
| Smoke (mock, real MedQA subset) | 60 | 7 | 0 GPU | 2 min |
| Tiny real-model parse check | 6 | 1 (H1) | ~72 | 10–30 min |
| First pass, one dataset | 60 | 4 (H1,H4,H5,SYC) | ~2,700 | 1–3 h |
| Full pilot, one dataset | 300 | 7 | ~22,500 | 6–20 h |
| Five datasets, one seed | 1,500 | 7 | ~112,500 | 30–100 h |
| Five datasets, two seeds | 3,000 | 7 | ~225,000 | 60–200 h |

The two-week plan below fits one seed on all five datasets plus a second seed on
PubMedQA and MedQA. If time is tight, run 60-case first passes on the three
secondary datasets and reserve full runs for PubMedQA and MedQA.

---

## 2. Week 1: make the pipeline trustworthy

### Day 1 — environment and real-data smoke test

```bash
cd pilot
pip install -r requirements.txt
pip install -e . --no-deps
bash scripts/smoke_test.sh
```

Check:

- all tests pass;
- `results/mock_medqa/report.md` exists and says **SMOKE ONLY**;
- P1 is consistent (consensus collapse);
- the per-cell table has no blank columns.

If the smoke test fails, fix that before touching a GPU. Do not debug model
inference and scoring at the same time.

### Day 2 — download and prepare all five benchmarks

```bash
python scripts/download_data.py all --out data/raw

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

For each dataset, check the stats file under `data/cases/*.stats.json`:

- 300 cases;
- option counts match the benchmark (MedQA 4, PubMedQA 3, MedMCQA 4,
  MedXpertQA mostly 10);
- `slice_kinds` is `abstract_sections` only for PubMedQA and `text_chunks` for
  the rest;
- `rarity_stratum` is `unknown` and `prevalence` is absent/`None` (this is
  correct);
- the gold answer is in the options.

Spot-check five cases per dataset by eye. For PubMedQA, inspect
`metadata.section_labels` and the bucket assignments. For MedXpertQA, confirm
the inline "Answer Choices:" block was stripped.

### Day 3 — tiny real-model parse check

Install the GPU stack:

```bash
pip install -r requirements-gpu.txt -r requirements.txt
pip install -e . --no-deps
```

Run a six-case, one-cell pass on the real backend:

```bash
python -m bts_audit.cli doctor --config configs/pilot_medqa.yaml
python -m bts_audit.cli run \
  --config configs/pilot_medqa.yaml \
  --limit 6 --cells H1
```

This is slow on a small card. Its purpose is to catch prompt/parse problems:
missing JSON, answers that do not map to an option, forecasts that do not sum to
1, chat-template failures. Inspect `answers.jsonl` and `forecasts.jsonl`.

Target: **> 90% parse rate** for answers and forecasts. If it is lower, shorten
the prompt or switch to a more instruction-tuned model before scaling.

### Day 4 — first real pass: MedQA

```bash
python -m bts_audit.cli run \
  --config configs/pilot_medqa.yaml \
  --limit 60 --cells H1,H4,H5,SYC
python -m bts_audit.cli analyze --run results/medqa_300
```

Why these cells: H1 is the negative control, H4/H5 are the positive candidates,
SYC is the mechanism check. MedQA's H4 is a text-chunk ablation, so treat any
recovery here as weak evidence.

Watch:

- answer and forecast parse rates;
- per-cell answer counts;
- whether forecasts are near-uniform or copy the model's own answer;
- whether H5 deliberation collapses the team.

### Day 5 — first real pass: PubMedQA (the decisive dataset)

```bash
python -m bts_audit.cli run \
  --config configs/pilot_pubmedqa.yaml \
  --limit 60 --cells H1,H4,H5,SYC
python -m bts_audit.cli analyze --run results/pubmedqa_300
```

This is the only dataset where H4/H5 test real evidence partitioning. Check the
section-reuse rate in `cases.jsonl`: abstracts with fewer than five sections give
some agents the same section. Record it; do not hide it.

If BTS cannot separate H1 from H4 here, the full run will not rescue the positive
claim.

### Day 6 — forecast quality and scorer sanity

Forecast quality decides whether P8 is interpretable. For each cell, record:

- forecast parse rate;
- mean Brier score against the leave-one-out answer distribution;
- mean Jensen–Shannon divergence from that distribution;
- ECE of the forecast distributions (`forecast_soft_ece`);
- whether forecasts are close to uniform or close to one-hot.

If forecasts are near-uniform, BTS's disagreement term is noise. If they are
near-one-hot on the model's own answer, BTS is just confidence in disguise. Both
are findings, not just plumbing.

Run the tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

All 41 tests should pass. Confirm:

- `h_G` is decreasing in `u`;
- the geometric and arithmetic variants differ only through the disagreement
  term;
- a case with identical answers has zero score variance;
- the oracle forecasts make the same-frequency gap zero.

If any of these fail, stop. The analysis is not ready.

### Day 7 — buffer

Use it for whatever broke. Something will: a model that emits prose, a tokenizer
without a chat template, a dataset column rename, an OOM at 7B, or a CUDA driver
mismatch.

---

## 3. Week 2: run the pilot and decide

### Day 8 — full runs: PubMedQA and MedQA

```bash
python -m bts_audit.cli run --config configs/pilot_pubmedqa.yaml --resume
python -m bts_audit.cli analyze --run results/pubmedqa_300

python -m bts_audit.cli run --config configs/pilot_medqa.yaml --resume
python -m bts_audit.cli analyze --run results/medqa_300
```

Do not watch the loss. Watch the parse rate and the per-cell answer counts.

### Day 9 — secondary datasets

Run full 300-case sweeps on MMLU-Med, MedMCQA, and MedXpertQA. If time is tight,
use 60-case first passes and say so.

```bash
python -m bts_audit.cli run --config configs/pilot_mmlu_med.yaml --resume
python -m bts_audit.cli run --config configs/pilot_medmcqa.yaml --resume
python -m bts_audit.cli run --config configs/pilot_medxpertqa.yaml --resume

python -m bts_audit.cli analyze --run results/mmlu_med_300
python -m bts_audit.cli analyze --run results/medmcqa_300
python -m bts_audit.cli analyze --run results/medxpertqa_300
```

These are replication and boundary checks. Their H4/H5 results are text-chunk
ablations, not evidence partitioning.

### Day 10 — ablations

1. **Oracle (P8).** Already computed in the report. If H1–H3 ΔAUC does not
   collapse under oracle forecasts, the apparent signal is real information, and
   the theory's A1 is violated for this team. That is a different paper.
2. **Scoring variant.** Re-run the analysis with `--variant arithmetic`. The
   rarity pattern should survive; the disagreement pattern should shrink. If it
   does not, the geometric aggregation is doing the work and you need to say so.
3. **Forecast smoothing.** Re-run with `forecast_smoothing: 0.001` and `0.05` in
   a copy of the config. The sign of the main result should not change.
4. **Case subset.** Recompute on first-half/second-half or by subject. A result
   that exists only in one slice of the benchmark is a sampling confound.

### Day 11 — second seed on the decisive datasets

Change `seed` and `output_dir` in copies of `pilot_pubmedqa.yaml` and
`pilot_medqa.yaml`, prepare new case files, and rerun. You are looking for the
**sign** of ΔAUC and the **direction** of ρ, not identical decimals. If the sign
flips between seeds, the effect is not stable enough for a paper.

### Days 12–13 — analysis and figures

Minimum figure set:

1. ΔAUC by cell, with case-level bootstrap intervals.
2. AUC(score) and AUC(confidence) by cell, paired.
3. ρ by cell, with majority accuracy as the reference line.
4. Within-case rarity slope by cell (elicited) and the oracle control.
5. The same-frequency gap: oracle (should be zero) and elicited (diagnostic).
6. P8: ΔAUC with elicited versus oracle forecasts, H1–H5.

Tables:

- per-cell accuracy, agreement, ΔAUC, ρ, ECE;
- P1–P8 scorecard;
- forecast quality (Brier, JS, ECE);
- selection accuracy by rule (majority, confidence, BTS, proper log score);
- dataset provenance and the PubMedQA section-reuse rate.

### Day 14 — write and decide

Use the decision tree below. Write the negative result first; it is the paper's
spine. The positive PubMedQA result is a boundary condition, not the headline.

---

## 4. Go/no-go decision tree

```
P1 passes?
├─ no  → the harness or the model is broken. Do not interpret anything.
└─ yes
   P8 passes?  (oracle drives H1–H3 ΔAUC to ~0)
   ├─ no  → BTS carries real information under shared evidence.
   │        A1 is violated for this team.
   │        Pivot: "BTS measures private information / forecast miscalibration".
   │        Still publishable, but not the rarity paper. Re-derive before writing.
   └─ yes
      P2 passes?  (H1 ΔAUC ≤ 0.10; negative is fine)
      ├─ no  → sampling noise alone creates a truth signal.
      │        Check forecast elicitation; this is usually a parser or prompt bug.
      └─ yes
         P6a passes?  (positive within-case rarity slope)
         ├─ no  → the score is not doing what Proposition 1 says.
         │        Check the scorer and the forecast smoothing before interpreting.
         └─ yes
            P5 recovers on PubMedQA?  (ΔAUC ≥ 0.15 and ρ rises)
            ├─ yes → GO. The story is:
            │        rarity under shared evidence, recovery under partitioned evidence.
            │        Run 3 seeds, then write.
            └─ no  → NO-GO for the positive claim.
                     The paper is the negative result + the monitor:
                     "BTS cannot be trusted as a clinical consensus rule;
                     here is the decomposition and the audit."
                     Workshop paper, or a strong methods section in a larger paper.
```

### What each outcome means for the paper

| Outcome | Paper |
|---|---|
| P1, P2, P6a, P8 pass; P5 recovers on PubMedQA | **Main paper.** Rarity audit + evidence-partition fix. |
| P1, P2, P6a, P8 pass; P5 fails | **Negative result.** BTS is a popularity statistic; partition does not rescue it at this scale. |
| P5 recovers only on text-chunk datasets | **Boundary result.** Suggestive, not evidence partitioning. Do not headline it. |
| P8 fails in H1–H3 | **Pivot.** BTS is picking up real information; the common-evidence assumption is wrong for LLM teams. |
| P2 fails | **Bug or A3 violation.** Do not write until you know which. |
| P6a fails | **Scorer bug.** The theory's most basic numerical prediction is not holding. |
| P7 fails | **Boundary condition.** Sycophancy is not enough to make the minority truthful. Report it; it weakens the "BTS looks good under bias" story. |

---

## 5. Pre-registration checklist

Before the full run, write a one-page note with:

- [ ] Dataset names, public sources, licenses, sample size, seed.
- [ ] Models, quantization, and generation settings.
- [ ] Cells and the exact manipulation in each.
- [ ] Primary endpoint and its direction.
- [ ] Secondary endpoints (ρ, rarity slope, gap, ECE, selection accuracy).
- [ ] Ablations (oracle, arithmetic, smoothing, subsets).
- [ ] Exclusion rules (parse failures, missing forecasts, invalid answers).
- [ ] The gate and the decision tree above.
- [ ] The statement that only PubMedQA supplies a genuine evidence partition.
- [ ] The statement that disease-prevalence claims are not testable here.
- [ ] The seed and the analysis code version.

Put it in the paper appendix. It is cheap and it protects the result.

---

## 6. Common ways the pilot can mislead you

1. **Uniform forecasts.** BTS then measures answer frequency only. Check
   `forecast_js` and `forecast_soft_ece`.
2. **Forecasts that copy the model's own answer.** BTS becomes a confidence
   proxy. The oracle ablation will show it.
3. **A team that is almost always right.** Correct answers are common, so ΔAUC is
   negative for a boring reason. That is the theory working, not a failure; say
   so.
4. **A team that is almost always wrong.** The minority is truth by accident.
   Check ρ and the sycophancy cell before claiming a truth signal.
5. **A single model family in H3.** If all three "families" are fine-tunes of the
   same base, H3 is H1 with extra steps.
6. **Leaking the gold answer into the prompt.** Do not include the gold option
   text in the evidence separately from the option list. Check the prepared
   `cases.jsonl`.
7. **Overlapping evidence slices.** H4 is only H4 if the slices are disjoint. On
   text-only benchmarks they are chunks of one stem; do not interpret an H4
   result there as evidence partitioning. On PubMedQA, sections are disjoint but
   may repeat when an abstract has fewer than five sections.
8. **Judging by accuracy alone.** BTS can improve accuracy while being
   anti-proper. The paper is about the mechanism, not a leaderboard.
9. **Treating MedMCQA validation as the official test set.** It is the public
   labelled split; say so.
10. **Reading P6a/P6b as disease-prevalence results.** They are mechanical
    properties of the score on answer frequency.

---

## 7. Minimum publishable package

- Theory note with the propositions and the impossibility result.
- The pilot on the five human-written benchmarks: PubMedQA (genuine evidence
  partition) plus MedQA, MMLU-Med, MedMCQA, and MedXpertQA (replication and
  text-chunk boundary checks), two seeds on the decisive datasets.
- P1–P8 scorecard for every dataset, with the text-chunk and validation-split
  caveats stated.
- One figure showing the rarity premium and the recovery under evidence
  partitioning.
- One figure showing the oracle ablation.
- A limitations section that names A1, A2, finite-team effects, text-chunk
  pseudo-partitions, PubMedQA section reuse, the absence of a prevalence prior,
  and the difference between posterior sampling and temperature sampling.

If the positive claim fails, the same package is a negative result. Do not throw
it away; the audit and the decomposition are the contribution.
