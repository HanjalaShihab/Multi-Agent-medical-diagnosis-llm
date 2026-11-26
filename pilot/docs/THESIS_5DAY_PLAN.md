# 5-day thesis completion plan (Bachelor, DIU SWE)

Deadline: **30 November 2026**. Degree: **Bachelor of Science** (black spiral binding, ≤30,000 words, plagiarism ≤40%, single source <3%).

## Decision: limit the scope honestly

Do **not** try all 25 full runs (5 models × 5 datasets). The defendable, finishable scope is:

| Role | Dataset | Cases | Cells | Output |
|---|---|---:|---|---|
| **Primary** | PubMedQA (`pqa_labeled`) | 300 | H0,H1,H2,H3,H4,H5,SYC | `results/colab_pubmedqa_small/` |
| Supporting 1 | MedQA (test) | 60 | H1,H4,H5,SYC | `results/colab_medqa_support/` |
| Supporting 2 | MedMCQA (validation) | 60 | H1,H4,H5,SYC | `results/colab_medmcqa_support/` |
| Not run | MMLU-Med, MedXpertQA | — | — | mention as future work |

Model team on all runs: `Qwen/Qwen2.5-0.5B-Instruct` + `Qwen/Qwen2.5-1.5B-Instruct` (cycled `0.5, 1.5, 0.5`), 5 agents, no API key.

Say this explicitly in Chapter 3 and Chapter 5: **thesis is a pilot/audit, not a SOTA diagnostic system; small open-weight models; the two supporting datasets are reduced-case probes, not full benchmark claims.**

## Compute (do this first, let it run in background)

1. Push the new files from the laptop:
   ```bash
   cd "/home/hanjala-shihab/Documents/Multi-Agent medical diagnosis llm"
   git add pilot/README.md \
     pilot/configs/colab_*_support.yaml \
     pilot/scripts/run_colab_thesis_scope.sh \
     pilot/scripts/md_to_thesis_docx.py \
     pilot/notebooks/colab_thesis_scope.ipynb \
     pilot/docs/THESIS_5DAY_PLAN.md \
     pilot/docs/COLAB.md \
     pilot/thesis
   git commit -m "Thesis-scope runs and DIU draft"
   git push origin main
   ```
   Do **not** use `git add -A`: the PDFs/template/docx in the repo root are untracked and large; the notebook only needs the files above. If earlier Colab configs were not pushed yet, add `pilot/configs/colab_*_small.yaml`, `pilot/notebooks/colab_small_models.ipynb`, `pilot/scripts/run_colab_all_small.sh` too.
2. Open `pilot/notebooks/colab_thesis_scope.ipynb` in Colab → **Runtime → T4 GPU**.
3. Cells: clone/pull → CUDA check → install → prepare data → **mount Drive** (`OUT_ROOT`) → run:
   ```python
   !OUT_ROOT='/content/drive/MyDrive/bts_audit_colab_results' bash scripts/run_colab_thesis_scope.sh
   ```
4. If disconnected: re-run setup cells + the same run cell. `--resume` continues from the last saved record in Drive.
5. Alternative: the laptop's `scripts/run_all_cpu_local.sh` is already running PubMedQA; let it continue, but do **not** depend on it for the deadline. The laptop GPU PC's `gpu_pubmedqa_*` runs are PubMedQA-only.

Expected outputs per run: `report.md`, `cell_metrics.csv`, `metrics.json`, `predictions.json`, `agent_scores.csv`, `answers.jsonl`, `forecasts.jsonl`, `figures/*.png`.

### Bringing GPU-PC big-model results into this laptop

`pilot/results/` is git-ignored, so pushing the GPU PC will **not** transfer its metrics. On the GPU PC, after runs finish, run:

```bash
cd pilot
python scripts/collect_results.py --src results --out results_summary \
  --pattern 'gpu_pubmedqa_*_full'
git add pilot/results_summary
git commit -m "Add GPU PubMedQA model comparison summary"
git push origin main
```

Then on this laptop: `git pull origin main`. Open `pilot/results_summary/summary_all.csv` for accuracy/AUC/delta-AUC/agreement/rarity/oracle columns per model, and `pilot/results_summary/<run>/report.md` for the P1-P8 verdict. `INDEX.md` marks any run that still needs `--resume`.

## Daily schedule

| Day | Date | Compute | Writing |
|---|---|---|---|
| 1 | 26 Nov | Start PubMedQA primary on Colab T4; prepare data | Chapter 1 (Introduction): background, problem, RQ1–RQ3, objectives, scope, organisation. Chapter 2 skeleton + related-work table. |
| 2 | 27 Nov | PubMedQA should finish; start MedQA + MedMCQA support | Chapter 3 (Methodology): BTS equations, H0–H5/SYC, P1–P8, sampling, models, metrics, threats to validity. |
| 3 | 28 Nov | All runs finish; export tables/figures | Chapter 4 (Results & Discussion): fill Tables 4.1–4.2 and Figures 4.1–4.2, interpret P1–P8, answer RQs. |
| 4 | 29 Nov | No more runs | Abstract, Chapter 5, References, front matter, format with the DIU template, similarity check. |
| 5 | 30 Nov | Print | Final proofread, spiral binding (**black**, Bachelor), CD label, submission. |

## Claim to defend (do not overclaim)

> Under shared evidence, BTS behaves like an answer-rarity statistic rather than a truth statistic. It recovers truth-signal only when correct answers are minority against genuinely private evidence.

This is an **audit**, so a negative/nuanced result is a valid result. Do not claim 90%+ accuracy or clinical deployment readiness.

## Mandatory caveats to include

- Only PubMedQA has genuine H4 evidence sections; H4/H5 on MedQA/MedMCQA are text-chunk ablations.
- `rarity_stratum`/`prevalence` are `None`; P6a/P6b are mechanical score properties, not disease-prevalence findings.
- Small models + parse failures (`answer_not_mapped`, `no_json`) affect cell metrics; report the count.
- Supporting runs use 60 cases and 4 cells only; no cross-dataset SOTA claim.
- No clinician evaluation, no real patient data, no cost/latency study.

## DIU formatting checklist (from `DIU_Thesis Guidelines.docx`)

- A4, 80 gm white paper, **single-sided**, clean laser print, 2 copies.
- Times New Roman 12 **or** Arial 11; **double spacing** for narrative; single spacing for tables, captions, references.
- Margins: **left 4.0 cm**, top/bottom/right 2.5 cm.
- Page numbers: no punctuation, 1 cm from bottom centre; prelims in lowercase Roman, body in Arabic.
- Tables numbered by chapter (`Table 4.1`), caption **above** table; figures caption **below** figure.
- One citation style throughout (this draft uses APA author–year). Similarity ≤40% (UG), <3% from any single source.
- Front matter: title page, declaration/copyright, supervisor & student declarations, acknowledgements, abstract, ToC, lists of tables/figures/symbols/abbreviations/appendices.
- Cover: full title, full name, "DAFFODIL INTERNATIONAL UNIVERSITY", year; gold lettering; black buckram/rexine with stiff boards for Bachelor; spine has title, name, degree, year. CD label: name, title, year, degree.

## Draft

`pilot/thesis/THESIS_DRAFT.md` and `pilot/thesis/BTS_Thesis_Draft.docx` are a chapter skeleton with ready-to-edit prose and clearly marked `[FILL: ...]` slots for numbers. Start from the `.docx`, then paste your final numbers/tables/figures into it.
