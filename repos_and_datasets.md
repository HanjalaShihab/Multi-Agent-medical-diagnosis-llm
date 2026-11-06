# Repositories and datasets you can actually reuse

**Compiled:** 2026-11-03. **Updated:** 2026-11-06.
**Method:** every GitHub link was extracted from the 17 papers in the folder,
then checked against the GitHub API (`/repos/{owner}/{repo}`) for stars, last
push, license, and language. Hugging Face datasets were checked against the HF
API and the public `resolve` URLs. Numbers are a snapshot, not a claim about
quality.

Four links in the PDFs were broken across lines or redirected. All are resolved
below:

- `github.com/Ayanami-E/HEAL-` + `KGGen` → **[Ayanami-E/HEAL-KGGen](https://github.com/Ayanami-E/HEAL-KGGen)**
- `github.com/qjdzj/` + `MACD` → **[qjdzj/MACD-Multi-Agent-Clinical-Diagnosis-with-Self-Learned-Knowledge-for-LLM](https://github.com/qjdzj/MACD-Multi-Agent-Clinical-Diagnosis-with-Self-Learned-Knowledge-for-LLM)** (the short URL redirects here)
- `huggingface.co/datasets/dux-` + `tecblic/symptom-disease-dataset` → **[dux-tecblic/symptom-disease-dataset](https://huggingface.co/datasets/dux-tecblic/symptom-disease-dataset)**
- `Teddy-XiongGZ/MIRAGE` redirects to **[gzxiong/MIRAGE](https://github.com/gzxiong/MIRAGE)**

---

## 1. Benchmark suite used by the pilot

The pilot uses **only human-written public benchmarks**. Every item is authored
by humans; none is generated, simulated, or restructured by an LLM. All five are
downloaded from public Hugging Face `resolve` URLs and converted locally.

| Benchmark | Public source | License | Rows | Role | Caveat |
|---|---|---|---|---|---|
| **MedQA-USMLE** | [`GBaker/MedQA-USMLE-4-options`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options), `phrases_no_exclude_test.jsonl` | CC-BY-4.0 | 1,273 | 4-option USMLE-style QA; primary replication for H1–H3/SYC | Text-only; H4 is a text-chunk ablation |
| **PubMedQA** | [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA), `pqa_labeled` | MIT | 1,000 | 3-way yes/no/maybe over real abstract sections; **the only genuine H4/H5 evidence partition** | Some abstracts have fewer than five sections, so agents may share a section |
| **MMLU-Med** | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu), six medical subjects | MIT | 1,111 | Breadth check across six medical subjects | Text-only; H4 is a text-chunk ablation |
| **MedMCQA** | [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa), `validation.parquet` | Apache-2.0 | 4,183 | Broad medical entrance-exam QA | Official test split withholds labels (`cop == -1`); the loader uses validation |
| **MedXpertQA** | [`TsinghuaC3I/MedXpertQA`](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA), `Text/test.jsonl` | MIT | 2,450 | Expert-level QA with up to 10 options | Text split only; the multimodal split is not used |

**No disease prevalence.** Real QA benchmarks do not carry a patient-population
prevalence prior. The loaders leave `rarity_stratum=None` and
`prevalence=None`; they never derive either from answer-option frequency.
P6a/P6b are mechanical properties of the BTS score, not disease-prevalence
tests.

### Download and prepare

```bash
cd pilot
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

The downloader uses the Python standard library, so `huggingface_hub` is not
required. Review each license before redistributing derived case files.

---

## 2. Excluded datasets and why

These were considered and deliberately left out. The exclusion is a data-policy
decision, not a judgement about their usefulness for other work.

| Dataset | What it is | Why it is excluded |
|---|---|---|
| **DDXPlus** | Peer-reviewed **synthetic patient simulator** (~1.3M simulated patients from a Bayesian network) | Not human-written benchmark data. Its structured evidence and prevalence prior are attractive for H4/P6, but using it would mix simulated patients into a human-benchmark audit. |
| **MedLA / MedDDx** | Public **constructed** benchmark built from the STaRK-Prime biomedical knowledge base: entity questions plus embedding-nearest distractors | Constructed, not collected. Many items are entity/basic-science queries, not clinical cases. It is also text-only, so its slices are text chunks. |
| **geteff1 302 cases** | 302 cases **derived from published MEDLINE case reports**, manually curated and restructured into JSON; at least four records contain an LLM reformatting preamble | A derivative, not verbatim source text, and not a human-written benchmark in the sense used here. |
| **RareBench** | Rare-disease benchmark (KDD 2024) | Not in the five-benchmark suite. The pilot's rarity claims are about answer frequency, not disease prevalence, so a rare-disease set would not test the stated mechanism without a prevalence prior. |
| **MIRAGE** | Multi-task medical benchmark | `NOASSERTION` license; read the LICENSE before redistributing derived splits. Not in the suite. |
| **Symptom–Disease** | Small symptom→disease mapping | Too easy for the main result; smoke-test quality only. |
| **BioASQ** | Biomedical QA benchmark | Public mirrors are gated or do not ship the benchmark file. Optional and manual only; not part of the reproducible suite. |

---

## 3. Repositories from the corpus

No code was copied into the pilot; the harness, scorer, and analysis are
original. These repositories informed the design. Licenses matter if you reuse
code later.

| Paper | Repo | Stars | Last push | License | What is in it | Use in this project |
|---|---|---|---|---|---|---|
| MedLA (AAAI-26 oral) | [alexander2618/MedLA](https://github.com/alexander2618/MedLA) | 27 | 2026-10-21 | MIT | Multi-agent harness: reasoning agents + LogicAgent + DecisionMaker; MedQA, PubMedQA, MMLU-med, MedDdx, BioASQ loaders | **Design reference only.** Its evidence-partition and evaluation ideas informed the ladder. MedDDx itself is excluded. |
| MIRAGE benchmark | [gzxiong/MIRAGE](https://github.com/gzxiong/MIRAGE) | 218 | 2026-10-10 | NOASSERTION | Official repo of the MIRAGE benchmark | Not used; license needs review before any reuse. |
| ConfAgents (2508.04915) | [PKU-AICare/ConfAgents](https://github.com/PKU-AICare/ConfAgents) | 15 | 2026-06-24 | **none** | Conformal confidence triage + escalated collaboration; cost accounting | **Re-implement only.** No license = all rights reserved. Useful as a baseline *design*, not as code. |
| DDO (2505.18630) | [zh-jia/DDO](https://github.com/zh-jia/DDO) | 9 | 2026-07-07 | **none** | Symptom-inquiry RL policy + diagnosis; HF adapters `zhjia/DDO` | **Re-implement only.** Good baseline for the "ask fewer questions" arm. |
| Rare-disease conversation (preprint) | [geteff1/Multi-agent-conversation-for-disease-diagnosis](https://github.com/geteff1/Multi-agent-conversation-for-disease-diagnosis) | 90 | 2026-09-23 | MIT | 302 curated rare-disease cases, primary + follow-up consultation; configurable doctor agents and supervisor | **Excluded data.** The case file is an LLM-restructured derivative; do not use it as human-written benchmark data. |
| MAM (ACL-25 Findings) | [yczhou001/MAM](https://github.com/yczhou001/MAM) | 60 | 2026-10-22 | MIT | Role-specialised multi-modal team (GP, specialists, radiologist, assistant, director) | **Prompt/role design reference.** Skip the imaging path for a text pilot. |
| MedChat | [Purdue-M2/MedChat](https://github.com/Purdue-M2/MedChat) | 20 | 2026-10-15 | MIT | Glaucoma reports from fundus images (SwinV2 + SegFormer + role agents + Director) | Not relevant to a text BTS pilot. |
| MACD | [qjdzj/MACD-…](https://github.com/qjdzj/MACD-Multi-Agent-Clinical-Diagnosis-with-Self-Learned-Knowledge-for-LLM) | 10 | 2026-06-10 | MIT | Clinical decision-making dataset + eval framework, ICD mappings, agent prompts | Optional ICD-coded evaluation reference. |
| HEAL-KGGen | [Ayanami-E/HEAL-KGGen](https://github.com/Ayanami-E/HEAL-KGGen) | 2 | 2026-06-24 | MIT | KG-enhanced medical QA evaluation loop | Only relevant if you add a KG-grounding arm. |
| ZODIAC | no repo | — | — | — | HF models only: `ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1`, `shenzhi-wang/Llama3-8B-Chinese-Chat` | No code to reuse. |
| Adaptive CDSS | no repo | — | — | — | Uses HF dataset `dux-tecblic/symptom-disease-dataset` | Dataset only; too easy for the main result. |

**No public repo found** for: KG4Diagnosis, MedSentry, MeDxAgent, MAGDA,
MDTeamGPT, the neuro-ophthalmic paper, and the two surveys. Do not assume code
exists because a paper has a "Code availability" section; the link was absent or
dead in the PDF text.

### License warnings

- **ConfAgents** and **DDO** have **no license file**. Under default copyright,
  that means all rights reserved. Read and re-implement the ideas; do not copy
  files into a released artifact.
- **MIRAGE** is `NOASSERTION`. Open the `LICENSE` file and check whether
  redistribution of derived splits is allowed.
- **MedLA, MAM, MACD, HEAL-KGGen, geteff1** are MIT. Safe to reuse with
  attribution, subject to the data caveats above.
- **MedChat** is MIT but JavaScript-heavy; the useful part is the role/director
  prompt pattern, not the app.

---

## 4. What to reuse, by pilot component

| Pilot component | Reuse | Why |
|---|---|---|
| Team harness (agents, rounds, logging) | **Write it yourself** (this repo) | The audit needs exact control of evidence slices, forecasts, and cell plans. The pilot harness is ~1,500 lines and dependency-light. |
| Role prompts | **MAM** (MIT) | GP / specialist / radiologist / director roles are plausible and reusable with attribution. |
| Evidence partition | **PubMedQA abstract sections** | The only human-written benchmark here with real sections; H4/H5 are genuine only there. |
| Confidence baseline | **ConfAgents** (re-implement) | The conformal triage baseline is the natural "cost-aware" comparison. Do not copy code (no license). |
| ICD-coded evaluation | **MACD** (MIT) | Optional if you want diagnosis-code-level metrics rather than option accuracy. |
| Sycophancy control | Replicate **2608.25267**'s prompt-pressure setup yourself | No public code; the setup is a short prompt template. This is P7. |
| BTS scorer | **Write it yourself** | No repo implements the clinical audit version. See `theory_prop1_prop2.md` §1.2 for the formula. |

---

## 5. A concrete 2-week pilot

The full day-by-day plan is in [`pilot/docs/PILOT_GUIDE.md`](pilot/docs/PILOT_GUIDE.md).
The short version:

**Week 1 — make the pipeline trustworthy**

| Day | Task |
|---|---|
| 1 | Install, run the real-data mock smoke test (`scripts/smoke_test.sh`), confirm P1 and the report. |
| 2 | Download and prepare all five benchmarks; inspect stats, slices, and labels. |
| 3 | Tiny real-model parse check (6 cases, H1); target > 90% answer/forecast parse rate. |
| 4 | First real pass on MedQA (60 cases, H1,H4,H5,SYC). |
| 5 | First real pass on PubMedQA (60 cases, H1,H4,H5,SYC); record section reuse. |
| 6 | Forecast quality audit and scorer sanity checks; all 41 tests pass. |
| 7 | Buffer for failures. |

**Week 2 — run and decide**

| Day | Task |
|---|---|
| 8 | Full PubMedQA and MedQA runs (300 cases, 7 cells). |
| 9 | Full MMLU-Med, MedMCQA, MedXpertQA runs (or 60-case first passes). |
| 10 | Ablations: oracle P8, arithmetic variant, smoothing, subsets. |
| 11 | Second seed on PubMedQA and MedQA. |
| 12–13 | Figures and tables; write the negative result first. |
| 14 | Go/no-go using the tree in `PILOT_GUIDE.md`. |

**Cost.** A full 300-case sweep is about 22,500 calls (12,000 answers + 10,500
forecasts). Five datasets, one seed, is about 112,500 calls; two seeds is about
225,000. On a 24 GB GPU with 4-bit 3B–7B models, budget 30–100 hours for one seed
across all five, or 60–200 hours for two seeds. No API cost.

**What you must build yourself.** There is no public implementation of BTS as a
*clinical audit statistic*. The scorer is trivial; the hard parts are (a)
eliciting forecasts that are actually about teammates rather than about the case,
and (b) constructing evidence partitions with no leakage. Budget most of day 3
and day 5 for those two.

---

## 6. Provenance notes for the paper

Cite the benchmark papers, not just the repositories:

- **MedQA** → Jin et al., 2021, *What Disease does this Patient Have? A
  Large-scale Open Domain Question Answering Dataset from Medical Exams*.
  Dataset card: CC-BY-4.0.
- **PubMedQA** → Jin et al., 2019, *PubMedQA: A Dataset for Biomedical Research
  Question Answering*. MIT. Use `pqa_labeled`, never `pqa_artificial`.
- **MMLU** → Hendrycks et al., 2021, *Measuring Massive Multitask Language
  Understanding*. MIT. Six medical subjects.
- **MedMCQA** → Pal et al., 2022, *MedMCQA: A Large-scale Multi-Subject
  Multi-Choice Dataset for Medical domain Question Answering*. Apache-2.0. This
  pilot uses the validation split because the test split withholds labels.
- **MedXpertQA** → Zuo et al., 2025, *MedXpertQA: Benchmarking Expert-Level
  Medical Reasoning and Understanding*. MIT. Text split only.

Keep a `pilot/data/README.md` record of the exact download date and the analysis
code version. Reviewers in this area have started asking.
