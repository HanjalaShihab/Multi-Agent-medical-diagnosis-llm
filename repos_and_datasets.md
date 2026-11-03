# Repositories and Datasets You Can Actually Reuse

**Compiled:** 2026-11-03
**Method:** every GitHub link was extracted from the 17 papers in the folder, then checked against the GitHub API (`/repos/{owner}/{repo}`) for stars, last push, license, and language. Hugging Face datasets were checked against the HF API. Numbers are a snapshot, not a claim about quality.

Four links in the PDFs were broken across lines or redirected. All are resolved below:

- `github.com/Ayanami-E/HEAL-` + `KGGen` → **[Ayanami-E/HEAL-KGGen](https://github.com/Ayanami-E/HEAL-KGGen)**
- `github.com/qjdzj/` + `MACD` → **[qjdzj/MACD-Multi-Agent-Clinical-Diagnosis-with-Self-Learned-Knowledge-for-LLM](https://github.com/qjdzj/MACD-Multi-Agent-Clinical-Diagnosis-with-Self-Learned-Knowledge-for-LLM)** (the short URL redirects here)
- `huggingface.co/datasets/dux-` + `tecblic/symptom-disease-dataset` → **[dux-tecblic/symptom-disease-dataset](https://huggingface.co/datasets/dux-tecblic/symptom-disease-dataset)**
- `Teddy-XiongGZ/MIRAGE` redirects to **[gzxiong/MIRAGE](https://github.com/gzxiong/MIRAGE)**

---

## 1. Repositories from the corpus

| Paper | Repo | Stars | Last push | License | Language | What is in it | Use for the BTS pilot? |
|---|---|---|---|---|---|---|---|
| MedLA (AAAI-26 oral) | [alexander2618/MedLA](https://github.com/alexander2618/MedLA) | 27 | 2026-10-21 | MIT | Python | Multi-agent harness: reasoning agents + LogicAgent + DecisionMaker; MedQA, PubMedQA, MMLU-med, MedDdx, BioASQ loaders | **Yes — best starting harness.** Fork the agent loop and eval script, replace the aggregation with your scorer. |
| MIRAGE benchmark | [gzxiong/MIRAGE](https://github.com/gzxiong/MIRAGE) | 218 | 2026-10-10 | NOASSERTION | — | Official repo of the MIRAGE benchmark; see README for tasks and splits | **Yes — evaluation set.** Read the LICENSE before redistributing. |
| ConfAgents (2508.04915) | [PKU-AICare/ConfAgents](https://github.com/PKU-AICare/ConfAgents) | 15 | 2026-06-24 | **none** | Python | Conformal confidence triage + escalated collaboration; cost accounting | **Re-implement only.** No license = all rights reserved. Useful as a baseline *design*, not as code. |
| DDO (2505.18630) | [zh-jia/DDO](https://github.com/zh-jia/DDO) | 9 | 2026-07-07 | **none** | Python | Symptom-inquiry RL policy + diagnosis; HF adapters `zhjia/DDO` | **Re-implement only.** Good baseline for the "ask fewer questions" arm. |
| Rare-disease conversation (preprint) | [geteff1/Multi-agent-conversation-for-disease-diagnosis](https://github.com/geteff1/Multi-agent-conversation-for-disease-diagnosis) | 90 | 2026-09-23 | MIT | Python (AutoGen) | **302 curated rare-disease cases**, primary + follow-up consultation; configurable number of doctor agents and supervisor | **Yes — the rare-disease test set.** This is the most directly reusable asset in the whole corpus. Cite their preprint. |
| MAM (ACL-25 Findings) | [yczhou001/MAM](https://github.com/yczhou001/MAM) | 60 | 2026-10-22 | MIT | Python | Role-specialised multi-modal team (GP, specialists, radiologist, assistant, director) | **Yes — prompt templates and role design.** Skip the imaging path for a text pilot. |
| MedChat | [Purdue-M2/MedChat](https://github.com/Purdue-M2/MedChat) | 20 | 2026-10-15 | MIT | JavaScript + Python | Glaucoma reports from fundus images (SwinV2 + SegFormer + role agents + Director) | **No for a text BTS pilot.** Only relevant if you extend to imaging. |
| MACD | [qjdzj/MACD-…](https://github.com/qjdzj/MACD-Multi-Agent-Clinical-Diagnosis-with-Self-Learned-Knowledge-for-LLM) | 10 | 2026-06-10 | MIT | — | Clinical decision-making dataset + eval framework, ICD mappings, agent prompts | **Yes — ICD-coded eval and prompt bank.** Check whether the dataset itself is in the repo or fetched. |
| HEAL-KGGen | [Ayanami-E/HEAL-KGGen](https://github.com/Ayanami-E/HEAL-KGGen) | 2 | 2026-06-24 | MIT | Python | KG-enhanced medical QA evaluation loop | **Maybe** — only if you add a KG-grounding arm. Small repo. |
| ZODIAC | no repo | — | — | — | — | HF models only: `ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1`, `shenzhi-wang/Llama3-8B-Chinese-Chat` | **No code to reuse.** Models only. |
| Adaptive CDSS | no repo | — | — | — | — | Uses HF dataset `dux-tecblic/symptom-disease-dataset` | Dataset only (see §2). |

**No public repo found** for: KG4Diagnosis, MedSentry, MeDxAgent, MAGDA, MDTeamGPT, the neuro-ophthalmic paper, and the two surveys. Do not assume code exists because a paper has a "Code availability" section; the link was absent or dead in the PDF text.

### License warnings (read before you copy)

- **ConfAgents** and **DDO** have **no license file**. Under default copyright, that means all rights reserved. You may read the code and re-implement the ideas; do not copy files into a released artifact.
- **MIRAGE** is `NOASSERTION`. Open the `LICENSE` file and check whether redistribution of derived splits is allowed.
- **MedLA, MAM, MACD, HEAL-KGGen, geteff1** are MIT. Safe to reuse with attribution.
- **MedChat** is MIT but JavaScript-heavy; the useful part is the role/director prompt pattern, not the app.

---

## 2. External datasets the pilot needs

These are not in the corpus but are the right substrates for the BTS experiment. All verified reachable on 2026-11-03.

| Dataset | Link | Size / signal | Why it fits |
|---|---|---|---|
| **DDXPlus** | [github.com/mila-iqia/ddxplus](https://github.com/mila-iqia/ddxplus) · [HF mirror](https://huggingface.co/datasets/aai530-group6/ddxplus) | 138★, last push 2026-11-01; ~1.3M synthetic patients, 51 diseases, structured evidence (symptoms, antecedents, exams) | **Best controlled substrate.** Evidence is structured, so you can partition it into disjoint slices deterministically (H4). Also gives you a prevalence prior, which is exactly what the rarity theory needs. |
| **MedQA (USMLE)** | [github.com/jind11/MedQA](https://github.com/jind11/MedQA) · [HF](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | 408★, MIT; 42,985 HF downloads | Standard 4-option QA. Good for scale and comparability with MedLA/HEAL. Weak for evidence partitioning. |
| **MedMCQA** | [github.com/medmcqa/medmcqa](https://github.com/medmcqa/medmcqa) · [HF](https://huggingface.co/datasets/openlifescienceai/medmcqa) | 305★, MIT; 78,239 HF downloads | Large Indian medical entrance QA; broad disease coverage. Good for prevalence stratification. |
| **RareBench** | [github.com/chenxz1111/RareBench](https://github.com/chenxz1111/RareBench) · [HF](https://huggingface.co/datasets/chenxz/RareBench) | 43★; KDD 2024; 789 HF downloads | Rare-disease benchmark. Use with geteff1's 302 cases for the rare-disease discrimination gap (P6b). |
| **MIRAGE** | [github.com/gzxiong/MIRAGE](https://github.com/gzxiong/MIRAGE) | 218★ | Official benchmark repo (see README for tasks); useful as a third evaluation set. |
| **Symptom–Disease** | [HF](https://huggingface.co/datasets/dux-tecblic/symptom-disease-dataset) | 189 downloads, 16 likes | Small, simple symptom→disease mapping. Only for a smoke test; too easy for the main result. |

**Not usable for this pilot:** the corpora behind KG4Diagnosis, MedSentry, MAGDA, MDTeamGPT, and the neuro-ophthalmic paper are either private or not linked.

---

## 3. What to reuse, by pilot component

| Pilot component | Reuse | Why |
|---|---|---|
| Team harness (agents, rounds, logging) | **MedLA** (MIT) | Already has multi-agent debate, a judge, and MedQA/MedDdx loaders. Strip it to ~300 lines. |
| Role prompts | **MAM** (MIT) | GP / specialist / radiologist / director roles are written out and clinically plausible. |
| Rare-disease cases | **geteff1 302 cases** (MIT) + **RareBench** | The only public rare-disease sets in the corpus. Essential for the rarity result. |
| Controlled evidence slices | **DDXPlus** | Structured evidence can be split into disjoint, non-overlapping slices with no leakage. This is H4. |
| Confidence baseline | **ConfAgents** (re-implement) | The conformal triage baseline is the natural "cost-aware" comparison. Do not copy code (no license). |
| ICD-coded evaluation | **MACD** (MIT) | If you want diagnosis-code-level metrics rather than string match. |
| Sycophancy control | Replicate **2608.25267**'s prompt-pressure setup yourself | No public code; the setup is a short prompt template. This is P7. |
| BTS scorer | **Write it yourself (~100 lines)** | No repo implements the clinical audit version. See `theory_prop1_prop2.md` §1.2 for the exact formula. |

---

## 4. A concrete 2-week pilot

**Week 1 — build and run the ladder**

| Day | Task |
|---|---|
| 1 | Pull DDXPlus. Sample 300 cases stratified by disease prevalence (100 common / 100 mid / 100 rare). Convert each case into 5 disjoint evidence slices (history, exam, labs, imaging, prior records). |
| 2 | Fork MedLA's loop into a single-file harness. Log for every agent: answer, top-1 confidence, full distribution, forecast over teammates, and the evidence slice it used. |
| 3 | Implement the BTS scorer (geometric and arithmetic variants side by side). Unit-test it on the numbers in `theory_prop1_prop2.md` §2.3. |
| 4 | Run H0 (T=0, same prompt) and H1 (T=0.7, 5 samples) on 200 cases. |
| 5 | Run H2 (5 role prompts) and H3 (3 model families). Compute AUC(score→correct), AUC(confidence→correct), ΔAUC, ECE. |

**Week 2 — the decisive cells**

| Day | Task |
|---|---|
| 6 | Run H4 (disjoint evidence slices) and H5 (H4 + citation requirement). Compute ρ, the minority-truth rate. |
| 7 | Oracle-forecast ablation (P8). This decides whether the signal is truth or miscalibration. |
| 8 | Rare-disease set: geteff1 302 cases + RareBench. Compute the rarity premium and the **rare-disease discrimination gap** (correct vs wrong *within* the same rarity stratum). |
| 9 | Sycophancy control (P7): user-pressure prompt, measure flip rate with and without BTS-guided selection. |
| 10 | Severity experiment (Prop 2.2): apply $w$ to the reward and to the reference; measure ranking change. |
| 11–12 | Write the results section around P1–P8 (with P6 split into P6a/P6b). |
| 13–14 | Go/no-go. |

**Cost.** 200 cases × 5 cells × 5 agents ≈ 5,000 calls per full sweep. With short prompts (≤2k tokens) this is a few dollars on any API. No GPU needed.

**What you must build yourself.** There is no public implementation of BTS as a *clinical audit statistic*. The scorer is trivial; the hard parts are (a) eliciting forecasts that are actually about teammates rather than about the case, and (b) constructing evidence slices with no leakage. Budget most of day 2 and day 6 for those two.

---

## 5. Provenance notes for the paper

If you use these assets, cite the corresponding papers, not just the repos:

- DDXPlus → Tchango et al., *DDXPlus: A New Dataset for Medical Diagnostic Reasoning*, NeurIPS 2022 Datasets & Benchmarks.
- RareBench → Chen et al., *RareBench: Can LLMs Serve as Rare Diseases Specialists?*, KDD 2024.
- MIRAGE → the MIRAGE benchmark paper (repo `gzxiong/MIRAGE`).
- geteff1 cases → *One is Not Enough: Multi-Agent Conversation Framework Enhances Rare Disease Diagnostic Capabilities of Large Language Models* (Research Square preprint).
- MedQA → Jin et al., 2021. MedMCQA → Pal et al., 2022.
- MedLA → Ma et al., AAAI 2026, arXiv 2509.23725.

Keep a `data/README.md` recording the exact commit hash and download date of each repo/dataset. Reviewers in this area have started asking.
