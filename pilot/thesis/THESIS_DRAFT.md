# Truth Serum or Popularity Contest? Auditing Bayesian Truth Serum Consensus in Multi-Agent LLM Medical Diagnosis

[FILL: Full name]
Bachelor of Science in Software Engineering (Major in Data Science)
Department of Software Engineering
DAFFODIL INTERNATIONAL UNIVERSITY
Daffodil Smart City, Ashulia, Dhaka
November 2026

[PAGEBREAK]

## DECLARATION OF THESIS AND COPYRIGHT

Author's full name: [FILL]
Date of birth: [FILL]
Title: Truth Serum or Popularity Contest? Auditing Bayesian Truth Serum Consensus in Multi-Agent LLM Medical Diagnosis
Academic session: [FILL]

I declare that this thesis is classified as: (tick one) CONFIDENTIAL / RESTRICTED / OPEN ACCESS.

I acknowledge that Daffodil International University reserves the rights stated in the university thesis declaration template.

Certified by:
Student's signature: ____________ Student ID: [FILL] Date: ____________
Supervisor's signature: ____________ Name of supervisor: [FILL] Date: ____________

[PAGEBREAK]

## SUPERVISOR'S DECLARATION

I/We have checked this thesis and, in my/our opinion, it is adequate in terms of scope and quality for the award of the degree of Bachelor of Science in Software Engineering.

Signature: ____________ Full name: [FILL] Position: [FILL] Date: ____________

## STUDENT'S DECLARATION

I hereby declare that the work in this thesis is based on my original work except for quotations and citations which have been duly acknowledged. I also declare that it has not been previously or concurrently submitted for any other degree at Daffodil International University or any other institution.

Signature: ____________ Full name: [FILL] ID number: [FILL] Date: ____________

[PAGEBREAK]

## ACKNOWLEDGEMENTS

[FILL: 1 paragraph. Acknowledge supervisor, department, and anyone who helped with data or compute. Keep it factual and short.]

[PAGEBREAK]

## ABSTRACT

Multi-agent large language model (LLM) systems for medical diagnosis usually combine teammate answers with majority voting or confidence-based selection. Both rules can fail when teammates share the same evidence or the same mistaken prior: the most popular answer is not necessarily the correct one. Bayesian Truth Serum (BTS) is a peer-prediction mechanism that rewards an answer that is surprisingly common relative to teammates' forecasts of the answer distribution. This thesis audits whether that reward is a truth signal or merely an answer-rarity statistic.

A local-first pilot was built in which five LLM agents answer human-written medical benchmark questions, report a confidence distribution, and forecast the other agents' answers. Seven controlled team conditions vary one source of heterogeneity at a time: shared-answer control (H0), sampling noise (H1), role prompts (H2), model heterogeneity (H3), evidence heterogeneity (H4), evidence plus deliberation (H5), and a wrong-answer sycophancy pressure condition (SYC). BTS selection is compared with majority, confidence, and a proper-scoring baseline using accuracy, selection accuracy, AUC, calibration error, agreement, and pre-registered audit checks P1-P8.

Because of the submission timeline, the primary evidence comes from PubMedQA (300 cases, all seven cells), with reduced supporting runs on MedQA and MedMCQA (60 cases, cells H1, H4, H5, SYC). Only human-written public benchmark data are used. Small open-weight models (Qwen2.5-0.5B and Qwen2.5-1.5B Instruct) are used without fine-tuning or paid APIs.

[FILL: 3-4 sentences with the actual numbers from Table 4.1 and Table 4.2, e.g. which cells showed a BTS advantage, whether the advantage survived the oracle-forecast ablation, and whether H4 differed from H3.]

The expected contribution is not a claim of clinical-grade accuracy. It is a boundary condition: BTS should be treated as a rarity-sensitive statistic under shared evidence, and as a possible truth signal only where teammate evidence is genuinely private. Limitations, including small models, answer-parsing failures, and the absence of disease-prevalence labels, are stated explicitly.

[PAGEBREAK]

## TABLE OF CONTENT

DECLARATION OF THESIS AND COPYRIGHT .................................. i
SUPERVISOR'S AND STUDENT'S DECLARATIONS .............................. ii
ACKNOWLEDGEMENTS .................................................... iii
ABSTRACT ............................................................ iv
TABLE OF CONTENT .................................................... v
LIST OF TABLES ....................................................... vi
LIST OF FIGURES ...................................................... vii
LIST OF SYMBOLS ....................................................... viii
LIST OF ABBREVIATIONS ................................................. ix

CHAPTER 1 INTRODUCTION ................................................ 1
CHAPTER 2 LITERATURE REVIEW ........................................... 6
CHAPTER 3 METHODOLOGY ................................................ 12
CHAPTER 4 RESULTS AND DISCUSSION ..................................... 20
CHAPTER 5 CONCLUSION AND FUTURE WORK ................................. 28
REFERENCES ........................................................... 31
APPENDICES ........................................................... 35

[PAGEBREAK]

## LIST OF TABLES

Table 2.1. Selected multi-agent medical diagnosis systems and their evaluation.
Table 2.2. Consensus rules compared in this thesis.
Table 3.1. Controlled team conditions (H0-H5, SYC).
Table 3.2. Audit checks P1-P8 and the evidence each one tests.
Table 3.3. Experimental matrix used in this thesis.
Table 4.1. PubMedQA primary results by cell. [FILL]
Table 4.2. Supporting results: MedQA and MedMCQA. [FILL]
Table 4.3. Answer-parsing failures by cell. [FILL]

[PAGEBREAK]

## LIST OF FIGURES

Figure 4.1. Selection accuracy by consensus rule across cells. [FILL]
Figure 4.2. BTS information score versus answer frequency (rarity premium). [FILL]
Figure 4.3. Oracle-forecast ablation: delta AUC with and without true forecasts. [FILL]

[PAGEBREAK]

## LIST OF SYMBOLS

x_i — answer reported by agent i
bar_x — empirical frequency of an answer in the team
bar_y — mean forecast of that frequency
I_i — BTS information score
P_i — BTS personal prediction score
r_i — total BTS reward, r_i = I_i + P_i
Delta AUC — AUC(BTS) minus AUC(confidence)

[PAGEBREAK]

## LIST OF ABBREVIATIONS

BTS — Bayesian Truth Serum
LLM — Large Language Model
MAS — Multi-Agent System
MCQ — Multiple-Choice Question
AUC — Area Under the ROC Curve
ECE — Expected Calibration Error
SYC — Sycophancy pressure condition
UG — Undergraduate
DIU — Daffodil International University

[PAGEBREAK]

## CHAPTER 1 INTRODUCTION

### 1.1 Background

Large language models are increasingly used as components of clinical decision-support pipelines. A single model can answer a medical multiple-choice question, but a team of models with different roles or prompts can be asked to deliberate, critique, and then converge on an answer. Published systems range from retrieval-augmented clinical decision support to role-based multi-agent consultation, knowledge-graph-guided diagnosis, conformal triage, and interactive inquiry agents (adaptive CDSS, 2025; ConfAgents, 2025; DDO, 2026; HEAL-KGGen, 2025; KG4Diagnosis, 2025; MaS-AD, 2026).

These systems almost always need a consensus rule. Majority voting is the simplest: the most frequent answer wins. Confidence-based selection chooses the answer attached to the highest self-reported confidence. Both are intuitive, but both conflate agreement with correctness. If every agent has read the same passage, shares the same training prior, or repeats the same misleading cue, high agreement can be high-confidence error. The multi-agent literature reports small gains over single agents in several evaluations, together with substantial cost and limited clinical validation (rare-disease evaluation, 2026; ConfAgents, 2025).

Bayesian Truth Serum (BTS) offers a different signal. In the peer-prediction version used here, each agent reports an answer and also predicts the distribution of answers the other agents will give. An agent receives a higher score when its answer is more frequent than the team expected, and a calibration term rewards accurate predictions. The intuition is that a genuinely informed minority can be identified if its forecast of the group is appropriately surprising, while a merely popular but uninformed answer should not be rewarded once forecasts are accounted for (Prelec, 2004).

### 1.2 Problem statement

It is not established that a BTS-based consensus rule selects correct medical answers. The mechanism can reward rarity as easily as truth: an answer that teammates did not expect becomes valuable under BTS even when it is wrong. Without a controlled comparison, a reported BTS gain could be explained by answer frequency, by role or prompt diversity, by model diversity, or by genuine private evidence. This thesis isolates those explanations.

### 1.3 Research questions

RQ1. Under shared evidence, does BTS-based selection outperform majority voting and confidence-based selection on medical benchmark questions?

RQ2. Does any BTS advantage depend on answer rarity rather than correctness, as shown by the rarity premium, fixed-frequency gap, and oracle-forecast ablation?

RQ3. Does genuine evidence heterogeneity (PubMedQA abstract sections) change the BTS signal relative to role, model, and text-chunk heterogeneity?

### 1.4 Objectives

1. Implement a reproducible local pilot in which agents produce answers, confidence distributions, and teammate forecasts.
2. Construct seven controlled team conditions that vary one heterogeneity source at a time, plus a wrong-answer sycophancy pressure condition.
3. Implement BTS scoring and four selection rules, and measure accuracy, selection accuracy, AUC, calibration, and agreement.
4. Run pre-registered audit checks P1-P8 on the primary dataset and on reduced supporting datasets.
5. Report boundary conditions under which BTS is a truth signal versus a popularity statistic.

### 1.5 Scope and limitations

This is a small-scale audit, not a clinical study. The models are open-weight instruct models of 0.5B-1.5B parameters on the thesis runs; no fine-tuning, no API key, and no patient data are used. Because of the submission deadline, the primary run covers PubMedQA (300 cases, seven cells) and the supporting runs cover MedQA and MedMCQA (60 cases, four cells each). MMLU-Med and MedXpertQA are prepared but not run, and are left to future work. A negative or null result is treated as a valid finding. The thesis makes no claim of diagnostic accuracy above a clinical threshold.

### 1.6 Thesis organisation

Chapter 2 reviews multi-agent medical diagnosis and consensus mechanisms. Chapter 3 presents the BTS formulation, team conditions, datasets, metrics, and audit checks. Chapter 4 reports results and interprets them against the research questions. Chapter 5 concludes with contributions, limitations, and future work.

[PAGEBREAK]

## CHAPTER 2 LITERATURE REVIEW

### 2.1 Multi-agent medical diagnosis

Recent work organises LLM agents into teams with distinct roles and interaction protocols. A survey of LLM-based multi-agent systems in medicine catalogues hierarchical, peer, and multidisciplinary-team patterns and notes the absence of a standard evaluation protocol (survey, 2025). MaS-AD gives each agent a syllogistic logic tree and aligns premises across agents, reporting gains on MedDDx and medical QA, but uses static MCQ settings and hand-specified logic trees (MaS-AD, 2026). An adaptive clinical decision-support system combines a retrieval agent, a web-evidence agent, and a fusion agent with a low-confidence retry loop and reports high accuracy on MedQA, PubMedQA, and MedBullets, without clinician validation or calibration guarantees (adaptive CDSS, 2025).

Other systems optimise cost or interaction. ConfAgents triages with conformal confidence and escalates only hard cases, reporting cost reductions of more than 50% while noting that multi-agent inference can be up to 50 times slower than a single call (ConfAgents, 2025). DDO decouples an inquiry agent from a diagnosis agent and is trained on real consultation data, but requires local training and has no resource-cost model (DDO, 2026). KG4Diagnosis and HEAL-KGGen ground teams in knowledge graphs, with the former lacking a completed quantitative benchmark and the latter reporting gains on a biomarker QA set without clinical validation (KG4Diagnosis, 2025; HEAL-KGGen, 2025).

A rare-disease evaluation of four topologies found differences of less than two percentage points between hierarchical, collaborative, and single-agent control conditions, with the adversarial topology much worse, and introduced a reasoning-gap metric (rare-disease evaluation, 2026). MACD reports gains from self-learned knowledge across cases but acknowledges knowledge drift and limited disease coverage (MACD, 2026). Across these papers, the common pattern is that team architecture is varied and accuracy is reported; the mechanism by which a consensus rule converts individual signals into a collective answer is rarely audited.

**Table 2.1. Selected multi-agent medical diagnosis systems and their evaluation.**

| System | Year | Main idea | Benchmark | Reported limitation |
|---|---|---|---|---|
| MaS-AD | 2026 | Premise-level logic alignment | MedDDx, MMLU-Med, MedQA-US | Static MCQ; no cost reporting |
| Adaptive CDSS | 2025 | Retrieval + evidence + adaptive retry | MedQA, PubMedQA, MedBullets | Heuristic confidence loop; no clinician study |
| ConfAgents | 2025 | Conformal triage, selective escalation | MedQA, MedBullets, MMLU | MCQ only; no test-ordering cost |
| DDO | 2026 | Inquiry/diagnosis decoupled, RL inquiry | Real consultation data | Requires local training; no cost model |
| KG4Diagnosis | 2025 | Hierarchical referral over knowledge graph | 362 diseases (planned) | No completed quantitative benchmark |
| Rare-disease evaluation | 2026 | Four topologies compared | 302 Orphanet cases | Small gains; single model; no cost normalisation |

### 2.2 Consensus rules

Majority voting and confidence-based selection are the dominant implicit consensus rules. Majority voting is robust to isolated outliers but vulnerable to correlated error. Confidence selection can exploit calibrated self-reports, but LLM confidence is often poorly calibrated (ConfAgents, 2025; adaptive CDSS, 2025). Neither rule asks agents to predict one another.

BTS comes from the literature on eliciting truthful subjective reports without ground truth (Prelec, 2004). The peer-prediction logic scores an answer relative to the distribution that agents expected. Related ideas appear in proper scoring rules and in conformal confidence estimation, but the audit question here is specific: does the BTS reward track correctness in a medical team, or does it track answer rarity?

**Table 2.2. Consensus rules compared in this thesis.**

| Rule | Signal used | Strength | Failure mode |
|---|---|---|---|
| Majority | Answer frequency | Simple, stable | Correlated mistakes become "truth" |
| Confidence | Self-reported confidence | Uses uncertainty | Miscalibration; overconfidence |
| BTS | Answer frequency vs teammates' forecasts | Rewards surprising-agreed answers | Can reward wrong rarity |
| Proper | Proper score on reported distribution | Theoretically aligned | Depends on forecast quality |

### 2.3 Research gap

The reviewed systems vary architectures and report accuracy, but they do not pre-register checks that separate truth from popularity, role effects, model effects, and evidence effects. The specific gap is a controlled audit of BTS as a consensus rule under shared versus private evidence in medical multi-agent QA. Existing work also lacks a clean separation between evidence heterogeneity and text-chunk ablation, and does not report an oracle-forecast ablation to test whether a BTS gain comes from private information or from forecast miscalibration. This thesis addresses that gap with a small, reproducible pilot.

[PAGEBREAK]

## CHAPTER 3 METHODOLOGY

### 3.1 Research design

The study is a controlled computational experiment. Each benchmark question is presented to five agents under a team condition. The condition changes one source of heterogeneity at a time. All agents produce (i) an answer, (ii) a confidence distribution over options, and (iii) on the terminal round, a forecast of the empirical answer distribution of the other agents. The same question set is used across cells, so differences between cells are attributable to the manipulated heterogeneity source rather than to different items.

### 3.2 Bayesian Truth Serum scoring

Let n be the number of agents and let x_i be the answer of agent i. For a candidate answer a:

bar_x(a) = (1/n) * sum_i 1[x_i = a]                                      (3.1)

bar_y(a) = (1/n) * sum_i y_i(a), where y_i(a) is agent i's forecast of the frequency of a                                                      (3.2)

I_i = sum_a 1[x_i = a] * log( bar_x(a) / bar_y(a) )                     (3.3)

P_i = sum_a y_i(a) * log( bar_x(a) / bar_y(a) )                         (3.4)

r_i = I_i + P_i                                                          (3.5)

Equation 3.3 rewards an answer that is more frequent than predicted; Equation 3.4 rewards accurate forecasts. The team-level BTS score of an answer is the aggregate reward of agents choosing it, and the BTS selection rule picks the answer with the highest aggregate reward. A geometric variant with an epsilon floor is used for numerical stability. Codes are available in `pilot/src/bts_audit/scoring.py`.

The implementation keeps the two terms separate so that the oracle-forecast ablation can replace y_i with the true distribution. If the BTS advantage disappears under true forecasts, the apparent signal was forecast miscalibration rather than private information.

### 3.3 Team conditions

**Table 3.1. Controlled team conditions.**

| Cell | Manipulation | What it tests |
|---|---|---|
| H0 | Same model, same shared context, greedy decoding | Collapse / no diversity baseline |
| H1 | Same model, sampling noise | Diversity from stochastic decoding |
| H2 | Same model, different clinical role prompts | Diversity from role framing |
| H3 | Different model backbones | Diversity from model heterogeneity |
| H4 | Different private evidence slices | Genuine evidence heterogeneity (PubMedQA abstract sections) |
| H5 | H4 plus citations and deliberation | Effect of deliberation on top of evidence |
| SYC | Wrong-answer pressure from the team | Sycophancy / minority-truth stress test |

For PubMedQA the evidence slices are the abstract sections (background, case, methods, results, conclusions, other), so H4 is a genuine evidence partition. For MedQA and MedMCQA the slices are contiguous text chunks of the question stem; H4 there is a text-chunk ablation and is interpreted as such.

### 3.4 Audit checks

**Table 3.2. Audit checks P1-P8.**

| Check | Test | Expected if BTS is a truth signal |
|---|---|---|
| P1 | H0 collapse | BTS and majority agree when agents are identical |
| P2 | Delta AUC vs confidence in H1-H2 | BTS AUC >= confidence AUC |
| P3 | Delta AUC vs confidence in H3 | BTS AUC >= confidence AUC with model diversity |
| P4 | Delta AUC vs confidence in H4-H5 | BTS AUC >= confidence AUC with private evidence |
| P5 | H4 vs H3 signal | BTS improves more with evidence than with model diversity |
| P6a | Rarity premium | BTS reward rises with answer rarity |
| P6b | Fixed-frequency gap | BTS separates correct and incorrect answers at equal frequency |
| P7 | SYC robustness | BTS resists wrong-answer pressure better than majority |
| P8 | Oracle forecast ablation | BTS signal survives when forecasts are replaced by truth |

Checks P6a and P6b are mechanical properties of the score. Because the benchmarks do not carry disease prevalence, this thesis makes no claim about disease rarity or population prevalence.

### 3.5 Data and sampling

Five human-written public benchmarks were prepared: MedQA-USMLE test (1,273 rows), PubMedQA `pqa_labeled` (1,000), MMLU medical test subsets (1,111), MedMCQA validation (4,183), and MedXpertQA text test (2,450). The official MedMCQA test split withholds labels, so validation is used. Synthetic or LLM-restructured datasets are excluded.

From each benchmark, 300 cases are deterministically sampled with seed 13 into `data/cases/<dataset>_300.jsonl`. Options are deterministically shuffled with the same seed while keeping the gold option. no labels are invented and no `rarity_stratum` or `prevalence` value is derived from answer frequency. For the deadline-limited thesis, PubMedQA uses all 300 cases and all seven cells; MedQA and MedMCQA use 60 cases and cells H1, H4, H5, SYC.

### 3.6 Models and implementation

The thesis runs use two open-weight instruct models, `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`, cycled as `0.5B, 1.5B, 0.5B, 0.5B, 1.5B` across five agents under H3-H5 and SYC. Runs are executed with Hugging Face transformers, `device_map: auto`, float16 on a T4 GPU, no 4-bit quantisation for the small models, and no API key. CPU runs of the same pipeline were used for debugging only. Each run writes `answers.jsonl` and `forecasts.jsonl` incrementally; `--resume` skips already completed `(cell, case, round, agent, task)` records, and each configuration uses a separate output directory.

### 3.7 Metrics

Per cell the pilot computes answer accuracy, selection accuracy for each consensus rule, AUC of the selection score against the binary correctness label, expected calibration error, teammate agreement rate, correlation between confidence and BTS score, and the oracle-forecast delta. Bootstrap confidence intervals use 1,000 resamples with a fixed seed. Significance is reported at alpha = 0.05, with the caveat that multiple comparisons across cells are exploratory.

### 3.8 Experimental matrix

**Table 3.3. Experimental matrix used in this thesis.**

| Dataset | Cases | Cells | Models | Role |
|---|---:|---|---|---|
| PubMedQA | 300 | H0,H1,H2,H3,H4,H5,SYC | 0.5B / 1.5B | Primary evidence |
| MedQA | 60 | H1,H4,H5,SYC | 0.5B / 1.5B | Supporting probe |
| MedMCQA | 60 | H1,H4,H5,SYC | 0.5B / 1.5B | Supporting probe |
| MMLU-Med | 300 | prepared, not run | — | Future work |
| MedXpertQA | 300 | prepared, not run | — | Future work |

### 3.9 Threats to validity

Four threats are explicit. First, small models may fail to follow the output format; failed parses are counted and reported (Table 4.3). Second, small-model accuracy is low, so AUC and selection accuracy can be unstable. Third, only PubMedQA provides genuine evidence sections. Fourth, no clinician or patient data are used, so clinical validity is outside the scope.

[PAGEBREAK]

## CHAPTER 4 RESULTS AND DISCUSSION

### 4.1 Output artefacts

Each completed run produces `report.md`, `cell_metrics.csv`, `metrics.json`, `predictions.json`, `agent_scores.csv`, `answers.jsonl`, `forecasts.jsonl`, and a `figures/` directory. The tables below should be filled directly from `cell_metrics.csv`; figures should be copied from `figures/`. Do not retype numbers by hand without checking them against the CSV.

### 4.2 Primary results: PubMedQA

**Table 4.1. PubMedQA primary results by cell. [FILL from results/colab_pubmedqa_small/cell_metrics.csv]**

| Cell | n | Accuracy | Majority | Confidence | BTS | Proper | AUC (conf.) | AUC (BTS) | Delta AUC | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 300 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| H1 | 300 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| H2 | 300 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| H3 | 300 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| H4 | 300 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| H5 | 300 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| SYC | 300 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

**Figure 4.1. Selection accuracy by consensus rule across cells. [FILL: paste figures/selection_accuracy.png]**

[FILL: 1-2 paragraphs. State which rule won in which cell, with numbers. Say explicitly whether P1 collapse appeared in H0.]

### 4.3 Supporting results

**Table 4.2. Supporting results: MedQA and MedMCQA. [FILL from the two support runs]**

| Dataset | Cell | n | Accuracy | Majority | Confidence | BTS | Delta AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| MedQA | H1 | 60 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| MedQA | H4 | 60 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| MedQA | H5 | 60 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| MedQA | SYC | 60 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| MedMCQA | H1 | 60 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| MedMCQA | H4 | 60 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| MedMCQA | H5 | 60 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| MedMCQA | SYC | 60 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

[FILL: 1 paragraph. These are 60-case probes; do not present them as benchmark-level results. For these datasets H4/H5 are text-chunk ablations, not evidence partitioning.]

### 4.4 Rarity, oracle ablation, and failure analysis

**Figure 4.2. BTS information score versus answer frequency. [FILL]**

**Figure 4.3. Delta AUC with and without oracle forecasts. [FILL]**

**Table 4.3. Answer-parsing failures by cell. [FILL]**

| Cell | Failed answers | `answer_not_mapped` | `no_json` |
|---|---:|---:|---:|
| H0 | [FILL] | [FILL] | [FILL] |
| H1 | [FILL] | [FILL] | [FILL] |
| H2 | [FILL] | [FILL] | [FILL] |
| H3 | [FILL] | [FILL] | [FILL] |
| H4 | [FILL] | [FILL] | [FILL] |
| H5 | [FILL] | [FILL] | [FILL] |
| SYC | [FILL] | [FILL] | [FILL] |

[FILL: interpret P6a/P6b/P8. If the BTS advantage disappears under oracle forecasts, say so plainly. If it survives, say that the signal is consistent with private information rather than mere forecast miscalibration.]

### 4.5 Answers to the research questions

RQ1. [FILL: one paragraph with numbers; state whether BTS beat majority and confidence under shared evidence.]

RQ2. [FILL: one paragraph on rarity premium, fixed-frequency gap and oracle ablation.]

RQ3. [FILL: one paragraph comparing H4 with H3 on PubMedQA, and the supporting datasets with the text-chunk caveat.]

### 4.6 Discussion

The central interpretation is conditional. If BTS fails under H0-H3 but improves under PubMedQA H4, the result supports the claim that the mechanism needs private evidence rather than role or model labels. If BTS does not improve even under H4, the stronger conclusion is that a small-model team does not have enough reliable private information for peer prediction to help, and that majority or confidence remains the safer default. Either outcome is informative because the design was built to allow the claim to fail.

The negative case is also practically useful. A hospital-style team with shared guidelines can easily produce correlated answers; a reward that mounts an extra premium on unexpected answers may amplify a wrong minority rather than correct it. The oracle ablation is therefore the decisive check: it separates a real information signal from a statistical artefact of forecast error.

[PAGEBREAK]

## CHAPTER 5 CONCLUSION AND FUTURE WORK

### 5.1 Conclusion

This thesis implemented and audited a local, reproducible multi-agent pipeline that elicits answers, confidence distributions, and teammate forecasts from LLM agents, then compares majority, confidence, BTS, and proper-scoring consensus rules. The audit varied diversity sources across seven controlled cells and applied eight pre-registered checks. The primary evidence came from PubMedQA with 300 cases and all cells, with reduced supporting probes on MedQA and MedMCQA.

[FILL: 2-3 sentences stating the actual finding. Avoid saying "BTS always works" or "the system is clinically accurate".]

The main conclusion is that BTS should not be treated as a truth serum by default. Under shared evidence it behaves as a rarity-sensitive statistic; it is defensible only where the team has genuinely private evidence and where forecast calibration has been checked.

### 5.2 Contributions

1. A local-first, API-free implementation of BTS scoring and consensus evaluation for medical multi-agent QA.
2. A controlled heterogeneity ladder that separates sampling noise, role prompts, model diversity, evidence diversity, deliberation, and sycophancy pressure.
3. An audit scorecard including an oracle-forecast ablation that distinguishes information from forecast miscalibration.
4. A clear statement of boundary conditions and dataset caveats for future multi-agent medical diagnosis work.

### 5.3 Limitations

The models are small and were not fine-tuned. Answer parsing failures were observed and are reported. Only PubMedQA has genuine evidence sections. The supporting datasets use 60 cases and four cells. No clinician evaluation, real patient data, prevalence labels, cost analysis, or prospective validation is included. MMLU-Med and MedXpertQA were prepared but not run within the deadline.

### 5.4 Future work

Future work should repeat the audit with 7B-class or stronger models, complete all five benchmarks at 300 cases, add clinician studies, and test real EHR-derived cases under appropriate governance. A prevalence-labelled dataset would allow P6a and P6b to be interpreted substantively rather than mechanically. Cost-normalised comparison and prospective validation are necessary before any clinical claim.

[PAGEBREAK]

## REFERENCES

Adaptive multi-agent LLM-based clinical decision support system. (2025). IEEE Access. [FILL: exact authors, volume, pages, DOI from the paper PDF.]

ConfAgents: Conformal-guided multi-agent framework for cost-efficient medical diagnosis. (2025). arXiv:2508.04915. [FILL: authors.]

DDO: Dual-decision optimization for LLM-based medical consultation. (2026). [FILL: authors and venue.]

Evaluating multi-agent LLM architectures for rare disease diagnosis. (2026). [FILL: authors and venue.]

HEAL-KGGen. (2025). bioRxiv. [FILL: authors and DOI.]

KG4Diagnosis: Hierarchical multi-agent LLM framework with knowledge graph enhancement for medical diagnosis. (2025). arXiv:2412.16833. [FILL: authors.]

MaS-AD: MedLA, a logic-driven multi-agent framework. (2026). AAAI. [FILL: authors.]

MACD: Multi-agent clinical diagnosis with self-learned knowledge. (2026). arXiv:2509.20067. [FILL: authors.]

MAGDA: Multi-agent guideline-driven diagnostic assistance. (2024). MICCAI. [FILL: authors.]

MDteamgpt. [FILL: year, authors, venue from PDF.]

MedChat. [FILL: year, authors, venue from PDF.]

MedSentry. [FILL: year, authors, venue from PDF.]

MeDxAgent: Multi-agent consultation for interactive medical diagnosis. [FILL: year, authors, venue from PDF.]

Prelec, D. (2004). A Bayesian truth serum for subjective data. Science, 306(5695), 462-466. https://doi.org/10.1126/science.1102081

Survey of LLM-based multi-agent systems in medicine. (2025). TechRxiv. [FILL: authors.]

ZODIAC. [FILL: year, authors, venue from PDF.]

[PAGEBREAK]

## APPENDIX A - REPRODUCTION COMMANDS

```bash
cd pilot
pip install -r requirements-gpu.txt
pip install -e . --no-deps
python scripts/download_data.py all --out data/raw
python -m bts_audit.cli prepare --dataset pubmedqa --raw data/raw/pubmedqa --out data/cases/pubmedqa_300.jsonl --n 300
OUT_ROOT=/content/drive/MyDrive/bts_audit_colab_results bash scripts/run_colab_thesis_scope.sh
```

Appendix B: raw `report.md` for PubMedQA.
Appendix C: raw `report.md` for MedQA and MedMCQA supporting runs.
Appendix D: config files used (`configs/colab_pubmedqa_small.yaml`, `configs/colab_medqa_support.yaml`, `configs/colab_medmcqa_support.yaml`).
