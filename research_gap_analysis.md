# Research Gap Analysis — Multi-Agent Medical Diagnosis with LLMs

Prepared from the 19 PDFs in this folder (17 unique papers: one survey and MeDxAgent are duplicated), plus a structured check of arXiv literature through **November 2026**. Companion file: `literature_matrix.csv`.

---

## 0. The short answer

**Do not build another architecture.** The "role-played specialists + debate + judge" design space is now saturated — in 2026 there are multiple papers for every module you could add (routing, triage, memory, conformal gating, graph grounding, dissent handling, sycophancy defence). A new architecture paper will be (a) scooped, and (b) rejected for "incremental".

The gaps that are still genuinely open are not *missing modules*. They are **missing claims the field cannot currently make**:

| Still cannot be claimed | Why it matters |
|---|---|
| "This system diagnoses under the resource constraints where most of the world's patients actually are" | Every paper in your corpus assumes the full workup (labs, imaging, specialists) is available |
| "This system improves *patient outcomes / clinician decisions*, not just answer accuracy" | Only MACD has any human arm; only ZODIAC has independent cardiologist review |
| "We know *why* the team beats the single model" | Gains of +2% to +22% are reported without compute normalisation or mechanism analysis |
| "The system is safe when the evidence is insufficient" | Safety papers (MedSentry) test adversarial prompts, not diagnostic harm from missing evidence |

**My recommendation: build the first resource-aware multi-agent diagnostic system + benchmark** (Section 5). It is clinically the largest untouched problem, it is cheap to run (no GPU cluster, no patient data for v1), and it gives you a *data/context moat* that a US lab cannot copy quickly.

---

## 1. What your corpus actually contains

Three clusters, not 17 independent contributions:

**Cluster A — Architecture variants on medical MCQA (the bulk)**
MedLA (AAAI-26), MAM, MDTeamGPT, MACD, ConfAgents, Adaptive CDSS, DDO, KG4Diagnosis, HEAL-KGGen. All report accuracy on MedQA / PubMedQA / MedBullets / MedMCQA / MMLU-Med, with a multi-agent pipeline on top.

**Cluster B — Modality- or specialty-specific systems**
MAGDA (chest X-ray), MedChat (glaucoma), ZODIAC (cardiology/ECG), neuro-ophthalmic MAS, MeDxAgent (interactive diagnosis benchmark).

**Cluster C — Evaluation and safety**
The two surveys (same content, TechRxiv + arXiv versions), the rare-disease topology study, MedSentry (adversarial safety benchmark).

### What the corpus collectively establishes

1. **Multi-agent > single agent, but the margin is small and unstable.** Rare-disease study: Hierarchical 50.0%, Collaborative 49.8%, single agent 48.5%. Adversarial topology collapses to 27.3%. MACD reports +18–22%, but against weak/no knowledge baselines and with a self-learned knowledge base that is itself the intervention.
2. **Cost is the binding constraint.** ConfAgents: MAS up to **50× slower** than a single call, with marginal gains on most cases. Every paper lists cost/latency as future work.
3. **Evaluation is exam-shaped.** Static MCQA with a single gold answer. No test ordering, no missing evidence, no time, no cost, no harm, no referral decision.
4. **Almost no clinical validation.** MACD uses "4,390 real-world cases" with a small physician subset; ZODIAC has independent cardiologist review; everything else is benchmark-only. The surveys name this explicitly.
5. **Consensus is assumed to be good.** The rare-disease study and MedSentry show the opposite: pressure toward consensus can destroy correct minority diagnoses.

---

## 2. The gap map

Status legend: **OPEN** = no direct competitor found · **CLOSING** = 2026 papers exist, room for a differentiated angle · **TAKEN** = do not enter.

| # | Gap | Status (Nov 2026) | Evidence |
|---|---|---|---|
| G1 | Real clinical outcomes / harm-weighted evaluation on EHR | **OPEN** (highest barrier) | No paper in corpus; NOHARM (2512.01241) evaluates single models + RAG on vignettes, not MAS on EHR |
| G2 | Resource-constrained / LMIC diagnosis | **OPEN** (nearest: MedAction) | Aletheia (2607.24814) is a single offline 3B model, not MAS, no test-ordering; MedAction (2605.07305) does single-LLM active test ordering with an unrestricted test menu; NSIDDx (2609.00256) is neuro-symbolic DDx, not resource-aware |
| G3 | Mechanism: what information flows between agents, and why gains appear | **OPEN for medicine** | General critiques exist (2606.13003, 2604.02460); no medical MAS information-flow study |
| G4 | Human–AI teaming / appropriate reliance with MAS | **CLOSING** | NOHARM (2512.01241) randomised physician+AI teaming study, also tests multi-agent AI teaming; g-AMIE (2507.15743) physician oversight |
| G5 | Cost-adaptive routing / when to collaborate | **TAKEN** | ConfAgents (2508.04915), MedRoute (2604.06180), SMART-Med (2610.33544), TeamMedAgents (2508.08115) |
| G6 | Conformal / calibrated abstention for MAS | **TAKEN (mostly)** | ConfAgents; 2609.25959 (MoA conformal); MedAbstain (2601.12471); 2603.24481 (calibration) |
| G7 | Dissent / sycophancy / conformity in agent debate | **TAKEN** | MedPRESS (2608.02520), 2605.23932, 2606.29270, 2610.34013, 2608.03744 |
| G8 | Guideline drift / outdated medical knowledge | **CLOSING** | 2610.30631, 2605.13045, 2505.07968 (all single-model/retrieval, **not MAS**) |
| G9 | Premature closure / diagnostic revision | **CLOSING** | 2605.15000, MedEinst (2601.06636), Sequential Diagnosis (2506.22405) |
| G10 | Multimodal conflict reconciliation | **CLOSING** | MASGR (2608.30938) for referral; 2604.17340 for treatment-guideline conflicts |
| G11 | Safety / adversarial robustness of MAS | **TAKEN** | MedSentry (2505.20824), 2610.22594 (prompt injection), 2610.32060 |
| G12 | Fairness across demographics | **CLOSING** | 2608.06949 (triage bias), 2604.18038 (racial bias via agentic workflows) |

### Verdict on the "obvious" ideas

- *"Add a knowledge graph"* → KG4Diagnosis, HEAL-KGGen, MedCollab, GraphDx, GuideSkill. Dead.
- *"Add a judge / debate protocol"* → MedLA, MDTeamGPT, Structured Debate-MoA (2609.05069). Dead.
- *"Make it cheaper"* → ConfAgents, MedRoute, SMART-Med, ECMR. Dead.
- *"Make it safer against attacks"* → MedSentry, 2610.22594. Dead.
- *"Add memory / self-evolution"* → MACD, EMR (2609.15161), MedCache (2608.29528), Vibe Medicine (2606.15504). Dead.
- *"Build a new interactive benchmark"* → MeDxBench, ClinVerse (2610.34878), ClinicalMC (2606.03157), MTDiag (2608.25085), CLIMB (2609.35462). Dead.

This is the single most useful thing to internalise: **the field is no longer short of architectures. It is short of clinical grounding.**

---

## 3. Three programs worth doing

Each is a 2–3 paper program, not a one-off. Scored 1–5.

| Criterion | P1: EHR silent trial | **P2: Resource-aware MAS (recommended)** | P3: Information-flow audit |
|---|---|---|---|
| Novelty | 4 | 4 | 5 |
| Feasibility (your resources) | 2 (needs PhysioNet + heavy phenotyping) | **5** (public structured data + APIs) | 4 (needs open-weight models) |
| Clinical importance | 5 | 5 | 3 |
| Scoop risk | 2 (low, data moat) | 3 (low–medium, context moat) | 4 (general version exists) |
| Time to first submission | 9–12 months | **5–7 months** | 4–6 months |
| Venue ceiling | npj Digit Med / JAMIA / MLHC | npj Digit Med / JAMIA / MLHC / CHIL | ACL / EMNLP / NeurIPS |

### P1 — Retrospective "silent trial" of MAS on real EHR (highest impact)
Use MIMIC-IV + MIMIC-CXR. Reconstruct the diagnostic question at an index time, hide later data, let the MAS order tests and produce a differential, score against discharge diagnoses, and add **harm-weighted endpoints** (missed can't-miss diagnosis, unsafe discharge, unnecessary invasive test). Gold: this is what a regulator would want. Barrier: PhysioNet credentialing, cohort construction, leakage control, ~9–12 months.

### P2 — Resource-aware multi-agent diagnosis (recommended, detailed below)
The system must diagnose *with the tests that exist where the patient is*, and must know when to stop, refer, or abstain.

### P3 — "What actually flows between agents?" (fastest, most scientific)
Instrument inter-agent messages and measure whether they carry **patient-specific evidence** or only **answer labels + confidence**. Counterfactual message surgery (replace a message with answer-only, or with a random peer's evidence) isolates whether gains come from deliberation or plain ensembling. If the answer is "ensembling", that is a field-level finding with a constructive fix (evidence-budgeted deliberation). Cheap, novel, but reviewer risk: "so what?" unless you deliver the fix.

---

## 4. Why P2 is the right bet for you

1. **The clinical problem is the biggest one.** Most diagnoses worldwide are made with history, examination, and a handful of point-of-care tests. Every system in your folder implicitly assumes a US/European tertiary centre.
2. **The objective function changes, which changes the science.** The current target is "accuracy on a fully specified vignette". The right target is "**risk-controlled diagnosis under an evidence budget**": maximise correct-and-safe decisions per unit of diagnostic cost, and escalate when the available evidence cannot support a diagnosis.
3. **It gives you a defensible moat.** Local clinical partners, Bangla-language cases, and real test menus/price lists cannot be copied by a lab in California. That is how you avoid being scooped while you write.
4. **It is cheap and buildable now.** DDXPlus is a structured synthetic patient dataset with evidence and differential diagnoses — you can simulate test availability, cost, and turnaround without any patient data. No GPU needed for the main experiments.
5. **It has an obvious second and third paper** (equity across resource tiers; real district-hospital deployment with clinicians).

---

## 5. Recommended program — full blueprint

**Working title:** *FrugalDx: Risk-Controlled Multi-Agent Diagnosis Under Resource Constraints*

**One-sentence claim:** *Existing multi-agent diagnostic systems are evaluated as if every test is available; we show that when evidence acquisition is constrained to what a real clinic can obtain, standard multi-agent collaboration becomes unsafe and cost-inefficient, and we introduce a resource-aware multi-agent framework that matches full-workup accuracy at a fraction of diagnostic cost while correctly escalating when a diagnosis cannot be supported.*

### 5.1 Research questions

- **RQ1 (measurement).** How do current MAS architectures degrade as the available evidence set shrinks (T0 history+exam → T1 +basic labs/RDT → T2 +imaging → T3 full workup), and do they *know* they have degraded?
- **RQ2 (mechanism).** Is the failure caused by knowledge gaps, or by the inability to plan evidence acquisition and to recognise insufficiency? (Ablate: give agents the missing knowledge vs. give them the ability to order tests.)
- **RQ3 (method).** Does an explicit resource planner (value-of-information under cost/availability constraints) + a safety/referral agent improve the accuracy–cost frontier and reduce harmful confident errors?
- **RQ4 (guarantee).** Can system-level conformal risk control deliver a user-specified bound on "confidently wrong diagnosis" across resource tiers?
- **RQ5 (equity).** Does the framework close the performance gap between low- and high-resource tiers, or widen it?

### 5.2 Hypotheses (falsifiable)

- H1: MAS accuracy advantage over a compute-matched single agent **shrinks or reverses** at T0/T1 (because the bottleneck becomes evidence, not reasoning).
- H2: Standard MAS **over-diagnose** at low tiers (confident answers from insufficient evidence) and **under-refer**.
- H3: VoI-based test selection reaches ≥90% of T3 accuracy at ≤40% of T3 cost.
- H4: Risk control holds empirical error ≤ α across tiers, and the abstention/referral rate rises monotonically as the tier drops.
- H5: The resource-aware system reduces the T0–T3 equity gap relative to static MAS.

### 5.3 Nearest neighbours and your delta

| Nearest work | What it does | Your delta |
|---|---|---|
| **MedAction** (2605.07305) | Single LLM; tree-structured distillation of multi-turn test-ordering trajectories; full test menu | Multi-agent; **test menu is site-specific and constrained**; cost-weighted; referral/escalation; risk guarantee |
| **Aletheia** (2607.24814) | Offline 3B model for DDx in sub-Saharan Africa | Multi-agent collaboration; explicit cost/availability model; safety-first objective |
| **ConfAgents** (2508.04915) | Conformal triage to decide *whether* to run the expensive consultation | Constraint is on **diagnostic** cost (tests/referrals), not inference cost; decides *what evidence to buy* |
| **GraphDx** (2607.15280) | Cost-aware sequential diagnosis over a KG | Explicit resource tiers, can't-miss safety, equity analysis, conformal risk control |
| **VoI for human–agent communication** (2601.06407) | VoI decides whether to ask the *user* a question | VoI decides which *clinical test* to order given availability, cost and turnaround |

If a reviewer can see this table in your introduction, your novelty is defensible.

### 5.4 System design (5 components)

1. **Evidence State** — structured findings with provenance: `{finding, value, source (history/exam/test), reliability, timestamp, cost_incurred}`. No free-text-only state.
2. **Differential Agent(s)** — 3–5 specialist agents produce ranked differentials *conditioned on the current evidence state only*, each with per-diagnosis confidence and the findings that support/refute it.
3. **Resource Planner** — the novel core. Maintains a belief over the differential; for each *available* test computes an estimate of value of information (expected reduction in decision risk) against cost, turnaround, invasiveness, and local availability. Orders a test **only if VoI > λ·cost**. Stops when marginal VoI < threshold or budget exhausted.
4. **Safety / Escalation Agent** — maintains a can't-miss list for the current differential; decides among `diagnose`, `order test`, `refer`, `abstain`. Optimised for *harm avoidance*, not accuracy.
5. **Risk-Controlled Aggregator** — combines specialist distributions with conformal calibration; emits a prediction set with a distribution-free guarantee at level α, or escalates.

### 5.5 Data

| Purpose | Dataset | Why |
|---|---|---|
| Core sandbox | **DDXPlus** | Structured evidence, differential diagnoses, prevalence — lets you simulate test availability/cost cleanly |
| External validity | **MedQA / MedMCQA / MedDDx** | Re-cast vignettes into tiers by masking evidence types |
| Realism check (optional) | **MIMIC-IV** (PhysioNet) | Sanity-check ordering patterns against real order sets |
| Guideline anchoring | **WHO IMCI / IMAI** syndromic pathways | Defines what a T0/T1 clinic can actually do |
| Extension | Local district-hospital cases (Bangla) | The moat; needs ethics approval |

### 5.6 Baselines (must all be compute-matched where possible)

1. Single LLM, zero-shot and CoT.
2. Single LLM + self-consistency at *equal token budget* to the MAS (the fair baseline the field usually omits).
3. Static role MAS (MAM-style).
4. Debate + judge MAS (MDTeamGPT-style).
5. ConfAgents-style triage; MedRoute-style routing.
6. Classical VoI / RL sequential diagnosis on DDXPlus.
7. **Oracle**: full workup, unlimited budget.

### 5.7 Metrics

Primary: **accuracy-at-fixed-cost** (area under the accuracy–cost curve), and **harm-adjusted utility** = correct·1 − missed-can't-miss·w − unnecessary-invasive-test·c.

Secondary: top-1/top-3 accuracy per tier; tests per case; cost per correct diagnosis; unnecessary test rate; missed can't-miss rate; appropriate referral rate; over-referral rate; ECE; selective risk at α; abstention rate by tier; equity gap (T0 − T3).

### 5.8 Experiment plan

- **E1 Main grid:** 4 tiers × 6 systems × 3 model families (one open 7–8B, two API) × 5 seeds. Report CIs, paired bootstrap tests.
- **E2 Ablation:** planner only / safety agent only / both / neither; VoI features (cost, invasiveness, turnaround) removed one at a time.
- **E3 Budget sweep:** λ from 0 to ∞ → accuracy–cost frontier.
- **E4 Robustness:** missing results, delayed results, false-negative tests, conflicting repeat tests.
- **E5 Risk control:** empirical coverage vs nominal α per tier; distribution shift across specialties.
- **E6 Error analysis:** clinician-coded 100 cases — was the *decision* (test/refer/diagnose) appropriate, independent of the final label?
- **E7 (optional, high value):** 10–15 clinicians rate 60 outputs blind for safety and actionability.

### 5.9 Threats to validity (write these in the paper before reviewers do)

- Synthetic cases (DDXPlus) ≠ real patients; mitigate with MedQA tiering + MIMIC order-pattern check.
- Test-availability tiers are a simplification; ground them in WHO IMCI and a real district-hospital test menu.
- LLM-as-judge for "appropriateness" is weak; use human coding for the E6 subset.
- Conformal guarantees assume exchangeability; report under shift honestly rather than claiming coverage you don't have.

### 5.10 Contributions (the paper's contribution list)

1. **Problem formulation:** diagnosis as risk-controlled evidence acquisition under resource constraints — a different objective from the accuracy-on-vignettes paradigm.
2. **FrugalDx benchmark:** the same cases evaluated under graded resource tiers with cost, availability, and can't-miss annotations.
3. **Method:** a planner + safety agent + risk-controlled aggregator that improves the accuracy–cost frontier and the equity gap.
4. **Evidence:** a systematic demonstration that current MAS advantages are partly an artefact of unlimited evidence access, with compute-matched baselines.

### 5.11 Target venues

npj Digital Medicine · JAMIA · MLHC · CHIL · MEDINFO · AAAI/ACL (methods track). Workshop first if you need a deadline: ML4H, CHIL short papers.

### 5.12 6-month timeline

| Month | Deliverable |
|---|---|
| 1 | Literature lock + novelty check; define tiers and cost model with a clinician; DDXPlus pipeline |
| 2 | Baseline harness (6 systems, compute matching, logging); E1 pilot on 200 cases |
| 3 | Resource Planner v1 + Safety Agent; E2 ablations |
| 4 | Full E1 + E3; risk control (E5); error taxonomy |
| 5 | Robustness (E4); clinician coding (E6); writing |
| 6 | Submit; release benchmark + code |

---

## 6. Optional quick-win first paper (2–3 months, APIs only)

If you want a publication on the board before the main project: **"The evidence budget matters: a compute-matched re-evaluation of multi-agent medical diagnosis."**

Take 4–5 representative systems from Cluster A, evaluate on 2–3 benchmarks, and compare against a single agent given the **same token budget** as self-consistency, plus the same retrieval. Report paired tests across 5 seeds. The general version of this critique already exists (2606.13003, 2604.02460) — the *medical* version does not, and the medical-specific twist is that evidence access, not reasoning depth, is the confound.

This is a workshop/short-paper contribution, not a main-track paper on its own. Use it to build the harness you need for P2.

---

## 7. How to defend novelty before you commit

Run these on arXiv, Semantic Scholar, and Google Scholar (title + abstract scan of the last 12 months). Record the date you ran them; put "novelty checked on <date>" in your notes.

```
"multi-agent" AND medical AND ("resource-constrained" OR "low-resource" OR "test availability")
"multi-agent" AND diagnosis AND ("test ordering" OR "value of information" OR "evidence acquisition")
"multi-agent" AND ("cost-aware" OR "budget") AND medical AND diagnosis
"multi-agent" AND ("referral" OR "escalation") AND ("conformal" OR "risk control")
"multi-agent" AND ("can't-miss" OR "red flag" OR "safety-netting") AND diagnosis
"frugal" OR "point-of-care" AND ("language model") AND diagnosis
```

Then check the three surveys' "future work" lists — if a direction appears there and no paper has claimed it, that is the strongest possible justification paragraph for your introduction.

**Specific papers to read before writing (they are the ones most likely to be compared to you):**
- 2508.04915 ConfAgents · 2604.06180 MedRoute · 2610.33544 SMART-Med — cost/routing
- 2607.24814 Aletheia · 2609.00256 NSIDDx — low-resource
- 2607.15280 GraphDx · 2610.34878 ClinVerse · 2606.03416 MeDxAgent · 2605.07305 MedAction — sequential/interactive/test-ordering
- 2512.01241 NOHARM · 2507.15743 g-AMIE — human teaming and oversight
- 2606.13003 Illusion of MAS advantage · 2604.02460 equal-token-budget — compute matching
- 2505.20824 MedSentry — safety

---

## 8. What to change in how you work

1. **Stop collecting architecture papers.** You have enough. Collect *evaluation* papers now.
2. **Write the related-work section around a claim, not a list.** "The field has optimised accuracy on fully specified vignettes; we show this assumption is the binding constraint."
3. **Fix your evaluation before your method.** Pre-register metrics, seeds, and the compute-matching rule. Reviewers in 2026 are armed with the compute-confound critique.
4. **Get one clinician co-author.** It is the cheapest way to raise the paper's credibility and it unlocks the E6/E7 studies.
