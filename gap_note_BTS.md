# Gap Note — In-Context Bayesian Truth Serum (BTS) for Medical Multi-Agent Consensus

**Verdict:** publishable, but not as "apply BTS to medical sycophancy". That version is (a) already published in general form and (b) theoretically broken in the setting you want to use it. The publishable version is: **"when does peer-prediction carry information in a clinical agent team, and what happens when you weight it for harm?"**

Status: novelty checked on arXiv, Nov 2026. Re-run §8 before submission.

---

## 1. What already exists (cite these or you will be rejected)

| Paper | What it claims | What it takes away from you |
|---|---|---|
| **2608.25267** (Aug 2026) *Mitigating LLM sycophancy with RL-based fine-tuning: Bayesian Truth Serum approach* | BTS as a GRPO reward; proves sycophantic answers earn lower expected reward; flip rate 23% → 4% | "BTS reduces sycophancy" — gone |
| **2601.20299** *Truthfulness Despite Weak Supervision* | Peer prediction for LLM evaluation + post-training, with guarantees, up to 405B | "Peer prediction trains truthful LLMs" — gone |
| **2610.33728** *Strategic Calibration in LLM Multi-Agent Systems: Audited Pivot Scoring* | Truthful confidence reporting when confidence determines allocation; unique best response | "Mechanism design for LLM MAS reporting" — gone |
| **2509.25184** *Incentive-Aligned Multi-Source LLM Summaries* | Multi-task peer prediction over conflicting sources; formal alignment guarantees | "Peer prediction for conflicting LLM evidence" — gone |
| **2607.12985** *Resist and Update* | Separates pressure from evidence via source reliability; incentive-compatible reporting | "Distinguishing deference from sycophancy" — partly gone |

**Not taken:** BTS × medical = 0 papers. BTS × multi-agent = 0 papers. Peer prediction × medical diagnosis = 0 papers.

So the *intersection* is empty. The question is whether it is empty because nobody thought of it (good) or because the assumptions fail (bad). It is mostly the second.

---

## 2. Formal setup

Agents `i = 1..n`. Each reports an answer `x_i ∈ X` (a diagnosis, or a ranked differential collapsed to a label) and a predicted distribution over the other agents' answers `y_i ∈ Δ(X)`.

- Empirical frequency: `ā(x) = (1/n) Σ_i 1[x_i = x]`
- Mean prediction: `ȳ(x) = (1/n) Σ_i y_i(x)`
- BTS information score: `s_i = ln( ā(x_i) / ȳ(x_i) )`
- Total score: `s_i + prediction_score(y_i, ā)` (quadratic rule)

Truthfulness is a Bayesian Nash equilibrium when (i) agents hold private information, (ii) they can predict each other, (iii) the correct answer is *surprisingly common* — more frequent than agents predict.

---

## 3. Proposition 1 — BTS collapses to majority voting on a homogeneous team

**Claim.** If agents are exchangeable (same base model, same prompt, same evidence, same decoding distribution `p`), then in expectation `ā(x) = ȳ(x) = p(x)` for every `x`, so `E[s_i] = 0` for every answer. The BTS ranking is then a noisy monotone function of the marginal frequency `p(x)` — i.e. **it can never outperform frequency-based aggregation, and it carries no information that voting does not already have.**

*Proof sketch.* Exchangeability ⇒ the best predictor of another agent's answer is `p` itself, so a calibrated agent reports `y_i = p`. The empirical frequency concentrates at `p` by the law of large numbers. Then `ln(p(x)/p(x)) = 0`. Finite-`n` sampling noise is mean-zero and independent of correctness. Therefore the only signal in the score is the ordering induced by `p(x)` — which is exactly majority voting's signal.

**Corollary (the design principle).** BTS adds value only to the extent that agents have **conditionally independent information given the diagnosis**. In a standard medical MAS every agent reads the same vignette, so conditional independence is ≈ 0 by construction.

**Implication for your design.** You cannot fix this with prompting. You must *create* private signals — e.g. **evidence partitioning**: the historian agent sees HPI and review of systems; the lab agent sees labs; the imaging agent sees the report; each specialist sees its own guideline excerpt. This is clinically natural (it mirrors a real chart being read by different people) and is the single design choice that makes the mechanism meaningful.

---

## 4. Proposition 2 — Severity weighting destroys truthfulness

**Claim.** Let `S` be a strictly proper scoring rule. Score the agent with `w(θ) · S(x, θ)` where `w` depends on the *true* diagnosis `θ` (e.g. `w = 5` for a can't-miss condition). Then the agent's best response is the tilted posterior `q ∝ p · w`, not its true posterior `p`. Strict propriety is lost; the mechanism now elicits a **decision policy**, not a **belief**.

*Why it matters clinically.* The whole point of a truth serum is to get the agent's honest belief, because honest beliefs are what let a clinician calibrate trust. If you weight for harm, you get an agent that strategically over-calls severe diagnoses. That may be *desirable* — but it is no longer truth elicitation, and you cannot also claim calibration.

**Consequence.** Harm asymmetry has to enter as a **constraint**, not a reweighted score. E.g. a lexicographic rule: *never allow the can't-miss diagnosis to be excluded from the differential, regardless of score*. That preserves propriety on the belief layer while enforcing safety on the decision layer.

This pairing — Proposition 1 + Proposition 2 — is the theoretical core of the paper. Neither is hard, but together they say something the field needs: **the two things people want from BTS in medicine (truthful beliefs and harm-aware decisions) cannot both be obtained from a single score.**

---

## 5. The paper this becomes

**Title:** *Truth Serum or Popularity Contest? Peer-Prediction Mechanisms in Clinical Multi-Agent Diagnosis*

**Contributions**
1. **C1 (theory).** BTS degenerates to frequency aggregation for exchangeable agents; value requires conditional independence (Prop. 1).
2. **C2 (theory).** Harm-weighted scoring breaks strict propriety, so safety must be a constraint (Prop. 2).
3. **C3 (benchmark/audit).** Prevalence-stratified, pressure-stratified evaluation of BTS vs voting/judge/debate in medicine — including the **rare-disease penalty**: does "surprisingly common" punish the correct atypical diagnosis?
4. **C4 (method).** *Evidence-partitioned BTS* (private signals created by chart partitioning) + severity-constrained scoring; shows where the mechanism becomes informative.
5. **C5 (negative result with a fix).** In the default medical MAS setup, BTS underperforms plain voting; report it.

**Figures**
- F1: score–correctness AUC vs. agent correlation ρ (the collapse curve).
- F2: accuracy by prevalence stratum (common / uncommon / rare) for each mechanism — the rare-disease penalty.
- F3: flip rate under patient pressure vs. peer pressure vs. authority pressure.
- F4: accuracy–cost frontier for voting / judge / debate / BTS / partitioned-BTS.

---

## 6. The go/no-go experiment (2 weeks)

Run this **before** writing anything.

**Setup.** 300–500 cases from MedQA + MedMCQA + the 302-case Orphanet rare-disease set you already have. Four heterogeneity levels:

| Level | Setup | Purpose |
|---|---|---|
| H0 | identical model, temp 0, identical prompt | the degeneracy prediction: AUC ≈ 0.50 |
| H1 | identical model, temp 0.7, independent samples | sampling diversity only |
| H2 | different personas/prompts, same model | prompt diversity |
| H3 | different model families | model diversity |
| H4 | **evidence-partitioned** agents (each sees a different slice) | genuine private information |

**Measure.** For each level: `AUC(BTS information score → correctness)`, accuracy of the BTS-selected answer vs. majority vote, and the **rare-disease penalty** (Δ accuracy common vs. rare).

**Decision rule**

- **GO** if: AUC(H0) ≤ 0.55, AUC(H4) ≥ 0.65, rare-disease penalty is measurable (> 5 points) and the partitioned variant closes ≥ half of it.
- **PIVOT** if: AUC is flat across H0–H4 (BTS carries no signal anywhere in medicine) → this is still a publishable *negative* result, but only as a workshop paper or as a section inside a larger paper.
- **STOP** if: BTS ≈ voting everywhere and there is no prevalence effect → there is no paper here; fold the effort into the resource-aware project.

Budget: ~$300–800 in API calls. Time: 10–14 days including analysis.

---

## 7. Effort, venue, timeline

| Item | Estimate |
|---|---|
| Go/no-go pilot | 2 weeks |
| Full experiment grid | 5–6 weeks |
| Theory write-up | 1–2 weeks (needs care; get a co-author who knows mechanism design) |
| Paper + revisions | 3–4 weeks |
| **Total to submission** | **~3–4 months** |

Venues (verify deadlines — they shift):
- **AAMAS** — natural home for the theory; mechanism design + MAS audience.
- **MLHC / CHIL** — if you lead with the clinical safety finding (rare-disease penalty).
- **npj Digital Medicine / JAMIA** — if you add a clinician-graded evaluation.
- **IJCAI / NeurIPS / ACL** — only if Prop. 1/2 are developed into something stronger than a two-page argument.

Realistic first target given Nov 2026: **CHIL or IJCAI (Jan–Feb)**, otherwise **MLHC (May)**. AAMAS 2027 is ~Oct 2027.

---

## 8. Risks and how to kill them

| Risk | Severity | Mitigation |
|---|---|---|
| Scooped by a medical BTS paper | Medium–high | Move fast; publish the pilot as an arXiv preprint once C1+C3 hold |
| Reviewer: "this is 2608.25267 with a medical dataset" | **High** | Lead with C1+C2 (theory), not with the application. Cite 2608.25267 in the abstract and state the delta explicitly |
| Reviewer: "LLMs don't have utilities, so incentive compatibility is a metaphor" | **High** | Address it head-on in a limitations subsection; test three instantiations: in-context instruction, score-based selection, GRPO-style training. Report all three |
| Homogeneous-team result is trivially expected | Medium | Make the *correlation–AUC curve* the contribution, not the point estimate |
| `n` too small → noisy estimates | Low | 9–15 agents; report CIs; 5 seeds |
| **"In-context" turns out to be the weakest lever** | Medium | Do not put "in-context" in the title. Treat it as one condition among three |

**Naming note.** Drop "In-Context" from the headline. It advertises the weakest part of the idea. "In-context" is a prompt; the interesting claim is about *information structure*.

---

## 9. Novelty re-check (run before submission)

```
"Bayesian truth serum" AND (medical OR clinical OR diagnosis)
"peer prediction" AND (medical OR clinical) AND ("language model" OR LLM)
"truthful elicitation" AND ("multi-agent" OR "agent team") AND clinical
"incentive" AND "sycophancy" AND ("multi-agent" OR consensus)
"proper scoring rule" AND "multi-agent" AND ("language model" OR LLM)
"evidence partitioning" AND ("multi-agent" OR "agent") AND diagnosis
```

Record the date. If 2608.25267 has a v2 that adds multi-agent or medical, re-scope immediately.

---

## 10. How this compares to the FrugalDx program

| | BTS paper | FrugalDx (resource-aware MAS) |
|---|---|---|
| Time to submission | 3–4 months | 5–7 months |
| Compute | APIs only | APIs only |
| Novelty type | Theory + audit | Task formulation + benchmark + method |
| Scoop risk | High (fast-moving mechanism-design area) | Low–medium (context moat) |
| Ceiling | AAMAS / MLHC | npj DM / JAMIA / MLHC |
| Failure mode | "Incremental over 2608.25267" | "Just an application paper" |
| Best used as | A sharp second paper, or a workshop paper first | The main project |

**Recommendation:** do the 2-week pilot now. If it passes the gate, it is a genuinely good paper and faster than FrugalDx. If it fails, you have lost two weeks and gained a strong related-work section for the main project — the degeneracy argument is exactly the kind of thing that makes a "why multi-agent consensus is fragile" subsection in the FrugalDx paper.
