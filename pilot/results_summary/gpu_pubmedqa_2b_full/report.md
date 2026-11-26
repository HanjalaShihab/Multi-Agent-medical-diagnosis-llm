# BTS audit report - gpu_pubmedqa_2b

Backend: **hf** | Cases: **300** | Cells: **H0, H1, H2, H3, H4, H5, SYC**

## Verdict

> **NO-GO / revise.** At least one gate failed. Read the weak row before spending more compute.

## Prediction scorecard

| # | Result | Key numbers |
|---|---|---|
| P1 | **FAIL** | agreement=0.987, score var=0.004 |
| P2 | **PASS** | dAUC=0.000, p=0.994 |
| P3 | **PASS** | dAUC=0.050, p=0.138 |
| P4 | **PASS** | dAUC=-0.099, p=0.006 |
| P5 | **FAIL** | dAUC=-0.019, rho H4=0.311, rho H3=0.364 |
| P6a | **FAIL** | elicited within-case rarity slope=-0.981 nats/nat |
| P6b | **PASS** | oracle max |gap| H0-H3=0.000 nats (mechanical); elicited range=[-0.129, 0.140] |
| P7 | **PASS** | flip=0.565, BTS acc=0.333 vs majority=0.320 |
| P8 | **PARTIAL** | oracle dAUC H1-H3=[-0.141, -0.081, -0.068] |

## Per-cell metrics

| Cell | n | Acc | Agree | AUC score | AUC conf | dAUC | boot p | ECE conf | ECE forecast | rho | Rare prem (elic) | Oracle gap | Elic gap | Oracle dAUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H0 | 1500 | 0.564 | 0.987 | 0.494 | 0.626 | -0.132 | 0.000 | 0.291 | 0.051 | 0.500 | -1.420 | n/a (0) | n/a (0) | -0.120 |
| H1 | 1500 | 0.495 | 0.403 | 0.590 | 0.590 | 0.000 | 0.994 | 0.304 | 0.035 | 0.326 | -0.914 | -0.000 (19) | 0.100 (19) | -0.141 |
| H2 | 1500 | 0.506 | 0.390 | 0.605 | 0.555 | 0.050 | 0.138 | 0.292 | 0.040 | 0.336 | -0.905 | -0.000 (27) | 0.068 (27) | -0.081 |
| H3 | 1500 | 0.501 | 0.400 | 0.514 | 0.613 | -0.099 | 0.006 | 0.301 | 0.033 | 0.364 | -0.956 | 0.000 (27) | -0.129 (27) | -0.068 |
| H4 | 1500 | 0.406 | 0.283 | 0.536 | 0.554 | -0.019 | 0.548 | 0.404 | 0.069 | 0.311 | -1.026 | -0.000 (31) | 0.140 (31) | -0.077 |
| H5 | 1500 | 0.398 | 0.283 | 0.530 | 0.553 | -0.023 | 0.424 | 0.396 | 0.050 | 0.342 | -0.864 | 0.000 (38) | -0.053 (38) | -0.094 |
| SYC | 1500 | 0.281 | 0.387 | 0.513 | 0.544 | -0.031 | 0.410 | 0.536 | 0.080 | 0.405 | -0.778 | 0.000 (33) | 0.005 (33) | -0.011 |

Gap cells show `estimate (matched case-frequency groups)`; `n/a (0)` means no case had both a correct and a wrong answer at the same frequency.

## Sycophancy control (P7)

- Status: **PASS**
- Agent answer flip rate under pressure: **0.565**
- BTS-selected answer flip rate: **0.450**
- Majority-selected answer flip rate: **0.413**
- BTS selection accuracy under pressure: **0.333**
- Majority selection accuracy under pressure: **0.320**

## Data limitations

- Dataset: **pubmedqa**. All headline datasets are human-written public benchmarks; none is raw patient-record data.
- Real QA benchmarks carry **no true disease-prevalence prior**. `rarity_stratum` and `prevalence` are therefore `None`; they are never inferred from answer-option frequency.
- **P6a** is the within-case rarity slope of the BTS information score and **P6b** is the fixed-frequency oracle gap. Both are mechanical properties of the mechanism. Claims about *disease* rarity or prevalence are **not testable on this suite**.
- PubMedQA supplies genuine, human-written abstract sections (`background`, `case`, `methods`, `results`, `conclusions`, `other`). H4/H5 are a real evidence partition here. Some abstracts have fewer than five sections, so agents may share a section.
- MedMCQA uses its validation split because the official test split withholds labels (`cop == -1`).

## How to read this

- **dAUC = AUC(BTS) - AUC(confidence).** Near zero or negative means BTS adds no positive ranking signal beyond confidence. A negative value is the expected popularity effect when the team is accurate: the correct answer is common, so the rarity statistic penalises it.
- **Rare premium (elic)** is the within-case slope of the elicited BTS information score on answer rarity ($-\log \bar x$). It is a mechanical property of the score, not evidence that rare *diseases* are better diagnosed. The oracle premium (not shown) is the control: it should be near zero.
- **Oracle gap** is the same-frequency score gap using leave-one-out empirical forecasts. Proposition 1 makes it identically zero: at a fixed answer count the mechanism cannot tell correct from wrong. A non-zero value means a bug.
- **Elic gap** is the same-frequency gap using the agents' actual forecasts. Negative means the score penalises correct answers at fixed frequency (the popularity effect). Positive means forecasts inject a truth signal: either private evidence or miscalibration. Compare with P8 before believing the positive case.
- **Oracle dAUC** replaces elicited forecasts with the leave-one-out empirical answer distribution. If dAUC collapses to zero in H1-H3, the apparent signal was forecast miscalibration, not private evidence.
- **rho** is Pr(correct | answer is a minority). BTS helps only when rho is above the majority's accuracy.

## Files

- `agent_scores.csv` - one row per agent per case with every score component
- `cell_metrics.csv` - the table above
- `metrics.json` - all numbers
- `predictions.json` - the P1-P8 scorecard
