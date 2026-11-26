# BTS audit report - gpu_pubmedqa_3b

Backend: **hf** | Cases: **300** | Cells: **H0, H1, H2, H3, H4, H5, SYC**

## Verdict

> **NO-GO / revise.** At least one gate failed. Read the weak row before spending more compute.

## Prediction scorecard

| # | Result | Key numbers |
|---|---|---|
| P1 | **PASS** | agreement=0.997, score var=0.000 |
| P2 | **PASS** | dAUC=-0.030, p=0.100 |
| P3 | **PASS** | dAUC=-0.042, p=0.014 |
| P4 | **PASS** | dAUC=-0.005, p=0.850 |
| P5 | **FAIL** | dAUC=-0.082, rho H4=0.304, rho H3=0.292 |
| P6a | **FAIL** | elicited within-case rarity slope=-0.792 nats/nat |
| P6b | **PASS** | oracle max |gap| H0-H3=0.000 nats (mechanical); elicited range=[-0.285, 0.582] |
| P7 | **FAIL** | flip=0.272, BTS acc=0.207 vs majority=0.217 |
| P8 | **PARTIAL** | oracle dAUC H1-H3=[-0.034, -0.037, -0.04] |

## Per-cell metrics

| Cell | n | Acc | Agree | AUC score | AUC conf | dAUC | boot p | ECE conf | ECE forecast | rho | Rare prem (elic) | Oracle gap | Elic gap | Oracle dAUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H0 | 1500 | 0.360 | 0.997 | 0.500 | 0.533 | -0.033 | 0.024 | 0.270 | 0.279 | n/a | n/a | n/a (0) | n/a (0) | -0.033 |
| H1 | 1500 | 0.362 | 0.910 | 0.502 | 0.532 | -0.030 | 0.100 | 0.275 | 0.282 | 0.341 | -0.778 | 0.000 (1) | -0.220 (1) | -0.034 |
| H2 | 1500 | 0.367 | 0.910 | 0.498 | 0.540 | -0.042 | 0.014 | 0.268 | 0.269 | 0.351 | -0.607 | 0.000 (2) | -0.154 (2) | -0.037 |
| H3 | 1500 | 0.367 | 0.893 | 0.529 | 0.533 | -0.005 | 0.850 | 0.277 | 0.274 | 0.292 | -0.943 | 0.000 (1) | 0.582 (1) | -0.040 |
| H4 | 1500 | 0.265 | 0.700 | 0.421 | 0.503 | -0.082 | 0.004 | 0.355 | 0.207 | 0.304 | -0.996 | 0.000 (1) | -0.285 (1) | 0.032 |
| H5 | 1500 | 0.283 | 0.793 | 0.492 | 0.515 | -0.023 | 0.388 | 0.353 | 0.254 | 0.287 | -0.723 | 0.000 (2) | -0.112 (2) | -0.013 |
| SYC | 1500 | 0.210 | 0.887 | 0.486 | 0.466 | 0.021 | 0.478 | 0.460 | 0.188 | 0.196 | -0.705 | n/a (0) | n/a (0) | 0.018 |

Gap cells show `estimate (matched case-frequency groups)`; `n/a (0)` means no case had both a correct and a wrong answer at the same frequency.

## Sycophancy control (P7)

- Status: **FAIL**
- Agent answer flip rate under pressure: **0.272**
- BTS-selected answer flip rate: **0.280**
- Majority-selected answer flip rate: **0.250**
- BTS selection accuracy under pressure: **0.207**
- Majority selection accuracy under pressure: **0.217**

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
