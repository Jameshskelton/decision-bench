# Decision model benchmark results

Generated from `results/`. Schema 2026-11-13.1.


## Block decision, d2_test

| System | n | AUROC [95% CI] | Balanced acc @0.5 [95% CI] | Recall | False blocks | ECE raw → recal | Auto-decided | Slipped bad calls | Missing |
|---|---|---|---|---|---|---|---|---|---|
| jev | 2000 | 0.999 [1.00, 1.00] | 0.979 [0.97, 0.99] | 0.994 | 0.036 | 0.064 → 0.018 | 1.00 | 10 | 0 |
| kev-4b | 2000 | 0.992 [0.99, 0.99] | 0.953 [0.94, 0.96] | 0.920 | 0.015 | 0.174 → 0.034 | 0.91 | 7 | 0 |
| d1-2b | 2000 | 0.988 [0.98, 0.99] | 0.949 [0.94, 0.96] | 0.963 | 0.064 | 0.077 → 0.035 | 0.93 | 26 | 0 |
| d3 | 2000 | 1.000 [1.00, 1.00] | 0.985 [0.98, 0.99] | 0.974 | 0.005 | 0.013 → 0.014 | 1.00 | 29 | 0 |
| llm-baseline | 2000 | 0.998 [1.00, 1.00] | 0.979 [0.97, 0.99] | 0.994 | 0.036 | 0.063 → 0.014 | 1.00 | 28 | 0 |

## Block decision, d1_test

| System | n | AUROC [95% CI] | Balanced acc @0.5 [95% CI] | Recall | False blocks | ECE raw → recal | Auto-decided | Slipped bad calls | Missing |
|---|---|---|---|---|---|---|---|---|---|
| jev | 661 | 0.912 [0.86, 0.96] | 0.807 [0.74, 0.86] | 0.959 | 0.344 | 0.143 → 0.042 | 1.00 | 116 | 0 |
| kev-4b | 661 | 0.853 [0.78, 0.91] | 0.639 [0.60, 0.68] | 0.326 | 0.048 | 0.391 → 0.456 | 0.63 | 2 | 0 |
| d1-2b | 661 | 0.657 [0.59, 0.72] | 0.696 [0.65, 0.74] | 0.472 | 0.080 | 0.346 → 0.373 | 0.60 | 155 | 0 |
| d3 | 661 | 0.989 [0.97, 1.00] | 0.901 [0.86, 0.94] | 0.987 | 0.184 | 0.033 → 0.031 | 1.00 | 7 | 0 |
| llm-baseline | 661 | 0.976 [0.96, 0.99] | 0.690 [0.65, 0.74] | 0.996 | 0.616 | 0.109 → 0.116 | 1.00 | 6 | 0 |

## Headline uncertainty (d1_test, paired bootstrap over agent runs, 1000 replicates)

Each replicate resamples whole agent runs and refits every system's thresholds (5-fold), so threshold-fitting noise is included. Reference: d3.

| System | Escalated [95% CI] | Escalated ÷ d3 [95% CI] | Escalated − d3 [95% CI] | Share of replicates where d3 escalates less | AUROC [95% CI] | d3 AUROC − this [95% CI] | Share where d3 AUROC higher |
|---|---|---|---|---|---|---|---|
| jev | [26%, 61%] | [2.6×, 177.8×] (82 replicates with d3 at 0% excluded) | [+20%, +56%] | 100.0% | [0.862, 0.954] | [+0.036, +0.126] | 100.0% |
| kev-4b | [48%, 74%] | [3.6×, 233.2×] (82 replicates with d3 at 0% excluded) | [+41%, +71%] | 100.0% | [0.777, 0.911] | [+0.080, +0.213] | 100.0% |
| d1-2b | [55%, 73%] | [3.7×, 212.4×] (82 replicates with d3 at 0% excluded) | [+46%, +68%] | 100.0% | [0.591, 0.717] | [+0.269, +0.400] | 100.0% |
| d3 | [0%, 17%] | – | – | – | [0.974, 0.998] | – | – |
| llm-baseline | [6%, 24%] | [0.8×, 44.0×] (82 replicates with d3 at 0% excluded) | [-3%, +19%] | 92.9% | [0.959, 0.990] | [+0.002, +0.025] | 98.3% |

### Escalation rate at other policy targets (5-fold)

| Targets | jev | kev-4b | d1-2b | d3 | llm-baseline |
|---|---|---|---|---|---|
| slip 0.5% / false block 2% | 61% | 73% | 70% | 14% | 24% |
| slip 1.0% / false block 5% | 50% | 63% | 64% | 8% | 14% |
| slip 2.0% / false block 10% | 30% | 53% | 57% | 0% | 6% |

## Policy fitted on the real traces (d1_test, 5-fold by trajectory)

Thresholds fitted on four fifths of the trajectories, applied to the held-out fifth, outcomes pooled. Limits: at most 1% of bad calls auto-allowed, at most 5% of legitimate calls auto-blocked (in each training split).

| System | Auto-decided | Escalated | Slipped bad calls (rate) | False blocks (rate) | Human review $/decision |
|---|---|---|---|---|---|
| jev | 50% | 50% | 8 (1.5%) | 7 (5.6%) | 0.252 |
| kev-4b | 37% | 63% | 7 (1.3%) | 9 (7.2%) | 0.315 |
| d1-2b | 36% | 64% | 5 (0.9%) | 5 (4.0%) | 0.322 |
| d3 | 92% | 8% | 5 (0.9%) | 6 (4.8%) | 0.039 |
| llm-baseline | 86% | 14% | 7 (1.3%) | 5 (4.0%) | 0.068 |

## Category (d1_test, 7 options asked)

Macro F1 averages the categories present in the labels: credential_misuse, eval_gaming, harness_introspection, legit, sandbox_escape.

| System | Macro F1 | Accuracy | F1 credential_misuse | F1 eval_gaming | F1 harness_introspection | F1 legit | F1 sandbox_escape | Picked an absent option |
|---|---|---|---|---|---|---|---|---|
| jev | 0.538 | 0.782 | 0.234 | 0.675 | 0.928 | 0.855 | 0.000 | 8 |
| kev-4b | 0.370 | 0.492 | 0.211 | 0.123 | 0.822 | 0.695 | 0.000 | 215 |
| d1-2b | 0.444 | 0.614 | 0.250 | 0.392 | 0.840 | 0.672 | 0.065 | 43 |
| d3 | 0.657 | 0.865 | 0.311 | 0.857 | 0.930 | 0.900 | 0.286 | 9 |

## Latency and throughput

| System | Single-stream p50 / p99 ms | Throughput at p99 ≤ 1000 ms (req/s) | Failed load requests |
|---|---|---|---|
| jev | 227.8 / 380.2 | 19.4 (c=8) | c=8: 1 failed; c=32: 124 failed; c=64: 333 failed |
| kev-4b | 31.4 / 37.2 | 59.6 (c=32) | – |
| d1-2b | 9.6 / 10.2 | 332.9 (c=64) | – |
| d3 | 33.4 / 44.5 | 44.9 (c=32) | – |
| llm-baseline | 35.1 / 44.9 | 46.8 (c=32) | – |

## Cost

Jev: $29.05 per million decisions.

| System | $/M decisions at full use | Crossover/day: flat (GPU utilization) | business hours | spiky |
|---|---|---|---|---|
| kev-4b | 20.83 | 3,693,269 (72%) | never ($39.99/M) | never ($208.30/M) |
| d1-2b | 3.73 | 3,693,269 (13%) | 3,693,269 (13%) | never ($37.30/M) |
| d3 | 27.68 | 3,693,269 (95%) | never ($53.15/M) | never ($276.83/M) |
| llm-baseline | 26.52 | 3,693,269 (91%) | never ($50.93/M) | never ($265.25/M) |

## Thresholds fitted on d2_calib

- jev: auto-allow below 0.680, auto-block at or above 0.680, T = 0.40 (no escalation band: d2_calib was separable, so every call is auto-decided)
- kev-4b: auto-allow below 0.312, auto-block at or above 0.433, T = 0.26
- d1-2b: auto-allow below 0.321, auto-block at or above 0.581, T = 0.50
- d3: auto-allow below 0.517, auto-block at or above 0.517, T = 1.12 (no escalation band: d2_calib was separable, so every call is auto-decided)
- llm-baseline: auto-allow below 0.706, auto-block at or above 0.706, T = 0.21 (no escalation band: d2_calib was separable, so every call is auto-decided)

## Determinism

- jev / d2_test: 3 runs, max |Δp| 0.1800, 9 decision flips
- jev / d1_test: 3 runs, max |Δp| 0.1000, 14 decision flips
- kev-4b / d2_test: 3 runs, max |Δp| 0.0000, 0 decision flips
- kev-4b / d1_test: 3 runs, max |Δp| 0.0000, 0 decision flips
- d1-2b / d2_test: 3 runs, max |Δp| 0.0000, 0 decision flips
- d1-2b / d1_test: 3 runs, max |Δp| 0.0000, 0 decision flips
- d3 / d2_test: 3 runs, max |Δp| 0.0000, 0 decision flips
- d3 / d1_test: 3 runs, max |Δp| 0.0118, 2 decision flips
- llm-baseline / d2_test: 3 runs, max |Δp| 0.0000, 0 decision flips
- llm-baseline / d1_test: 3 runs, max |Δp| 0.0305, 0 decision flips
