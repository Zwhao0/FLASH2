# ETP=1 dual-view 36-run matrix score

Shorter wall-clock observation replaces150steps; fresh workers eachcase from CPU-only torch-preloaded forkserver. Different observation length and padding from previous cohorts; no pooled accuracy or improvement claim. Fixed host delays are proxies, not DVFS or paper percentage labels.

The original 36 row names and order define 28 positive cases and 8 controls. The amendment uses a 25 second wall horizon after synchronized step 45, zero padding, and a 10,000 step safety cap. Each run's measurement length comes from matching `run_summary.actual_steps` across all 16 ranks. Fixed host-delay values remain 14, 100, and 200 ms.

Valid runs: 28/36; exact positives: 19/28; clean controls: 3/8; conditional precision/recall: 95.0%/90.5%; conservative precision/recall: 95.0%/67.9%; observed wrong-rank predictions across all rows: 1; refusals: 4.

Conditional metrics use valid runs; valid controls contribute false positives. Invalid or unrun positives receive no TP credit, while any readable terminal false positive remains counted. Exact accuracy keeps all 28 planned positives in its denominator. This matrix is separate from both the earlier 22-run matrix and the three-run pilot.

Timing uses rank0 step 10–40 for baseline, step 50 through the last executed step for post-fault, steps 50–59 for early post-fault, and the final 10 executed steps. These are iteration-time measurements and an inverse-iteration-time throughput proxy only; `time_ms` excludes the Gloo stop broadcast, the following batch, and its initialization barrier, so it is not an end-to-end cost. Planned delays are not interpreted as paper percentages.

## Per-run results

| Run | Topology | Node / PP stage | Delay ms | Actual steps | Attempt | Valid | Online union | TP | Wrong ranks | Refusals | Profile | Baseline → post ms | Early post ms | Last 10 ms | Verify ranks |
|---|---|---|---:|---:|---|---|---|---:|---|---:|---|---:|---:|---:|---:|
| fdm-1021-e8t1p1-control-before | 8/1/1 | — / — | 0 | 667 | completed | yes | [] | 0 | [] | 0 | not_triggered (0/0) | 22.77 → 24.62 | 24.27 | 25.23 | 16/16 |
| fdm-1021-e8t1p1-low-n0 | 8/1/1 | 0 / 0 | 14 | 435 | completed | no | [0] | 0 | [] | 0 | incomplete (0/1) | 98.43 → 48.20 | 52.39 | 248.00 | 16/16 |

- fdm-1021-e8t1p1-low-n0 failure reasons: one or more triggered profiles failed profile acceptance
| fdm-1021-e8t1p1-high-n1 | 8/1/1 | 1 / 0 | 200 | 146 | completed | no | [9] | 0 | [] | 0 | incomplete (0/1) | 43.27 → 235.61 | 236.54 | 297.50 | 16/16 |

- fdm-1021-e8t1p1-high-n1 failure reasons: one or more triggered profiles failed profile acceptance
| fdm-1021-e8t1p1-medium-n0 | 8/1/1 | 0 / 0 | 100 | 218 | completed | no | [0] | 0 | [] | 0 | incomplete (0/1) | 98.43 → 135.01 | 135.10 | 146.77 | 16/16 |

- fdm-1021-e8t1p1-medium-n0 failure reasons: one or more triggered profiles failed profile acceptance
| fdm-1021-e8t1p1-low-n1 | 8/1/1 | 1 / 0 | 14 | 444 | completed | yes | [] | 0 | [] | 0 | not_triggered (0/0) | 25.64 → 58.98 | 50.84 | 59.61 | 16/16 |
| fdm-1021-e8t1p1-high-n0 | 8/1/1 | 0 / 0 | 200 | 146 | completed | no | [0] | 0 | [] | 0 | incomplete (0/1) | 25.12 → 235.83 | 235.55 | 275.93 | 16/16 |

- fdm-1021-e8t1p1-high-n0 failure reasons: one or more triggered profiles failed profile acceptance
| fdm-1021-e8t1p1-medium-n1 | 8/1/1 | 1 / 0 | 100 | 216 | completed | no | [9] | 0 | [] | 0 | incomplete (0/1) | 26.61 → 135.42 | 135.24 | 130.01 | 16/16 |

- fdm-1021-e8t1p1-medium-n1 failure reasons: one or more triggered profiles failed profile acceptance
| fdm-1021-e8t1p1-control-after | 8/1/1 | — / — | 0 | 539 | completed | yes | [] | 0 | [] | 0 | not_triggered (0/0) | 50.29 → 43.82 | 57.34 | 69.80 | 16/16 |
| fdm-1021-e4t2p1-control-before | 4/2/1 | — / — | 0 | 275 | completed | no | [] | 0 | [] | 0 | incomplete (0/1) | 98.88 → 98.68 | 98.58 | 97.66 | 16/16 |

- fdm-1021-e4t2p1-control-before failure reasons: missing controller_stopped; missing training_complete; one or more triggered profiles failed profile acceptance
| fdm-1021-e4t2p1-low-n0 | 4/2/1 | 0 / 0 | 14 | 296 | completed | no | [0] | 0 | [] | 0 | incomplete (0/1) | 94.40 → 97.32 | 98.03 | 97.28 | 16/16 |

- fdm-1021-e4t2p1-low-n0 failure reasons: one or more triggered profiles failed profile acceptance
| fdm-1021-e4t2p1-high-n1 | 4/2/1 | 1 / 0 | 200 | — | engineering_failure | no | [] | 0 | [] | 0 | unavailable (0/0) | — → — | — | — | 16/16 |

- fdm-1021-e4t2p1-high-n1 failure reasons: attempt not completed successfully: status='engineering_failure', exit_code=124; could not derive one shared actual_steps value from all 16 ranks; fault wall-time anchor missing; cannot evaluate pre-fault alarms; missing training_complete; node0: one or more worker exits are missing, failed, or still alive; node0: workers-node0.json success is not true; node1: one or more worker exits are missing, failed, or still alive; node1: workers-node1.json success is not true; rank0: expected one run_summary, found 0; rank10: expected one run_summary, found 0; rank11: expected one run_summary, found 0; rank12: expected one run_summary, found 0; rank13: expected one run_summary, found 0; rank14: expected one run_summary, found 0; rank15: expected one run_summary, found 0; rank1: expected one run_summary, found 0; rank2: expected one run_summary, found 0; rank3: expected one run_summary, found 0; rank4: expected one run_summary, found 0; rank5: expected one run_summary, found 0; rank6: expected one run_summary, found 0; rank7: expected one run_summary, found 0; rank8: expected one run_summary, found 0; rank9: expected one run_summary, found 0
| fdm-1021-e4t2p1-medium-n0 | 4/2/1 | 0 / 0 | 100 | 207 | completed | yes | [0] | 1 | [] | 0 | valid (1/1) | 35.88 → 152.33 | 153.43 | 147.28 | 16/16 |
| fdm-1021-e4t2p1-low-n1 | 4/2/1 | 1 / 0 | 14 | 444 | completed | yes | [9] | 1 | [] | 0 | valid (1/1) | 36.63 → 62.91 | 57.35 | 50.41 | 16/16 |
| fdm-1021-e4t2p1-high-n0 | 4/2/1 | 0 / 0 | 200 | 143 | completed | yes | [0] | 1 | [] | 0 | valid (1/1) | 37.50 → 252.99 | 254.14 | 246.95 | 16/16 |
| fdm-1021-e4t2p1-medium-n1 | 4/2/1 | 1 / 0 | 100 | 208 | completed | yes | [9] | 1 | [] | 0 | valid (1/1) | 37.88 → 152.51 | 151.56 | 146.55 | 16/16 |
| fdm-1021-e4t2p1-control-after | 4/2/1 | — / — | 0 | 641 | completed | yes | [] | 0 | [] | 1 | valid (1/1) | 36.73 → 39.81 | 40.38 | 39.43 | 16/16 |
| fdm-1021-e4t1p2-control-before | 4/1/2 | — / — | 0 | 721 | completed | yes | [] | 0 | [] | 1 | valid (1/1) | 21.30 → 36.81 | 30.11 | 25.16 | 16/16 |
| fdm-1021-e4t1p2-low-n0 | 4/1/2 | 0 / 0 | 14 | 545 | completed | yes | [0] | 1 | [] | 0 | valid (1/1) | 19.53 → 50.20 | 37.15 | 36.55 | 16/16 |
| fdm-1021-e4t1p2-high-n1 | 4/1/2 | 1 / 0 | 200 | 151 | completed | yes | [9] | 1 | [] | 0 | valid (1/1) | 19.85 → 236.51 | 236.42 | 223.49 | 16/16 |
| fdm-1021-e4t1p2-medium-n0 | 4/1/2 | 0 / 0 | 100 | 227 | completed | yes | [0] | 1 | [] | 0 | valid (1/1) | 19.63 → 137.45 | 137.22 | 123.51 | 16/16 |
| fdm-1021-e4t1p2-low-n1 | 4/1/2 | 1 / 0 | 14 | 549 | completed | yes | [9] | 1 | [] | 0 | valid (1/1) | 19.70 → 50.48 | 37.08 | 37.05 | 16/16 |
| fdm-1021-e4t1p2-high-n0 | 4/1/2 | 0 / 0 | 200 | 151 | completed | yes | [0] | 1 | [] | 0 | valid (1/1) | 19.82 → 236.32 | 236.45 | 223.56 | 16/16 |
| fdm-1021-e4t1p2-medium-n1 | 4/1/2 | 1 / 0 | 100 | 228 | completed | yes | [9] | 1 | [] | 0 | valid (1/1) | 19.49 → 136.91 | 137.01 | 123.64 | 16/16 |
| fdm-1021-e4t1p2-high-n0-stage1 | 4/1/2 | 0 / 1 | 200 | 151 | completed | yes | [] | 0 | [] | 1 | valid (1/1) | 19.84 → 236.90 | 236.42 | 222.23 | 16/16 |
| fdm-1021-e4t1p2-high-n1-stage1 | 4/1/2 | 1 / 1 | 200 | 151 | completed | yes | [13] | 1 | [] | 0 | valid (1/1) | 19.21 → 235.60 | 235.33 | 222.82 | 16/16 |
| fdm-1021-e4t1p2-control-after | 4/1/2 | — / — | 0 | 769 | completed | yes | [10] | 0 | [10] | 0 | valid (1/1) | 19.78 → 35.74 | 31.59 | 20.73 | 16/16 |
| fdm-1021-e4t2p2-control-before | 4/2/2 | — / — | 0 | 1038 | completed | yes | [] | 0 | [] | 0 | not_triggered (0/0) | 21.47 → 22.28 | 21.83 | 23.93 | 16/16 |
| fdm-1021-e4t2p2-low-n0 | 4/2/2 | 0 / 0 | 14 | 542 | completed | yes | [0] | 1 | [] | 0 | valid (1/1) | 25.74 → 50.89 | 51.52 | 37.25 | 16/16 |
| fdm-1021-e4t2p2-high-n1 | 4/2/2 | 1 / 0 | 200 | 150 | completed | yes | [9] | 1 | [] | 0 | valid (1/1) | 20.77 → 238.35 | 238.60 | 225.33 | 16/16 |
| fdm-1021-e4t2p2-medium-n0 | 4/2/2 | 0 / 0 | 100 | 225 | completed | yes | [0] | 1 | [] | 0 | valid (1/1) | 21.11 → 138.26 | 138.26 | 124.92 | 16/16 |
| fdm-1021-e4t2p2-low-n1 | 4/2/2 | 1 / 0 | 14 | 534 | completed | yes | [9] | 1 | [] | 0 | valid (1/1) | 21.04 → 52.17 | 50.72 | 37.35 | 16/16 |
| fdm-1021-e4t2p2-high-n0 | 4/2/2 | 0 / 0 | 200 | 150 | completed | yes | [0] | 1 | [] | 0 | valid (1/1) | 21.02 → 239.08 | 237.19 | 220.77 | 16/16 |
| fdm-1021-e4t2p2-medium-n1 | 4/2/2 | 1 / 0 | 100 | 223 | completed | yes | [9] | 1 | [] | 0 | valid (1/1) | 21.10 → 139.65 | 138.99 | 129.20 | 16/16 |
| fdm-1021-e4t2p2-high-n0-stage1 | 4/2/2 | 0 / 1 | 200 | 149 | completed | yes | [5] | 1 | [] | 0 | valid (1/1) | 20.80 → 238.60 | 238.09 | 234.53 | 16/16 |
| fdm-1021-e4t2p2-high-n1-stage1 | 4/2/2 | 1 / 1 | 200 | 149 | completed | yes | [13] | 1 | [] | 0 | valid (1/1) | 22.49 → 238.60 | 239.31 | 246.68 | 16/16 |
| fdm-1021-e4t2p2-control-after | 4/2/2 | — / — | 0 | 723 | completed | yes | [] | 0 | [] | 1 | valid (1/1) | 26.16 → 37.33 | 36.34 | 22.08 | 16/16 |

## Per topology

| Group | Planned + / controls | Valid | TP / conditional FP / FN | Conditional P / R | Observed FP | Exact | Refusals |
|---|---:|---:|---:|---:|---:|---:|---:|
| EP4/TP1/PP2 | 8 / 2 | 10/10 | 7 / 1 / 1 | 87.5% / 87.5% | 1 | 7/8 | 2 |
| EP4/TP2/PP1 | 6 / 2 | 5/8 | 4 / 0 / 0 | 100.0% / 100.0% | 0 | 4/6 | 1 |
| EP4/TP2/PP2 | 8 / 2 | 10/10 | 8 / 0 / 0 | 100.0% / 100.0% | 0 | 8/8 | 1 |
| EP8/TP1/PP1 | 6 / 2 | 3/8 | 0 / 0 / 1 | — / 0.0% | 0 | 0/6 | 0 |

## Per injection node

| Group | Planned + / controls | Valid | TP / conditional FP / FN | Conditional P / R | Observed FP | Exact | Refusals |
|---|---:|---:|---:|---:|---:|---:|---:|
| node0 | 14 / 0 | 10/14 | 9 / 0 / 1 | 100.0% / 90.0% | 0 | 9/14 | 1 |
| node1 | 14 / 0 | 11/14 | 10 / 0 / 1 | 100.0% / 90.9% | 0 | 10/14 | 0 |

## Per fixed delay

| Group | Planned + / controls | Valid | TP / conditional FP / FN | Conditional P / R | Observed FP | Exact | Refusals |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | 8 / 0 | 6/8 | 6 / 0 / 0 | 100.0% / 100.0% | 0 | 6/8 | 0 |
| 14 | 8 / 0 | 6/8 | 5 / 0 / 1 | 100.0% / 83.3% | 0 | 5/8 | 0 |
| 200 | 12 / 0 | 9/12 | 8 / 0 / 1 | 100.0% / 88.9% | 0 | 8/12 | 1 |
| control | 0 / 8 | 7/8 | 0 / 1 / 0 | 0.0% / — | 1 | 0/0 | 3 |

## Per pipeline stage

| Group | Planned + / controls | Valid | TP / conditional FP / FN | Conditional P / R | Observed FP | Exact | Refusals |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp0 | 24 / 0 | 17/24 | 16 / 0 / 1 | 100.0% / 94.1% | 0 | 16/24 | 0 |
| pp1 | 4 / 0 | 4/4 | 3 / 0 / 1 | 100.0% / 75.0% | 0 | 3/4 | 1 |

## Controls by topology

| Group | Planned + / controls | Valid | TP / conditional FP / FN | Conditional P / R | Observed FP | Exact | Refusals |
|---|---:|---:|---:|---:|---:|---:|---:|
| EP4/TP1/PP2 | 0 / 2 | 2/2 | 0 / 1 / 0 | 0.0% / — | 1 | 0/0 | 1 |
| EP4/TP2/PP1 | 0 / 2 | 1/2 | 0 / 0 / 0 | — / — | 0 | 0/0 | 1 |
| EP4/TP2/PP2 | 0 / 2 | 2/2 | 0 / 0 / 0 | — / — | 0 | 0/0 | 1 |
| EP8/TP1/PP1 | 0 / 2 | 2/2 | 0 / 0 / 0 | — / — | 0 | 0/0 | 0 |

## Failures, refusals, and wrong predictions

Invalid or incomplete rows: 8/36. Online `no_localization` events: 4. Unique wrong-rank predictions summed by run: 1; these include controls, invalid profiles, malformed runs with readable terminal records, and any partial run that emitted a terminal output.

| Run | Status | Refusal events | Wrong prediction ranks | Failure / parse evidence |
|---|---|---:|---|---|
| fdm-1021-e8t1p1-low-n0 | invalid | 0 | [] | one or more triggered profiles failed profile acceptance |
| fdm-1021-e8t1p1-high-n1 | invalid | 0 | [] | one or more triggered profiles failed profile acceptance |
| fdm-1021-e8t1p1-medium-n0 | invalid | 0 | [] | one or more triggered profiles failed profile acceptance |
| fdm-1021-e8t1p1-high-n0 | invalid | 0 | [] | one or more triggered profiles failed profile acceptance |
| fdm-1021-e8t1p1-medium-n1 | invalid | 0 | [] | one or more triggered profiles failed profile acceptance |
| fdm-1021-e4t2p1-control-before | invalid | 0 | [] | missing controller_stopped; missing training_complete; one or more triggered profiles failed profile acceptance |
| fdm-1021-e4t2p1-low-n0 | invalid | 0 | [] | one or more triggered profiles failed profile acceptance |
| fdm-1021-e4t2p1-high-n1 | invalid | 0 | [] | attempt not completed successfully: status='engineering_failure', exit_code=124; could not derive one shared actual_steps value from all 16 ranks; fault wall-time anchor missing; cannot evaluate pre-fault alarms; missing training_complete; node0: one or more worker exits are missing, failed, or still alive; node0: workers-node0.json success is not true; node1: one or more worker exits are missing, failed, or still alive; node1: workers-node1.json success is not true; rank0: expected one run_summary, found 0; rank10: expected one run_summary, found 0; rank11: expected one run_summary, found 0; rank12: expected one run_summary, found 0; rank13: expected one run_summary, found 0; rank14: expected one run_summary, found 0; rank15: expected one run_summary, found 0; rank1: expected one run_summary, found 0; rank2: expected one run_summary, found 0; rank3: expected one run_summary, found 0; rank4: expected one run_summary, found 0; rank5: expected one run_summary, found 0; rank6: expected one run_summary, found 0; rank7: expected one run_summary, found 0; rank8: expected one run_summary, found 0; rank9: expected one run_summary, found 0 |
| fdm-1021-e4t2p1-control-after | valid | 1 | [] | — |
| fdm-1021-e4t1p2-control-before | valid | 1 | [] | — |
| fdm-1021-e4t1p2-high-n0-stage1 | valid | 1 | [] | — |
| fdm-1021-e4t1p2-control-after | valid | 0 | [10] | — |
| fdm-1021-e4t2p2-control-after | valid | 1 | [] | — |

The JSON file contains the fixed matrix, per-row attempt and run-summary evidence, helper schema checks, raw terminal outputs, worker guards, timing windows, and subgroup counts.
