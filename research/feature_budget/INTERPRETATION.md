# OPS-SAT 2578D MSF-MOFS feature-budget tradeoff — interpretation

**Status:** all 14 configurations completed in one GitHub Actions run; 3 unit tests passed. Original delivery release remains unchanged. All audit results are exploratory because this same source data has already supported earlier random-split studies.

## Research question

Can allowing more than the earlier 13 selected MSF-MOFS features preserve binary anomaly detection performance while reducing the downstream model computation? Every model uses the **same fixed LogisticRegression classifier**; only feature subsets differ. Only original OPS-SAT official train=1 rows are partitioned 956/319/319; original official test=0 is excluded.

### Selected representative results (actual measured)

| Method | Features | Selector ms | Fit ms | First select+fit ms | Audit F1 | Audit AP | Audit Recall | Normal false-positive rate | Calibration F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 2578 | 0.0 | 521.2 | 521.2 | 0.8667 | 0.9484 | 0.8125 | 0.0157 | **0.8800** |
| MSF budget 16 | 8 | 292.8 | 2.4 | 295.2 | 0.8305 | 0.9222 | 0.7656 | 0.0196 | 0.7899 |
| MSF budget 32 | 16 | 326.2 | 2.9 | 329.1 | 0.8640 | 0.9547 | 0.8438 | 0.0275 | 0.8125 |
| **MSF budget 64** | **37** | **399** | **4.9** | **404** | **0.8926** | **0.9531** | **0.8438** | **0.0118** | **0.8167** |
| MSF budget 128 | 66 | 595 | 7.5 | 602 | 0.8871 | 0.9409 | 0.8594 | 0.0196 | 0.8244 |
| MSF budget 256 | 128 | 1022 | 18.5 | 1040 | 0.8595 | 0.9236 | 0.8125 | 0.0196 | 0.8413 |
| MSF budget 512 | 256 | 2288 | 45.1 | 2333 | 0.8387 | 0.9401 | 0.8125 | 0.0314 | 0.8030 |
| Fisher Top 32 | 32 | 14.2 | 4.2 | 18.4 | 0.8689 | 0.9358 | 0.8281 | 0.0196 | 0.8136 |
| Fisher Top 512 | 512 | 14.2 | 88.8 | 103.0 | 0.8819 | 0.9511 | 0.8750 | 0.0275 | 0.8271 |
| Official original 18 | 18 | 0.0 | 3.5 | 3.5 | 0.7627 | 0.8934 | 0.7031 | 0.0353 | 0.8640 |

All values in the full metrics JSON are authoritative; displayed times above are rounded.

## What the data support

1. Feature-budget constraints address excessive sparsification: MSF budget64 picked 37/2578 descriptors; on this fixed internal audit it achieved F1 **0.8926** and AP **0.9531** versus full-feature F1 **0.8667** and AP **0.9484**.
2. Downstream training time reduced from **521.2 ms** to **4.9 ms**, a ~106x training speedup, but the feature search alone required **~399 ms**. The first search+fit (~404 ms) is only moderately faster than full training (~521 ms). The common 2560-feature engineering time (~5.19 seconds) is **not** eliminated.
3. Fisher Top512 is a strong simple filter baseline: audit F1 **0.8819**, AP **0.9511**, fit time **88.8 ms**, selector time **14.2 ms**, first select+fit **103.0 ms**. MSF is *not* unambiguously more efficient once search cost is included.
4. More features are not automatically better: MSF budget512 selected 256 features but performed worse than budget64 on this audit. Search is stochastic and optimization criteria differ from audit F1.

## Critical audit-versus-calibration mismatch

**Do not declare MSF budget64 the best final model based on the audit table.** It has calibration F1 **0.8167**, while full-feature calibration F1 is **0.8800**. The preregistered calibration-only recommendation admitted only original18 and full2578; it selected original18 for the least projected 10-retrain cost. Yet original18 scored just **0.7627** F1 on the audit versus full **0.8667**.

This demonstrates unstable model/budget selection under the current single randomized split. The calibration-only rule selecting original18 is part of the actual recorded outcome and should **not be retroactively changed to budget64** after looking at the audit. We can use the budget64 result as a *hypothesis* for a future repeated-split stability study, not as confirmatory proof.

## Next decision gate (on the same satellite dataset)

Run repeated **stratified grouped segment** resampling with predeclared budget candidates and wrapper capacity, report paired distributions of F1, AP, recall, training and selection time. A valid performance-cost improvement should persist on most splits and not be chosen after inspecting each split's test subset. Do not introduce other datasets or cross-domain generalization.

## Reproducibility

- Frozen protocol: `research/feature_budget/PROTOCOL_FROZEN.json`
- Experiment code: `research/feature_budget/run_budget.py`
- Complete metrics: `results/feature_budget/metrics.json`
- All 14 rows: `results/feature_budget/comparison.csv`
- Four actual generated plots: `results/feature_budget/*png`
- Successful GitHub CI: https://github.com/Fan-1-boop/Health-assessment-algorithm-/actions/runs/40079618217

**Scope constraints:** Results demonstrate retrospective anomaly classification of independent OPS-SAT segments. No trustworthy three-state physical-fault labels, no online detection delay evaluation, and no patent novelty claim.
