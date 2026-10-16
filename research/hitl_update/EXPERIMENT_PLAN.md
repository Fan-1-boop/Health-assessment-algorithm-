# Controlled supervised model-update research: public experiment plan v0.1

This is an **experimental protocol**, not a patent filing, patentability opinion, or description of a novel claim.

## Research objective

Evaluate whether a candidate **human-reviewed, label-specific, controlled-update workflow** can preserve existing anomaly and health-state performance while incorporating trustworthy new labeled samples.

The goal is to address erroneous updates arising from conflating *ordinary operating-mode changes* with *true device anomalies/faults*. No unsupervised model drift adaptation or auto-generated labels are permitted for an update.

## Fixed foundations

- **Delivery baseline:** `release/delivery-mvp-v1.0-frozen`. Do not modify code, models, reported numbers, or input files there.
- **Demonstration data:** the synthetic MVP has simulated H/D/F state and anomaly labels; it can demonstrate three-state behavior but cannot validate physical spacecraft degradation or failures.
- **Real satellite evidence:** OPS-SAT-AD supplies expert-informed normal/abnormal segment labels only. It can test the anomaly classifier; it cannot validate physical H/D/F or continuity-based HMM transitions.
- Never compare real binary anomaly F1 with synthetic three-state Macro-F1 as though they were the same task.
- Use one predeclared update batch and one held-out evaluation set per experiment; no optimization using the held-out test.

## Minimum annotation/data contract

A candidate supervised upload must provide:

| Required attribute | Example semantics |
|---|---|
| Data source and acquisition interval | satellite/device ID, timestamps, channel schema |
| Evidence source | expert-reviewed observation, verified maintenance record, controlled fault-injection experiment |
| Target type | `anomaly_binary`, `physical_state_3class`, or `health_score` |
| Label and label meaning | task-specific, with clear start/end boundaries where relevant |
| Review status | verified by a named reviewer/organization process, with immutable review timestamp |
| Quality metadata | completeness, data gaps, sampling validity, duplicate/overlap checks |
| Access and version metadata | batch ID, original immutable data checksum, parent model/version |

**Target restrictions**: binary anomaly supervision alone cannot train a physical `degraded/fault` classifier; a subjective 0–100 rating updates only a scoring map unless accompanied by independently confirmed physical labels. An unreviewed prediction is never promoted to ground truth.

## Basic public baselines

| Arm | Update procedure | Evaluation purpose |
|---|---|---|
| A: Freeze | Apply old model unchanged | Baseline error for new verified operating conditions |
| B: Ordinary supervised retrain | Train a candidate from existing plus newly verified labels, without branch-specific control | Typical supervised-update control |
| C: Candidate controlled update | Route verified annotations to the authorized update target, evaluate compatibility of candidate artifacts, require operator sign-off | Whether controlled release reduces unwanted performance regressions |

B and C must use **identical available supervised batches and identical reserved evaluation data**. Training cost, storage, and latency should be recorded if meaningfully different.

## Primary acceptance/reporting metrics

- **New verified operating mode:** false alert rate; number of correctly accepted verified-normal segments.
- **Confirmed anomaly data:** abnormal-class Recall, Precision, F1, AP, and raw false positive/false negative counts.
- **Three-state synthetic-only study:** H/D/F Macro-F1, Recall per class, Brier; never claim real physical fault sensitivity.
- **Simulated operator score study:** MAE and monotonic order of anchors (Healthy >= Degraded >= Fault).
- **Regression check:** original reference-set metrics must be reported side by side with new-data metrics. If changes are detrimental, candidly mark the update as rejected rather than moving thresholds after inspecting the final evaluation.
- **Safety behavior:** reject insufficient labeling, unknown label type, inconsistent timestamps, unreviewed uploads, and candidate artifacts whose independent validation or human approval is absent.

## Study status

- `[x]` Freeze full delivery snapshot and initialize independent research branch.
- `[x]` Record task semantics and high-level A/B/C experiment protocol.
- `[x]` Implement basic supervised upload-data validator with tests (public infrastructure, 2026-10-16).
- `[x]` Produce real OPS-SAT existing-label A/B/C pilot with **simulated** manual upload, explicit limitations, and frozen audit (2026-10-16; see PILOT_RESULTS.md).
- `[ ]` Obtain independent labels separating newly confirmed nominal modes from actual faults; then test a differentiating model-compatibility mechanism on the **same** input batches and evaluation sets. Screen prior art before publishing novel details.
- `[ ]` Prepare a private patent-sensitive technical specification before any public disclosure.

## Publication guardrail

Do **not** add an undisclosed candidate patented mechanism, mathematical acceptance rule, specialized compatibility construction, or claim text to this public branch until an intellectual-property filing strategy has been decided. The public protocol above intentionally stays at the level of conventional supervised model development and acceptance.
