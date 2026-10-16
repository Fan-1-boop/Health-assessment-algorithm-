# Human-in-the-loop supervised update — research branch

**Branch:** `research/hitl-quality-gated-update`  
**Delivery baseline:** `release/delivery-mvp-v1.0-frozen` (read-only reference)  
**Stage:** research initiation, **not** a validated invention or a shipped update mechanism.

## Narrow research question

When trusted operators provide newly labeled satellite telemetry, can we update only the appropriate candidate model components **without silently turning new nominal operating modes into faults, or genuine faults into nominal background**?

The research addresses **controlled supervised updates**, *not* autonomous drift adaptation:

- Inference on unlabeled telemetry does **not** train or update any parameter.
- A model-generated pseudo-label is **never** equivalent to a human-confirmed label.
- A supervised data submission must carry provenance, label semantics, timestamps, and quality-review status.
- Normal/anomaly event labels, healthy/degraded/fault **state** labels, and 0–100 health assessments are distinct targets; do not convert one into another.
- Any new trained model remains a **candidate** pending independent evaluation and explicit approval; retain previous working artifacts for rollback.

## Existing baseline to preserve

The current deployed proof-of-concept uses a frozen, fully **synthetic** 3-state health MVP and an independent **real** OPS-SAT-AD 0/1 anomaly classification demo. OPS-SAT-AD does not contain verified physical three-state labels, so its anomaly labels must not be repurposed as fault severity labels.

The frozen delivery branch, existing datasets, published model metrics, and artifacts remain untouched.

## Initial research milestones

1. Inventory existing labels, model components, and permissible update targets.
2. Define the **input-data contract** and quality checks; reject missing or ambiguous supervision.
3. Establish comparable no-update, supervised full-retraining, and candidate controlled-update experiments on *identical* labeled batches.
4. Compare newly labeled data performance and original reference-set degradation, including false alarms, missed anomalies, per-state recall, Brier, and score MAE where applicable.
5. Propose a candidate technical differentiation only after a documented prior-art review and measured improvements.
6. Human review before any new version is designated for delivery.

See `EXPERIMENT_PLAN.md` for the public experimental protocol. **Do not publish potentially novel algorithms, mathematical rules, parameterized claims, or draft patent material to this public repository before an IP review.**

## IP/publication status

This GitHub repository is **public**, so every commit in this research branch is publicly visible even if it is never merged to main. This directory contains only high-level, established engineering research patterns and public experiment planning. For patent-sensitive, specific implementation work, first use a separate private repository and coordinate filing/publication timing with qualified patent counsel.
