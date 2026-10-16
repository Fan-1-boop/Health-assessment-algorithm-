# V1.4 ESA telecommand-context exploratory ablation

Execution: https://github.com/Fan-1-boop/Health-assessment-algorithm-/actions/runs/38884811049 — SUCCESS.

The experiment used genuine ESA Mission1 \`channel_21\` 10-minute data and eight real, anonymized telecommand streams selected by Priority >=3 and source archive compressed size <=250k, **without using anomaly labels for command selection**. Causal command pulses cover the preceding 10 minutes. 8 selected streams had only 20–66 nonzero resampled occurrences each. Command data checksum and source SHA-256s are in \`data/esa_v14_channel21/command_manifest.json\`.

**This is exploratory, not a new unbiased held-out test.** The same channel_21 test data were disclosed in the earlier official-ID experiment, so apparent improvements would require an entirely new untouched event/channel protocol to verify.

## Actual result

| Model | Validation AP | Previously exposed test AP | Test ROC-AUC | Test ID event recall under frozen 0.2%-nominal + three-alert policy |
|---|---:|---:|---:|---:|
| Ridge target history only | 0.054665 | 0.017803 | 0.601740 | 0/10 |
| Ridge target history + eight telecommand streams | 0.054932 | 0.017797 | 0.601590 | 0/10 |

Both produce zero frozen-policy alarms in the old test stage. Test positive prevalence is approximately 0.009866. Command context provided no practically meaningful gain here. The changes in AP are tiny; there is no evidence of significance or causality.

## Interpretation and limits

- Picking files by archive size was a deterministic no-label pilot, *not* a physically informed command-to-equipment mapping.
- Very sparse command pulses plus coarse 10-minute binning may fail to represent actual spacecraft operating modes.
- More relevant contextual variables would require subsystem association, synchronized command states, and potentially event-phase information.
- These are NOT official ESA-ADB 30-second or Telemanom-ESA scores.
- True anomalies and rare-normal events are correctly separated in target semantics, but reliable identification of unknown faults is not demonstrated.
- No automated model updates, health score mapping, or cross-branch fusion were attempted.

Next experimental gate: choose physically/operationally relevant telecommands from official subsystem metadata or expert-defined rule, freeze global official-ID groups across channels, and run a new prospective event-class/channel protocol. No more tuning on the already-exposed channel_21 test stage.
