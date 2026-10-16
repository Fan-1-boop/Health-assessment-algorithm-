# V1.4 independent ESA anomaly branch: official event-ID audit and conditional-command pilot

**2026-10-16 research status:** official-ID single-channel channel_21 experiment succeeded and was committed. Model quality **not approved**. No late fusion, physical fault severity inference or automatic model adaptation.

## 1. Historical reproducibility

- V1/V1.1/V1.2 and the Proxy-V1 three-state rules are untouched.
- V1.3 \`channel_41\` and its exposed test data remain an *exploratory* benchmark, not a repeatedly tunable holdout.
- V1.4's new \`channel_21\` was chosen from published channel event counts **before inspecting its model test results**, and the candidate models, validation AP selection, nominal 0.2% cutoff quantile, and three-consecutive-alert rule were committed before executing the new test. Subsequent diagnostics cannot be promoted to fresh confirmatory tests.
- Mission1 official source record: https://zenodo.org/records/15237121, CC BY 3.0 IGO. See per-file extraction SHA-256 manifests.

## 2. Dataset / label protocol

The public data are ESA satellite telemetry and official event labels:
- 0 Nominal (negative and training-normal reference)
- 1 Anomaly (positive for the strict anomaly task)
- 2 Rare Event (separate negative subset in this project's task, NOT the default official ESA benchmark semantics)
- 3 Communication gap, 4 Invalid (excluded from causal windows)

Data are as-of sampled every 10 minutes, with maximum 60-minute freshness, one measured telemetry target channel and **no telecommands** in the first independent experiment. This is **NOT** ESA official 30-second multi-channel preprocessing or its corrected event-wise metric.

For ID auditing, \`scripts/run_v14_esa.py\` reads the actual \`labels.csv\` and \`anomaly_types.csv\` original event IDs, merges multirow annotations with the same ID, and chooses chronological boundaries that do not cut an event belonging to the selected channel. No model train/validation/test causal window shares raw source observations. This does **not** mean different official IDs represent statistically independent physical fault origins; it also does not guarantee multi-channel global-ID separation for future joint-channel models. That will require one common cross-channel partition.

## 3. V1.4 channel_21 actual outcomes

Official anomaly IDs: **29 train / 9 validation / 10 test**; rare event IDs: 20 / 10 / 9. Windows evaluated: 107,744 / 40,464 / 35,780. The selected model is robust level deviation (validation AP 0.112096 vs 0.101337 for iForest, 0.054665 for autoregressive Ridge).

| Model (diagnostic comparison) | Test point AP | Test ROC-AUC | ID recall with three-in-a-row alert | Normal alert windows | Rare-event alert windows |
|---|---:|---:|---:|---:|---:|
| Robust level deviation (validation-selected) | 0.017602 | 0.472218 | 4/10 | 6,188 / 33,038 | 1,646 / 2,389 |
| Windowed iForest | 0.015083 | see JSON | 0/10 | 15 / 33,038 | 22 / 2,389 |
| Autoregressive Ridge | 0.017803 | see JSON | 0/10 | 0 / 33,038 | 0 / 2,389 |

Test anomaly prevalence 353/35,780 = 0.009866. The selected robust detector's test ROC-AUC is below 0.5; positive ranking has failed overall. Test average precision only modestly exceeds random-prevalence reference. **This model is not deployable.**

The alarm rule reduced selected-model \`false_alarm_episodes\` from 101 to 14 by merging contiguous alarms, but it barely changed false-alarm time occupancy: normal 6,303 -> 6,188, rare nominal 1,648 -> 1,646. Thus **14 alarm episodes is not evidence of low false-alert burden**. Sustained false alarms can span long intervals and appear as few episodes. All alarm-rate denominators are sampled evaluation windows, **not physical hours**.

**Important:** An event classified as "detected" by the current coverage criterion can overlap an alarm which was already active before the true anomaly began. The next evaluation iteration should report alarm-onset-based event recall as a separate, stricter metric, not silently replace the historical coverage metric.

## 4. Why the result matters

V1.3 \`channel_41\` had relatively high test point AP with major false positives. V1.4 \`channel_21\` is substantially harder and its score ordering is poor. The difference is not a direct before/after method comparison, because underlying channel and anomaly events differ. The result underscores the need for multi-channel / multi-event validation and command-aware anomaly evidence, not further tuning on these exposed tests.

## 5. Preliminary context pilot (post-hoc, separately marked)

The official ZIP contains 698 telecommand streams. A small Deflate64 patch allows \`channels.csv\` and other previously unavailable members to be read. \`scripts/extract_esa_commands.py\` deterministically selects 8 higher-priority streams solely by archive size and priority, **not event labels**. Extracted command flags use the preceding 10 minutes only. \`scripts/run_v14_command_ablation.py\` compares a matched target-only Ridge forecast with a target+command Ridge forecast; because the channel_21 test has already been revealed, **any ensuing result is a descriptive diagnostic, not new unbiased evidence**.

Anonymized command IDs do not prove relevance to the selected telemetry channel. The pilot is intentionally simple, **not Telemanom-ESA**, and does not establish fault causality.

## 6. Remaining gates

1. Unify all selected channels on a global official-ID split; record annotation overlap and same-event cross-channel correlations.
2. Add higher time resolution, e.g. official ESA 30-second preprocessing, plus exact event-level metrics, false alarms per unit wall-clock time, and onset-only recall.
3. Validate true command context relevance; then compare forecasting with/without commands on newly frozen event groups, not already-opened tests.
4. Reproduce the actual Telemanom-ESA LSTM/dynamic-thresholding implementation before invoking its published name in performance claims.
5. Maintain a **separate** state/HMM branch. There is no real health/degradation/fault three-way ground truth in ESA-ADB. Proxy-V1 results do not prove physical fault classification.
