# V1.3 independent branches: protocol and implementation review

**Status (2026-10-16): implemented and unit tested; ESA-Mission1 real-data benchmark NOT run.**

## 0. Historical continuity

V1/V1.1/V1.2 are left untouched. `configs/proxy_label_rules_v1.json` is frozen; its classes are artificial interval-position proxies, **not physical faults**. V1.3 does not do fusion, health scoring, supervised repair confirmation, online updates or claim a new patentable algorithm.

## 1. Confirmed upstream ESA format

Official `kplabs-pl/ESA-ADB` preprocessing `Mission1_semisupervised_prep_from_raw.py` produces preprocessed CSV files with `timestamp`, telemetry/telecommand inputs and `is_anomaly_<channel>` labels.

Official `utils.AnnotationLabel`: `0=NOMINAL, 1=ANOMALY, 2=RARE_EVENT, 3=GAP, 4=INVALID`.

The repository's default benchmark **treats rare nominal events as anomalies**. Our *separate strict research task* evaluates class 1 against classes 0+2 and reports false positives on rare events separately. We **do not claim that these metrics match ESA's official corrected F0.5/ADTQC or full benchmark**.

Official source: https://github.com/kplabs-pl/ESA-ADB and https://zenodo.org/records/15237121; download version and zip checksum must be recorded. The archive `ESA-Mission1.zip` is ~3.8 GB. A lightweight channel subset is generated **after** obtaining/extracting the archive, not a small publicly posted CSV.

## 2. Correct split and evaluation gate

`src/v13_protocol.py` uses official per-channel **contiguous event episodes** for a chronological partition with >=2 anomaly episodes in train/val/test; rare nominal is a separate type. Real ESA annotation IDs may group overlap/multichannel events differently; for publication-quality benchmark use original `labels.csv` event IDs instead of the per-channel contiguous-run approximation.

Split **before** window generation; each window's raw measurements are fully contained in one partition. Gap/invalid rows within lookback result in exclusion. No random overlapping-window shuffle. `--stride` reduces computation but may worsen early anomaly coverage. Normal pretraining windows must contain only code 0.

The V1.2 SMAP P-1 stream only has three independent anomaly episodes, failing this protocol. It is not re-split to make results look better. The event-bounded split is the minimum admissibility gate; *multiple channels corresponding to one physical event are not independent*.

Validation AP chooses the winner; the test is evaluated once for reporting. `test_all_candidates` results are diagnostic only, not a new model-selection signal. Report point AP, event recall, false alarm episodes, detection delay, and rare-normal false-positive points. **No point adjustment**. AP is over a chronological partition containing normal context, **not AP of positive-only snippets**.

## 3. Anomaly models currently implemented

- `RobustLevelDeviation`: raw sensor sample relative to median/MAD of nominal training windows.
- `WindowedIForest`: IsolationForest on causal statistics of raw windows (internal encoding, not supervised MSF features).
- `ForecastRidge`: supervised only by temporal forecasting targets in confirmed nominal windows; residual anomaly score.

**Not implemented:** Telemanom-ESA official Docker algorithm, TranAD, MTGFlow, official ESA metrics VUS-PR / affiliation. Do not represent `ForecastRidge` as Telemanom, which is an LSTM forecasting baseline with a dynamic threshold.

### Run after obtaining ESA raw dataset

```bash
# First preprocess externally following OFFICIAL ESA-ADB instructions:
# python notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py ../../data/ESA-Mission1
python scripts/run_v13_anomaly.py \
  --esa-preprocessed-csv /path/to/ESA-ADB/data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \
  --channel <actual_ESA_telemetry_column> \
  --out results/v13/esa_mission1
```

Use `pd.read_csv(..., nrows=0)` to inspect actual preprocessed path/schema before adapting the CLI. The exact official directory naming depends on script's `learning_type` and version; do not assume the example path exists. Raw data are NOT bundled or downloaded automatically by CI.

## 4. Hierarchical state branch: what is implemented and what is blocked

`src/v13_state.py` implements an experiment-ready `select_grouped_msf` (two independent binary tasks), a dedicated Fisher *seed event-group* excluded from GroupKFold wrapper evaluation, calibrated hierarchical LR (separate event-group calibration set), train-only event-respecting transition estimates, causal HMM posterior/prior evidence filter with group resets and Brier/NLL/rare-class metrics.

This is a **separate V1.3 research adaptation**, not an exact reproduction of the original paper's original 5-NN selector. Inner search fitness is **not unbiased** even with independent seed group; an *outer untouched event group* is essential.

The module has real functionality and unit tests using **synthetic known classes**. It is **not verified on true spacecraft health/degradation/fault labels** because ESA/SMAP do not have them. No state model accuracy numbers are claimed for V1.3 real data.

State evaluation requires disjoint `search/fit`, `calibration`, `HMM tuning`, `test` event groups, and >=3 classes per group split. If insufficient, return **BLOCKED**, not random leakage.

For the first branch a lower-dimensional train-only simple feature selector should be compared against grouped MSF, with balanced accuracy and feature count recorded. Calibration should be compared to uncalibrated LR; HMM should be accepted only if improvement is consistent without loss of class recall or probability calibration.

## 5. Current completion status

| Work item | Status |
|---|---|
| ESA official label mapping checked | DONE (verified against upstream source) |
| Separate true-anomaly and rare-normal semantics | DONE |
| Chronological event-first split + causal windows | DONE, synthetic unit tests |
| Small-data adequacy gate | DONE; P-1 fails |
| Three lightweight anomaly baselines | DONE, synthetic smoke only |
| Grouped two-task MSF, probability calibration, HMM primitives | DONE, synthetic tests only |
| ESA real data download/preprocessing | **PENDING**; 3.8GB download/external compute required |
| Telemanom-ESA official Docker reproduction | **PENDING** |
| Full real-data V1.3 scores | **NOT AVAILABLE** |
| Fusion/online incremental learning | intentionally disabled |

Next gate: real ESA-Mission1 preprocessing + enough events to select model without a single-event validation bottleneck. Preserve raw checksums, code commit, seed list, per-event rankings and once-only test metrics.
