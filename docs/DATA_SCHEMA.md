# V1 schema and feedback separation

## Ground-truth labels

NASA SMAP `anomaly_sequences` yields only 0/1 anomaly intervals. Our `degraded_proxy` / `fault_proxy` are **artificial partitions** of true anomaly segments; they should **never** be interpreted as real degradation or mechanical failure labels.

## Annotation input schema (for future real data)

| Field | Type | Comment |
|---|---|---|
| `sample_id`, `device_id` | string | references original signal |
| `window_start`, `window_end` | integer/timestamp | timestamp or index |
| `raw_data_ref` | string | pointer to real time series |
| `operating_condition` | object/null | speed/load/mode when available |
| `anomaly_label` | bool/null | time-series statistical anomaly |
| `health_state_label` | enum/null | genuine `healthy/degraded/fault`, NEVER inferred from anomaly label |
| `health_score_label` | number/null (0..100) | from human assessment |
| `label_source` | string | engineer / repair_record / simulation / NASA_anomaly_annotation |
| `confirmation_status` | enum | pending/confirmed/ruled_out/not_inspected |
| `confirmed_at` | timestamp/null | after original event |
| `score_basis` | string/null | maintenance/measurement/assessment |
| `annotation_confidence` | number/null | optional human confidence |

## Model outputs

- `base_probabilities`: HMM/DBN status posterior, 3-vector.
- `anomaly_evidence`: calibrated score, channel/time interval, model version.
- `fused_probabilities`: 3-vector, sum to one.
- `health_score`: model probability-to-score mapping.
- `score_profile_version`: supports user calibration without touching state prediction.

## Preventing leakage

- Train MSF ranking, scalers, classification and HMM transitions on training only.
- Train fusion and user mapping on validation only; never fit to final test.
- Human inspection confirmation may arrive later; `confirmed_at` must not leak into predictions for earlier windows.
- All feedback in this V1 is programmatically simulated and explicitly marked so in results.
