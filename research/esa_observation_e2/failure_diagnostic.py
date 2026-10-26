#!/usr/bin/env python3
"""Read-only diagnostic of WHY fixed ESA E2 real event alarms were missed.

This is an exploratory audit of already exposed real ESA event IDs, not a new
independent test, not an optimized model and not a patented decision mechanism.
All thresholds come UNCHANGED from E2 validation-only metrics.json. No fitting,
reweighting, new labels, score calibration or changes to frozen protocol occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from v13_protocol import Partition  # noqa: E402
from v14_alarm import sustained_alert, alert_statistics  # noqa: E402
from v15_global_events import get_global_event_records, event_alarm_metrics  # noqa: E402

SPLIT = [431304, 593284]
MODEL_NAMES = (
    "original_robust_level",
    "forecast_innovation",
    "local_change",
    "fixed_combined_static_risk",
    "fixed_combined_static_3tier",
    "fixed_combined_EMA_3tier",
    "fixed_combined_HMM_3tier",
)


def event_audit(records, part, ends, codes, scores, threshold):
    """Pure read-only threshold/3-window decomposition; no fitted quantities."""
    ends = np.asarray(ends, dtype=int)
    codes = np.asarray(codes, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(ends) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid score shape or values")
    above = scores >= float(threshold)
    on = sustained_alert(scores, threshold, ends, min_run=3, expected_stride=4)
    adjacent = np.r_[False, np.diff(ends) == 4]
    rising = on & (~np.r_[False, on[:-1]] | ~adjacent)
    result = []
    for r in records:
        if not (r["target_evaluable"] and
                part.start <= r["global_start"] and r["global_stop"] <= part.stop):
            continue
        inside = (ends >= r["target_start"]) & (ends < r["target_stop"]) & (codes[ends] == 1)
        candidates = np.flatnonzero(inside & above)
        covered = np.flatnonzero(inside & on)
        onset = np.flatnonzero(inside & rising)
        max_run = 0
        run = 0
        last_index = -2
        for idx in candidates:
            run = run + 1 if idx == last_index + 1 and ends[idx] - ends[last_index] == 4 else 1
            max_run = max(max_run, run)
            last_index = idx
        if len(onset):
            reason = "in_event_new_onset"
        elif len(covered):
            reason = "alarm_preexisting"
        elif not len(candidates):
            reason = "no_in_event_single_threshold_crossing"
        else:
            reason = "threshold_crossings_without_in_event_sustained_alarm"
        result.append({
            "id": r["id"],
            "target_anomaly_evaluation_windows": int(inside.sum()),
            "instantaneous_validation_threshold_crossings": int(len(candidates)),
            "max_consecutive_in_event_crossings": int(max_run),
            "sustained_alarm_covered": bool(len(covered)),
            "strict_new_alarm_onset": bool(len(onset)),
            "first_crossing_delay_10min_grid_steps": (
                int(ends[candidates[0]] - r["target_start"]) if len(candidates) else None
            ),
            "strict_alarm_delay_10min_grid_steps": (
                int(ends[onset[0]] - r["target_start"]) if len(onset) else None
            ),
            "observed_failure_type": reason,
        })
    return result, on


def run(result_dir, source, catalog):
    result_dir = Path(result_dir)
    metric_path = result_dir / "metrics.json"
    m = json.loads(metric_path.read_text(encoding="utf-8"))
    cfg_path = ROOT / "research/esa_observation_e2/PROTOCOL_FROZEN.json"
    if hashlib.sha256(cfg_path.read_bytes()).hexdigest() != m["protocol_SHA256"]:
        raise ValueError("E2 frozen protocol hash changed")
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != m["real_ESA_source_SHA256"]:
        raise ValueError("Actual ESA source hash does not match frozen E2 output")
    if hashlib.sha256(Path(catalog).read_bytes()).hexdigest() != m["real_catalog_SHA256"]:
        raise ValueError("Official ESA catalog hash mismatch")
    if m["source_window_end_counts"]["test"] != 35779:
        raise ValueError("Incorrect frozen test evaluation population")
    if set(m["results"]) != set(MODEL_NAMES):
        raise ValueError("Missing frozen model comparison")
    src = pd.read_csv(source, usecols=["timestamp", "eval_label"])
    codes = src.eval_label.to_numpy(dtype=int)
    times = pd.to_datetime(src.timestamp, utc=True, format="mixed")
    records = get_global_event_records(pd.read_csv(catalog), times, codes)
    frame = pd.read_csv(result_dir / "real_timepoint_scores.csv")
    ends = frame["raw_global_grid_indices"].to_numpy(dtype=int)
    if (len(ends) != m["source_window_end_counts"]["test"] or
            not np.all(np.diff(ends) > 0) or np.any(ends < SPLIT[1]) or
            np.any(ends >= len(codes))):
        raise ValueError("Frozen ESA test-window ends changed")
    if (not np.array_equal(frame["official_anomaly"].to_numpy(int), (codes[ends] == 1).astype(int))
            or not np.array_equal(frame["official_rare_nominal"].to_numpy(int),
                                  (codes[ends] == 2).astype(int))):
        raise ValueError("Official labels and test predictions misaligned")
    part = Partition("test", SPLIT[1], len(codes), ())
    expected_ids = set(m["official_event_ID_partition"]["test"]["target_evaluable_anomaly"])
    output = {}
    for name in MODEL_NAMES:
        scores = frame["score_" + name].to_numpy(dtype=float)
        row = m["results"][name]
        threshold = row["alarm_cutoff_from_validation"]
        details, alarm = event_audit(records, part, ends, codes, scores, threshold)
        if {x["id"] for x in details} != expected_ids:
            raise AssertionError("Official target event IDs differ from frozen metrics")
        assert len(details) == 10, "Frozen test event count changed"
        stats = alert_statistics(ends, alarm, codes, expected_stride=4)
        official = event_alarm_metrics(records, part, ends, codes, alarm, 4, grid_step_minutes=10)
        old = row["alarm_occupancy"]
        for field in ("normal_alert_windows", "rare_nominal_alert_windows",
                      "true_anomaly_alert_windows", "false_alarm_episodes"):
            if stats[field] != old[field]:
                raise AssertionError("Alarm replay does not match original E2 metrics: " + name + " " + field)
        for field in ("coverage_ID_recall", "strict_new_onset_ID_recall",
                      "false_alarm_new_onsets"):
            if not np.isclose(official[field], row["official_ID_metrics"][field], atol=1e-12):
                raise AssertionError("Official event replay differs: " + name + " " + field)
        reason_counts = {k: sum(z["observed_failure_type"] == k for z in details) for k in (
            "no_in_event_single_threshold_crossing",
            "threshold_crossings_without_in_event_sustained_alarm",
            "alarm_preexisting",
            "in_event_new_onset",
        )}
        output[name] = {
            "frozen_validation_threshold": threshold,
            "events_with_at_least_one_single_crossing": sum(
                z["instantaneous_validation_threshold_crossings"] > 0 for z in details),
            "events_with_strict_new_alarm": sum(z["strict_new_alarm_onset"] for z in details),
            "failure_reason_counts": reason_counts,
            "per_official_event": details,
        }
    diagnostic = {
        "study": "Frozen real ESA E2 missed-event threshold attribution audit",
        "source": "previously exposed real ESA Mission1 official event IDs, exploratory ONLY",
        "model_weights_thresholds_splits_modified": False,
        "prediction_or_model_refit": False,
        "not_an_independent_holdout": True,
        "no_true_three_state_physical_fault_labels": True,
        "no_expert_health_scores": True,
        "e2_metrics_SHA256": hashlib.sha256(metric_path.read_bytes()).hexdigest(),
        "results": output,
        "caution": (
            "Single-window threshold crossings are diagnostic only; they are NOT deployed alarms. "
            "An event absent above the frozen threshold does NOT prove absence of a physical "
            "signature. This replay cannot estimate blind generalization or justify tuning."
        ),
    }
    (result_dir / "FAILURE_DIAGNOSTIC.json").write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# ESA E2真实遥测：异常未触发告警的固定阈值归因审计",
        "",
        "**只读取既有真实ESA窗口分数；阈值、三连告警、数据划分、模型权重和HMM参数完全不变。所有测试事件此前已被观察，本诊断不是新盲测，也不证明故障状态。**",
        "",
        "| 模型 | 至少单次跨验证阈值的事件/10 | 无任何单次越线 | 越线但无持续告警 | 原有告警覆盖而无新起点 | 事件内真正新告警 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in output.items():
        counts = item["failure_reason_counts"]
        lines.append(
            f"| {name} | {item['events_with_at_least_one_single_crossing']}/10 | "
            f"{counts['no_in_event_single_threshold_crossing']} | "
            f"{counts['threshold_crossings_without_in_event_sustained_alarm']} | "
            f"{counts['alarm_preexisting']} | {counts['in_event_new_onset']} |"
        )
    lines += [
        "",
        "单窗越线只是失败归因观察，不是有效告警；实际报警严格保留连续三个40分钟评价窗口。",
        "如大多数事件连单窗阈值都未越过，优先检验工程状态观测的可识别性；如越线却没有新起点，需要区分持续性要求及此前已存在的告警。",
        "不能根据已看过的官方测试事件选择新阈值、修改组合权重或声称独立验收。",
        "",
        "逐事件统计详见同目录 FAILURE_DIAGNOSTIC.json。",
        "",
    ]
    (result_dir / "FAILURE_DIAGNOSTIC.md").write_text("\n".join(lines), encoding="utf-8")
    print("ESA_E2_FAILURE_DIAGNOSTIC", json.dumps(
        {name: data["failure_reason_counts"] for name, data in output.items()},
        ensure_ascii=False
    ), flush=True)
    return diagnostic


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", default="results/esa_observation_e2")
    p.add_argument("--source", default="data/esa_v15_multichannel/three_channels.csv.gz")
    p.add_argument("--catalog", default="data/esa_v15_multichannel/mission1_event_catalog.csv")
    args = p.parse_args()
    run(args.result_dir, args.source, args.catalog)
