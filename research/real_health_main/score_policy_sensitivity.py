#!/usr/bin/env python3
"""Compare two explicitly hypothetical final display score POLICIES on REAL records.

Not expert calibration: No operator ratings in OPS-SAT. Different policy
demonstrates scale sensitivity only, and must not modify underlying telemetry
anomaly probability/alarms or HMM state vectors.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from display_score import DisplayScorePolicy


def run(source="results/real_health_main/audit_segment_predictions.csv",
        out="results/real_health_main/score_policy_sensitivity.json"):
    p=Path(source)
    df=pd.read_csv(p)
    required=["p_hmm_healthy","p_hmm_deviation","p_hmm_highrisk",
        "p_fused_anomaly","p_aux_anomaly","alarm_input_p","binary_y"]
    if not set(required).issubset(df):
        raise RuntimeError("Not a verified real-telemetry benchmark prediction table")
    states=df[["p_hmm_healthy","p_hmm_deviation","p_hmm_highrisk"]].to_numpy(float)
    fusionprob=df["p_fused_anomaly"].to_numpy(float)
    alarmprob=df["p_aux_anomaly"].to_numpy(float)
    before=alarmprob.copy()
    version="opssat_training_group_based_fusion_v0.1"
    original=DisplayScorePolicy(100,60,0,15,"INIT_100_60_0",version)
    alternative=DisplayScorePolicy(100,70,0,15,"HYPOTHETICAL_SCALE_100_70_0_NOT_EXPERT",version)
    score_a=original.evaluate(states,fusionprob,fusion_version=version)
    score_b=alternative.evaluate(states,fusionprob,fusion_version=version)
    assert np.array_equal(before,alarmprob)
    assert np.array_equal(before,df.alarm_input_p.to_numpy(float))
    assert np.allclose(states.sum(axis=1),1)
    # No threshold/alert recomputation from subjective health score.
    alarms_a=alarmprob>=.5
    alarms_b=alarmprob>=.5
    assert np.array_equal(alarms_a,alarms_b)
    sdelta=score_b-score_a
    output={
      "grounding":"TRUE published OPS-SAT-AD original train==1, heldout internal segments only",
      "rows":len(df),"genuine_binary_anomalies":int(df.binary_y.sum()),
      "two_PRESCRIBED_not_expert_calibrated_policy_versions":[
       original.expert_mapping_version,alternative.expert_mapping_version],
      "expert_score_groundtruth_count":0,
      "cannot_report_expert_MAE":True,
      "original_policy_weight":[100,60,0],"new_hypothetical_policy_weight":[100,70,0],
      "raw_main_state_probs_unchanged":True,
      "raw_aux_anomaly_probs_unchanged":True,
      "alarm_decisions_unchanged":bool(np.array_equal(alarms_a,alarms_b)),
      "alarm_rate_at_arbitrary_0_5_threshold":float(np.mean(alarms_a)),
      "score_A_mean":float(score_a.mean()),"score_B_mean":float(score_b.mean()),
      "mean_score_B_minus_A":float(sdelta.mean()),
      "nonzero_score_delta_count":int(np.sum(abs(sdelta)>1e-9)),
      "score_A_minmax":[float(score_a.min()),float(score_a.max())],
      "score_B_minmax":[float(score_b.min()),float(score_b.max())],
      "interpretation":"A policy-sensitivity/invariance check, NOT measured human-score agreement or validation of initial policy 15 point anomaly penalty. Objective anomaly alarm is separate."
    }
    Path(out).write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print("SCORE_POLICY_REAL",json.dumps(output,ensure_ascii=False),flush=True)
    return output


if __name__=="__main__":run()
