#!/usr/bin/env python3
"""Post-hoc REAL ESA failure DIAGNOSTICS only; never use to retune model.

Analyses original E1 archived real evaluation outputs. Separate genuine ESA
target Anomaly, Rare Nominal, and ordinary Nominal. The engineering H/D/R
risk tier is NOT a physical fault label or expert-rated health score.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/"results/real_esa_health_hmm"

def run():
    f=pd.read_csv(OUTPUT/"real_timepoint_scores.csv")
    metrics=json.loads((OUTPUT/"metrics.json").read_text())
    roles=np.where(f.official_anomaly_label_at_endpoint.to_numpy(int)==1,"anomaly",
             np.where(f.official_rare_nominal_at_endpoint.to_numpy(int)==1,
             "rare_nominal","ordinary_nominal"))
    prob=f[["main_state_H_weak","main_state_D_weak","main_state_R_weak"]].to_numpy(float)
    hmm_risk=1-prob[:,0]
    raw=f["score_robust_level_reference"].to_numpy(float)
    static=f["score_static_three_tier"].to_numpy(float)
    summaries={}
    for group in ["ordinary_nominal","rare_nominal","anomaly"]:
        hit=roles==group
        quant=lambda x:[float(v) for v in np.quantile(x,[.1,.5,.9])]
        summaries[group]={
          "n_real_windows":int(hit.sum()),
          "raw_robust_risk_quantile_10_50_90":quant(raw[hit]),
          "static_risk_quantile_10_50_90":quant(static[hit]),
          "HMM_risk_quantile_10_50_90":quant(hmm_risk[hit]),
          "HMM_R_argmax_fraction":float(np.mean(np.argmax(prob[hit],axis=1)==2)),
          "HMM_H_argmax_fraction":float(np.mean(np.argmax(prob[hit],axis=1)==0)),
        }
    report={
      "study":"Posthoc failure diagnostic of already-used ESA Mission1 continuous real E1 data",
      "not_new_confirmatory_experiment":True,
      "real_ESA_official_labels":True,
      "labels":["target anomaly","rare nominal (negative under study convention)","ordinary nominal"],
      "source_metrics_key":"results/real_esa_health_hmm/metrics.json",
      "n_real_evaluated_windows":int(len(f)),
      "label_class_summaries":summaries,
      "static_HMM_risk_average_absolute_difference":float(np.mean(np.abs(hmm_risk-static))),
      "static_HMM_risk_correlation":float(np.corrcoef(hmm_risk,static)[0,1]),
      "transition_mat":metrics["weak_HMM_transition"],
      "HMM_high_risk_state_self_transition":float(metrics["weak_HMM_transition"][2][2]),
      "HMM_normal_reference_self_transition":float(metrics["weak_HMM_transition"][0][0]),
      "weak_state_definitions_do_not_come_from_physical_fault_annotators":True,
      "forbidden":"No threshold change/model selection on these exposed real ESA test event results",
      "interpretation_limit":"A lower alarm occupancy paired with zero strict event onset recall is not a safe operational improvement; this diagnostic cannot verify expert score calibration."
    }
    (OUTPUT/"failure_diagnostics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("ESA_FAILURE_DIAGNOSTIC",json.dumps(report,ensure_ascii=False),flush=True)
    return report


if __name__=="__main__":run()
