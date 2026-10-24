#!/usr/bin/env python3
"""Public baseline: candidate anomaly-model update vs stale/refitted fusion.

SIMULATED labels ONLY; no human verified satellite data or deployment.
This experiment uses ordinary RF supervised retraining, conventional logistic
late-fusion recalibration, and ordinary validation-only rollback decisions.
No patent-specific algorithm is implemented here.

Original synthetic state classifier, MSF selectors, transition matrix, and
HMM remain frozen. Only anomaly branch and candidate fusion are refitted.

DATA PARTITION
 - missions 0,1: historical detector training (544 windows)
 - mission 2 cycle 8: simulated newly reviewed update (68 windows)
 - mission 2 cycle 9: anomaly threshold calibration (68 windows)
 - mission 2 cycle 10: identical old/new fusion fitting labels (68 windows)
 - mission 2 cycle 11: relative candidate acceptance labels (68 windows)
 - mission 3 cycles12..15: audit (272 windows)

Legacy state Platt calibrators/prior were fitted to all mission2 during MVP
training, including acceptance cycle11. Thus acceptance is NOT fully held out
from the legacy state branch; any differences are exploratory only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score,average_precision_score,precision_score,
                             recall_score,roc_auc_score,confusion_matrix)

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"mvp"))
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"research/hitl_update"))
from synthetic import generate,make_windows,highdim_features
from v13_state import predict_hierarchical,filter_groups,evaluation
from gate import validate_supervised_upload,canonical_csv
from run import stacking_inputs,DEFAULT_WEIGHTS,choose_threshold

PROTOCOL=ROOT/"research/probability_compatibility/PROTOCOL_FROZEN.json"
FROZEN=ROOT/"results/mvp"


def metrics_state(y,p):
    s=evaluation(y,p)
    return {k:s[k] for k in ("macro_f1","recalls","brier","nll")}


def metrics_binary(y,p,t):
    y=np.asarray(y,int)
    p=np.asarray(p,float)
    pred=p>=t
    return {
      "n":int(len(y)),"positives":int(y.sum()),
      "f1":float(f1_score(y,pred,zero_division=0)),
      "ap":float(average_precision_score(y,p)),
      "auc":float(roc_auc_score(y,p)) if len(set(y))==2 else None,
      "precision":float(precision_score(y,pred,zero_division=0)),
      "recall":float(recall_score(y,pred,zero_division=0)),
      "fpr":float(np.mean(pred[y==0])) if np.any(y==0) else None,
      "FP":int(np.sum(pred&(y==0))),
      "FN":int(np.sum((~pred)&(y==1))),
      "threshold":float(t),
      "prob_brier":float(np.mean((p-y)**2)),
      "prob_nll":float(-np.mean(y*np.log(np.clip(p,1e-8,1))+
                      (1-y)*np.log(np.clip(1-p,1e-8,1))))}


def _jensen_shannon_bernoulli(a,b):
    a=np.clip(np.asarray(a,float),1e-8,1-1e-8)
    b=np.clip(np.asarray(b,float),1e-8,1-1e-8)
    m=(a+b)*.5
    def kl(p,q):
        return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
    return float(np.mean(.5*kl(a,m)+.5*kl(b,m)))


def js_categorical(a,b):
    a=np.clip(np.asarray(a,float),1e-9,1)
    b=np.clip(np.asarray(b,float),1e-9,1)
    a=a/a.sum(axis=1,keepdims=True)
    b=b/b.sum(axis=1,keepdims=True)
    m=(a+b)*.5
    kl=lambda p,q:np.sum(p*np.log(p/q),axis=1)
    return float(np.mean(.5*(kl(a,m)+kl(b,m))))


def transform_stress(p):
    """Public diagnostic only: artificial post-update interface probability shift."""
    z=logit(np.clip(np.asarray(p,float),1e-6,1-1e-6))
    return expit(1.8*z+0.7)


def fusion_estimator():
    return LogisticRegression(C=.5,max_iter=1200,random_state=42)


def monotime(f,repeats=7):
    f()
    durations=[]
    for _ in range(repeats):
        now=time.perf_counter()
        f()
        durations.append(1000*(time.perf_counter()-now))
    return {"median_ms":float(np.median(durations)),
            "q25_ms":float(np.percentile(durations,25)),
            "q75_ms":float(np.percentile(durations,75)),
            "repeats":repeats}


def qualify_gate(before,after,rules):
    checks={
      "state_macro_f1":bool(after["state"]["macro_f1"]>=before["state"]["macro_f1"]-rules["max_macro_f1_drop"]),
      "state_brier":bool(after["state"]["brier"]<=before["state"]["brier"]+rules["max_brier_increase"]),
      "state_nll":bool(after["state"]["nll"]<=before["state"]["nll"]+rules["max_state_nll_increase"]),
      "anomaly_recall":bool(after["anomaly"]["recall"]>=before["anomaly"]["recall"]-rules["max_anomaly_recall_drop"]),
      "anomaly_fpr":bool(after["anomaly"]["fpr"]<=before["anomaly"]["fpr"]+rules["max_anomaly_fpr_increase"]),
      "health_score_mae":bool(after["health_score_MAE"]<=before["health_score_MAE"]+rules["max_health_MAE_increase"])
    }
    return checks,all(checks.values())


def probability_shift(before,after,base_state_fused=None,other_state_fused=None):
    before=np.asarray(before,float);after=np.asarray(after,float)
    output={
      "anomaly_probability_mean_abs_change":float(np.mean(np.abs(after-before))),
      "anomaly_probability_p95_abs_change":float(np.quantile(np.abs(after-before),.95)),
      "anomaly_bernoulli_JS_nats":_jensen_shannon_bernoulli(before,after),
      "anomaly_binary_rank_preserved":bool(
         np.all(np.argsort(before,kind="stable")==np.argsort(after,kind="stable"))) }
    if base_state_fused is not None and other_state_fused is not None:
        output["fused_state_JS_nats"]=js_categorical(base_state_fused,other_state_fused)
        output["fused_state_probability_mean_abs_change"]=float(
            np.mean(np.abs(np.asarray(base_state_fused)-np.asarray(other_state_fused))))
    return output


def evaluate_arm(ys,ya,score_truth,post,anomaly_scores,threshold,fusion_model):
    fused=fusion_model.predict_proba(stacking_inputs(post,anomaly_scores))
    assert fused.shape==(len(ys),3)
    if np.any(~np.isfinite(fused)) or not np.allclose(fused.sum(axis=1),1):
        raise AssertionError("Invalid fused probability vector")
    return {
        "state":metrics_state(ys,fused),
        "anomaly":metrics_binary(ya,anomaly_scores,threshold),
        "health_score_MAE":float(np.mean(np.abs(fused@DEFAULT_WEIGHTS-score_truth))),
        "fused_probabilities":fused
    }


def _public_arm(row):
    return {k:v for k,v in row.items() if k!="fused_probabilities"}


def run(out_path):
    out=Path(out_path);out.mkdir(parents=True,exist_ok=True)
    protocol_bytes=PROTOCOL.read_bytes()
    spec=json.loads(protocol_bytes)
    existing_metrics=json.loads((FROZEN/"metrics.json").read_text())
    models=joblib.load(FROZEN/"model_bundle.joblib")
    if models["spec_sha256"]!=existing_metrics["protocol_sha256"]:
        raise AssertionError("Frozen simulator/artifact mismatched")
    windows,meta=make_windows(generate())
    x,names=highdim_features(windows)
    if names!=models["feature_names"]:raise AssertionError("State feature schema mismatch")
    cycle=meta.cycle_id.to_numpy(int)
    mission=meta.mission_id.to_numpy(int)
    state=meta.health_state_simulated.to_numpy(int)
    anomaly=meta.anomaly_simulated.to_numpy(int)
    scores=meta.operator_score_simulated.to_numpy(float)
    raw=windows.reshape(len(windows),-1)
    masks={
      "historical_train":mission<=1,
      "simulated_operator_batch":cycle==8,
      "anomaly_threshold":cycle==9,
      "fusion_fit":cycle==10,
      "gate_validation":cycle==11,
      "mission3_audit":mission==3
    }
    lengths={k:int(np.sum(v)) for k,v in masks.items()}
    assert lengths=={
      "historical_train":544,"simulated_operator_batch":68,
      "anomaly_threshold":68,"fusion_fit":68,
      "gate_validation":68,"mission3_audit":272}
    chosen=np.r_[*[np.flatnonzero(m) for k,m in masks.items() if k!="historical_train"]]
    if len(chosen)!=len(np.unique(chosen)):raise AssertionError("Reserved groups overlap")
    orig_model=models["anomaly_detector"]
    orig_probs=orig_model.predict_proba(raw)[:,1]

    # Simulated uploader: sends only normal/anomaly labels, never physical
    # healthy/degraded/fault states and never artificial operator-score truth.
    new_ids=np.flatnonzero(masks["simulated_operator_batch"])
    feature_columns=[f"sample_{i:03d}" for i in range(raw.shape[1])]
    upload=pd.DataFrame(raw[new_ids],columns=feature_columns)
    upload.insert(0,"label",anomaly[new_ids])
    upload.insert(0,"sample_id",[f"sim_m2_cycle8_window_{k}" for k in new_ids])
    batch_digest=hashlib.sha256(canonical_csv(upload)).hexdigest()
    manifest={
      "batch_id":"SIMULATED_OPERATOR_V01_CYCLE8",
      "target_task":"anomaly_binary",
      "label_source":"simulator_defined_anomaly_truth_not_real_human",
      "review_status":"simulated_review",
      "reviewer_id":"SIMULATED_REVIEWER_NOT_AUTHENTICATED",
      "reviewed_at":"2026-10-16T00:00:00Z",
      "parent_model_version":"frozen-synthetic-mvp-v1",
      "data_sha256":batch_digest,
      "approval_status":"pending",
      "data_unit":"annotated_segment",
      "demo_only":True
    }
    all_reserved=np.flatnonzero(masks["anomaly_threshold"]|masks["fusion_fit"]|
                   masks["gate_validation"]|masks["mission3_audit"])
    gate_result=validate_supervised_upload(
       upload,manifest,feature_columns,
       blocked_sample_ids=[f"sim_m2_cycle8_window_{i}" for i in all_reserved],
       expected_parent_model="frozen-synthetic-mvp-v1",
       allow_simulated_review=True)
    assert gate_result.candidate_allowed and not gate_result.release_allowed
    assert gate_result.route=="anomaly_detector_candidate"
    assert not manifest["approval_status"]=="approved"

    pool=masks["historical_train"]|masks["simulated_operator_batch"]
    def refit_detector():
        candidate=clone(orig_model)
        candidate.fit(raw[pool],anomaly[pool])
        return candidate
    candidate_fit=monotime(refit_detector,repeats=spec["timing"]["model_fit_repeats"])
    updated_model=refit_detector()
    new_probs=updated_model.predict_proba(raw)[:,1]
    base_post=filter_groups(models["hierarchical"],predict_hierarchical(
         models["hierarchical"],x),cycle,gamma=.6,memory=.5)
    test_idx=np.flatnonzero(masks["mission3_audit"])
    # Verify frozen upstream model matches previously reported HMM scores.
    historic=evaluation(state[test_idx],base_post[test_idx])
    for key in ("macro_f1","brier","nll"):
        if not np.isclose(historic[key],existing_metrics["base_hmm_state_metrics"][key],atol=1e-10):
            raise AssertionError("Historical HMM path unexpectedly changed: "+key)

    def fit_fusion(d):
        xfit=stacking_inputs(base_post[masks["fusion_fit"]],d[masks["fusion_fit"]])
        yf=state[masks["fusion_fit"]]
        if set(yf)!={0,1,2}:raise AssertionError("Fusion fitting lacks 3 states")
        return fusion_estimator().fit(xfit,yf)
    # Common compatible reference trained once on cycle10. Unlike original
    # published MVP fusion (trained all mission2), this avoids using cycle11
    # gate labels when fitting the two compared fusion versions.
    fusion_time_old=monotime(lambda:fit_fusion(orig_probs),
                  repeats=spec["timing"]["fusion_fit_repeats"])
    fusion_old=fit_fusion(orig_probs)
    old_threshold=choose_threshold(orig_probs[masks["anomaly_threshold"]],
                   anomaly[masks["anomaly_threshold"]])

    output={
      "study":"Public generic two-branch probability interface maintenance, synthetic-only",
      "simulated_review_only":True,"real_human_reviewed_labels":False,
      "candidate_model_deployed":False,
      "original_delivery_model_untouched":True,
      "protocol_sha256":hashlib.sha256(protocol_bytes).hexdigest(),
      "frozen_state_msfs":"Original synthetic feature selector 50+10 (59 unique); no feature refit",
      "real_OPSSAT_budget_max64_status":"Frozen earlier budget64, not transferred to this synthetic state task",
      "counts":lengths,
      "gate_data_validation":{"candidate_allowed":gate_result.candidate_allowed,
            "production_release_allowed":gate_result.release_allowed,
            "route":gate_result.route,"label_origin":"simulated",
            "new_anomaly_labels":int(anomaly[new_ids].sum()),
            "new_normal_labels":int(len(new_ids)-anomaly[new_ids].sum()),
            "sha256":gate_result.digest},
      "runtime":{"candidate_RF_train_with_68_simulated_supervision":candidate_fit,
                 "reference_fusion_fit_only_cycle10":fusion_time_old},
      "scenarios":{},
      "limitations":[
        "All new binary anomalies and three-class labels were generated by frozen simulator; not satellite expert labels",
        "No new normal regime has actually been discovered or verified by a human",
        "Candidate RF uses ordinary supervised refit on history plus exactly one 68-window simulated labeled cycle",
        "Baseline and refreshed fusion both fit on exactly same reserved cycle10 three-class labels",
        "Original frozen hierarchy probabilities/Platt calibration and prior already saw ALL mission2 labels including gate cycle11; inherited reference state-model leakage makes gate comparative and not independent full-pipeline validation",
        "Original publication fusion is NOT reused as comparison baseline; it used whole mission2, so we refit both old/new fusion on cycle10 for fairer pairwise comparison",
        "Mission3 was already observed in historical MVP publications; this is exploratory, not a fresh unknown audit set",
        "Probability logit scaling is a deliberately artificial interface stress, not a real trained detector",
        "No proprietary patent mechanism is tested; ordinary recalibration/rollback alone is not a patentable algorithm by itself",
        "No model is auto deployed; no real authorized approval or human reviewed data",
        "Do not compare with physical spacecraft degradation or real health score labels"
      ]
    }
    rows=[]
    predictions=pd.DataFrame({
      "mission":meta.mission_id.to_numpy()[test_idx],
      "cycle_id":cycle[test_idx],
      "sample_in_mission":meta.sample_in_mission.to_numpy(int)[test_idx],
      "state_truth_simulated":state[test_idx],
      "anomaly_truth_simulated":anomaly[test_idx],
    })
    times=spec["timing"]["inference_repeats"]
    for scenario in ("natural","logit_scale_stress"):
        candidate_score= new_probs if scenario=="natural" else transform_stress(new_probs)
        update_threshold=choose_threshold(
              candidate_score[masks["anomaly_threshold"]],
              anomaly[masks["anomaly_threshold"]])
        new_fusion_time=monotime(lambda:fit_fusion(candidate_score),
             repeats=spec["timing"]["fusion_fit_repeats"])
        new_fusion=fit_fusion(candidate_score)
        scen_report={
          "stress_is_artificial":scenario!="natural",
          "threshold_fit_on_cycle9":{"old":float(old_threshold),"candidate":float(update_threshold)},
          "fusion_fit_times":{"old":fusion_time_old,"candidate":new_fusion_time},
          "gate_validation":{},"audit_mission3":{},
          "probability_shifts":{},
          "gate_checks":{},
          "stage_only":True
        }
        def arm_evaluations(mask):
            ys=state[mask];ya=anomaly[mask];score_truth=scores[mask]
            post=base_post[mask]
            options={
              "A_legacy_detector_compatible_fusion":(orig_probs[mask],old_threshold,fusion_old),
              "B_updated_detector_stale_fusion":(candidate_score[mask],update_threshold,fusion_old),
              "C_updated_detector_refit_fusion":(candidate_score[mask],update_threshold,new_fusion)
            }
            return {k:evaluate_arm(ys,ya,score_truth,post,*v)
                    for k,v in options.items()}
        hold=arm_evaluations(masks["gate_validation"])
        base=hold["A_legacy_detector_compatible_fusion"]
        candidate=hold["C_updated_detector_refit_fusion"]
        checks,accepted=qualify_gate(base,candidate,spec["acceptance_validation"])
        chosen="C_updated_detector_refit_fusion" if accepted else "A_legacy_detector_compatible_fusion"
        scen_report["gate_checks"]=checks
        scen_report["candidate_allowed_for_human_approval"]=accepted
        scen_report["selected_undeployed_arm"]=chosen
        audit_rows=arm_evaluations(masks["mission3_audit"])
        audit_rows["D_staged_validation_selection"]=audit_rows[chosen]
        hold["D_staged_validation_selection"]=hold[chosen]
        scen_report["gate_validation"]={k:_public_arm(v) for k,v in hold.items()}
        scen_report["audit_mission3"]={k:_public_arm(v) for k,v in audit_rows.items()}
        for part,which in [("gate_validation",masks["gate_validation"]),
                            ("audit_mission3",masks["mission3_audit"])]:
            earlier=(hold if part=="gate_validation" else audit_rows)
            scen_report["probability_shifts"][part]={
               "stale":probability_shift(orig_probs[which],candidate_score[which],
                     earlier["A_legacy_detector_compatible_fusion"]["fused_probabilities"],
                     earlier["B_updated_detector_stale_fusion"]["fused_probabilities"]),
               "refit":probability_shift(orig_probs[which],candidate_score[which],
                     earlier["A_legacy_detector_compatible_fusion"]["fused_probabilities"],
                     earlier["C_updated_detector_refit_fusion"]["fused_probabilities"])
            }
        def f_original(mask,arm):
            s=orig_probs[mask] if arm=="A" else candidate_score[mask]
            fuse=fusion_old if arm in ("A","B") else new_fusion
            return fuse.predict_proba(stacking_inputs(
               filter_groups(models["hierarchical"],predict_hierarchical(
                models["hierarchical"],x[mask]),cycle[mask],gamma=.6,memory=.5),s))
        stage_functions={
           "A_legacy_detector_compatible_fusion":lambda:f_original(masks["mission3_audit"],"A"),
           "B_updated_detector_stale_fusion":lambda:f_original(masks["mission3_audit"],"B"),
           "C_updated_detector_refit_fusion":lambda:f_original(masks["mission3_audit"],"C")
        }
        # Full raw classifier inference timing separately, avoiding repeated
        # hidden state model fitting.
        timed_data={}
        raw_audit=raw[masks["mission3_audit"]]
        for tag,m in [("old",orig_model),("candidate",updated_model)]:
            timed_data[tag+"_anomaly_predict"]=monotime(
               lambda mdl=m:mdl.predict_proba(raw_audit)[:,1],repeats=times)
        for label,fun in stage_functions.items():
            timed_data[label+"_state_hmm_and_fusion_given_cached_anomaly"]=monotime(fun,repeats=times)
        scen_report["runtime_audit"]=timed_data
        output["scenarios"][scenario]=scen_report
        for arm,val in scen_report["audit_mission3"].items():
            rows.append({
              "scenario":scenario,"arm":arm,
              "is_synthetic":True,"selected_undeployed_arm":chosen,
              "candidate_gate_accepted":accepted,
              "state_macro_f1":val["state"]["macro_f1"],
              "state_brier":val["state"]["brier"],"state_nll":val["state"]["nll"],
              "health_MAE":val["health_score_MAE"],
              "anomaly_f1":val["anomaly"]["f1"],"anomaly_ap":val["anomaly"]["ap"],
              "anomaly_recall":val["anomaly"]["recall"],"anomaly_fpr":val["anomaly"]["fpr"]
            })
        for arm,v in audit_rows.items():
            predictions[f"{scenario}__{arm}__pred_state"]=np.argmax(v["fused_probabilities"],axis=1)
            for cls,label in enumerate(("healthy","degraded","fault")):
                predictions[f"{scenario}__{arm}__p_{label}"]=v["fused_probabilities"][:,cls]
        predictions[f"{scenario}__old_anomaly_probability"]=orig_probs[test_idx]
        predictions[f"{scenario}__candidate_anomaly_probability"]=candidate_score[test_idx]
        print("SCENARIO",json.dumps({
             "scenario":scenario,"candidate_gate_accepted":accepted,
             "selected_undeployed_arm":chosen,
             "gate_checks":checks,
             "audit":{k:{key:v["state"][key] for key in ("macro_f1","brier","nll")}
                for k,v in scen_report["audit_mission3"].items()}
           },ensure_ascii=False),flush=True)
    (out/"metrics.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    pd.DataFrame(rows).to_csv(out/"arm_metrics.csv",index=False)
    predictions.to_csv(out/"simulated_test_predictions.csv",index=False)
    create_charts(output,out)
    render_report(output,out)
    return output


def create_charts(result,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    label=["A original","B stale","C refit","D gated"]
    arms=list(result["scenarios"]["natural"]["audit_mission3"])
    for metric,filename,title in [
          ("brier","state_brier_by_update_arm.png",
           "Simulator state Brier: anomaly-model update and fusion interface"),
          ("macro_f1","state_macro_f1_by_update_arm.png",
           "Simulator three-state Macro-F1 for update/recalibration arms")]:
        fig,ax=plt.subplots(figsize=(11,6))
        x=np.arange(len(arms))
        width=.37
        for k,scenario in enumerate(("natural","logit_scale_stress")):
            vals=[result["scenarios"][scenario]["audit_mission3"][a]["state"][metric]
                  for a in arms]
            ax.bar(x+(k-.5)*width,vals,width,label=scenario)
        ax.set_xticks(x,label)
        ax.set_title(title);ax.set_ylabel(metric)
        ax.grid(axis="y",alpha=.25);ax.legend()
        fig.tight_layout()
        fig.savefig(out/filename,dpi=175)
        plt.close(fig)
    # Provider-agnostic probability drift magnitude per scenario and arm.
    fig,ax=plt.subplots(figsize=(10,5))
    ss=("natural","logit_scale_stress")
    ax.bar(np.arange(2)-0.15,[
       result["scenarios"][s]["probability_shifts"]["audit_mission3"]["stale"]["fused_state_JS_nats"]
       for s in ss],.3,label="stale fusion")
    ax.bar(np.arange(2)+0.15,[
       result["scenarios"][s]["probability_shifts"]["audit_mission3"]["refit"]["fused_state_JS_nats"]
       for s in ss],.3,label="refitted fusion")
    ax.set_xticks(np.arange(2),ss)
    ax.set_ylabel("Jensen-Shannon divergence (nats), baseline vs candidate")
    ax.set_title("Probability-interface drift across frozen and updated model components")
    ax.legend();ax.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(out/"fusion_probability_JS_drift.png",dpi=175)
    plt.close(fig)


def render_report(result,out):
    lines=[
      "# 人工监督更新后的跨分支概率兼容性：常规工程对照基线",
      "",
      "**重要：全部标签均来自冻结模拟器，没有真实操作员上传新标签，专利差异化算法尚未实现，模型没有发布。**",
      "",
      "本轮固定了既有两级健康状态分类器、MSF-MOFS 特征、HMM 转移参数；仅使用模拟新增二分类标注更新异常RF，再对比是否重训普通 LogisticRegression 融合器。",
      "",
      "数据分区：训练 missions0/1(544)；模拟新增标注 cycle8(68)；阈值校准cycle9(68)；两种融合器共同用cycle10(68)拟合；candidate验收cycle11(68)；最后mission3(272)观察。",
      "",
      "**重大限制：**冻结健康状态分类器的Platt校准与HMM状态先验最初使用了整个mission2，其中包含cycle11。"
      "故本轮验收只证明同基线下常规分支比较的工程流程，不构成全系统独立验证。",
      "",
      "两种场景：natural=真实发生的模拟监督RF重训；logit_scale_stress=对新RF输出进行人为固定logit缩放，用来测试接口数值变化的脆弱性，不是新卫星工况或新的有标签模型。",
      "",
      "| 场景 | 对照 | 三级Macro-F1 ↑ | 三级Brier ↓ | NLL ↓ | 模拟健康分MAE ↓ | 异常F1 | 异常AP | 异常Recall | 正常FPR |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    names={
      "A_legacy_detector_compatible_fusion":"A 旧异常分支+兼容融合",
      "B_updated_detector_stale_fusion":"B 更新异常分支+旧融合",
      "C_updated_detector_refit_fusion":"C 更新异常分支+常规重新校准融合",
      "D_staged_validation_selection":"D 门控后的候选（不发布）"
    }
    for scenario,rec in result["scenarios"].items():
        for key,v in rec["audit_mission3"].items():
            z=v["state"];a=v["anomaly"]
            lines.append(f"| {scenario} | {names[key]} | {z['macro_f1']:.4f} | {z['brier']:.5f} | "
              f"{z['nll']:.5f} | {v['health_score_MAE']:.3f} | {a['f1']:.4f} | "
              f"{a['ap']:.4f} | {a['recall']:.4f} | {a['fpr']:.4f} |")
    lines.extend([
      "",
      "## 版本兼容和常规验收结论",
      ""
    ])
    for scenario,rec in result["scenarios"].items():
        s=rec["probability_shifts"]["audit_mission3"]
        lines.extend([
            f"- {scenario}: 新旧异常概率平均绝对变化={s['stale']['anomaly_probability_mean_abs_change']:.4f}，"
            f"熵量化JS={s['stale']['anomaly_bernoulli_JS_nats']:.5f}；"
            f"旧融合输入/输出JS={s['stale']['fused_state_JS_nats']:.5f}，"
            f"重新拟合融合后输出JS={s['refit']['fused_state_JS_nats']:.5f}。"
            f"验证门控通过={rec['candidate_allowed_for_human_approval']}；"
            f"选择={names[rec['selected_undeployed_arm']]}。"
        ])
    lines+=["",
      "![新旧版本Brier比较](state_brier_by_update_arm.png)",
      "",
      "![Macro-F1比较](state_macro_f1_by_update_arm.png)",
      "",
      "![两分支输出分布变化](fusion_probability_JS_drift.png)",
      "",
      "更新异常检测器的概率接口不等于更新HMM状态转移概率；本次保持健康分支/HMM不变，普通融合器需要评估是否重校准。",
      "观察正向改进不代表专利创新：普通融合重训和验证门控都是标准工程方案。",
      "没有任何真实人工身份认证或真实卫星三级状态标签；即使候选验证通过也没有自动部署。",
      "",
      "仅保留一个真实卫星数据集（OPS-SAT）上的二分类证据与一个固定模拟全架构的三级证据；跨数据集迁移学习属于未来方向。",
      ""
    ]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--out",default="results/probability_compatibility")
    o=args.parse_args()
    run(o.out)
