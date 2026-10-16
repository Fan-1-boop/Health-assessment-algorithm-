#!/usr/bin/env python3
"""Public baseline experiment: A(no update), B(supervised retrain), C(quality gate).

This is a *non-novel* engineering reference. OPS-SAT-AD already-labeled
segments are repackaged as a SIMULATED human upload; no real operator
re-reviewed them here, no production deployment is allowed.

Only original OPS-SAT official train=1 rows are used, including for audit.
Do not use the already published original official test=0 for this pilot.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gate import canonical_csv, validate_supervised_upload

def _split(df,seed,details):
    """Disjoint labeled partitions from the original official training rows."""
    eligible=np.flatnonzero(df.train.to_numpy(int)==1)
    trainpool,reserved=train_test_split(eligible,
         test_size=details["reserved_fraction"],random_state=seed,
         stratify=df.anomaly.to_numpy(int)[eligible])
    old,batch=train_test_split(trainpool,
         test_size=details["candidate_batch_fraction_of_training_pool"],
         random_state=seed+1,stratify=df.anomaly.to_numpy(int)[trainpool])
    validation,audit=train_test_split(reserved,
         test_size=details["audit_fraction_of_reserved"],
         random_state=seed+2,stratify=df.anomaly.to_numpy(int)[reserved])
    return tuple(np.sort(z) for z in (old,batch,validation,audit))


def _estimator():
    return HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=11,
           learning_rate=.055,l2_regularization=.4,random_state=42)

def _imputer_fit(x):
    v=np.nanmedian(np.where(np.isfinite(x),x,np.nan),axis=0)
    return np.nan_to_num(v,nan=0.0,posinf=0.0,neginf=0.0)

def _imputer_apply(x,median):
    return np.where(np.isfinite(x),x,median[None,:])

def _training(df,fit_idx,features):
    x=df[list(features)].to_numpy(float)
    y=df.anomaly.to_numpy(int)
    impute=_imputer_fit(x[fit_idx])
    estimator=_estimator()
    estimator.fit(_imputer_apply(x[fit_idx],impute),y[fit_idx])
    return estimator,impute

def _score(df,idx,features,model):
    estimator,impute=model
    raw=df[list(features)].to_numpy(float)[idx]
    return estimator.predict_proba(_imputer_apply(raw,impute))[:,1]


def _validation_threshold(y,score):
    pre,rec,threshold=precision_recall_curve(y,score)
    best=2*pre[:-1]*rec[:-1]/np.maximum(pre[:-1]+rec[:-1],1e-12)
    return float(threshold[np.flatnonzero(best>=best.max()-1e-12)[-1]])


def _metrics(y,s,threshold):
    pred=np.asarray(s)>=threshold
    y=np.asarray(y,int)
    return {
       "samples":int(len(y)),"positives":int(y.sum()),
       "f1":float(f1_score(y,pred,zero_division=0)),
       "precision":float(precision_score(y,pred,zero_division=0)),
       "recall":float(recall_score(y,pred,zero_division=0)),
       "accuracy":float(accuracy_score(y,pred)),
       "ap":float(average_precision_score(y,s)),
       "roc_auc":float(roc_auc_score(y,s)),
       "normal_fpr":float(np.mean(pred[y==0])) if np.any(y==0) else None,
       "false_positive":int(np.sum(pred&(y==0))),
       "false_negative":int(np.sum((~pred)&(y==1))),
       "threshold":float(threshold)}


def run(out_path):
    out=Path(out_path);out.mkdir(parents=True,exist_ok=True)
    specfile=ROOT/"research/hitl_update/PILOT_FROZEN.json"
    config=json.loads(specfile.read_text())
    feature_config=json.loads((ROOT/"real_mvp/PROTOCOL_FROZEN.json").read_text())
    features=feature_config["features"]
    source=ROOT/config["source_relative_path"]
    df=pd.read_csv(source)
    if len(df)!=2123 or df.segment.duplicated().any():
        raise ValueError("Real OPS-SAT source schema changed; blocked")
    seed=int(config["seed"])
    old,batch,val,audit=_split(df,seed,config["pool_split"])
    y=df.anomaly.to_numpy(int)
    original_test=np.flatnonzero(df.train.to_numpy(int)==0)
    assert len(set(old)|set(batch)|set(val)|set(audit))==int((df.train==1).sum())
    assert not (set(old)|set(batch)|set(val)|set(audit))&set(original_test)
    samples=set(map(str,df.segment.to_numpy()[val]))
    samples.update(map(str,df.segment.to_numpy()[audit]))
    samples.update(map(str,df.segment.to_numpy()[original_test]))
    columns=["sample_id","label",*features]
    uploaded=df.loc[batch,["segment","anomaly",*features]].copy()
    uploaded.columns=columns
    if uploaded.sample_id.duplicated().any():raise AssertionError("Duplicate upload ID")
    verified_checksum=hashlib.sha256(canonical_csv(uploaded)).hexdigest()
    # Source labels are historical expert-informed annotations. The new
    # operator review & manual upload is SIMULATED, not represented as genuine.
    manifest={
        "batch_id":"DEMO_OPSSAT_BATCH_001",
        "target_task":"anomaly_binary",
        "label_source":"official_public_labels_simulated_operator_upload",
        "review_status":"simulated_review",
        "reviewer_id":"DEMO_REVIEWER_NOT_AUTHENTICATED",
        "reviewed_at":"2026-10-16T00:00:00Z",
        "parent_model_version":"real-opssat-mvp-v1",
        "data_sha256":verified_checksum,
        "approval_status":"pending",
        "data_unit":"annotated_segment",
        "demo_only":True
    }
    # A: historical-only reference.
    old_model=_training(df,old,features)
    sv_old=_score(df,val,features,old_model)
    t_old=_validation_threshold(y[val],sv_old)
    v_old=_metrics(y[val],sv_old,t_old)
    # B: ordinary supervised retrain on *exact* old+new samples.
    combined=np.sort(np.r_[old,batch])
    candidate=_training(df,combined,features)
    sv_new=_score(df,val,features,candidate)
    t_new=_validation_threshold(y[val],sv_new)
    v_new=_metrics(y[val],sv_new,t_new)
    # C: task-specific verified upload -> same B model with conventional
    # validation-only regression checks. Candidate only, NEVER deploy.
    check=validate_supervised_upload(uploaded,manifest,features,
           blocked_sample_ids=samples,
           expected_parent_model="real-opssat-mvp-v1",
           allow_simulated_review=True,
           max_missing_fraction=.02)
    assert check.route=="anomaly_detector_candidate" and not check.release_allowed
    rules=config["c_acceptance"]
    checks={
        "validation_f1":bool(v_new["f1"]>=v_old["f1"]-rules["max_f1_regression"]),
        "validation_recall":bool(v_new["recall"]>=v_old["recall"]-rules["max_recall_regression"]),
        "validation_nominal_fpr":bool(
            v_new["normal_fpr"]<=v_old["normal_fpr"]+rules["max_normal_false_alarm_rate_increase"])
    }
    accepted_for_staging=bool(all(checks.values()))
    cmodel,c_threshold=(candidate,t_new) if accepted_for_staging else (old_model,t_old)
    # Audit is untouched during fit/selection/threshold/acceptance.
    va={"A_frozen":v_old,"B_ordinary_retrain":v_new,
         "C_controlled_candidate":v_new if accepted_for_staging else v_old}
    models={"A_frozen":(old_model,t_old),
            "B_ordinary_retrain":(candidate,t_new),
            "C_controlled_candidate":(cmodel,c_threshold)}
    audit_metric={}
    for arm,(m,t) in models.items():
        audit_metric[arm]=_metrics(y[audit],_score(df,audit,features,m),t)
    report={
       "experiment":"PUBLIC PILOT / NOT NOVEL IP / NOT PRODUCTION",
       "dataset":"Real OPS-SAT-AD official preannotated 18-feature satellite segments",
       "important_caveat":"Update review/upload is SIMULATED. No newly supplied authentic human labels, no real production approval.",
       "split_source":"Original official train=1 only. Historical official test=0 completely untouched.",
       "counts":{"old":len(old),"batch":len(batch),"validation":len(val),"audit":len(audit),
                 "official_test_excluded":len(original_test)},
       "ids_sha256":{name:hashlib.sha256(",".join(map(str,df.segment.to_numpy()[idx])).encode()).hexdigest()
                     for name,idx in [("old",old),("batch",batch),("validation",val),("audit",audit)]},
       "upload_review":{"digest":check.digest,"task":manifest["target_task"],
                        "route":check.route,"demo_only":check.demo_only,
                        "candidate_allowed":check.candidate_allowed,
                        "release_allowed":check.release_allowed,
                        "rows":check.rows,"anomaly_labels":check.positive,
                        "missing_fraction":check.incomplete_fraction,
                        "warnings":check.issues},
       "validation_comparison":va,
       "C_ordinary_acceptance_checks":checks,
       "C_candidate_accepted_for_staging":accepted_for_staging,
       "audit_no_tuning":audit_metric,
       "new_model_deployed":False,
       "interpretation":"C chooses B or A; C cannot intrinsically outperform both on every held-out set. This pilot tests gating and rollback, not patent-level algorithmic novelty.",
       "config_sha256":hashlib.sha256(specfile.read_bytes()).hexdigest(),
       "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
       "limitations":[
          "No new operator-labeled satellite samples: archived labels repackaged in simulated reviewed batch",
          "No actual satellite mode-change episode was supplied, so mode-vs-fault distinction is not established",
          "OPS-SAT-AD labels are binary anomaly, no real physical health/degradation/fault states",
          "C uses ordinary staged validation/rollback rules only; not a novel mathematical mechanism",
          "No autonomous learning, live model replacement, or release approval",
          "No original official test=0 examples were used in the pilot"]
    }
    (out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=int)+"\n")
    (out/"simulated_upload_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    artifact={"candidate_model":candidate[0],"candidate_imputer":candidate[1],
       "candidate_threshold":t_new,"feature_names":features,
       "candidate_stage_only":True,"production_approved":False,
       "source_batch_digest":check.digest}
    joblib.dump(artifact,out/"candidate_only.joblib",compress=3)
    (out/"RESULTS.md").write_text(
        "# 有监督受控更新 A/B/C 基线（研究试验）\n\n"
        "**这不是新发明的实验验证，也没有真实人工新增标注。** "
        "原始 OPS-SAT 数据集既有专家参与标注；本次将其中一批封装为模拟人工上传，只在测试环境允许模拟审核，生产接口默认拒绝。\n\n"
        f"数据划分（均为官方训练段）：旧训练={len(old)}，新批次={len(batch)}，验证={len(val)}，最终审计={len(audit)}。"
        f"原始官方测试 {len(original_test)} 段本次完全未使用。\n\n"
        "| 对照 | 验证 F1 | 审计 F1 | 审计 Recall | 审计正常误报率 |\n"
        "|---|---:|---:|---:|---:|\n"
        +"\n".join(f"| {k} | {va[k]['f1']:.4f} | {audit_metric[k]['f1']:.4f} | "
                    f"{audit_metric[k]['recall']:.4f} | {audit_metric[k]['normal_fpr']:.4f} |"
                    for k in models)
        +"\n\n"
        f"候选门控条件：{checks}；是否允许候选进入人工审批阶段：{accepted_for_staging}；"
        "**生产发布：未批准，未进行。**\n\n"
        "标签验证只针对真实正常/异常二分类；不能转换为物理健康/退化/故障。"
        "C 只是在 A 与 B 之间做常规的验证集判断，不代表已经形成专利创新。\n")
    print(json.dumps({"counts":report["counts"],"gate":checks,
       "stage_candidate":accepted_for_staging,
       "validation_F1":{k:round(v["f1"],5) for k,v in va.items()},
       "audit_F1":{k:round(v["f1"],5) for k,v in audit_metric.items()},
       "production_release":False},indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/hitl_public_pilot")
    args=p.parse_args()
    run(args.out)
