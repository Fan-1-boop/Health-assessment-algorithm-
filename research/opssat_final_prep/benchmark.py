#!/usr/bin/env python3
"""One-OPS-SAT realistic label batch contamination test, non-novel baseline.

Uses TRUE raw OPS-SAT-AD telemetry (2578 retrospective features), fixed max
64-feature NSGA2/MSF budget, authentic pre-existing binary ground truth.
"New human labeled batch" is simulated using 319 historical OPS-SAT labels.
No real new operator signoff; original official test=0 EXCLUDED ENTIRELY.

The ordinary quality gate tests *only validation performance*. This file does
not implement confidential patent mechanisms, physical 3-class truth, or
autonomous production deployment.
"""
from __future__ import annotations
import argparse,hashlib,io,json,os,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import (average_precision_score,f1_score,recall_score,
                             precision_score,matthews_corrcoef)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from research.highdim_opssat import benchmark as highdim
from research.feature_budget.run_budget import selection_with_budget
from research.hitl_update.ablation import _split
from msf_mofs import fisher_scores

SPEC=ROOT/"research/opssat_final_prep/PROTOCOL_FROZEN.json"
PILOT=ROOT/"research/hitl_update/PILOT_FROZEN.json"
HIGH_DIM_SPEC=ROOT/"research/highdim_opssat/PROTOCOL_FROZEN.json"


def estimator():
    return make_pipeline(StandardScaler(),LogisticRegression(
        C=.5,class_weight="balanced",solver="liblinear",
        max_iter=350,random_state=42))


def choose_threshold(y,p):
    from sklearn.metrics import precision_recall_curve
    precision,recall,thresholds=precision_recall_curve(y,p)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(
            precision[:-1]+recall[:-1],1e-12)
    return float(thresholds[int(np.argmax(f1))])


def metrics(y,p,threshold):
    y=np.asarray(y,int);p=np.asarray(p,float)
    pred=p>=threshold
    return {"rows":int(len(y)), "anomalies":int(y.sum()),
      "f1":float(f1_score(y,pred)),
      "AP":float(average_precision_score(y,p)),
      "recall":float(recall_score(y,pred)),
      "precision":float(precision_score(y,pred,zero_division=0)),
      "FPR":float(np.mean(pred[y==0])),
      "MCC":float(matthews_corrcoef(y,pred)),
      "FP":int(np.sum(pred&(y==0))),
      "FN":int(np.sum((~pred)&(y==1))),
      "brier":float(np.mean((p-y)**2)),
      "threshold":float(threshold)}


def gate(a,b,config):
    tests={
      "f1":b["f1"] >=a["f1"]+config["min_f1_relative"],
      "AP":b["AP"]>=a["AP"]+config["min_AP_relative"],
      "recall":b["recall"]>=a["recall"]+config["min_recall_relative"],
      "FPR":b["FPR"]<=a["FPR"]+config["max_FPR_relative"]
    }
    return {k:bool(v) for k,v in tests.items()},bool(all(tests.values()))


def timer(f,reps):
    f();elapsed=[]
    for _ in range(reps):
        now=time.perf_counter()
        f()
        elapsed.append(1000*(time.perf_counter()-now))
    return {"median_ms":float(np.median(elapsed)),"q25_ms":float(np.quantile(elapsed,.25)),
            "q75_ms":float(np.quantile(elapsed,.75)),"repeats":int(reps)}


def plot(rows,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenario=list(dict.fromkeys(x["scenario"] for x in rows))
    arm=["A_old","B_direct_retrain","C_validation_gated"]
    for key,filename,desc in [
        ("f1","audit_F1_update_vs_noise.png","Anomaly F1 (audit)"),
        ("brier","audit_Brier_update_vs_noise.png","Anomaly probability Brier (audit)"),
        ("FPR","audit_FPR_update_vs_noise.png","False positive rate (audit)")]:
        fig,ax=plt.subplots(figsize=(10,5))
        x=np.arange(len(scenario));width=.25
        for i,method in enumerate(arm):
            values=[next(z[key] for z in rows if z["scenario"]==s and z["arm"]==method)
                    for s in scenario]
            ax.bar(x+(i-1)*width,values,width,label=method)
        ax.set_xticks(x,scenario,rotation=18,ha="right")
        ax.set_ylabel(desc);ax.set_title("OPS-SAT-AD: real binary labels, simulated upload corruption")
        ax.legend();ax.grid(axis="y",alpha=.2)
        fig.tight_layout();fig.savefig(out/filename,dpi=170);plt.close(fig)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    spec_raw=SPEC.read_bytes();spec=json.loads(spec_raw)
    proto=json.loads(PILOT.read_text())
    df_proto=json.loads(HIGH_DIM_SPEC.read_text())
    X,df,feature_names,source=highdim.load_and_engineer(out,df_proto)
    labels=df.anomaly.to_numpy(int)
    old,batch,val,audit=_split(df,int(proto["seed"]),proto["pool_split"])
    assert (len(old),len(batch),len(val),len(audit))==(876,319,239,160)
    official_test=np.flatnonzero(df.train.to_numpy(int)==0)
    union=set(old)|set(batch)|set(val)|set(audit)
    assert len(union)==1594 and union.isdisjoint(official_test)
    # Reproduce one fixed prior-declared feature BUDGET, but learn its actual
    # subset using ONLY the old historical rows of THIS experiment.
    start=time.perf_counter()
    fisher=np.nan_to_num(fisher_scores(X[old],labels[old]),nan=0.,posinf=0.)
    rank=np.argsort(-fisher,kind="stable")
    cfg=spec["selector"]
    sel,meta=selection_with_budget(X[old],labels[old],rank,
       cfg["budget"],seed=cfg["seed"],population=cfg["population"],
       generations=cfg["generations"],folds=cfg["inner_cv_folds"],
       tolerance=cfg["tolerance"])
    feature_selection_ms=1000*(time.perf_counter()-start)
    assert 32<=len(sel)<=64 and len(np.unique(sel))==len(sel)
    x_old=X[old][:,sel];x_val=X[val][:,sel];x_audit=X[audit][:,sel]
    def fit_old():return estimator().fit(x_old,labels[old])
    t_old=timer(fit_old,spec["timing"]["training_repeats"])
    old_model=fit_old()
    old_val=old_model.predict_proba(x_val)[:,1]
    old_t=choose_threshold(labels[val],old_val)
    old_cal=metrics(labels[val],old_val,old_t)
    old_audit=metrics(labels[audit],
                     old_model.predict_proba(x_audit)[:,1],old_t)
    scores_new=old_model.predict_proba(X[batch][:,sel])[:,1]
    scenarios={}
    all_rows=[]
    for condition in spec["scenarios"]:
        name=condition["name"];rate=condition["noise"]
        rng=np.random.default_rng(int(spec["random_noise_seed"]))
        changed=np.zeros(len(batch),bool)
        if rate==1:changed[:]=True
        elif rate>0:
            choose=rng.choice(len(batch),size=round(rate*len(batch)),replace=False)
            changed[choose]=True
        submitted=labels[batch].copy()
        submitted[changed]=1-submitted[changed]
        candidate_idx=np.r_[old,batch]
        candidate_labels=np.r_[labels[old],submitted]
        def fit_candidate():
            return estimator().fit(X[candidate_idx][:,sel],candidate_labels)
        timing=timer(fit_candidate,spec["timing"]["training_repeats"])
        model=fit_candidate()
        candidate_val=model.predict_proba(x_val)[:,1]
        new_t=choose_threshold(labels[val],candidate_val)
        new_cal=metrics(labels[val],candidate_val,new_t)
        checks,approved=gate(old_cal,new_cal,spec["gate"])
        can_audit=metrics(labels[audit],
                          model.predict_proba(x_audit)[:,1],new_t)
        stage_audit=can_audit if approved else old_audit
        ref_disagreement={
            "old_threshold_label_disagreement_rate":float(np.mean((scores_new>=old_t)!=submitted)),
            "old_score_confident_disagreement_rate":float(np.mean(
               ((scores_new<.2)&(submitted==1))|((scores_new>.8)&(submitted==0)))),
            "old_score_confident_agreement_rate":float(np.mean(
               ((scores_new>.8)&(submitted==1))|((scores_new<.2)&(submitted==0)))),
            "note":"Old-model confidence disagreements are only a warning: legitimate new regimes can break old predictions, no authenticity inference"
        }
        scenarios[name]={
          "original_expert_labels_flipped_only_for_fault_injection":int(changed.sum()),
          "batch_simulated_review_not_real_operator":True,
          "original_old_model_held_constant":True,
          "candidate_fit":timing,
          "validation":{"A":old_cal,"B":new_cal},
          "acceptance_checks":checks,"candidate_passed":approved,
          "staged_selection":"B" if approved else "A",
          "audit":{"A":old_audit,"B":can_audit,"C":stage_audit},
          "model_disagreement_diagnostics":ref_disagreement,
        }
        for a,m in [("A_old",old_audit),("B_direct_retrain",can_audit),
                    ("C_validation_gated",stage_audit)]:
            all_rows.append({"scenario":name,"arm":a,"simulated_label_flip_count":int(changed.sum()),
                    "gate_accepted":approved,"staged_choice":("B" if approved else "A"),
                    **m,"feature_count":int(len(sel)),
                    "model_fit_ms":timing["median_ms"] if a!="A_old" else t_old["median_ms"]})
        print("SCENARIO",json.dumps({
         "name":name,"flip_count":int(changed.sum()),
         "gate":approved,"A_AUDIT_F1":old_audit["f1"],
         "B_AUDIT_F1":can_audit["f1"],"C_AUDIT_F1":stage_audit["f1"],
         "A_Brier":old_audit["brier"],"B_Brier":can_audit["brier"],
         "C_Brier":stage_audit["brier"],
         "confident_label_disagreement":ref_disagreement["old_score_confident_disagreement_rate"]
        },ensure_ascii=False),flush=True)
    report={
      "study":spec["version"],
      "data_scope":"TRUE OPS-SAT observations and existing expert normal/anomaly labels. Simulated upload, not newly human verified.",
      "prototype_only":True,"production_deployed":False,
      "protocol_sha256":hashlib.sha256(spec_raw).hexdigest(),
      "counts":{"old":len(old),"repackaged_update":len(batch),
                "validation":len(val),"internal_audit":len(audit),
                "official_test_excluded":len(official_test)},
      "original_real_data_source":source,
      "selector":{"fixed_max_budget":64,"actual_selected_count":len(sel),
                  "feature_indices":[int(i) for i in sel],
                  "selected_feature_names":[feature_names[i] for i in sel],
                  "search_ms":feature_selection_ms,"meta":meta},
      "classifier_old":{"fit_timing":t_old,"validation":old_cal,"audit":old_audit},
      "scenarios":scenarios,
      "caveats":[
        "This is an exploratory real-satellite binary anomaly study, with original test=0 completely excluded",
        "Existing real expert labels were rearranged into pseudo-upload; no newly human-confirmed data or actual deployment",
        "Scenarios flip original labels deliberately only for engineering robustness tests",
        "High-dimensional budget64 selected after previous experiments, not an independently optimal budget claim",
        "Feature-selection mask is recomputed on old876 because prior study's mask was learned with a different training split; count may differ from earlier 37",
        "Selected features describe complete segments, not true real-time early detection",
        "No trusted physically verified 3-class states/HMM on OPS-SAT",
        "No patent-specific algorithm implemented; only conventional calibration gate",
        "Rejecting a real honest new operating condition merely due to old-model prediction disagreement is unsafe",
        "Internal split may contain correlated segments of one channel/operation; do not assume source-independent temporal generalization"
      ]
    }
    (out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(out/"audit_metrics.csv",index=False)
    plot(all_rows,out)
    md=[
     "# 真实 OPS-SAT-AD：固定 MSF 最大64特征预算的有标签更新稳健性预验收",
     "",
     "**真实来源是卫星遥测与已经存在的正常/异常二分类标签；“上传的人工审核标签”是把已有标签模拟包装。没有真实操作员重新审核。**",
     "",
     f"内部分区：历史训练 {len(old)}、模拟上传 {len(batch)}、阈值/验收 {len(val)}、内部审计 {len(audit)}；原始官方测试 {len(official_test)} 全程未用于此轮。",
     f"高维2578特征，MSF固定预算上限64；本次仅由历史训练集选取 **{len(sel)}维**，一次性搜索耗时{feature_selection_ms:.1f}ms。",
     "",
     "| 模拟更新标签 | 注入反转条数 | 常规验证门控通过 | A 不更新 F1 | B 普通重训 F1 | C 暂存后 F1 | C 审计 Brier |",
     "|---|---:|---|---:|---:|---:|---:|"]
    for name,r in scenarios.items():
        md.append(f"| {name} | {r['original_expert_labels_flipped_only_for_fault_injection']} | {r['candidate_passed']} | "
         f"{r['audit']['A']['f1']:.4f} | {r['audit']['B']['f1']:.4f} | "
         f"{r['audit']['C']['f1']:.4f} | {r['audit']['C']['brier']:.4f} |")
    md+=["",
      "![真实卫星二分类 F1 与标签注入](audit_F1_update_vs_noise.png)",
      "",
      "![真实卫星 Brier 与标签注入](audit_Brier_update_vs_noise.png)",
      "",
      "![真实卫星正常误报率](audit_FPR_update_vs_noise.png)",
      "",
      "旧模型高置信与上传标签不一致**不能直接证明标签错误**，因为它也可能是合法的工况变化；只能用于后续人工复核。",
      "所有模型/标签门控输出都不会自动部署，专利差异化机制仍待私有验证。",
      ""]
    (out/"RESULTS.md").write_text("\n".join(md),encoding="utf-8")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/opssat_final_prep")
    a=p.parse_args()
    run(a.out)
