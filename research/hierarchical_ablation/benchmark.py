#!/usr/bin/env python3
"""Comparative simulation ablation: frozen two-stage vs direct 3-way classifier.

All state labels here come from mvp/synthetic.py, NOT satellite hardware.
Mission 3 is NEVER used for fitting, temperature scaling, or hyperparameter
choices. Saved delivery artifact is loaded read-only. The original 64 max
feature budget belongs to a different real OPS-SAT binary anomaly benchmark.
"""
from __future__ import annotations
import argparse,hashlib,io,json,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp,softmax
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"mvp"))
sys.path.insert(0,str(ROOT/"src"))
from synthetic import generate,make_windows,highdim_features
from v13_state import (fit_hierarchical,predict_hierarchical,filter_groups,evaluation)
from research.state_hmm_ablation.run import true_transition_delays,timer

SPEC=ROOT/"research/hierarchical_ablation/PROTOCOL_FROZEN.json"
FROZEN=ROOT/"results/mvp"


def make_direct():
    # Direct multinomial model, unlike hierarchical pair of binary logits.
    return make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=700,C=.5,class_weight="balanced",
             solver="lbfgs",random_state=42))


def temperature_calibration(prob,y):
    """Calibrate a *single* scalar on independent simulation mission 2."""
    y=np.asarray(y,int)
    prob=np.asarray(prob,float)
    if prob.shape!=(len(y),3) or set(np.unique(y))!={0,1,2}:
        raise ValueError("Need 3-class supervised calibration data")
    z=np.log(np.clip(prob,1e-12,1.))
    def loss(logt):
        logits=z/np.exp(logt)
        return float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y]))
    fit=minimize_scalar(loss,bounds=(np.log(.2),np.log(8.)),
                         method="bounded",options={"xatol":1e-7})
    if not fit.success or not np.isfinite(fit.x):raise RuntimeError("Temperature fit failed")
    return float(np.exp(fit.x))


def apply_temperature(prob,temp):
    if temp<=0:raise ValueError("Temperature must be positive")
    p=np.asarray(prob,float)
    z=np.log(np.clip(p,1e-12,1.))/temp
    return softmax(z,axis=1)


def quality(y,p,groups,times):
    q=evaluation(y,p)
    pred=np.asarray(p).argmax(axis=1)
    q.update({
       "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
       "confusion_matrix":confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
       "class_recall_names":["healthy","degraded","fault"],
       "transition":true_transition_delays(y,pred,groups,times)
    })
    return q


def pickle_bytes(model):
    buf=io.BytesIO()
    joblib.dump(model,buf,compress=3)
    return len(buf.getvalue())


def plot(report,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series=list(report["methods"])
    plotted=[report["methods"][x] for x in series]
    fig,ax=plt.subplots(figsize=(10,6))
    for i,(name,q) in enumerate(zip(series,plotted)):
        ms=report["inference_timing"][name]["median_ms"]
        ax.scatter(ms,q["brier"],s=90)
        ax.annotate(name,(ms,q["brier"]),xytext=(5,7 if i%2 else -16),
                    textcoords="offset points",fontsize=8)
    ax.set_xlabel("Inference time / 272 simulated windows (ms; lower better)")
    ax.set_ylabel("3-class Brier (lower better)")
    ax.set_title("State classifier structure and HMM: inference cost vs calibrated probability error")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/"inference_cost_vs_brier.png",dpi=170)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,6))
    for i,(name,q) in enumerate(zip(series,plotted)):
        ms=report["inference_timing"][name]["median_ms"]
        ax.scatter(ms,q["macro_f1"],s=90)
        ax.annotate(name,(ms,q["macro_f1"]),xytext=(5,7 if i%2 else -16),
                    textcoords="offset points",fontsize=8)
    ax.set_xlabel("Inference time / 272 simulated windows (ms; lower better)")
    ax.set_ylabel("3-state Macro-F1 (higher better)")
    ax.set_title("State classifier structure and HMM: cost vs multiclass identification")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/"inference_cost_vs_macro_f1.png",dpi=170)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,6))
    labels=list(report["methods"])
    positions=np.arange(len(labels))
    for i,(c,name) in enumerate([(0,"Healthy"),(1,"Degraded"),(2,"Fault")]):
        ax.bar(positions+(i-1)*.24,[report["methods"][key]["recalls"][c] for key in labels],
               width=.23,label=name)
    ax.set_xticks(positions,labels,rotation=20,ha="right")
    ax.set_ylim(0,1.1)
    ax.set_ylabel("Recall on SIMULATED mission 3")
    ax.set_title("Healthy / Degraded / Fault recall: direct vs hierarchical")
    ax.grid(axis="y",alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"per_state_recall.png",dpi=170)
    plt.close(fig)


def run(out_path):
    out=Path(out_path)
    out.mkdir(parents=True,exist_ok=True)
    spec_raw=SPEC.read_bytes()
    spec=json.loads(spec_raw)
    original=json.loads((FROZEN/"metrics.json").read_text())
    frozen=joblib.load(FROZEN/"model_bundle.joblib") # trusted project-generated file
    if frozen["spec_sha256"]!=original["protocol_sha256"]:
        raise AssertionError("Delivery model does not match frozen metrics")
    windows,meta=make_windows(generate())
    X,names=highdim_features(windows)
    assert names==frozen["feature_names"]
    missions=meta.mission_id.to_numpy(int)
    fit=missions<=1;cal=missions==2;test=missions==3
    y=meta.health_state_simulated.to_numpy(int)
    groups=meta.cycle_id.to_numpy(int)
    timeline=meta.sample_in_mission.to_numpy(int)
    assert (fit.sum(),cal.sum(),test.sum())==(
         spec["data"]["expected_train_windows"],
         spec["data"]["expected_cal_windows"],
         spec["data"]["expected_test_windows"])
    assert set(np.unique(y[fit]))==set(np.unique(y[cal]))==set(np.unique(y[test]))=={0,1,2}
    h=frozen["hierarchical"]
    idx1=np.asarray(h.features_stage1,int)
    idx2=np.asarray(h.features_stage2,int)
    union=np.union1d(idx1,idx2)
    assert len(idx1)==50 and len(idx2)==10 and len(union)==59

    # Original trained reference, 100% reloaded; never refit a new version of
    # the delivery model for any reported original score.
    def h_prob():return predict_hierarchical(h,X[test])
    ph=h_prob()
    historic=evaluation(y[test],ph)
    for metric in ("macro_f1","brier","nll"):
        if not np.isclose(historic[metric],original["base_static_state_metrics"][metric],atol=1e-10):
            raise AssertionError("Frozen reference metric changed")

    trained_direct={}
    direct_temps={}
    direct_probs={}
    fit_time={}
    repeat=spec["timing"]["repeats"]

    def hierarchical_refit():
        return fit_hierarchical(X[fit],y[fit],groups[fit],X[cal],y[cal],idx1,idx2)

    fit_time["two_stage_retrain_without_feature_search"]=timer(hierarchical_refit,repeat)
    check_refit=hierarchical_refit()
    check_prob=predict_hierarchical(check_refit,X[test])
    # Training deterministic; if differs record transparent mismatch.
    refit_max_diff=float(np.max(np.abs(check_prob-ph)))

    for key,cols in (
        ("direct_union59",union),
        ("direct_all400",np.arange(X.shape[1]))
    ):
        xfit=X[fit][:,cols]
        xcal=X[cal][:,cols]
        xtest=X[test][:,cols]
        def fit_direct():
            mdl=make_direct().fit(xfit,y[fit])
            temp=temperature_calibration(mdl.predict_proba(xcal),y[cal])
            return mdl,temp
        fit_time[key+"_train_and_calibrate"]=timer(fit_direct,repeat)
        mdl,temp=fit_direct()
        if list(mdl.classes_)!=[0,1,2]:
            raise AssertionError("Unexpected direct classes")
        predicted=apply_temperature(mdl.predict_proba(xtest),temp)
        trained_direct[key]=(mdl,cols)
        direct_temps[key]=temp
        direct_probs[key]=predicted

    T_groups=groups[test]
    def posthmm(p):
        # All models share the SAME transition matrix / prior estimated from
        # fit & calibration mission, and exact same gamma / memory. Causal
        # sequence resets at independently generated cycle boundaries.
        return filter_groups(h,p,T_groups,gamma=.6,memory=.5)

    probs={
       "two_stage_static":ph,
       "direct_union59":direct_probs["direct_union59"],
       "direct_all400":direct_probs["direct_all400"],
       "two_stage_hmm":posthmm(ph),
       "direct_union59_hmm":posthmm(direct_probs["direct_union59"]),
       "direct_all400_hmm":posthmm(direct_probs["direct_all400"])
    }
    gt=y[test]
    for name,p in probs.items():
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-9):
            raise AssertionError("Invalid three-state probabilities "+name)
    scores={key:quality(gt,p,T_groups,timeline[test]) for key,p in probs.items()}
    if not np.isclose(scores["two_stage_hmm"]["macro_f1"],
                      original["base_hmm_state_metrics"]["macro_f1"],atol=1e-12):
        raise AssertionError("Frozen HMM reference changed")

    prediction_functions={
      "two_stage_static":h_prob,
      "direct_union59":lambda:apply_temperature(
          trained_direct["direct_union59"][0].predict_proba(
             X[test][:,union]),direct_temps["direct_union59"]),
      "direct_all400":lambda:apply_temperature(
          trained_direct["direct_all400"][0].predict_proba(
             X[test]),direct_temps["direct_all400"]),
    }
    prediction_functions.update({
      key+"_hmm":(lambda orig=key:posthmm(prediction_functions[orig]()))
      for key in ("two_stage_static","direct_union59","direct_all400")
    })
    # Name alias for frozen pipeline from loop above.
    prediction_functions["two_stage_hmm"]=prediction_functions.pop("two_stage_static_hmm")
    inference_timing={key:timer(fn,repeat) for key,fn in prediction_functions.items()}

    size={
      "two_stage_classifier_calibrators_and_HMM":pickle_bytes(h),
      "direct_union59_classifier_and_temperature":pickle_bytes({
          "model":trained_direct["direct_union59"][0],
          "temperature":direct_temps["direct_union59"],
          "feature_indices":union}),
      "direct_all400_classifier_and_temperature":pickle_bytes({
          "model":trained_direct["direct_all400"][0],
          "temperature":direct_temps["direct_all400"],
          "feature_indices":np.arange(X.shape[1])}),
    }
    report={
      "study":"Frozen synthetic satellite hierarchical classifier vs direct multinomial comparison",
      "source_truth":"SIMULATED 3-state healthy/degraded/fault, not real satellite diagnosis",
      "exploratory_not_fresh_confirmation":True,
      "protocol_sha256":hashlib.sha256(spec_raw).hexdigest(),
      "frozen_mvp_spec_sha256":frozen["spec_sha256"],
      "training_rows":int(fit.sum()),"calibration_rows":int(cal.sum()),
      "independent_mission3_test_rows":int(test.sum()),
      "feature_setup":{"stage1":len(idx1),"stage2":len(idx2),
         "task_specific_union":len(union),"all_dimensions":X.shape[1],
         "direct_union_uses_supervised_hierarchical_selected_features":True,
         "real_opssat_64_budget_not_applied_here":True},
      "temperature_calibration":direct_temps,
      "retrained_hierarchy_max_probability_difference_from_frozen":refit_max_diff,
      "methods":scores,
      "model_fit_timings_including_prob_calibration_excluding_feature_search":fit_time,
      "inference_timing":inference_timing,
      "compressed_model_sizes_bytes":size,
      "methodological_limits":[
       "Only synthetic state truth from frozen simulator; no real physical satellite faults",
       "Mission 3 outcomes not used for model fitting or temperature calibration",
       "Union59 direct baseline shares class-supervised features originally chosen for the two binary tasks and is not an independently optimized feature selection",
       "Direct multiclass is multinomial lbfgs, hierarchical uses separate liblinear binary models; regularization and class weighting similar but not perfectly identical optimization",
       "Both direct models use scalar temperature scaling fitted on the same simulation mission2 that hierarchical uses for its two Platt calibrators",
       "Fixed HMM state transition/prior shared across all models, not fit on mission3",
       "Feature selection search cost excluded from training; full feature extraction excluded from inference timing",
       "Prior frozen synthetic test has been inspected; this is descriptive exploratory ablation, not a new unbiased data-independent model comparison",
       "No claims of statistical significance from one synthetic test mission",
       "No patent-level mechanism or invention proven by familiar hierarchy/HMM model structures"]
    }
    (out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    rows=[]
    for name,q in scores.items():
        timing=inference_timing[name]
        t=q["transition"]
        rows.append({
          "method":name,"features_used":59 if "union59" in name or "two_stage" in name else 400,
          "macro_f1":q["macro_f1"],"brier":q["brier"],"nll":q["nll"],
          "balanced_accuracy":q["balanced_accuracy"],
          "healthy_recall":q["recalls"][0],
          "degraded_recall":q["recalls"][1],"fault_recall":q["recalls"][2],
          "transitions":t["true_change_events"],
          "transitions_detected":t["transition_detected"],
          "transitions_missed":t["transition_missed"],
          "median_detected_transition_delay_min":t["median_detected_delay_minutes"],
          "spurious_state_switches":t["spurious_prediction_state_switches"],
          "predict_272_windows_median_ms":timing["median_ms"]})
    pd.DataFrame(rows).to_csv(out/"comparison.csv",index=False)
    df=pd.DataFrame({
      "mission":missions[test],
      "cycle":T_groups,
      "minute_in_mission":timeline[test],
      "simulated_state_label":gt
    })
    for key,p in probs.items():
        df[key+"_prediction"]=p.argmax(axis=1)
        for i,state in enumerate(("healthy","degraded","fault")):
            df[key+"_p_"+state]=p[:,i]
    df.to_csv(out/"test_window_predictions.csv",index=False)
    plot(report,out)

    chinese={
     "two_stage_static":"两级静态",
     "direct_union59":"直接三分类（59维）",
     "direct_all400":"直接三分类（400维）",
     "two_stage_hmm":"两级 + HMM",
     "direct_union59_hmm":"直接59维 + HMM",
     "direct_all400_hmm":"直接400维 + HMM"}
    lines=[
      "# 两级健康状态分类 vs 直接三级分类：模拟数据多目标消融",
      "",
      "**声明：此项研究使用同一冻结的全模拟卫星数据。测试任务 272 个窗口来自模拟 mission3，绝非真实物理卫星故障标签。**",
      "",
      "直接分类器在模拟 missions0,1 拟合，在 mission2 通过温度缩放进行概率校准。"
      "两级分类器来自冻结 MVP，不重新选特征。三个静态模型均接入相同的已经训练好的 HMM 转移矩阵和先验，单独比较 HMM 增益。",
      "",
      "| 方法 | 特征数 | Macro-F1 | Brier ↓ | NLL ↓ | 健康召回 | 退化召回 | 故障召回 | 272窗推断ms |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for row in rows:
        lines.append(
         f"| {chinese[row['method']]} | {row['features_used']} | {row['macro_f1']:.4f} | "
         f"{row['brier']:.4f} | {row['nll']:.4f} | "
         f"{row['healthy_recall']:.4f} | {row['degraded_recall']:.4f} | "
         f"{row['fault_recall']:.4f} | {row['predict_272_windows_median_ms']:.3f} |")
    lines+=["",
      "## 模型拟合/校准代价",
      "",
      "| 模型 | 中位拟合+概率校准时间 (ms) |",
      "|---|---:|"]
    for k,v in fit_time.items():lines.append(f"| {k} | {v['median_ms']:.2f} |")
    lines+=["",
      "训练计时不包括事先完成的特征选择搜索；推断计时不包含原始遥测特征生成。",
      "",
      "![推断成本 vs Brier](inference_cost_vs_brier.png)",
      "",
      "![推断成本 vs Macro-F1](inference_cost_vs_macro_f1.png)",
      "",
      "![三级状态召回对比](per_state_recall.png)",
      "",
      "结论应结合各类别召回、概率 Brier/NLL、训练与推断成本，不以单一 Macro-F1 判定层级结构必然更优。"
      "本研究先前已经见过相同模拟任务的测试结果，因而属于事后探索性消融。"
    ]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({
       "rows":[{k:r[k] for k in ("method","macro_f1","brier","nll",
                "healthy_recall","degraded_recall","fault_recall",
                "predict_272_windows_median_ms")} for r in rows],
       "train_ms":{k:v["median_ms"] for k,v in fit_time.items()},
       "size_bytes":size,"refit_maxdiff":refit_max_diff
       },ensure_ascii=False,indent=2))
    return report


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--out",default="results/hierarchical_ablation")
    opts=cli.parse_args()
    run(opts.out)
