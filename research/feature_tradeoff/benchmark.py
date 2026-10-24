#!/usr/bin/env python3
"""Single-dataset OPS-SAT feature-selection efficiency Pareto diagnostic.

Important: Original official test=0 never participates. Uses new disjoint
train/calibration/audit subsets of the original train=1. Descriptive research,
NOT a replacement for the frozen OPS-SAT delivery result.

We time SELECT, FIT, and PREDICT separately. Precomputed feature extraction
time is NOT included; raw telemetry transformations are not timed here.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import (average_precision_score, f1_score,
     matthews_corrcoef, precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from msf_mofs import select_features

FEATURES_SPEC=ROOT/"real_mvp/PROTOCOL_FROZEN.json"
SPEC=ROOT/"research/feature_tradeoff/PROTOCOL_FROZEN.json"
SOURCE=ROOT/"results/real_ops_sat/dataset.csv"

def split_ids(y,eligible,seed):
    """60/20/20 fit, validation-threshold, audit; all source train=1."""
    fit,other=train_test_split(eligible,test_size=.40,stratify=y[eligible],random_state=seed)
    cal,audit=train_test_split(other,test_size=.50,stratify=y[other],random_state=seed+1)
    return tuple(np.sort(z) for z in (fit,cal,audit))

def train_classifier():
    return HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=11,
        learning_rate=.055,l2_regularization=.4,random_state=42)

def calibration_threshold(y,p):
    pr,re,th=precision_recall_curve(y,p)
    f1=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-12)
    if not len(th):raise ValueError("Empty precision-recall thresholds")
    return float(th[np.flatnonzero(f1>=f1.max()-1e-12)[-1]])

def audit_metrics(y,s,threshold):
    y=np.asarray(y,int);s=np.asarray(s,float);pred=(s>=threshold).astype(int)
    if len(set(y))!=2:raise ValueError("Audit must include both true classes")
    return dict(
       f1=float(f1_score(y,pred)),ap=float(average_precision_score(y,s)),
       recall=float(recall_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
       fpr=float(np.mean(pred[y==0])),
       mcc=float(matthews_corrcoef(y,pred)),
       false_positives=int(np.sum((pred==1)&(y==0))),
       false_negatives=int(np.sum((pred==0)&(y==1))))

def med_quantile(timing):
    vals=np.asarray(timing,float)*1000
    return {"median_ms":float(np.median(vals)),"q25_ms":float(np.percentile(vals,25)),
            "q75_ms":float(np.percentile(vals,75)),"repeats":len(vals)}

def pareto_front(records,xkey="model_fit_median_ms",ykey="audit_f1"):
    out=[]
    for a in records:
        dominated=any(
          b is not a and b[xkey]<=a[xkey] and b[ykey]>=a[ykey]
          and (b[xkey]<a[xkey] or b[ykey]>a[ykey])
          for b in records)
        if not dominated:out.append(a["name"])
    return out

def plot(records,out,xfield,title,xlabel,filename):
    # The figure is an actual chart generated from measured results, not schematic.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,6.8))
    for i,r in enumerate(records):
        size=70+float(r["feature_count"])*12
        ax.scatter(r[xfield],r["audit_f1"],s=size,alpha=.8)
        ax.annotate(r["name"].replace("fisher_","Fisher "),(
                  r[xfield],r["audit_f1"]),xytext=(5,7 if i%2 else -17),
                  textcoords="offset points",fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Independent internal audit F1 (higher better)")
    ax.grid(alpha=.23)
    ax.set_ylim(max(0,min(r["audit_f1"] for r in records)-.06),
                min(1.01,max(r["audit_f1"] for r in records)+.06))
    ax.set_xlim(left=0)
    ax.text(.02,.02,"Marker size = # selected features; original official test=0 excluded",
            transform=ax.transAxes,fontsize=9,alpha=.7)
    fig.tight_layout()
    fig.savefig(out/filename,dpi=180)
    plt.close(fig)

def run(out="results/feature_tradeoff"):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sraw=SPEC.read_bytes()
    spec=json.loads(sraw)
    feature_names=json.loads(FEATURES_SPEC.read_text())["features"]
    source=pd.read_csv(SOURCE)
    if len(source)!=2123 or not {"train","anomaly","segment"}.issubset(source.columns):
        raise ValueError("OPS-SAT source changed")
    if source.segment.duplicated().any():raise ValueError("Duplicate segments")
    raw=source[feature_names].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    y=source.anomaly.to_numpy(int)
    pool=np.flatnonzero(source.train.to_numpy(int)==1)
    excluded=np.flatnonzero(source.train.to_numpy(int)==0)
    fit,cal,audit=split_ids(y,pool,int(spec["frozen_seed"]))
    if not (set(fit)|set(cal)|set(audit))==set(pool):raise AssertionError("Invalid pool split")
    if (set(fit)|set(cal)|set(audit))&set(excluded):raise AssertionError("Official holdout touched")

    # Fits the median on TRAIN FIT only; retains identical preprocessing for all
    # feature subset configurations; report this shared overhead separately.
    ts=time.perf_counter()
    clean=np.where(np.isfinite(raw[fit]),raw[fit],np.nan)
    med=np.nanmedian(clean,axis=0)
    med=np.nan_to_num(med,nan=0.,posinf=0.,neginf=0.)
    X=np.where(np.isfinite(raw),raw,med)
    prep_ms=(time.perf_counter()-ts)*1000.

    ts=time.perf_counter()
    sel,search_meta=select_features(X[fit],y[fit],
                 population=12,generations=5,seed=42,tolerance=.03)
    msf_search_ms=(time.perf_counter()-ts)*1000.
    ts=time.perf_counter()
    with np.errstate(divide="ignore",invalid="ignore"):
        scores,_=f_classif(X[fit],y[fit])
    scores=np.nan_to_num(scores,nan=-np.inf)
    rank=np.argsort(-scores,kind="stable")
    fisher_ms=(time.perf_counter()-ts)*1000.
    if not len(sel):raise ValueError("MSF selected empty subset")
    methods={"all_18":(np.arange(18),0.0),
            "msf_mofs_selected":(sel,msf_search_ms)}
    methods.update({f"fisher_top{k}":(rank[:k],fisher_ms) for k in (3,6,9,12,15)})
    if set(methods)!=set(spec["comparison_methods"]):
        raise ValueError("Experiment methods don't match preregistration")

    # One untimed warm-up precedes timing; timed values report model FIT vs
    # PREDICT on the fixed audit set for 7 independent repeated fits.
    warm=train_classifier()
    warm.fit(X[fit][:,methods["all_18"][0]],y[fit])
    warm.predict_proba(X[audit][:,methods["all_18"][0]])
    records=[]; timing_n=int(spec["training_repeat_count"])
    for name in spec["comparison_methods"]:
        idx,selection_ms=methods[name]
        tx=X[fit][:,idx]
        vx=X[cal][:,idx]
        hx=X[audit][:,idx]
        # No threshold may be learned from the held-out internal audit set.
        fits=[];preds=[]
        model=None;val_probs=None;audit_probs=None
        for j in range(timing_n):
            start=time.perf_counter()
            m=train_classifier()
            m.fit(tx,y[fit])
            fits.append(time.perf_counter()-start)
            if j==0:
                val_probs=m.predict_proba(vx)[:,1]
            start=time.perf_counter()
            probs=m.predict_proba(hx)[:,1]
            preds.append(time.perf_counter()-start)
            if j==0:
                model=m
                audit_probs=probs
            else:
                if not np.allclose(probs,audit_probs,rtol=1e-12,atol=1e-12):
                    raise AssertionError("Non-deterministic repeated model for "+name)
        threshold=calibration_threshold(y[cal],val_probs)
        q=audit_metrics(y[audit],audit_probs,threshold)
        buf=io.BytesIO();joblib.dump(model,buf,compress=3)
        fit_stats=med_quantile(fits);pred_stats=med_quantile(preds)
        rec={
           "name":name,
           "feature_count":int(len(idx)),
           "feature_fraction":float(len(idx)/len(feature_names)),
           "feature_names":[feature_names[k] for k in idx],
           "selection_ms_once":float(selection_ms),
           "model_fit_median_ms":fit_stats["median_ms"],
           "model_fit_q25_ms":fit_stats["q25_ms"],
           "model_fit_q75_ms":fit_stats["q75_ms"],
           "predict_audit_median_ms":pred_stats["median_ms"],
           "predict_audit_q25_ms":pred_stats["q25_ms"],
           "predict_audit_q75_ms":pred_stats["q75_ms"],
           "predict_per_segment_median_ms":float(pred_stats["median_ms"]/len(audit)),
           "select_plus_first_fit_ms":float(selection_ms+fit_stats["median_ms"]),
           "model_size_bytes":len(buf.getvalue()),
           "validation_threshold":threshold,
           "audit_f1":q["f1"],"audit_ap":q["ap"],
           "audit_recall":q["recall"],"audit_precision":q["precision"],
           "audit_fpr":q["fpr"],"audit_mcc":q["mcc"],
           "audit_fp":q["false_positives"],"audit_fn":q["false_negatives"]
        }
        records.append(rec)
    pd.DataFrame(records).to_csv(out/"tradeoff_table.csv",index=False)
    plot(records,out,"model_fit_median_ms","Classifier training time vs anomaly F1",
         "Classifier FIT time (ms, median of 7)", "tradeoff_fit_time_vs_f1.png")
    plot(records,out,"select_plus_first_fit_ms","First-run select + train time vs anomaly F1",
         "Feature-selection (once) + classifier FIT time (ms)",
         "tradeoff_selection_plus_fit_vs_f1.png")
    report={
      "experiment":spec["study"],
      "scope":"Single real satellite dataset OPS-SAT-AD, official 18 full-segment features",
      "purpose":"Feature count vs downstream supervised fit cost and audit performance",
      "data_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      "config_sha256":hashlib.sha256(sraw).hexdigest(),
      "split":{"fit":len(fit),"calibration":len(cal),"internal_audit":len(audit),
               "official_test_excluded":len(excluded),
               "fit_positive":int(y[fit].sum()),"cal_positive":int(y[cal].sum()),
               "audit_positive":int(y[audit].sum())},
      "measurement":{"runs_per_model":timing_n,"preprocessing_once_ms":prep_ms,
         "timer":"time.perf_counter wall clock, same single CPU runner; no GPU",
         "selection_timed_once":True,"cpu_count":os.cpu_count(),
         "blas_thread_env":{k:os.environ.get(k) for k in
              ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},
         "feature_source":"Precomputed official 18 statistical features; input feature engineering cost unmeasured",
         "timing_precision":"Small millisecond times are OS scheduling sensitive; median and quartiles are reported"},
      "msf_internal_selection":{"selected":len(sel),"search_metadata":search_meta},
      "pareto_front_fit_time_vs_F1":pareto_front(records),
      "pareto_front_initial_total_vs_F1":pareto_front(records,xkey="select_plus_first_fit_ms"),
      "records":records,
      "warnings":[
        "The original official test=0 was already reported in earlier delivery, but is excluded entirely here",
        "This is an internal one-dataset tradeoff diagnostic; it cannot prove out-of-domain transfer",
        "18 input features are already precomputed; extraction-time savings are not measured",
        "MSF uses logistic regression wrapper; the comparison model here is HistGradientBoosting",
        "To claim selection improves end-to-end cost, amortize the selector's up-front search cost over a specified number of retrains",
        "Exact runtime ratios depend on CPU and workload; same hardware used for all methods",
        "No feature-count/latency tradeoff is itself a novel patent mechanism",
        "The pretrained frozen MVP and its reported best test model are unchanged"
      ]
    }
    (out/"tradeoff_metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,default=int)+"\n")
    full=next(r for r in records if r["name"]=="all_18")
    msf=next(r for r in records if r["name"]=="msf_mofs_selected")
    fmt=lambda r:f"{r['feature_count']} | {r['selection_ms_once']:.2f} | {r['model_fit_median_ms']:.2f} | {r['predict_per_segment_median_ms']:.4f} | {r['audit_f1']:.4f} | {r['audit_ap']:.4f}"
    md=[
       "# 单一真实卫星数据集：特征数—计算成本—分类性能对照",
       "",
       "**范围：只评价 OPS-SAT-AD，原始官方 test=0 未使用，冻结交付结果不改。**",
       "",
       "以下是实际运行计时，单位毫秒；不同 CPU 上绝对运行时间可能不同。",
       "",
       "| 方法 | 特征数 | 选择耗时（一次）ms | 分类器训练中位数 ms | 单片段预测中位数 ms | 内部审计 F1 | 内部审计 AP |",
       "|---|---:|---:|---:|---:|---:|---:|",
    ]
    md += ["| "+r["name"]+" | "+fmt(r)+" |" for r in records]
    md+=["",
      f"MSF 相比完整特征，特征数从 18 减为 {msf['feature_count']}；",
      f"其训练时间是 {msf['model_fit_median_ms']:.2f}ms vs {full['model_fit_median_ms']:.2f}ms；"
      f" 审计 F1 是 {msf['audit_f1']:.4f} vs {full['audit_f1']:.4f}。",
      "",
      "## 图表",
      "",
      "![分类器训练时间 vs F1](tradeoff_fit_time_vs_f1.png)",
      "",
      "![初次特征选择加训练时间 vs F1](tradeoff_selection_plus_fit_vs_f1.png)",
      "",
      "第一张图检验“选少量特征后，下游模型是否训练更快”；"
      "第二张图检验“把特征搜索过程也算进去后，第一次训练总时间是否仍占优”。",
      "每个点表示一种方法；面积表示特征数。建议在论文中同时报告 AP/Recall、模型大小和预测耗时，不只给 F1。",
      "",
      "## 必要解释",
      "",
      "- 没有测量官方 18 项统计特征从原始卫星遥测提取的时间，不能因此宣称完整在线推理延迟改善。",
      "- 18 维是低维情况，MSF 搜索自身可能远比一次模型训练耗时；在这一数据集上不应预设 MSF 是最优。",
      "- 计时来自 GitHub Actions 同一次固定环境的 7 次重复；用 median + 四分位数区间降低时钟噪声。",
      "- 这张图表达的是单数据集内的成本—效果权衡，不涉及跨数据集迁移，也不构成专利创造性证明。",
      ""
    ]
    (out/"RESULTS.md").write_text("\n".join(md))
    print(json.dumps({"split":report["split"],"pareto":report["pareto_front_fit_time_vs_F1"],
          "summary":[{k:r[k] for k in ("name","feature_count","selection_ms_once",
          "model_fit_median_ms","predict_per_segment_median_ms","audit_f1","audit_ap")}
          for r in records]},ensure_ascii=False,indent=2))
    return report

if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--out",default="results/feature_tradeoff")
    args=a.parse_args()
    run(args.out)
