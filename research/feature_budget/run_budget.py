#!/usr/bin/env python3
"""Reproducible, one-OPS-SAT-dataset feature budget tradeoff study.

Reuses exact real raw-telemetry-derived 2578-dimensional engineering
of research/highdim_opssat/benchmark.py. No synthetic sensor features.
This explores a NEW budget-constrained Fisher-seeded NSGA-II selector
using the existing project's MSF-MOFS evolutionary primitives. It is
an engineering variant, not the exact frozen original algorithm.

Use only original official train=1, with new fit/calibration/internal-audit
segments. Internal audits on this dataset have been viewed in prior studies;
this is exploratory, not confirmatory independent evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/highdim_opssat"))
sys.path.insert(0,str(ROOT/"src"))
import benchmark as highdim
from msf_mofs import fisher_scores, _fronts, _crowding, _survivors

PROTOCOL=ROOT/"research/feature_budget/PROTOCOL_FROZEN.json"


def model():
    return highdim.make_model()


def selection_with_budget(X,y,rank,budget,seed,population=8,generations=3,folds=2,tolerance=.025):
    """Fisher-seeded budget-constrained multiobjective NSGA-II feature selection.

    Min objective 1: 1-CV Gmean, min objective 2: #features / total dims.
    Search enforces ceil(budget/2) <= cardinality <= budget for meaningful
    fixed budget studies. Evaluation fits a StandardScaler inside each fold.
    """
    X=np.asarray(X,float);y=np.asarray(y,int)
    d=X.shape[1];k=int(budget);low=max(1,int(math.ceil(k/2)))
    if len(rank)!=d or not 1<=k<=d:raise ValueError("Invalid budget/ranking")
    rng=np.random.default_rng(seed)
    inv_rank=np.empty(d,dtype=int)
    inv_rank[rank]=np.arange(d)
    folds_ix=list(StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed).split(X,y))
    pool=np.asarray(rank[:min(d,max(64,k*4))],int)
    cache={}
    def evaluate(z):
        selected=tuple(np.flatnonzero(z))
        if selected in cache:return cache[selected]
        if not (low<=len(selected)<=k):
            raise AssertionError("Selector candidate violated cardinality constraints")
        scores=[]
        for ti,vi in folds_ix:
            a=X[ti][:,selected];b=X[vi][:,selected]
            sc=StandardScaler()
            a=sc.fit_transform(a);b=sc.transform(b)
            clf=LogisticRegression(C=.5,class_weight="balanced",
                      solver="liblinear",max_iter=350,random_state=42).fit(a,y[ti])
            pred=clf.predict(b)
            cm=confusion_matrix(y[vi],pred,labels=[0,1])
            sens=cm[1,1]/max(1,cm[1].sum())
            spec=cm[0,0]/max(1,cm[0].sum())
            scores.append(math.sqrt(spec*sens))
        value=(1-float(np.mean(scores)),len(selected)/d)
        cache[selected]=value
        return value

    def repair(z,target=None):
        z=np.asarray(z,bool).copy()
        active=np.flatnonzero(z)
        if target is None:target=int(np.clip(len(active),low,k))
        else:target=int(np.clip(target,low,k))
        if len(active)>target:
            # Retain a mixture of Fisher-best and stochastic survivors to
            # avoid replacing the evolutionary stage with pure ranking.
            shuffle_order=active[rng.permutation(len(active))]
            weights=.70*inv_rank[shuffle_order]/max(1,d-1)+.30*rng.random(len(active))
            keep=shuffle_order[np.argsort(weights)[:target]]
            z[:]=False;z[keep]=True
        elif len(active)<target:
            # Candidate pool combines Fisher seed candidates and global diversity.
            choices=pool[~z[pool]]
            n=target-len(active)
            candidates=[]
            if len(choices):
                rank_sample=choices[:min(len(choices),n//2+1)]
                candidates.extend(map(int,rank_sample))
            rest=np.flatnonzero(~z)
            rest=np.setdiff1d(rest,np.asarray(candidates,dtype=int),assume_unique=False)
            if n>len(candidates):
                candidates.extend(map(int,rng.choice(rest,size=n-len(candidates),replace=False)))
            z[np.asarray(candidates[:n],int)]=True
        return z

    pop=[]
    for j,n in enumerate([low,max(low,3*k//4),k]):
        z=np.zeros(d,dtype=bool);z[rank[:n]]=True
        pop.append(z)
    while len(pop)<population:
        n=int(rng.integers(low,k+1))
        fisher_n=int(math.ceil(n*.75))
        rank_candidate=rng.choice(pool,size=fisher_n,replace=False)
        rand_candidate=rng.choice(d,size=n-fisher_n,replace=False)
        z=np.zeros(d,dtype=bool);z[rank_candidate]=True;z[rand_candidate]=True
        pop.append(repair(z,n))
    pop=np.asarray(pop,bool)
    objs=np.asarray([evaluate(z) for z in pop],float)
    for generation in range(generations):
        fronts=_fronts(objs)
        rank_index=np.empty(len(pop),int)
        crowd={}
        for i,front in enumerate(fronts):
            rank_index[front]=i;crowd.update(_crowding(front,objs))
        def pick():
            a,b=rng.integers(len(pop),size=2)
            return int(a if (rank_index[a],-crowd[a])<=(rank_index[b],-crowd[b]) else b)
        children=[]
        while len(children)<population:
            a,b=pop[pick()].copy(),pop[pick()].copy()
            if rng.random()<.9:
                cut=int(rng.integers(1,d))
                a,b=np.r_[a[:cut],b[cut:]],np.r_[b[:cut],a[cut:]]
            for child in (a,b):
                flips=int(rng.integers(1,5))
                indexes=rng.choice(d,size=flips,replace=False)
                child[indexes]=~child[indexes]
                target=int(rng.integers(low,k+1))
                child=repair(child,target)
                children.append(child)
                if len(children)==population:break
        cp=np.asarray(children,bool)
        co=np.asarray([evaluate(z) for z in cp],float)
        pop,objs=_survivors(np.vstack([pop,cp]),np.vstack([objs,co]),population)
    front=_fronts(objs)[0]
    best=max(1-objs[i,0] for i in front)
    viable=[i for i in front if 1-objs[i,0]>=best-tolerance]
    choice=min(viable,key=lambda i:(int(pop[i].sum()),objs[i,0]))
    selected=np.flatnonzero(pop[choice])
    return selected,{"budget":k,"selected":int(len(selected)),
         "inner_cv_best_gmean":float(best),"chosen_gmean":float(1-objs[choice,0]),
         "unique_cv_wrapper_evaluations":len(cache),
         "population":population,"generations":generations,"cv_folds":folds,
         "objectives":["1 - CV Gmean","feature count / 2578"]}


def timed_method(name,indices,selector_ms,selector_info,X,y,fit,cal,audit,repeat):
    inds=np.asarray(indices,int)
    if not len(inds):raise ValueError("Empty feature set")
    tx=X[fit][:,inds];cx=X[cal][:,inds];ax=X[audit][:,inds]
    train_times=[];pred_times=[]
    calibration=None;audit_score=None;size_bytes=None
    for j in range(repeat):
        start=time.perf_counter()
        m=model().fit(tx,y[fit])
        train_times.append((time.perf_counter()-start)*1000.)
        if j==0:calibration=m.predict_proba(cx)[:,1]
        start=time.perf_counter()
        p=m.predict_proba(ax)[:,1]
        pred_times.append((time.perf_counter()-start)*1000.)
        if j==0:
            audit_score=p
            dest=io.BytesIO();joblib.dump(m,dest,compress=3)
            size_bytes=len(dest.getbuffer())
        elif not np.allclose(audit_score,p,atol=1e-12,rtol=1e-12):
            raise AssertionError("Repeated model predict mismatch")
    threshold=highdim.choose_threshold(y[cal],calibration)
    audit_quality=highdim.quality(y[audit],audit_score,threshold)
    calibration_quality=highdim.quality(y[cal],calibration,threshold)
    tr=float(np.median(train_times));pr=float(np.median(pred_times))
    return {
      "method":name,"features":int(len(inds)),
      "feature_count_ratio":float(len(inds)/X.shape[1]),
      "selector_ms":float(selector_ms),
      "fit_median_ms":tr,
      "fit_q25_ms":float(np.percentile(train_times,25)),
      "fit_q75_ms":float(np.percentile(train_times,75)),
      "predict_319_segments_median_ms":pr,
      "predict_per_segment_ms":pr/len(audit),
      "initial_select_plus_fit_ms":float(tr+selector_ms),
      "10x_retrains_plus_selection_ms":float(selector_ms+10*tr),
      "model_size_bytes":size_bytes,
      "threshold_from_cal":threshold,
      "cal_F1":calibration_quality["f1"],
      "cal_AP":calibration_quality["ap"],
      "cal_recall":calibration_quality["recall"],
      "audit_F1":audit_quality["f1"],
      "audit_AP":audit_quality["ap"],
      "audit_recall":audit_quality["recall"],
      "audit_precision":audit_quality["precision"],
      "audit_fpr":audit_quality["fpr"],
      "audit_mcc":audit_quality["mcc"],
      "audit_FP":audit_quality["false_positives"],
      "audit_FN":audit_quality["false_negatives"],
      "selector_metadata":selector_info
    }


def plot(rows,out,x,y,name,title,xlabel,ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,7))
    style={"full_2578":"black","official_18":"#777777"}
    for j,r in enumerate(rows):
        if r["method"].startswith("msf_"):marker="o"
        elif r["method"].startswith("fisher_"):marker="s"
        else:marker="D"
        ax.scatter(r[x],r[y],marker=marker,s=50+10*np.sqrt(r["features"]),
                   color=style.get(r["method"],None),alpha=.8)
        ax.annotate(r["method"],(r[x],r[y]),xytext=(4,8 if j%2 else -16),
                    textcoords="offset points",fontsize=8)
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
    ax.grid(alpha=.22)
    ax.set_xlim(left=0)
    ax.set_title(title)
    ax.text(.02,.02,"Circles MSF; squares Fisher; diamonds full/official. Same OPS-SAT, same classifier.",
            transform=ax.transAxes,fontsize=8,alpha=.65)
    fig.tight_layout()
    fig.savefig(out/name,dpi=180)
    plt.close(fig)


def pareto(rows,xkey,ykey):
    result=[]
    for a in rows:
        dominated=any(b is not a and b[xkey]<=a[xkey] and b[ykey]>=a[ykey] and
                      (b[xkey]<a[xkey] or b[ykey]>a[ykey]) for b in rows)
        if not dominated:result.append(a["method"])
    return result


def run(outpath="results/feature_budget"):
    out=Path(outpath)
    out.mkdir(parents=True,exist_ok=True)
    config_bytes=PROTOCOL.read_bytes()
    config=json.loads(config_bytes)
    # Reuse exactly the existing engineered multi-scale feature definitions.
    source_config=json.loads((ROOT/"research/highdim_opssat/PROTOCOL_FROZEN.json").read_text())
    X,df,feature_names,source_manifest=highdim.load_and_engineer(out,source_config)
    y=df.anomaly.to_numpy(int)
    fit,cal,audit,excluded=highdim.get_partition(df,config["split"]["seed"])
    assert len(set(np.r_[fit,cal,audit]))==int((df.train==1).sum())
    assert set(np.r_[fit,cal,audit]).isdisjoint(excluded)
    X,med=highdim.standardize_fitted(X,fit)
    start=time.perf_counter()
    rank=np.argsort(-np.nan_to_num(fisher_scores(X[fit],y[fit]),nan=0.,posinf=0.),
                    kind="stable")
    fisher_ms=(time.perf_counter()-start)*1000.
    mcfg=config["budgeted_msf"]
    selections={}
    all_budgets=config["feature_budgets"]
    for b in all_budgets:
        start=time.perf_counter()
        mask,info=selection_with_budget(X[fit],y[fit],rank,b,
             seed=mcfg["random_seed"]+b,
             population=mcfg["population"],
             generations=mcfg["generations"],
             folds=mcfg["inner_cv_folds"],
             tolerance=.025)
        cost=(time.perf_counter()-start)*1000.
        selections[b]=(mask,cost,info)
        print("SELECTOR",json.dumps({"budget":b,"features":len(mask),
                     "selector_ms":cost,"wrapper_evaluations":info["unique_cv_wrapper_evaluations"]}),flush=True)
    # All metrics below use the identical classifier and untouched audit labels.
    methods=[("official_18",np.arange(18),0.,None),
             ("full_2578",np.arange(X.shape[1]),0.,None)]
    for b in all_budgets:
        methods.append((f"fisher_{b}",rank[:b],fisher_ms,
                        {"selection":"train-only Fisher filter"}))
        inds,cost,info=selections[b]
        methods.append((f"msf_{b}",inds,cost,info))
    model().fit(X[fit][:,:18],y[fit]).predict_proba(X[cal][:,:18])
    rows=[]
    for name,inds,search_ms,info in methods:
        result=timed_method(name,inds,search_ms,info,X,y,fit,cal,audit,
             repeat=config["timing"]["train_repeats"])
        rows.append(result)
        print("MEASURED",json.dumps({k:result[k] for k in (
            "method","features","selector_ms","fit_median_ms",
            "cal_F1","cal_AP","audit_F1","audit_AP","audit_recall","audit_fpr")}),flush=True)
    # Predefined selection rule using calibration metrics ONLY. Audit is merely
    # reported afterwards. If no candidate matches, explicitly say so.
    baseline=next(r for r in rows if r["method"]=="full_2578")
    thresholds=config["calibration_only_recommendation"]
    feasible=[r for r in rows if
         r["cal_F1"]>=baseline["cal_F1"]-thresholds["max_F1_drop_from_full"]
         and r["cal_AP"]>=baseline["cal_AP"]-thresholds["max_AP_drop_from_full"]
         and r["cal_recall"]>=baseline["cal_recall"]-thresholds["max_recall_drop_from_full"]]
    suggested=min(feasible,key=lambda r:(r["10x_retrains_plus_selection_ms"],
                           -r["cal_F1"]))["method"] if feasible else None
    report={
      "experiment":config["version"],
      "protocol_sha256":hashlib.sha256(config_bytes).hexdigest(),
      "source":source_manifest,
      "partitions":{"train_fit":len(fit),"calibration":len(cal),"internal_audit":len(audit),
             "official_test_0_excluded":len(excluded)},
      "scientific_limitations":[
        "This is one-dataset exploratory research only; previous studies already viewed multiple reseeded internal audits on the same official train pool",
        "All multi-scale features use full raw segment, so not an online early-warning experiment",
        "Added 2560 dimensions encode correlated deterministic summaries of 9 individual single-channel OPS-SAT segment groups, not 2560 sensors",
        "Fisher-based and MSF wrapper feature selection both use FIT ONLY; calibration chooses thresholds and a descriptive recommendation",
        "Model selection under known prior same-dataset analyses is exploratory, not a fresh confirmatory claim",
        "Budgeted MSF selector is an NSGA-II/Fisher-guided adaptation; not a reproduction of the original MSF-MOFS paper",
        "Original official test=0 is fully excluded and delivery release branch not modified",
        "Absolute wall clock depends on hardware; same single-thread CI job, training repeat median",
        "Initial selector cost included; classifier fit acceleration alone does not prove total training speedup"
      ],
      "hardware_threads":{k:os.environ.get(k) for k in
                      ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
      "source_feature_extraction_s":source_manifest["feature_engineering_seconds"],
      "fisher_rank_ms":fisher_ms,
      "rows":rows,
      "pareto_fit_vs_audit_F1":pareto(rows,"fit_median_ms","audit_F1"),
      "pareto_selectplusfit_vs_audit_F1":pareto(rows,"initial_select_plus_fit_ms","audit_F1"),
      "recommendation_rule":"CALIBRATION ONLY: F1/AP/Recall within predeclared gap from full 2578; among feasible minimum selection+10x fits",
      "calibration_feasible":[r["method"] for r in feasible],
      "recommended_by_calibration":suggested,
      "recommended_internal_audit":next(({
           "audit_F1":r["audit_F1"],"audit_AP":r["audit_AP"],
           "audit_recall":r["audit_recall"],"audit_fpr":r["audit_fpr"]}
           for r in rows if r["method"]==suggested),None)
    }
    (out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=int)+"\n")
    tab=pd.DataFrame([{k:v for k,v in r.items() if k!="selector_metadata"} for r in rows])
    tab.to_csv(out/"comparison.csv",index=False)
    for x,y,name,title in [
      ("fit_median_ms","audit_F1","fit_time_vs_F1.png","Classifier fit time vs F1 (OPS-SAT budgets)"),
      ("initial_select_plus_fit_ms","audit_F1","first_cost_vs_F1.png","First selector+fit time vs F1 (OPS-SAT budgets)"),
      ("fit_median_ms","audit_AP","fit_time_vs_AP.png","Classifier fit time vs AP (OPS-SAT budgets)"),
      ("features","audit_F1","feature_count_vs_F1.png","Number of features vs F1 (OPS-SAT budgets)")]:
        plot(rows,out,x,y,name,title,x.replace("_"," ")+" (milliseconds if time)",
             "audit F1" if y=="audit_F1" else "audit AP")
    md=[
     "# OPS-SAT 2578 维特征：MSF-MOFS 不同预算的时间—性能平衡",
     "",
     "**只使用原官方训练片段；实验属于同一数据集上的探索性对照，之前的多次随机内部审计已被观察，不等同于新独立测试。**",
     "",
     f"样本数：拟合 {len(fit)}，校准 {len(cal)}，内部审计 {len(audit)}；原官方测试 {len(excluded)} 段未使用。",
     f"原始遥测派生特征统一提取 {source_manifest['feature_engineering_seconds']:.2f} 秒，本次预算控制仅决定下游使用哪些维度；不能声称省下全部特征提取时间。",
     "",
     "| 方法 | 特征数 | MSF/Fisher选择 ms | 分类器训练 ms | 首次合计 ms | F1 | AP | 异常召回 | 正常误报率 |",
     "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for r in rows:
        md.append(f"| {r['method']} | {r['features']} | {r['selector_ms']:.1f} | {r['fit_median_ms']:.1f} | {r['initial_select_plus_fit_ms']:.1f} | {r['audit_F1']:.4f} | {r['audit_AP']:.4f} | {r['audit_recall']:.4f} | {r['audit_fpr']:.4f} |")
    md += [
      "",
      f"**预先冻结的校准集规则推荐**：{suggested if suggested else '没有候选符合约束'}；此推荐没有用内部审计指标选择。",
      "",
      "![训练成本—F1](fit_time_vs_F1.png)",
      "",
      "![首次搜索+训练成本—F1](first_cost_vs_F1.png)",
      "",
      "![训练成本—AP](fit_time_vs_AP.png)",
      "",
      "![特征维数—F1](feature_count_vs_F1.png)",
      "",
      "这是同一真实卫星数据集上的研究对照，没有跨数据集迁移，也没有真实三级状态真值；MSF 的高压缩和精度之间的代价需如实报告。",
      ""
    ]
    (out/"RESULTS.md").write_text("\n".join(md),encoding="utf-8")
    print("FINAL",json.dumps({"recommended":suggested,"feasible":report["calibration_feasible"],
         "pareto_fit":report["pareto_fit_vs_audit_F1"],
         "pareto_first":report["pareto_selectplusfit_vs_audit_F1"]},ensure_ascii=False),flush=True)
    return report


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--out",default="results/feature_budget")
    parsed=args.parse_args()
    run(parsed.out)
