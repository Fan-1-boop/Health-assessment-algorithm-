#!/usr/bin/env python3
"""Real OPS-SAT raw 2578-dimensional feature engineering and efficiency audit.

The raw observations are real satellite telemetry; the 2560 new descriptors
are engineered from each original SINGLE-CHANNEL segment. Interpolating the
signal to 128 normalized points cannot create independent sensor information.
All features describe an entire segment (RETROSPECTIVE, NOT early prediction).

Original OPS-SAT official test=0 is excluded from all model comparisons.
"""
from __future__ import annotations
import argparse
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
import requests
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (f1_score,precision_score,recall_score,average_precision_score,
                            roc_auc_score,precision_recall_curve,matthews_corrcoef)
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from msf_mofs import select_features, fisher_scores

SPEC=ROOT/"research/highdim_opssat/PROTOCOL_FROZEN.json"


def _lagcorr(z,lag):
    a=z[:,:-lag]
    b=z[:,lag:]
    a=a-a.mean(axis=1,keepdims=True)
    b=b-b.mean(axis=1,keepdims=True)
    numerator=(a*b).sum(axis=1)
    denominator=np.sqrt(np.maximum(np.sum(a*a,axis=1)*np.sum(b*b,axis=1),0))
    return np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-20)


def features_for_one_segment(raw,window_widths,anchors=16,num_grid=128):
    """2560 true raw-signal derived multi-scale and multi-location descriptors.

    Sample count varies from 8 to ~1040. Linear interpolation ensures every
    segment maps to the SAME feature schema, but does not add any true samples.
    Absolute amplitude, duration and original 18 features are not discarded.
    """
    raw=np.asarray(raw,float)
    if len(raw)<2 or not np.all(np.isfinite(raw)):
        raise ValueError("Segment requires >=2 finite observed samples")
    z=np.interp(np.linspace(0,len(raw)-1,num_grid),np.arange(len(raw)),raw)
    all_features=[]
    descriptor_names=[
        "mean","std","min","max","p10","p90","rms","slope","delta",
        "diff_abs_mean","diff_std","lag1_corr","lag2_corr",
        "spectral_low_ratio","spectral_high_ratio","iqr","median","mad",
        "center_value","energy"]
    for width in window_widths:
        starts=np.linspace(0,num_grid-width,anchors).round().astype(int)
        windows=np.stack([z[start:start+width] for start in starts])
        means=windows.mean(axis=1)
        std=windows.std(axis=1)
        diff=np.diff(windows,axis=1)
        centered=windows-means[:,None]
        t=np.arange(width,dtype=float)-(width-1)*.5
        slopes=(centered@t)/np.dot(t,t)
        transformed=np.abs(np.fft.rfft(centered,axis=1))**2
        spectrum=transformed[:,1:] # DC removed
        powers=spectrum.sum(axis=1)
        cutoff=max(1,spectrum.shape[1]//2)
        low=spectrum[:,:cutoff].sum(axis=1)
        high=spectrum[:,cutoff:].sum(axis=1)
        p10=np.percentile(windows,10,axis=1)
        p90=np.percentile(windows,90,axis=1)
        med=np.median(windows,axis=1)
        vals=[
            means,std,windows.min(axis=1),windows.max(axis=1),
            p10,p90,np.sqrt(np.mean(windows**2,axis=1)),
            slopes,windows[:,-1]-windows[:,0],
            np.abs(diff).mean(axis=1),diff.std(axis=1),
            _lagcorr(windows,1),_lagcorr(windows,2),
            np.divide(low,powers,out=np.zeros_like(low),where=powers>1e-20),
            np.divide(high,powers,out=np.zeros_like(high),where=powers>1e-20),
            p90-p10,med,np.median(np.abs(windows-med[:,None]),axis=1),
            windows[:,width//2],np.mean(windows**2,axis=1)]
        arr=np.stack(vals,axis=1)
        all_features.append(arr.flatten())
    return np.concatenate(all_features),descriptor_names


def load_and_engineer(out,config):
    source=ROOT/config["source"]["original_features_path"]
    existing=pd.read_csv(source)
    existing_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if len(existing)!=2123:
        raise ValueError("Unexpected OPS-SAT official feature dataframe")
    t_download=time.perf_counter()
    url=config["source"]["raw_url"]
    res=requests.get(url,timeout=130)
    res.raise_for_status()
    raw_payload=res.content
    download_s=time.perf_counter()-t_download
    raw_sha=hashlib.sha256(raw_payload).hexdigest()
    raw=pd.read_csv(io.BytesIO(raw_payload),parse_dates=["timestamp"])
    want=config["source"]
    if len(raw)!=want["expected_raw_rows"] or set(want["expected_raw_columns"])!=set(raw.columns):
        raise ValueError("Official raw segment file changed")
    if raw.segment.nunique()!=len(existing):
        raise ValueError("Raw/dataset segment count mismatch")
    if raw.groupby("segment").anomaly.nunique().max()!=1 or raw.groupby("segment").train.nunique().max()!=1:
        raise ValueError("Some source segments have inconsistent labels or split")
    raw=raw.sort_values(["segment","timestamp"],kind="stable")
    grouped=raw.groupby("segment",sort=False)
    segment_map={int(x):gr.value.to_numpy(float) for x,gr in grouped}
    seglabels=raw.groupby("segment",sort=False).agg(anomaly=("anomaly","first"),
                                                  train=("train","first"))
    idx=existing.segment.to_numpy(int)
    if np.any(seglabels.loc[idx,"anomaly"].to_numpy(int)!=existing.anomaly.to_numpy(int)):
        raise ValueError("Official source segment labels mismatch")
    if np.any(seglabels.loc[idx,"train"].to_numpy(int)!=existing.train.to_numpy(int)):
        raise ValueError("Official source train flags mismatch")

    props=config["high_dim_features"]
    t0=time.perf_counter()
    eng=[]
    for s in idx:
        features,descs=features_for_one_segment(
            segment_map[int(s)],props["window_widths"],props["positions_per_width"],
            props["normalized_resampling_points"])
        eng.append(features)
    engineered=np.asarray(eng,dtype=float)
    extraction_s=time.perf_counter()-t0
    if engineered.shape!=(2123,2560) or not np.all(np.isfinite(engineered)):
        raise ValueError("Raw multiscale features contain invalid values or wrong width")
    official_features=json.loads((ROOT/"real_mvp/PROTOCOL_FROZEN.json").read_text())["features"]
    official=existing[official_features].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    if official.shape[1]!=18:
        raise ValueError("Official source dimension mismatch")
    X=np.column_stack([official,engineered])
    if X.shape[1]!=props["total_features"]:
        raise ValueError("Unexpected total feature count")
    labels=list(official_features)+[
        f"width{w}_start{a:02d}_{kind}"
        for w in props["window_widths"]
        for a in range(props["positions_per_width"])
        for kind in descs
    ]
    if len(labels)!=len(set(labels)) or len(labels)!=X.shape[1]:
        raise ValueError("Duplicated feature labels")
    manifest={
        "source_raw_records":len(raw),"raw_signal_segments":len(seglabels),
        "signal_channels":int(raw.channel.nunique()),
        "raw_segment_size_min":int(raw.groupby("segment").size().min()),
        "raw_segment_size_median":float(raw.groupby("segment").size().median()),
        "source_raw_sha256":raw_sha,
        "official_18d_sha256":existing_sha,
        "derived_matrix_sha256":hashlib.sha256(np.ascontiguousarray(engineered).tobytes()).hexdigest(),
        "resampled_points":props["normalized_resampling_points"],
        "derived_features":engineered.shape[1],
        "combined_features":X.shape[1],
        "feature_engineering_seconds":extraction_s,
        "official_raw_download_seconds":download_s,
        "derived_feature_generation":"window statistics of deterministically interpolated original raw per-segment signal; no synthetic added training features",
        "important_limitations":[
            "Entire segment is observed for features: retrospective, not online early anomaly onset",
            "Interpolation does not create independent observations; thousands of correlated features likely",
            "Official precomputed 18 features are appended to 2560 derived features",
            "No model ever uses segment/channel ID or original official test samples as training features"],
        "source_url":url,
    }
    (out/"source_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    return X,existing,labels,manifest


def get_partition(df,seed):
    y=df.anomaly.to_numpy(int)
    base=np.flatnonzero(df.train.to_numpy(int)==1)
    excluded=np.flatnonzero(df.train.to_numpy(int)==0)
    fit,remain=train_test_split(base,test_size=.4,stratify=y[base],random_state=seed)
    cal,audit=train_test_split(remain,test_size=.5,stratify=y[remain],random_state=seed+1)
    fit,cal,audit=(np.sort(x) for x in (fit,cal,audit))
    if not set(np.r_[fit,cal,audit])==set(base) or set(np.r_[fit,cal,audit])&set(excluded):
        raise ValueError("Partition leakage")
    return fit,cal,audit,excluded


def standardize_fitted(X,fit):
    train=np.where(np.isfinite(X[fit]),X[fit],np.nan)
    med=np.nanmedian(train,axis=0)
    med=np.nan_to_num(med,nan=0.,posinf=0.,neginf=0.)
    x=np.where(np.isfinite(X),X,med[None,:])
    return x,med


def make_model():
    return make_pipeline(StandardScaler(),
        LogisticRegression(C=.5,class_weight="balanced",solver="liblinear",
                           max_iter=350,random_state=42))


def choose_threshold(y,score):
    p,r,t=precision_recall_curve(y,score)
    f1=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    if not len(t):raise ValueError("No decision thresholds")
    return float(t[np.flatnonzero(f1>=f1.max()-1e-12)[-1]])


def quality(y,score,t):
    pred=np.asarray(score)>=t
    return {"f1":float(f1_score(y,pred)),
            "ap":float(average_precision_score(y,score)),
            "auc":float(roc_auc_score(y,score)),
            "precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred,zero_division=0)),
            "mcc":float(matthews_corrcoef(y,pred)),
            "fpr":float(np.mean(pred[y==0])),
            "false_positives":int(np.sum(pred&(y==0))),
            "false_negatives":int(np.sum((~pred)&(y==1)))}


def chart(records,out,xkey,qualitykey,name,title,xlabel,ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,7))
    for i,row in enumerate(records):
        x=row[xkey];y=row[qualitykey]
        size=35+1.7*np.sqrt(row["features"])
        ax.scatter(x,y,s=size,alpha=.8)
        ax.annotate(row["model"],(x,y),xytext=(4,8 if i%2 else -13),
                    textcoords="offset points",fontsize=8)
    ax.set_xlim(left=0);ax.set_ylim(min(0.95,max(0,min(r[qualitykey] for r in records)-.08)),1.005)
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
    ax.grid(alpha=.24);ax.set_title(title)
    ax.text(.02,.02,"OPS-SAT official-train internal audit; not original official held-out test",
            transform=ax.transAxes,fontsize=8,alpha=.7)
    fig.tight_layout()
    fig.savefig(out/name,dpi=170)
    plt.close(fig)


def run(outpath):
    out=Path(outpath);out.mkdir(parents=True,exist_ok=True)
    config_raw=SPEC.read_bytes()
    config=json.loads(config_raw)
    X,df,labels,manifest=load_and_engineer(out,config)
    y=df.anomaly.to_numpy(int)
    fit,cal,audit,excluded=get_partition(df,config["protocol"]["random_seed"])
    ts=time.perf_counter()
    X,med=standardize_fitted(X,fit)
    prep_ms=(time.perf_counter()-ts)*1000
    selector_cfg=config["selectors"]["msf"]
    ts=time.perf_counter()
    selected,msf_metadata=select_features(X[fit],y[fit],
        population=selector_cfg["population"],generations=selector_cfg["generations"],
        seed=selector_cfg["seed"],tolerance=selector_cfg["tolerance"])
    msf_search_ms=(time.perf_counter()-ts)*1000
    ts=time.perf_counter()
    ranking=np.argsort(-np.nan_to_num(fisher_scores(X[fit],y[fit]),nan=0.,posinf=0.),kind="stable")
    fisher_ms=(time.perf_counter()-ts)*1000
    methods={"official_18d":(np.arange(18),0.),
            "all_2578d":(np.arange(X.shape[1]),0.),
            "msf_mofs":(selected,msf_search_ms)}
    methods.update({f"fisher_{k}d":(ranking[:k],fisher_ms) for k in (64,128,256,512,1024)})
    print("SELECTOR_RESULTS",json.dumps({"selected":len(selected),"search_ms":msf_search_ms,
                         "fisher_ms":fisher_ms,"cache_eval":msf_metadata["unique_wrapper_evaluations"]}),flush=True)

    # Untimed warm-up shared model class ensures import overhead not counted.
    warm=make_model().fit(X[fit][:,:18],y[fit])
    warm.predict_proba(X[audit][:,:18])
    rows=[]
    for method,(columns,search_ms) in methods.items():
        if len(columns)==0:raise ValueError("Empty selected subset")
        xfit=X[fit][:,columns]; xcal=X[cal][:,columns];xaudit=X[audit][:,columns]
        repeats=config["timing"]["repeat_model_fit"]
        fit_times=[];pred_times=[];final_scores=None;val_scores=None;model_bytes=None
        for iteration in range(repeats):
            now=time.perf_counter()
            model=make_model().fit(xfit,y[fit])
            fit_times.append((time.perf_counter()-now)*1000)
            if iteration==0:
                val_scores=model.predict_proba(xcal)[:,1]
            now=time.perf_counter()
            pred=model.predict_proba(xaudit)[:,1]
            pred_times.append((time.perf_counter()-now)*1000)
            if iteration==0:
                final_scores=pred
                s=io.BytesIO()
                joblib.dump(model,s,compress=3)
                model_bytes=len(s.getvalue())
            elif not np.allclose(pred,final_scores,atol=1e-11,rtol=1e-11):
                raise AssertionError("Repeated fitting changed scores for "+method)
        threshold=choose_threshold(y[cal],val_scores)
        result=quality(y[audit],final_scores,threshold)
        fitms=float(np.median(fit_times)); predms=float(np.median(pred_times))
        row={
            "model":method,"features":int(len(columns)),
            "selected_feature_fraction":float(len(columns)/X.shape[1]),
            "search_ms":float(search_ms),"fit_median_ms":fitms,
            "fit_q25_ms":float(np.percentile(fit_times,25)),
            "fit_q75_ms":float(np.percentile(fit_times,75)),
            "predict_audit_median_ms":predms,
            "predict_per_segment_ms":predms/len(audit),
            "initial_select_plus_fit_ms":fitms+search_ms,
            "select_plus_10_fits_ms":search_ms+10*fitms,
            "model_size_bytes":int(model_bytes),
            "audit_f1":result["f1"],"audit_ap":result["ap"],
            "audit_auc":result["auc"],"audit_recall":result["recall"],
            "audit_fpr":result["fpr"],"audit_mcc":result["mcc"],
            "audit_precision":result["precision"],
            "audit_fp":result["false_positives"],"audit_fn":result["false_negatives"],
            "validation_threshold":threshold
        }
        rows.append(row)
        print("METHOD",json.dumps({k:row[k] for k in
                ("model","features","search_ms","fit_median_ms","audit_f1","audit_ap","audit_recall")}),flush=True)
    pd.DataFrame(rows).to_csv(out/"comparison.csv",index=False)
    for x,key,filename,title in [
        ("fit_median_ms","audit_f1","cost_vs_f1_fit.png","High-dimensional feature selection: classifier fit vs F1"),
        ("initial_select_plus_fit_ms","audit_f1","cost_vs_f1_initial.png","High-dimensional feature selection: FIRST training cost vs F1"),
        ("fit_median_ms","audit_ap","cost_vs_ap_fit.png","High-dimensional feature selection: classifier fit vs AP")
    ]:
        chart(rows,out,x,key,filename,title,
              "Measured milliseconds (smaller is better)",
              ("Audit F1" if key=="audit_f1" else "Audit Average Precision"))
    full=next(r for r in rows if r["model"]=="all_2578d")
    msf=next(r for r in rows if r["model"]=="msf_mofs")
    saving=full["fit_median_ms"]-msf["fit_median_ms"]
    breakeven=int(np.ceil(msf_search_ms/saving)) if saving>0 else None
    report={
        "study":"Pre-registered 2578-feature real OPS-SAT retrospective classification cost-performance",
        "protocol_sha256":hashlib.sha256(config_raw).hexdigest(),
        "source":manifest,
        "partition":{"fit":int(len(fit)),"calibration":int(len(cal)),
                     "audit":int(len(audit)),"original_test_0_never_used":int(len(excluded)),
                     "fit_anomaly":int(y[fit].sum()),"cal_anomaly":int(y[cal].sum()),
                     "audit_anomaly":int(y[audit].sum())},
        "timing":{"preprocessing_ms":prep_ms,"selector_ms":msf_search_ms,
                  "fisher_rank_ms":fisher_ms,"repeated_fits_per_model":repeats,
                  "timing_type":"single-threaded Linux GitHub Actions wall clock medians",
                  "threads":{key:os.environ.get(key) for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}},
        "msf_metadata":msf_metadata,
        "msf_selected_feature_count":len(selected),
        "msf_feature_names":[labels[int(i)] for i in selected],
        "records":rows,
        "msf_vs_full":{
            "feature_reduction_fraction":1-len(selected)/len(labels),
            "classifier_fit_saved_ms":saving,
            "classifier_fit_speedup_fraction":1-msf["fit_median_ms"]/full["fit_median_ms"],
            "F1_change":msf["audit_f1"]-full["audit_f1"],
            "AP_change":msf["audit_ap"]-full["audit_ap"],
            "break_even_refits_excluding_extraction":breakeven,
        },
        "qualifications":[
            "A single original real satellite dataset; 529 previously reported original official test segments excluded from this experiment",
            "Thousands of derived descriptors from one raw univariate channel segment; not thousands of independent sensors",
            "All features use full observed segments; cannot claim online predictive onset or early fault detection",
            "Feature extraction costs included separately, not in classification-only fit",
            "Selection cost paid before training; first-run total should be reported",
            "MSF selector internally optimizes logistic wrapper CV; classifier comparison uses same logistic class",
            "No patent novelty is established merely by feature dimensionality",
            "No cross-dataset generalization or domain adaptation studied"
        ]
    }
    (out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,default=lambda z:int(z) if isinstance(z,np.integer) else float(z))+"\n")
    t=[
       "# OPS-SAT 真实卫星原始遥测：2578 维特征成本—性能对照",
       "",
       "**注意：本实验使用真实单通道原始遥测片段生成 2560 项派生特征，并附加官方 18 项摘要特征；没有生成随机假特征。插值不增加独立信息，且全部为片段结束后的回顾性统计。**",
       "",
       f"样本数：训练 {len(fit)}、阈值校准 {len(cal)}、内部审计 {len(audit)}；此前官方测试 {len(excluded)} 条不使用。",
       f"真实原始点数 {manifest['source_raw_records']}，9 个通道，派生特征提取总耗时 {manifest['feature_engineering_seconds']:.2f} 秒。",
       f"一次性 MSF-MOFS 搜索耗时 {msf_search_ms:.1f} ms，选出 {len(selected)} / 2578 维；同一逻辑回归分类器重复训练 {repeats} 次取中位数。",
       "",
       "| 方案 | 特征数 | 搜索时间 ms | 分类器训练 ms | 首次总成本 ms | F1 | AP | Recall | 正常误报率 |",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        t.append(f"| {r['model']} | {r['features']} | {r['search_ms']:.1f} | {r['fit_median_ms']:.1f} | {r['initial_select_plus_fit_ms']:.1f} | {r['audit_f1']:.4f} | {r['audit_ap']:.4f} | {r['audit_recall']:.4f} | {r['audit_fpr']:.4f} |")
    t += [
        "",
        f"MSF vs 全部2578维：特征压缩率 **{100*report['msf_vs_full']['feature_reduction_fraction']:.1f}%**；"
        f"单次分类器训练节省 **{saving:.1f} ms**；"
        f"审计 F1 变化 **{report['msf_vs_full']['F1_change']:+.4f}**；"
        f"AP 变化 **{report['msf_vs_full']['AP_change']:+.4f}**。",
        f"按单次训练成本简单摊销，不计特征提取，MSF 需要约 **{breakeven if breakeven is not None else '无法摊销'}** 次分类器重训才抵消一次性的特征搜索。",
        "",
        "![训练时间 vs F1](cost_vs_f1_fit.png)",
        "",
        "![含特征搜索的首次成本 vs F1](cost_vs_f1_initial.png)",
        "",
        "![训练时间 vs AP](cost_vs_ap_fit.png)",
        "",
        "特征选择后下游训练更快，不代表含搜索和全量特征提取的总流程一定更快。"
        "此实验既报告成本节省，也如实报告 F1/AP/Recall 等性能变化。",
        ""
    ]
    (out/"RESULTS.md").write_text("\n".join(t))
    print("FINAL_SUMMARY",json.dumps(report["msf_vs_full"],ensure_ascii=False),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/highdim_opssat")
    a=p.parse_args()
    run(a.out)
