#!/usr/bin/env python3
"""Public ordinary grouped same-OPS-SAT robustness and v0.2 replay evidence.

NO confidential/private candidate criteria in this file. No new model variants.
4 label scenarios x fixed grouped folds; 3 leave-channel-out stress folds.
All original OPS-SAT test=0 records excluded from every part of this run.
Reference conflict counters are observations, NOT true-label authentication.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from research.highdim_opssat import benchmark as highdim
from research.feature_budget.run_budget import selection_with_budget
from research.opssat_final_prep.benchmark import estimator,choose_threshold,timer
from research.strict_acceptance.benchmark import (
   alpha_from_calibration,guard_quality,make_metrics)
from msf_mofs import fisher_scores

SPEC=ROOT/"research/grouped_update_robustness/PROTOCOL_FROZEN.json"
FEATURE_PROTO=ROOT/"research/highdim_opssat/PROTOCOL_FROZEN.json"

def channel_order_blocks(df,block=16):
    """Same-channel consecutive original segment IDs, not physical UTC."""
    result=np.empty(len(df),dtype=object)
    for channel,sub in df.groupby("channel",sort=True):
        inds=sub.sort_values("segment",kind="stable").index.to_numpy(int)
        for rank,index in enumerate(inds):
            result[index]=f"{channel}__segment_order_{rank//block:04d}"
    if any(x is None for x in result):raise RuntimeError("Missing temporal-proxy group")
    return result

def holdout_by_group(indices,group,frac,seed):
    indices=np.asarray(indices,int)
    unique=len(np.unique(group[indices]))
    if unique<3:raise ValueError("Too few independent channel-segment groups")
    split=GroupShuffleSplit(n_splits=1,test_size=frac,random_state=int(seed))
    left,right=next(split.split(np.zeros(len(indices)),groups=group[indices]))
    a,b=np.sort(indices[left]),np.sort(indices[right])
    if set(group[a])&set(group[b]):raise AssertionError("Group overlap")
    return a,b


def blocked_fold(df,group,seed):
    original=np.flatnonzero(df.train.to_numpy(int)==1)
    old,remainder=holdout_by_group(original,group,.43,seed)
    rest,batch=holdout_by_group(remainder,group,.36,seed+7)
    last,cal=holdout_by_group(rest,group,.33,seed+19)
    accept,audit=holdout_by_group(last,group,.5,seed+31)
    return {"old_fit":old,"batch":batch,"cal":cal,"accept":accept,"audit":audit}


def channel_shift_fold(df,group,channel,seed):
    """Alternate sensitivity: no target-channel samples in historical fit.

    Batch/gate/audit in monotone segment-ID order with 8-record embargo.
    This segment order is not guaranteed raw physical timestamps.
    """
    original=np.flatnonzero(df.train.to_numpy(int)==1)
    src=df.iloc[original]
    target=src[src.channel==channel].sort_values("segment")
    if len(target)<100:raise ValueError("Target channel too small for stress partition")
    n=len(target)
    p=target.index.to_numpy(int)
    boundary_a=round(n*.50)
    boundary_b=round(n*.75)
    if boundary_b-boundary_a<=16 or n-boundary_b<=16:
        raise ValueError("Insufficient shifted channel after temporal-proxy embargo")
    batch=np.sort(p[:boundary_a])
    accept=np.sort(p[boundary_a+8:boundary_b])
    audit=np.sort(p[boundary_b+8:])
    other=np.flatnonzero((df.train.to_numpy(int)==1)&(df.channel.to_numpy()!=channel))
    old,cal=holdout_by_group(other,group,.15,seed)
    # No same target channel in original old fit / calibration.
    if channel in set(df.iloc[np.r_[old,cal]].channel):
        raise AssertionError("Target channel contaminates historical training")
    return {"old_fit":old,"batch":batch,"cal":cal,"accept":accept,"audit":audit}


def validate_fold(parts,group,y,official_test,minimums):
    labels={}
    used=[]
    for key,ix in parts.items():
        ix=np.asarray(ix,int)
        n_pos=int(np.sum(y[ix]==1));n_neg=int(np.sum(y[ix]==0))
        labels[key]={"n":int(len(ix)),"anomalies":n_pos,
                     "normal":n_neg,"channels":int(len(set(group[ix].tolist())))}
        if n_pos<minimums.get(key,3) or n_neg<3:
            return False,f"{key}: fewer than required anomalies or normals",labels
        used.extend(map(int,ix))
    if len(set(used))!=len(used):
        raise AssertionError("Different study phases have shared segment IDs")
    if set(used)&set(map(int,official_test)):
        raise AssertionError("Official heldout test leakage")
    return True,"OK",labels


def label_conflicts(labels,old_scores):
    labels=np.asarray(labels,int);old_scores=np.asarray(old_scores,float)
    return int(np.sum((labels==1)&(old_scores<.2)|
                      (labels==0)&(old_scores>.8)))


def fit_one_fold(X,df,feature_names,parts,conf,foldname,group):
    y=df.anomaly.to_numpy(int)
    old,batch,cal,accept,audit=(parts[z] for z in
                  ("old_fit","batch","cal","accept","audit"))
    t0=time.perf_counter()
    rank=np.argsort(-np.nan_to_num(fisher_scores(X[old],y[old]),
            nan=0.,posinf=0.),kind="stable")
    subset,sel_meta=selection_with_budget(X[old],y[old],rank,64,
                   seed=141,population=8,generations=3,folds=2,tolerance=.025)
    search_ms=(time.perf_counter()-t0)*1000
    assert 32<=len(subset)<=64
    # Multiscale X derives only from this segment; identity/channel not predictors.
    sx=X[:,subset]
    old_fit=estimator().fit(sx[old],y[old])
    old_p=old_fit.predict_proba(sx)[:,1]
    old_thr=choose_threshold(y[cal],old_p[cal])
    before_accept=make_metrics(y[accept],old_p[accept],old_thr)
    before_audit=make_metrics(y[audit],old_p[audit],old_thr)
    reference=label_conflicts(y[cal],old_p[cal])
    snap={
       "fold_name":foldname,
       "source":"REAL OPS-SAT-AD old model reference on disjoint grouped records",
       "reference_label_conflicts":reference,
       "reference_count":len(cal),
       "sample_counts":{"batch":len(batch),"old":len(old),"cal":len(cal),
          "accept":len(accept),"audit":len(audit)},
       "base":{"acceptance":before_accept,"audit":before_audit},
       "scenarios":{}
    }
    results=[]
    config=conf["acceptance_limits"]
    limit={"max_f1_drop":config["max_f1_drop"],"max_AP_drop":config["max_AP_drop"],
       "max_recall_drop":config["max_recall_drop"],"max_FPR_increase":config["max_normal_FPR_increase"],
       "max_brier_increase":config["max_brier_increase"],"max_NLL_increase":config["max_nll_increase"]}
    corruption={"correct_existing_binary_labels":0,"flip_random_10pct":.1,
                "flip_random_30pct":.3,"flip_all_labels":1}
    for condition,rate in corruption.items():
        labels=y[batch].copy()
        flipped=np.zeros(len(batch),bool)
        if rate==1:flipped[:]=True
        elif rate:
            seed=int(conf["label_flip_seed"])+sum(ord(c) for c in foldname)
            rng=np.random.default_rng(seed)
            changed=rng.choice(len(batch),size=int(round(rate*len(batch))),replace=False)
            flipped[changed]=True
        labels[flipped]^=1
        pool=np.r_[old,batch]
        target=np.r_[y[old],labels]
        def candidate_fit():return estimator().fit(sx[pool],target)
        fit_ms=timer(candidate_fit,3)["median_ms"]
        clf=candidate_fit()
        new_p=clf.predict_proba(sx)[:,1]
        new_thr=choose_threshold(y[cal],new_p[cal])
        after_accept=make_metrics(y[accept],new_p[accept],new_thr)
        after_audit=make_metrics(y[audit],new_p[audit],new_thr)
        checks_c,passed_c=guard_quality(before_accept,after_accept,limit,False)
        checks_d,passed_d=guard_quality(before_accept,after_accept,limit,True)
        alpha,alphaloss,_=alpha_from_calibration(y[cal],old_p[cal],new_p[cal],
                 conf["mixture_alpha"],.0001)
        mixed=(1-alpha)*old_p+alpha*new_p
        mixed_thr=choose_threshold(y[cal],mixed[cal])
        m_accept=make_metrics(y[accept],mixed[accept],mixed_thr)
        checks_e,passed_e=guard_quality(before_accept,m_accept,limit,True)
        if alpha==0:passed_e=False
        m_audit=make_metrics(y[audit],mixed[audit],mixed_thr)
        results.append({
          "fold":foldname,"condition":condition,
          "injected_label_flips":int(flipped.sum()),
          "old_conflicts_batch":label_conflicts(labels,old_p[batch]),
          "old_conflicts_reference":reference,
          "N_batch":len(batch),"N_reference":len(cal),
          "feature_count":len(subset),"feature_selection_ms":search_ms,
          "candidate_fit_ms":fit_ms,
          "A_F1":before_audit["f1"],"B_F1":after_audit["f1"],
          "A_Brier":before_audit["brier"],"B_Brier":after_audit["brier"],
          "A_NLL":before_audit["nll"],"B_NLL":after_audit["nll"],
          "C_accepted":passed_c,"D_accepted":passed_d,
          "E_alpha":alpha,"E_accepted":passed_e,
          "C_F1":after_audit["f1"] if passed_c else before_audit["f1"],
          "D_F1":after_audit["f1"] if passed_d else before_audit["f1"],
          "E_F1":m_audit["f1"] if passed_e else before_audit["f1"],
          "C_Brier":after_audit["brier"] if passed_c else before_audit["brier"],
          "D_Brier":after_audit["brier"] if passed_d else before_audit["brier"],
          "E_Brier":m_audit["brier"] if passed_e else before_audit["brier"]
        })
        snap["scenarios"][condition]={
          "conflicts":label_conflicts(labels,old_p[batch]),
          "flipped":int(flipped.sum()),
          "ordinary_accept":passed_c,
          "prob_guard_accept":passed_d,
          "acceptance":after_accept,"audit":after_audit,
          "calibration_mixture_alpha":alpha,
          "public_checks":{"C":checks_c,"D":checks_d,"E":checks_e}
        }
    return {"snapshot":snap,"rows":results,
         "selected_features":int(len(subset)),"search_ms":search_ms,
         "fit_segments":len(old),"fit_anomalies":int(y[old].sum()),
         "source_feature_selection":"old historical group only",
         "selector_inner_cv_evaluations":sel_meta["unique_cv_wrapper_evaluations"]}


def figures(table,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for col,title,fname in [
      ("B_F1","Correct/noisy label retrain F1 by grouped fold","group_fold_F1.png"),
      ("B_Brier","Correct/noisy label retrain probability Brier by grouped fold","group_fold_Brier.png")]:
        fig,ax=plt.subplots(figsize=(10,5.8))
        for name,sub in table.groupby("condition"):
            ax.scatter(sub["fold"],sub[col],label=name)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=35,ha="right")
        ax.set_ylabel(col);ax.set_title(title);ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/fname,dpi=160);plt.close(fig)


def run(outpath):
    out=Path(outpath);out.mkdir(parents=True,exist_ok=True)
    config_bytes=SPEC.read_bytes()
    cfg=json.loads(config_bytes)
    feature_cfg=json.loads(FEATURE_PROTO.read_text())
    X,df,names,source=highdim.load_and_engineer(out,feature_cfg)
    group=channel_order_blocks(df,cfg["blocked_group"]["block_size"])
    y=df.anomaly.to_numpy(int)
    official=np.flatnonzero(df.train.to_numpy(int)==0)
    minclass={"old_fit":5,"batch":3,"cal":3,"accept":3,"audit":3}
    snapshots=[]
    allrows=[]
    skipped=[]
    fold_summary=[]
    for seed in cfg["blocked_group"]["repeat_seeds"]:
        fold=f"block_{seed}"
        try:
            parts=blocked_fold(df,group,seed)
            eligible,reason,counts=validate_fold(parts,group,y,official,minclass)
            if not eligible:raise ValueError(reason)
            result=fit_one_fold(X,df,names,parts,cfg,fold,group)
            allrows+=result["rows"]
            snapshots.append(result["snapshot"])
            fold_summary.append({"fold":fold,"type":"grouped_nearby_segments",
                      "phases":counts,"selector_count":result["selected_features"],
                      "search_ms":result["search_ms"]})
            print("FOLD",fold,json.dumps({k:counts[k]["n"] for k in counts}),
                  "sel",result["selected_features"],flush=True)
        except ValueError as exc:
            skipped.append({"fold":fold,"why":str(exc)})
            print("INFEASIBLE",fold,str(exc),flush=True)
    for channel in cfg["channel_shift"]["target_channels"]:
        fold="heldout_channel_"+channel
        try:
            parts=channel_shift_fold(df,group,channel,97)
            eligible,reason,counts=validate_fold(parts,group,y,official,minclass)
            if not eligible:raise ValueError(reason)
            result=fit_one_fold(X,df,names,parts,cfg,fold,group)
            allrows+=result["rows"]
            snapshots.append(result["snapshot"])
            fold_summary.append({"fold":fold,"type":"heldout_channel_timeproxy_embargo",
                     "phases":counts,"selector_count":result["selected_features"],
                     "search_ms":result["search_ms"],
                     "heldout_channel":channel})
            print("FOLD",fold,json.dumps({k:counts[k]["n"] for k in counts}),
                  "sel",result["selected_features"],flush=True)
        except ValueError as exc:
            skipped.append({"fold":fold,"why":str(exc)})
            print("INFEASIBLE",fold,str(exc),flush=True)
    if len(snapshots)<2:
        raise RuntimeError("Too few feasible independent grouped folds")
    # Only aggregated public baseline artifacts. Confidential candidate logic
    # will later be run OFFLINE using the snapshot data; public code does not
    # contain any private decision rule.
    table=pd.DataFrame(allrows)
    table.to_csv(out/"public_grouped_comparison.csv",index=False)
    (out/"private_replay_inputs.json").write_text(json.dumps({
      "source":"OPS-SAT REAL BINARY prior true labels; grouped internal split",
      "protocol_sha256":hashlib.sha256(config_bytes).hexdigest(),
      "research_only":True,
      "snapshots":snapshots},indent=2)+"\n",encoding="utf-8")
    results={
      "study":"Grouped OPS-SAT single-dataset ordinary update baseline (private decisions NOT included)",
      "protocol_sha256":hashlib.sha256(config_bytes).hexdigest(),
      "source":source,
      "n_complete_folds":len(snapshots),
      "n_infeasible_folds":len(skipped),
      "infeasible":skipped,
      "folds":fold_summary,
      "all_original_official_test_0_excluded":True,
      "public_summary_by_condition":{},
      "known_evidence_limitations":cfg["boundary_conditions"]
    }
    for scenario,block in table.groupby("condition"):
        good=block["B_F1"]-block["A_F1"]
        damage=block["B_Brier"]-block["A_Brier"]
        results["public_summary_by_condition"][scenario]={
          "folds":len(block),
          "ordinary_C_accepted_count":int(block.C_accepted.sum()),
          "strict_D_accepted_count":int(block.D_accepted.sum()),
          "E_nonzero_weight_count":int((block.E_alpha>0).sum()),
          "E_accepted_count":int(block.E_accepted.sum()),
          "mean_B_minus_A_F1":float(good.mean()),
          "median_B_minus_A_F1":float(good.median()),
          "mean_B_minus_A_Brier":float(damage.mean()),
          "median_B_minus_A_Brier":float(damage.median()),
          "B_F1_positive_count":int((good>0).sum()),
          "B_Brier_lower_count":int((damage<0).sum())
        }
    (out/"summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    figures(table,out)
    lines=[
      "# OPS-SAT 单真实数据集：按通道与相邻片段分组的监督更新稳定性",
      "",
      "**本研究只实现非保密普通基线及它们的真实OPS-SAT结果。任何待专利的私有候选规则均未提交本仓库。**",
      "",
      f"固定的原真实OPS-SAT数据，2,578维输入，MSF预算上限64。共完成 {len(snapshots)} 组，预设分组不可执行 {len(skipped)} 组。",
      "每组从原官方训练样本中分离旧训练、模拟新增标签、阈值/参考校准、独立验收、审计；原529条官方测试始终不用。",
      "每个通道先按segment ID排序，再以16个相邻ID条目为一组。此排序只是近邻代理，不保证卫星实际UTC时间。",
      "额外留出整个目标通道，评价合法新通道/工况上的旧模型置信冲突，不能将冲突自动定性为假标签。",
      "",
      "| 标签条件 | 完成组数 | 普通门控C接受 | 严格Brier/NLL门控D接受 | E非0混合次数 | 平均B-A F1 | 平均B-A Brier | B F1改善次数 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for case,stats in results["public_summary_by_condition"].items():
        lines.append(
         f"| {case} | {stats['folds']} | {stats['ordinary_C_accepted_count']} | "
         f"{stats['strict_D_accepted_count']} | {stats['E_nonzero_weight_count']} | "
         f"{stats['mean_B_minus_A_F1']:+.4f} | {stats['mean_B_minus_A_Brier']:+.5f} | "
         f"{stats['B_F1_positive_count']} |"
        )
    lines +=["","![F1](group_fold_F1.png)","","![Brier](group_fold_Brier.png)","",
      "研究结论以实际分组与稀疏故障类别限制为边界；小通道或异常数不足的预冻结分组如实记为infeasible，不得事后挑选表现良好的随机种子。",
      "正确标签更新被拒绝并不等于适应成功；错误标签有时被允许同样不能省略。",
      "本项目以一个真实卫星数据集为目标，不涉及跨数据集迁移或真实三级物理状态验证。",
      ""]
    (out/"RESULTS.md").write_text("\n".join(lines))
    print("FINAL",json.dumps({k:results[k] for k in
       ("n_complete_folds","n_infeasible_folds","public_summary_by_condition","infeasible")}),flush=True)
    return results


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="results/grouped_update_robustness")
    x=ap.parse_args()
    run(x.out)
