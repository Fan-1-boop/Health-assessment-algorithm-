#!/usr/bin/env python3
"""Ordinary public baseline: phase-isolated threshold vs acceptance, OPS-SAT.

Only real OPS-SAT-AD binary labels; simulated uploads/flip errors.
Fixed maximum MSF budget64; selection uses old historical train only.
This is not a novel patent mechanism: it compares ordinary nested holdouts,
non-inferiority gates and conventional predictive probability averaging.
No original official OPS-SAT train==0 samples used here.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from scipy.special import xlogy
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from research.highdim_opssat import benchmark as highdim
from research.feature_budget.run_budget import selection_with_budget
from research.hitl_update.ablation import _split
from research.opssat_final_prep.benchmark import estimator,choose_threshold,metrics,timer
from msf_mofs import fisher_scores

CONFIG=ROOT/"research/strict_acceptance/PROTOCOL_FROZEN.json"
OLD_SPLIT=ROOT/"research/hitl_update/PILOT_FROZEN.json"
FEATURE_SPEC=ROOT/"research/highdim_opssat/PROTOCOL_FROZEN.json"


def add_nll(y,p):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    y=np.asarray(y,int)
    return float(-np.mean(xlogy(y,p)+xlogy(1-y,1-p)))


def guard_quality(base,candidate,limits,probability_metrics=False):
    checks={
      "F1":candidate["f1"]>=base["f1"]-limits["max_f1_drop"],
      "AP":candidate["AP"]>=base["AP"]-limits["max_AP_drop"],
      "recall":candidate["recall"]>=base["recall"]-limits["max_recall_drop"],
      "normal_FPR":candidate["FPR"]<=base["FPR"]+limits["max_FPR_increase"]
    }
    if probability_metrics:
        checks["Brier"]=candidate["brier"]<=base["brier"]+limits["max_brier_increase"]
        checks["NLL"]=candidate["nll"]<=base["nll"]+limits["max_NLL_increase"]
    return {k:bool(v) for k,v in checks.items()},bool(all(checks.values()))


def make_metrics(y,p,threshold):
    d=metrics(y,p,threshold)
    d["nll"]=add_nll(y,p)
    return d


def alpha_from_calibration(y,p_old,p_candidate,candidates,min_gain):
    """Known prediction shrinkage baseline, not a new model or patent algorithm.

    Choose alpha using CALIBRATION data alone; never audit or independent gate.
    If gain below a strictly prespecified margin, use alpha0 and report no update.
    """
    vals=[]
    for z in candidates:
        p=(1-z)*p_old+z*p_candidate
        vals.append(float(np.mean((p-np.asarray(y))**2)))
    baseline=vals[0]
    optimum=int(np.argmin(vals))
    if baseline-vals[optimum] < min_gain:
        return 0,dict(zip(map(str,candidates),vals)),float(baseline)
    return float(candidates[optimum]),dict(zip(map(str,candidates),vals)),float(vals[optimum])


def decision_output(p_old,p_candidate,accepted):
    """On failed accept, output ORIGINAL identical array, not rounded."""
    if np.shape(p_old)!=np.shape(p_candidate):
        raise ValueError("Incompatible output shapes")
    return np.asarray(p_candidate).copy() if accepted else np.asarray(p_old).copy()


def plot(rows,out,metric,filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenarios=list(dict.fromkeys(r["scenario"] for r in rows))
    arms=["A_old","B_direct_retrain","C_conventional_gate","D_prob_quality_gate","E_calibration_shrinkage"]
    fig,ax=plt.subplots(figsize=(12,6))
    w=.17;x=np.arange(len(scenarios))
    for j,name in enumerate(arms):
        vals=[next(r[metric] for r in rows if r["arm"]==name and r["scenario"]==s)
            for s in scenarios]
        ax.bar(x+(j-2)*w,vals,w,label=name)
    ax.set_xticks(x,scenarios,rotation=13,ha="right")
    ax.set_ylabel(metric);ax.set_title("One real OPS-SAT dataset, simulated label upload faults")
    ax.grid(axis="y",alpha=.23);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/filename,dpi=175);plt.close(fig)


def run(out_path):
    out=Path(out_path);out.mkdir(parents=True,exist_ok=True)
    config_bytes=CONFIG.read_bytes()
    spec=json.loads(config_bytes)
    pilot=json.loads(OLD_SPLIT.read_text())
    feature_config=json.loads(FEATURE_SPEC.read_text())
    X,df,names,manifest=highdim.load_and_engineer(out,feature_config)
    y=df.anomaly.to_numpy(int)
    old,batch,prev_val,audit=_split(df,int(pilot["seed"]),pilot["pool_split"])
    official_test=np.flatnonzero(df.train.to_numpy(int)==0)
    assert (len(old),len(batch),len(prev_val),len(audit),len(official_test))==(876,319,239,160,529)
    assert len(set(old)|set(batch)|set(prev_val)|set(audit))==1594
    assert (set(old)|set(batch)|set(prev_val)|set(audit)).isdisjoint(set(official_test))
    cal,accept=train_test_split(prev_val,test_size=.5,
       stratify=y[prev_val],random_state=int(spec["split"]["calibration_seed"]))
    cal,accept=map(np.sort,(cal,accept))
    assert len(cal)==119 and len(accept)==120 and set(cal).isdisjoint(accept)
    assert not(set(cal)|set(accept))&set(audit)
    ts=time.perf_counter()
    ranking=np.argsort(-np.nan_to_num(fisher_scores(X[old],y[old]),nan=0.,posinf=0.),kind="stable")
    cfg=spec["msf"]
    selected,meta=selection_with_budget(X[old],y[old],ranking,
             cfg["max_budget"],seed=cfg["seed"],population=cfg["population"],
             generations=cfg["generations"],folds=2,tolerance=.025)
    selector_ms=(time.perf_counter()-ts)*1e3
    assert 32<=len(selected)<=64
    xx=X[:,selected]
    def fitold():return estimator().fit(xx[old],y[old])
    t_old=timer(fitold,int(spec["timing"]["retrain_repeats"]))
    original=fitold()
    pold=original.predict_proba(xx)[:,1]
    t_old_thresh=choose_threshold(y[cal],pold[cal])
    old_accept=make_metrics(y[accept],pold[accept],t_old_thresh)
    old_audit=make_metrics(y[audit],pold[audit],t_old_thresh)
    values=[];scenarios={}
    for name,rate in [(z,{"correct_known_labels":0,"random_flip_10pct":.10,
                     "random_flip_30pct":.30,"all_labels_flipped":1}[z]) for z in spec["label_conditions"]]:
        corrupt=np.zeros(len(batch),bool)
        rng=np.random.default_rng(20261024)
        if rate==1:corrupt[:]=True
        elif rate>0:corrupt[rng.choice(len(batch),round(rate*len(batch)),replace=False)]=True
        batch_y=y[batch].copy();batch_y[corrupt]^=1
        pool=np.r_[old,batch]
        pool_y=np.r_[y[old],batch_y]
        def fitnew():return estimator().fit(xx[pool],pool_y)
        cost=timer(fitnew,int(spec["timing"]["retrain_repeats"]))
        candidate=fitnew()
        pnew=candidate.predict_proba(xx)[:,1]
        t_new=choose_threshold(y[cal],pnew[cal])
        # Old and candidate thresholds both estimated on CAL alone.
        new_accept=make_metrics(y[accept],pnew[accept],t_new)
        checks_c,accepted_c=guard_quality(old_accept,new_accept,spec["safety_constraints"],False)
        checks_d,accepted_d=guard_quality(old_accept,new_accept,spec["safety_constraints"],True)
        alpha,losses,best_brier=alpha_from_calibration(y[cal],pold[cal],pnew[cal],
              spec["alpha_selection"]["candidates"],
              spec["alpha_selection"]["minimum_calibration_brier_gain"])
        p_mix=(1-alpha)*pold+alpha*pnew
        t_mix=choose_threshold(y[cal],p_mix[cal])
        mix_accept=make_metrics(y[accept],p_mix[accept],t_mix)
        checks_e,accepted_e=guard_quality(old_accept,mix_accept,spec["safety_constraints"],True)
        if alpha==0:accepted_e=False
        p_c=decision_output(pold[audit],pnew[audit],accepted_c)
        p_d=decision_output(pold[audit],pnew[audit],accepted_d)
        p_e=decision_output(pold[audit],p_mix[audit],accepted_e)
        if not accepted_c and not np.array_equal(p_c,pold[audit]):
            raise AssertionError("C rollback not bitwise")
        if not accepted_d and not np.array_equal(p_d,pold[audit]):
            raise AssertionError("D rollback not bitwise")
        if not accepted_e and not np.array_equal(p_e,pold[audit]):
            raise AssertionError("E rollback not bitwise")
        evaluations={
            "A_old":old_audit,
            "B_direct_retrain":make_metrics(y[audit],pnew[audit],t_new),
            "C_conventional_gate":make_metrics(y[audit],p_c,t_new if accepted_c else t_old_thresh),
            "D_prob_quality_gate":make_metrics(y[audit],p_d,t_new if accepted_d else t_old_thresh),
            "E_calibration_shrinkage":make_metrics(y[audit],p_e,t_mix if accepted_e else t_old_thresh)
        }
        scenarios[name]={
            "flipped_labels_for_stress_test":int(corrupt.sum()),
            "data_simulated_upload_not_human_review":True,
            "old_threshold_from_cal_only":float(t_old_thresh),
            "candidate_threshold_from_cal_only":float(t_new),
            "safety_acceptance_checks":checks_c,
            "safety_acceptance_passed":accepted_c,
            "quality_acceptance_checks":checks_d,
            "quality_acceptance_passed":accepted_d,
            "shrinkage":{"alpha":float(alpha),"calibration_losses":losses,"min_cal_brier":best_brier,
                  "candidate_test_gate_checks":checks_e,
                  "candidate_test_gate_passed":accepted_e,
                  "effective_update_applied":bool(alpha>0 and accepted_e)},
            "independent_acceptance":{
              "A":old_accept,"B":new_accept,"E_shrinkage":mix_accept},
            "audit":evaluations,
            "training_ms":cost
        }
        for arm,v in evaluations.items():
            values.append({"scenario":name,"arm":arm,
                 "accepted":bool({"A_old":False,"B_direct_retrain":True,
                    "C_conventional_gate":accepted_c,
                    "D_prob_quality_gate":accepted_d,
                    "E_calibration_shrinkage":accepted_e}[arm]),
                 "shrinkage_alpha":float(alpha) if arm=="E_calibration_shrinkage" else None,
                 "flipped_input_labels":int(corrupt.sum()),
                 "selector_dim":len(selected),
                 **v})
        print("SCENARIO",json.dumps({"name":name,"flip":int(corrupt.sum()),
              "C":accepted_c,"D":accepted_d,"E":accepted_e,"alpha":alpha,
              "audit_F1":{k:v["f1"] for k,v in evaluations.items()},
              "audit_Brier":{k:v["brier"] for k,v in evaluations.items()}},ensure_ascii=False),flush=True)
    pd.DataFrame(values).to_csv(out/"strict_table.csv",index=False)
    summary={
      "study":"OPS-SAT honest/contaminated update, independent threshold vs acceptance, ordinary quality gate and shrinkage",
      "exploratory_only":True,
      "protocol_sha256":hashlib.sha256(config_bytes).hexdigest(),
      "source":manifest,
      "partition":{"old_train":len(old),"upload":len(batch),"threshold_calibration":len(cal),
         "independent_acceptance":len(accept),"historical_internal_audit":len(audit),
         "official_test_fully_excluded":len(official_test),
         "cal_positives":int(y[cal].sum()),"accept_positives":int(y[accept].sum()),
         "audit_positives":int(y[audit].sum())},
      "selector":{"max_budget":64,"selected_count":len(selected),"indices":list(map(int,selected)),
         "search_ms":selector_ms,"meta":meta},
      "old":{"training_ms":t_old,"calibration_only_threshold":t_old_thresh,
              "independent_acceptance":old_accept,"audit":old_audit},
      "scenarios":scenarios,
      "notes":[
         "Ordinary baseline, not patent claim",
         "Real OPS-SAT normal/abnormal labels, not physical healthy/degraded/fault labels",
         "Previously audited internal data mean this is not a fresh unseen confirmatory study",
         "Small half-validation acceptance subset has few positive samples and correlated telemetry risk",
         "E alpha0 signifies NO REAL UPDATE; not a successful new model improvement",
         "Original 529 official holdout excluded entirely and previous official test already had prior-public exposure",
         "MSF max64 is fixed engineering budget, not statistical global optimum",
         "Threshold calibration, update acceptance, and audit have disjoint segments within this experiment",
         "No new authenticated human annotations, no approval or production deployment"
      ]}
    (out/"strict_metrics.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    for field,file in [("f1","strict_f1.png"),("brier","strict_brier.png")]:
        plot(values,out,field,file)
    lines=[
      "# OPS-SAT真实二分类：完全隔离阈值校准与候选验收的常规强基线",
      "",
      "**这些都是常规标准工程方法，不包含私有专利候选机制。原始官方测试完全排除，不构成新的盲态确认实验。**",
      "",
      f"样本：旧训练{len(old)}、上传模拟{len(batch)}、阈值校准{len(cal)}、独立候选验收{len(accept)}、内部审计{len(audit)}；原始官方测试529未使用。",
      f"MSF固定最大64预算，独立从旧训练学得{len(selected)}个特征；特征搜索{selector_ms:.1f}ms，旧模型训练{t_old['median_ms']:.1f}ms。",
      "",
      "| 标签条件 | 普通门控C接受 | 含Brier/NLL门控D接受 | 保守概率混合E系数 | E实际更新 | 审计A F1 | 审计B F1 | 审计C F1 | 审计D F1 | 审计E F1 | 审计A Brier | 审计E Brier |",
      "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k,r in scenarios.items():
        a=r["audit"]
        lines.append(
         f"| {k} | {r['safety_acceptance_passed']} | {r['quality_acceptance_passed']} | "
         f"{r['shrinkage']['alpha']:.2f} | {r['shrinkage']['effective_update_applied']} | "
         + " | ".join(f"{a[z]['f1']:.4f}" for z in
           ("A_old","B_direct_retrain","C_conventional_gate",
            "D_prob_quality_gate","E_calibration_shrinkage"))
         +f" | {a['A_old']['brier']:.4f} | {a['E_calibration_shrinkage']['brier']:.4f} |")
    lines+=["",
      "![F1](strict_f1.png)","",
      "![Brier](strict_brier.png)","",
      "**解读规则：** 拒绝候选后维持旧模型不是「新模型更新成功」，且过度保守导致正确更新全部拒绝，也不应视为完成受控适应。"
      " 独立验收防止把同一批样本同时用于阈值与接受判断，但小样本仍可能导致波动。",
      " 所有真实标签均为官方原有二分类标签，新增人工审核是模拟上传，三级状态+HMM只能由既有模拟状态实验支撑。"
    ]
    (out/"RESULTS.md").write_text("\n".join(lines))
    return summary


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="results/strict_acceptance")
    opts=ap.parse_args()
    run(opts.out)
