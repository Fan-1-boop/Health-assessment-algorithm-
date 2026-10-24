#!/usr/bin/env python3
"""Strictly phase-isolated public supervised update + rollback study.

REAL HUMAN REVIEW: none. PHYSICAL SATELLITE FAULT TRUTH: none.
Only one fixed original simulator. No proprietary patent algorithm is present.
Frozen delivery artifacts are loaded read-only; all refits stay in RAM.

Avoid prior MVP's state-calibrator/gate overlap:
* missions0-1: MSF-selected feature IDs and supervised base training
* cycle8: NEW state Platt calibration and independent anomaly threshold
* cycle9: new normal/anomaly label batch for candidate RF only
* cycle10: old/new identical fusion supervised training data
* cycle11: untouched phase-independent gate
* mission3: descriptive audit, never training, calibration or gating

The frozen MSF indices were selected from missions0-1 only, so cycle8 also
remains independent of feature selection, unlike our prior v0.1 audit.
"""
from __future__ import annotations
import argparse,hashlib,io,json,sys,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score,f1_score,precision_score,
    recall_score,roc_auc_score)
from scipy.special import logit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"mvp"))
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"research/hitl_update"))
from synthetic import generate,make_windows,highdim_features
from v13_state import fit_hierarchical,predict_hierarchical,filter_groups,evaluation
from gate import canonical_csv,validate_supervised_upload

SPEC=ROOT/"research/isolated_update_rollback/PROTOCOL_FROZEN.json"
DELIVERY=ROOT/"results/mvp"
DEFAULT_SCORE_WEIGHTS=np.array([100.,60.,0.])


def stacking_inputs(state_probs,anom_prob):
    p=np.clip(np.asarray(state_probs,float),1e-6,1)
    a=np.clip(np.asarray(anom_prob,float),1e-6,1-1e-6)
    return np.column_stack([np.log(p),logit(a)])


def threshold_F1(y,score):
    from sklearn.metrics import precision_recall_curve
    pre,rec,thresholds=precision_recall_curve(y,score)
    scores=2*pre[:-1]*rec[:-1]/np.maximum(pre[:-1]+rec[:-1],1e-10)
    if not len(thresholds):raise ValueError("No threshold")
    return float(thresholds[int(np.argmax(scores))])


def fit_fusion(state_post,anom_probs,state_labels,indices):
    p=state_post[indices]
    x=stacking_inputs(p,anom_probs[indices])
    y=state_labels[indices]
    assert set(y)=={0,1,2}
    mdl=LogisticRegression(C=.5,max_iter=1200,random_state=42)
    mdl.fit(x,y)
    assert list(mdl.classes_)==[0,1,2]
    return mdl


def probability_js(p,q):
    p=np.clip(np.asarray(p,float),1e-9,1)
    q=np.clip(np.asarray(q,float),1e-9,1)
    p=p/p.sum(axis=1,keepdims=True)
    q=q/q.sum(axis=1,keepdims=True)
    m=.5*(p+q)
    return float(.5*np.mean(np.sum(p*np.log(p/m),axis=1)+np.sum(q*np.log(q/m),axis=1)))


def eval_arms(state_true,anom_true,score_true,raw_anom,thr,state_prob,fusion,idx):
    pa=fusion.predict_proba(stacking_inputs(state_prob[idx],raw_anom[idx]))
    return evaluate_probs(state_true[idx],anom_true[idx],score_true[idx],
                          raw_anom[idx],thr,pa)


def evaluate_probs(ys,ya,score_truth,d,thr,p):
    ys=np.asarray(ys,int);ya=np.asarray(ya,int)
    d=np.asarray(d,float);p=np.asarray(p,float)
    if p.shape!=(len(ys),3) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=1),1):
        raise ValueError("Invalid three-state model probability interface")
    decision=d>=thr
    es=evaluation(ys,p)
    return {
       "state_macro_f1":es["macro_f1"],
       "state_brier":es["brier"],
       "state_nll":es["nll"],
       "state_recalls":es["recalls"],
       "anomaly_f1":float(f1_score(ya,decision,zero_division=0)),
       "anomaly_AP":float(average_precision_score(ya,d)),
       "anomaly_recall":float(recall_score(ya,decision,zero_division=0)),
       "anomaly_precision":float(precision_score(ya,decision,zero_division=0)),
       "anomaly_normal_fpr":float(np.mean(decision[ya==0])) if np.any(ya==0) else None,
       "anomaly_FP":int(np.sum(decision & (ya==0))),
       "anomaly_FN":int(np.sum((~decision)&(ya==1))),
       "score_default_weights_MAE":float(np.mean(np.abs(p@DEFAULT_SCORE_WEIGHTS-score_truth))),
       "threshold":float(thr)
    }


def stage_accept(base,update,rules):
    checks={
       "three_state_macro_f1":bool(update["state_macro_f1"]>=base["state_macro_f1"]-rules["max_state_macro_f1_drop"]),
       "three_state_Brier":bool(update["state_brier"]<=base["state_brier"]+rules["max_state_Brier_increase"]),
       "three_state_NLL":bool(update["state_nll"]<=base["state_nll"]+rules["max_state_NLL_increase"]),
       "anomaly_recall":bool(update["anomaly_recall"]>=base["anomaly_recall"]-rules["max_anomaly_recall_drop"]),
       "anomaly_normal_FPR":bool(update["anomaly_normal_fpr"]<=base["anomaly_normal_fpr"]+rules["max_anomaly_FPR_increase"]),
       "default_score_MAE":bool(update["score_default_weights_MAE"]<=base["score_default_weights_MAE"]+rules["max_health_MAE_increase"])
    }
    return checks,bool(all(checks.values()))


def stage_select(base_candidate_proba,updated_candidate_proba,passed):
    """No production action; on rejection return exact unchanged old array."""
    old=np.asarray(base_candidate_proba)
    new=np.asarray(updated_candidate_proba)
    if old.shape!=new.shape:raise ValueError("Shape mismatch")
    return new.copy() if passed else old.copy()


def timeit(fn,n):
    fn();v=[]
    for _ in range(n):
        st=time.perf_counter()
        fn()
        v.append((time.perf_counter()-st)*1e3)
    return {"median_ms":float(np.median(v)),"q25_ms":float(np.quantile(v,.25)),
            "q75_ms":float(np.quantile(v,.75)),"runs":int(n)}


def checked_upload(raw,indices,labels,source_tag):
    """Simulated reviewed batch, always forbidden for real release."""
    names=[f"raw_window_{i:03d}" for i in range(raw.shape[1])]
    frame=pd.DataFrame(raw[indices],columns=names)
    frame.insert(0,"label",labels)
    frame.insert(0,"sample_id",[f"sim_cycle9_index_{k}" for k in indices])
    reviewed_sha=hashlib.sha256(canonical_csv(frame)).hexdigest()
    m={
      "batch_id":"DEMO_"+source_tag,
      "target_task":"anomaly_binary",
      "label_source":"SIMULATOR_"+source_tag,
      "review_status":"simulated_review",
      "reviewer_id":"SIMULATED_NOT_AUTHENTICATED",
      "reviewed_at":"2026-10-16T00:00:00Z",
      "parent_model_version":"frozen-synthetic-v1",
      "data_sha256":reviewed_sha,
      "approval_status":"pending",
      "data_unit":"annotated_segment","demo_only":True
    }
    result=validate_supervised_upload(frame,m,names,
               expected_parent_model="frozen-synthetic-v1",
               allow_simulated_review=True)
    if not (result.candidate_allowed and not result.release_allowed):
        raise AssertionError("Simulated upload unexpectedly authorized production")
    return {"digest":reviewed_sha,"task":result.route,
            "demo_only":result.demo_only,"production_release_allowed":result.release_allowed,
            "rows":len(indices),"claimed_review_is_authenticated":False}


def chart(report,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys=["honest_simulated_update","incorrect_label_batch","semantic_probability_channel_fault"]
    label=["A baseline","B old fusion","C updated fusion","D gated"]
    fields=[("state_brier","Brier (smaller better)","state_brier_and_rollback.png"),
            ("state_macro_f1","Macro-F1 (larger better)","macro_f1_and_rollback.png")]
    for f,ylabel,file in fields:
        fig,axs=plt.subplots(1,3,figsize=(15,5.5),sharey=True)
        for ax,s in zip(axs,keys):
            a=report["scenarios"][s]["audit_mission3"]
            values=[a[k][f] for k in ("A","B","C","D")]
            ax.bar(range(4),values)
            ax.set_xticks(range(4),label,rotation=35,ha="right")
            ax.set_title(s.replace("_"," "),fontsize=10)
            ax.grid(alpha=.2,axis="y")
            if s!="honest_simulated_update":
                ax.text(.05,.95,"fault injected",transform=ax.transAxes,va="top",fontsize=9)
        axs[0].set_ylabel(ylabel)
        fig.suptitle("Ordinary gated update/rollback, synthetic satellite states only")
        fig.tight_layout()
        fig.savefig(out/file,dpi=170);plt.close(fig)


def run(out_path):
    out=Path(out_path);out.mkdir(parents=True,exist_ok=True)
    frozen=joblib.load(DELIVERY/"model_bundle.joblib")
    old_metrics=json.loads((DELIVERY/"metrics.json").read_text())
    if frozen["spec_sha256"]!=old_metrics["protocol_sha256"]:
        raise RuntimeError("Frozen reference artifact checksum mismatch")
    raw_config=SPEC.read_bytes();config=json.loads(raw_config)
    win,table=make_windows(generate())
    x,names=highdim_features(win)
    if names!=frozen["feature_names"]:raise RuntimeError("400D frozen schema changed")
    raw=win.reshape(len(win),-1)
    mission=table.mission_id.to_numpy(int)
    cycle=table.cycle_id.to_numpy(int)
    y=table.health_state_simulated.to_numpy(int)
    abnormal=table.anomaly_simulated.to_numpy(int)
    health_score=table.operator_score_simulated.to_numpy(float)
    phases={
      "model_fit":np.flatnonzero(mission<=1),
      "state_calibration_and_anomaly_threshold":np.flatnonzero(cycle==8),
      "update_batch":np.flatnonzero(cycle==9),
      "fusion_train":np.flatnonzero(cycle==10),
      "acceptance_gate":np.flatnonzero(cycle==11),
      "mission3_audit":np.flatnonzero(mission==3)
    }
    cfg=config["dataset_partition"]
    counts={k:len(v) for k,v in phases.items()}
    assert counts=={"model_fit":544,"state_calibration_and_anomaly_threshold":68,
           "update_batch":68,"fusion_train":68,"acceptance_gate":68,"mission3_audit":272}
    assert len(set(map(int,np.concatenate(list(phases.values())))))==len(table)
    for k,indices in phases.items():
        assert set(np.unique(y[indices]))=={0,1,2},f"Missing state in phase {k}"
        assert set(np.unique(abnormal[indices]))=={0,1},f"Missing anomaly in phase {k}"
    fit=phases["model_fit"];cal=phases["state_calibration_and_anomaly_threshold"]
    upload=phases["update_batch"];fusion_fit_idx=phases["fusion_train"]
    gate_idx=phases["acceptance_gate"];audit_idx=phases["mission3_audit"]
    frozen_selections=(np.asarray(frozen["stage1_selector"],int),np.asarray(frozen["stage2_selector"],int))
    if (len(frozen_selections[0]),len(frozen_selections[1]))!=(50,10):
        raise RuntimeError("Fixed selector must be 50,10")
    # Rebuild state classifier/Platt on genuinely phase-disjoint train and cal
    # while retaining original train-only MSF selected feature indices.
    hierarchy=fit_hierarchical(x[fit],y[fit],cycle[fit],x[cal],y[cal],
                   *frozen_selections)
    raw_post=predict_hierarchical(hierarchy,x)
    post=filter_groups(hierarchy,raw_post,cycle,gamma=.6,memory=.5)

    original_forest=frozen["anomaly_detector"]
    probs_old=original_forest.predict_proba(raw)[:,1]
    thr_old=threshold_F1(abnormal[cal],probs_old[cal])
    print("PHASE_COUNTS",json.dumps(counts),flush=True)
    out_report={
       "experiment":config["version"],
       "simulated_only":True,
       "real_operator_verified":False,
       "model_production_deployed":False,
       "protocol_sha256":hashlib.sha256(raw_config).hexdigest(),
       "frozen_model_spec":frozen["spec_sha256"],
       "phases":counts,
       "feature_selectors":{"stage1":50,"stage2":10,
          "unique":len(set(frozen_selections[0])|set(frozen_selections[1])),
          "selected_only_from_base_fit_missions":True},
       "state_calibration_and_HMM_group_check":"Platt: cycle8 only; T: model_fit mission0/1 only; all HMM groups reset at cycle boundaries",
       "scenario_definitions":config["scenarios"],
       "scenarios":{},
       "limitations":[
          "Whole dataset is simulated; human-reviewed labels are only simulated manifests",
          "All earlier synthetic mission3 results have been inspected. Audit is a disjoint phase within this run, but not a fresh unseen test",
          "State feature selector indices are reused from frozen MVP selected on missions0/1 labels only",
          "True calibration independence here: cycle8 is not included in original MSF search, state classifier fit, fusion fit or validation gate",
          "Cycle11 gate is excluded from all model fitting, selector, probability calibrator, fusion fitting and alarm threshold estimation",
          "Original RF trained missions0/1 (not recalibrated), kept frozen as A",
          "Injected 0/1 label reversal and class-column swap are negative engineering tests, not physically generated fault behavior",
          "Quality checks reject adverse candidates but cannot cryptographically verify a human's diagnosis",
          "Original on-orbit three-state truth unavailable; no real digital-twin field validation",
          "Approval remains pending; D is staging only, not a deployed model",
          "Public baseline uses ordinary re-training and regression gates, no proprietary/patent-specific new mechanism"
       ]
    }

    base_reference_fusion=fit_fusion(post,probs_old,y,fusion_fit_idx)
    time_reference_fusion=timeit(lambda:fit_fusion(post,probs_old,y,fusion_fit_idx),
                          config["timing"]["fusion_fit_repeats"])
    for scenario in ("honest_simulated_update","incorrect_label_batch",
                     "semantic_probability_channel_fault"):
        defective_labels=(1-abnormal[upload]) if scenario=="incorrect_label_batch" else abnormal[upload]
        gate_meta=checked_upload(raw,upload,defective_labels,scenario)
        def make_candidate():
            m=clone(original_forest)
            pool=np.r_[fit,upload]
            labels=np.r_[abnormal[fit],defective_labels]
            m.fit(raw[pool],labels)
            return m
        train_times=timeit(make_candidate,config["timing"]["RF_retrain_repeats"])
        detector=make_candidate()
        probs_new=detector.predict_proba(raw)[:,1]
        thr_new=threshold_F1(abnormal[cal],probs_new[cal])
        fusion_candidate=fit_fusion(post,probs_new,y,fusion_fit_idx)
        fusion_times=timeit(lambda:fit_fusion(post,probs_new,y,fusion_fit_idx),
                            config["timing"]["fusion_fit_repeats"])
        def for_phase(indices):
            a=base_reference_fusion.predict_proba(stacking_inputs(post[indices],probs_old[indices]))
            b=base_reference_fusion.predict_proba(stacking_inputs(post[indices],probs_new[indices]))
            c=fusion_candidate.predict_proba(stacking_inputs(post[indices],probs_new[indices]))
            if scenario=="semantic_probability_channel_fault":
                # Swap healthy/fault at consumer interface; valid normalization
                # but WRONG semantic labels. Checks must reject at gate.
                c=c[:,[2,1,0]]
            return a,b,c
        ga,gb,gc=for_phase(gate_idx)
        gate={
          "A":evaluate_probs(y[gate_idx],abnormal[gate_idx],health_score[gate_idx],
                  probs_old[gate_idx],thr_old,ga),
          "B":evaluate_probs(y[gate_idx],abnormal[gate_idx],health_score[gate_idx],
                  probs_new[gate_idx],thr_new,gb),
          "C":evaluate_probs(y[gate_idx],abnormal[gate_idx],health_score[gate_idx],
                  probs_new[gate_idx],thr_new,gc)
        }
        checks,approved_for_stage=stage_accept(gate["A"],gate["C"],
                                         config["regression_rules"])
        chosen="C" if approved_for_stage else "A"
        gd=stage_select(ga,gc,approved_for_stage)
        if not approved_for_stage and not np.array_equal(gd,ga):
            raise AssertionError("Gate rejection must return same original probabilities")
        gate["D"]=gate[chosen]
        aa,ab,ac=for_phase(audit_idx)
        a={
          "A":evaluate_probs(y[audit_idx],abnormal[audit_idx],health_score[audit_idx],
                  probs_old[audit_idx],thr_old,aa),
          "B":evaluate_probs(y[audit_idx],abnormal[audit_idx],health_score[audit_idx],
                  probs_new[audit_idx],thr_new,ab),
          "C":evaluate_probs(y[audit_idx],abnormal[audit_idx],health_score[audit_idx],
                  probs_new[audit_idx],thr_new,ac)
        }
        dd=stage_select(aa,ac,approved_for_stage)
        audit_exact_rollback=bool(np.array_equal(aa,dd))
        if not approved_for_stage and not audit_exact_rollback:
            raise AssertionError("Independent audit predictions changed despite rollback")
        a["D"]=a[chosen]
        # Test only the explicitly pre-registered probability class mismatch
        # is rejected. It is a deterministic protocol fault, not an algorithm.
        if scenario=="semantic_probability_channel_fault" and approved_for_stage:
            raise AssertionError("Semantic channel fault escaped safety gate")
        report={
          "review":gate_meta,
          "candidate_train_time":train_times,
          "fusion_train_time":fusion_times,
          "old_fusion_train_time":time_reference_fusion,
          "thresholds":{"base":thr_old,"candidate":thr_new},
          "independent_gate":gate,
          "checks":checks,
          "passed_regression_gate":approved_for_stage,
          "staged_selected":chosen,
          "audit":a,
          "rejection_rollback_predictions_bitwise_identical":audit_exact_rollback if not approved_for_stage else None,
          "state_probability_JS_vs_A_on_audit":{
            "B_stale":probability_js(aa,ab),
            "C_new":probability_js(aa,ac)},
          "test_labels_used_for_selection":False,
          "production_deployed":False
        }
        out_report["scenarios"][scenario]=report
        print("SCENARIO",json.dumps({
          "scenario":scenario,"selected":chosen,"gate_pass":approved_for_stage,
          "gate_checks":checks,"audit_A":{k:a["A"][k] for k in
                ("state_macro_f1","state_brier","state_nll")},
          "audit_C":{k:a["C"][k] for k in
                ("state_macro_f1","state_brier","state_nll")},
          "audit_D":{k:a["D"][k] for k in
                ("state_macro_f1","state_brier","state_nll")},
          "bitwise_rollback":report["rejection_rollback_predictions_bitwise_identical"]
        },flush=True))
    (out/"metrics.json").write_text(json.dumps(out_report,ensure_ascii=False,indent=2)+"\n")
    rows=[]
    for scenario,res in out_report["scenarios"].items():
        for arm,metric in res["audit"].items():
            rows.append({"scenario":scenario,"arm":arm,
              "selected_by_validation":res["staged_selected"],
              "accepted_gate":res["passed_regression_gate"],
              **{k:v for k,v in metric.items() if k not in ("state_recalls","threshold")}})
    pd.DataFrame(rows).to_csv(out/"comparison.csv",index=False)
    make_report(out_report,out)
    chart(out_report,out)
    return out_report


def make_report(data,out):
    lines=[
      "# 严格阶段隔离的模拟受控更新与真实回退执行 v1",
      "",
      "**该实验不涉及真实卫星故障标签、真实人工审核和生产发布。**",
      "",
      "本次消除了旧对照中 **原三级状态校准器提前使用候选验收周期标签** 的问题。"
      "原MSF特征选择只看 missions0/1；两级分类器训练也用missions0/1；"
      "本次状态Platt校准与异常阈值只使用cycle8；cycle9模拟新标签更新异常RF；"
      "cycle10专门拟合新旧融合器；cycle11专用于候选完整系统验收；mission3仅描述性审计。",
      "",
      "所有健康、退化、故障、异常和人工健康分仍由模拟器生成，且此前同一 mission3 的结果已公开，不能称为首次独立的确认性测试。",
      "",
      "三类实验：可信模拟标签、**故意反转的错误异常标签**、"
      "**故意把三级概率输出的健康/故障列交换**。最后一种只检验故障注入下的拒绝与回退能力。",
      "",
      "| 场景 | 方法 | Macro-F1↑ | Brier↓ | NLL↓ | 异常Recall↑ | 正常FPR↓ | 健康分MAE↓ |",
      "|---|---|---:|---:|---:|---:|---:|---:|"
    ]
    chinese={
      "honest_simulated_update":"模拟正确标签",
      "incorrect_label_batch":"模拟错误标签反转",
      "semantic_probability_channel_fault":"融合类别语义通道错位"
    }
    for scenario,val in data["scenarios"].items():
        for arm,q in val["audit"].items():
            lines.append(f"| {chinese[scenario]} | {arm} | {q['state_macro_f1']:.4f} | "
                f"{q['state_brier']:.5f} | {q['state_nll']:.5f} | "
                f"{q['anomaly_recall']:.4f} | {q['anomaly_normal_fpr']:.4f} | "
                f"{q['score_default_weights_MAE']:.3f} |")
    lines+=["",
      "## 候选验收与回退",
      ""
    ]
    for scenario,val in data["scenarios"].items():
        lines.append(f"- {chinese[scenario]}：候选C验收通过={val['passed_regression_gate']}；"
            f"最终暂存={val['staged_selected']}；"
            f"拒绝后预测与旧模型逐字节一致={val['rejection_rollback_predictions_bitwise_identical']}；"
            f"验收项={val['checks']}。")
    lines+=["",
      "![不同更新/故障情形的Brier与回退](state_brier_and_rollback.png)",
      "",
      "![不同更新/故障情形的Macro-F1与回退](macro_f1_and_rollback.png)",
      "",
      "## 科研边界与下一步",
      "",
      "- 数据和任务属于同一固定模拟集合，不属于迁移学习。真实OPS-SAT只有异常二分类，不能直接宣称三级健康验证。",
      "- 候选只使用cycle9的二分类标签训练；没有用它伪造三级健康标签。",
      "- 质量数据门控最多检查元数据、摘要完整性、合法范围和样本隔离，无法证明模拟错误标签一定被发现；最终需要有真实审核身份和可靠物理标签。",
      "- 验收规则属于常规性能门控；不等于已实现发明专利核心算法。",
      "- \`D\` 是实际运行的内存模型输出选择，模拟了原子回退语义，但**没有部署/切换GitHub交付分支或真实生产系统**。",
      "- 原始合成样本的真实状态过于容易识别，各对照可能出现天花板效应；应如实报告负面/无差异结果。",
      "- 下一阶段若设计新的专利候选机制，必须比普通融合重训+独立门控有可复现的额外收益；任何专利专有公式不在此公开仓库公布。",
      ""
    ]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/isolated_update_rollback")
    opts=parser.parse_args()
    run(opts.out)
