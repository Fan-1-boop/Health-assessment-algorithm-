#!/usr/bin/env python3
"""One-command, single-synthetic-dataset satellite health MVP.

Usage: python mvp/run.py --out results/mvp
The dataset, all labels and all operator score corrections are SIMULATED.
Nothing here demonstrates validated physical spacecraft fault diagnostics.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,roc_auc_score,precision_recall_curve,
    f1_score,accuracy_score,recall_score,confusion_matrix,precision_score)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"mvp"))
from synthetic import generate,make_windows,highdim_features,CHANNELS,SEED
from v13_state import (select_grouped_msf,fit_hierarchical,
                       predict_hierarchical,filter_groups,evaluation)

DEFAULT_WEIGHTS=np.array([100.,60.,0.])
RNG=np.random.default_rng(SEED+9)

def _scores(p):
    return np.asarray(p,float)

def stacking_inputs(prob,anomaly):
    prob=np.clip(np.asarray(prob,float),1e-6,1)
    q=logit(np.clip(np.asarray(anomaly,float),1e-6,1-1e-6))
    return np.column_stack([np.log(prob),q])

def learn_mapping(prob,score):
    """Ordered 0<=F<=D<=H<=100 and L2 prior (human corrections only)."""
    prob=np.asarray(prob,float);score=np.asarray(score,float)
    def obj(w):
        e=prob@w-score
        return float(np.mean(e*e)+.01*np.sum((w-DEFAULT_WEIGHTS)**2))
    result=minimize(obj,DEFAULT_WEIGHTS.copy(),method="SLSQP",bounds=[(0,100)]*3,
        constraints=[{"type":"ineq","fun":lambda w:w[0]-w[1]},
                     {"type":"ineq","fun":lambda w:w[1]-w[2]}],
        options={"maxiter":1500,"ftol":1e-10})
    if not result.success:raise RuntimeError("Ordered feedback calibration failed: "+result.message)
    return np.asarray(result.x,float)

def choose_threshold(scores,y):
    precision,recall,threshold=precision_recall_curve(y,scores)
    if len(threshold)==0:raise ValueError("No anomaly thresholds")
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-10)
    return float(threshold[int(np.argmax(f1))])

def anomaly_report(y,score,threshold):
    pred=(np.asarray(score)>=threshold).astype(int)
    return {"n":len(y),"ap":float(average_precision_score(y,score)),
        "auroc":float(roc_auc_score(y,score)),
        "f1":float(f1_score(y,pred,zero_division=0)),
        "precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),
        "false_alarms":int(np.sum((pred==1)&(np.asarray(y)==0))),
        "missed_anomaly_windows":int(np.sum((pred==0)&(np.asarray(y)==1))),
        "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec=(ROOT/"mvp/SPEC_FROZEN.json").read_bytes()
    data=generate()
    raw=data.copy()
    raw["timestamp"]=raw["timestamp"].astype(str)
    raw.to_csv(out/"synthetic_satellite_telemetry.csv",index=False)
    windows,table=make_windows(data)
    X,feature_names=highdim_features(windows)
    if X.shape[1]!=400:raise AssertionError("Unexpected feature dimensions")
    mission=table.mission_id.to_numpy(int)
    cycle=table.cycle_id.to_numpy(int)
    state=table.health_state_simulated.to_numpy(int)
    anomaly=table.anomaly_simulated.to_numpy(int)
    train=mission<=1
    calibrate=mission==2
    test=mission==3
    assert np.all(train|calibrate|test)
    assert set(np.unique(state[train]))==set(np.unique(state[calibrate]))==set(np.unique(state[test]))=={0,1,2}
    assert len(set(cycle[train])&set(cycle[calibrate]))==0
    assert len(set(cycle[train])&set(cycle[test]))==0
    assert len(set(cycle[calibrate])&set(cycle[test]))==0

    # Two independently searched binary tasks:
    # healthy/unhealthy and degraded/fault. MSF-MOFS only sees TRAIN labels.
    group=cycle[train]
    s1,sel1=select_grouped_msf(X[train],(state[train]!=0).astype(int),
            group,population=8,generations=3,seed=42)
    unhealthy=train&(state!=0)
    s2,sel2=select_grouped_msf(X[unhealthy],(state[unhealthy]==2).astype(int),
            cycle[unhealthy],population=8,generations=3,seed=43)
    h=fit_hierarchical(X[train],state[train],cycle[train],
                       X[calibrate],state[calibrate],s1,s2)
    p_static_cal=predict_hierarchical(h,X[calibrate])
    p_static_test=predict_hierarchical(h,X[test])
    p_base_cal=filter_groups(h,p_static_cal,cycle[calibrate],gamma=.6,memory=.5)
    p_base_test=filter_groups(h,p_static_test,cycle[test],gamma=.6,memory=.5)

    # Anomaly branch operates on RAW time series windows, never MSF features.
    raw_train=windows[train].reshape(int(train.sum()),-1)
    raw_cal=windows[calibrate].reshape(int(calibrate.sum()),-1)
    raw_test=windows[test].reshape(int(test.sum()),-1)
    detector=RandomForestClassifier(n_estimators=120,max_depth=14,
              min_samples_leaf=2,class_weight="balanced_subsample",
              random_state=42,n_jobs=2).fit(raw_train,anomaly[train])
    d_cal=detector.predict_proba(raw_cal)[:,1]
    d_test=detector.predict_proba(raw_test)[:,1]
    threshold=choose_threshold(d_cal,anomaly[calibrate])

    # Transparent, simple probability-level late fusion: trained only on
    # calibration mission; no parameter chosen from final mission.
    fusion=LogisticRegression(C=.5,max_iter=1200,random_state=42)
    fusion.fit(stacking_inputs(p_base_cal,d_cal),state[calibrate])
    p_fused_test=fusion.predict_proba(stacking_inputs(p_base_test,d_test))
    p_fused_cal=fusion.predict_proba(stacking_inputs(p_base_cal,d_cal))
    if list(fusion.classes_)!=[0,1,2]:raise AssertionError("fusion classes mismatch")

    # 72 simulated human scores are supplied from CALIBRATION mission only.
    feedback_indices=np.sort(RNG.choice(np.flatnonzero(calibrate),size=72,replace=False))
    feedback_local=np.searchsorted(np.flatnonzero(calibrate),feedback_indices)
    operator_scores=table.operator_score_simulated.to_numpy(float)
    user_weights=learn_mapping(p_fused_cal[feedback_local],operator_scores[feedback_indices])
    predicted_scores=p_fused_test@user_weights
    initial_scores=p_fused_test@DEFAULT_WEIGHTS
    test_truth=operator_scores[test]
    feedback=[{"sample_id":int(i),"health_score_entered":round(float(operator_scores[i]),2),
               "source":"SIMULATED_OPERATOR","health_label_for_feedback":"SIMULATED",
               "model_stage":"calibration","confirmed_physical_fault":False}
              for i in feedback_indices]
    (out/"simulated_feedback.jsonl").write_text(
        "".join(json.dumps(e,ensure_ascii=False)+"\n" for e in feedback),encoding="utf-8")

    testrows=table.loc[test,["mission_id","sample_in_mission","cycle_id",
          "health_state_simulated","anomaly_simulated","benign_regime_event",
          "operator_score_simulated"]].copy().reset_index(drop=True)
    testrows["p_healthy"]=p_fused_test[:,0]
    testrows["p_degraded"]=p_fused_test[:,1]
    testrows["p_fault"]=p_fused_test[:,2]
    testrows["p_anomaly"]=d_test
    testrows["anomaly_flag"]=(d_test>=threshold).astype(int)
    testrows["health_score_predicted"]=predicted_scores
    testrows["state_predicted"]=p_fused_test.argmax(axis=1)
    testrows.to_csv(out/"demo_predictions.csv",index=False)

    result={
      "title":"Single synthetic satellite dataset, MVP end-to-end demonstration",
      "scope":"SIMULATION_ONLY; no verified satellite fault truth",
      "dataset":{"raw_rows":len(data),"sensor_channels":len(CHANNELS),
        "feature_dimensions":X.shape[1],"train_windows":int(train.sum()),
        "calibration_windows":int(calibrate.sum()),"test_windows":int(test.sum()),
        "state_train_counts":np.bincount(state[train],minlength=3).tolist(),
        "state_test_counts":np.bincount(state[test],minlength=3).tolist(),
        "simulated_anomaly_test_windows":int(anomaly[test].sum()),
        "seed":SEED},
      "msf_mofs":{"task1_healthy_vs_unhealthy":sel1,
        "task2_degraded_vs_fault":sel2,
        "feature_count_task1":int(len(s1)),
        "feature_count_task2":int(len(s2)),
        "features_union":int(len(set(s1)|set(s2))),
        "chosen_stage1_features":[feature_names[i] for i in s1],
        "chosen_stage2_features":[feature_names[i] for i in s2]},
      "base_static_state_metrics":evaluation(state[test],p_static_test),
      "base_hmm_state_metrics":evaluation(state[test],p_base_test),
      "anomaly_detector":{"model":"RandomForest_raw_window","threshold_from_calibration":threshold,
           "test":anomaly_report(anomaly[test],d_test,threshold)},
      "late_fusion_state_metrics":evaluation(state[test],p_fused_test),
      "simulated_feedback":{"n_calibration_ratings":len(feedback_indices),
          "default_score_weights":DEFAULT_WEIGHTS.tolist(),
          "corrected_score_weights":[float(v) for v in user_weights],
          "default_score_test_mae":float(np.mean(np.abs(initial_scores-test_truth))),
          "corrected_score_test_mae":float(np.mean(np.abs(predicted_scores-test_truth)))},
      "protocol_sha256":hashlib.sha256(spec).hexdigest(),
      "important_limitations":["Every state, fault, anomaly, operating mode and feedback score is synthetic.",
          "Train/calibration/test are distinct missions generated by the SAME simulator distribution.",
          "MSF-MOFS is the repository's LogisticRegression NSGA-II engineering adaptation, not a full paper reproduction.",
          "Fusion and calibration both fit on the calibration mission; this is a demo, not a final research evaluation.",
          "No claims of real satellite operating performance or maintenance efficiency."]}
    (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else float(x))+"\n",encoding="utf-8")
    artifact={"stage1_selector":s1,"stage2_selector":s2,"hierarchical":h,
       "anomaly_detector":detector,"anomaly_threshold":threshold,"fusion":fusion,
       "default_score_mapping":DEFAULT_WEIGHTS,"corrected_score_mapping":user_weights,
       "feature_names":feature_names,"raw_sensor_channels":CHANNELS,
       "spec_sha256":result["protocol_sha256"]}
    joblib.dump(artifact,out/"model_bundle.joblib",compress=3)
    summary=(
f"# 卫星健康度算法 MVP（全模拟数据）实验结果\n\n"
f"**声明：所有健康状态、故障、异常与操作员校正分数均由程序模拟，非真实卫星故障验证。**\n\n"
f"单数据集：{len(data)} 条遥测、{len(CHANNELS)} 个原始通道、{X.shape[1]} 个工程特征；"
f"训练/校准/测试窗口：{int(train.sum())}/{int(calibrate.sum())}/{int(test.sum())}。\n\n"
f"| 模块 | 测试结果 |\n|---|---:|\n"
f"| 两级静态分类 Macro-F1 | {result['base_static_state_metrics']['macro_f1']:.4f} |\n"
f"| 两级分类 + HMM Macro-F1 | {result['base_hmm_state_metrics']['macro_f1']:.4f} |\n"
f"| 独立异常检测 AP | {result['anomaly_detector']['test']['ap']:.4f} |\n"
f"| 独立异常检测 F1 | {result['anomaly_detector']['test']['f1']:.4f} |\n"
f"| 后期概率融合 Macro-F1 | {result['late_fusion_state_metrics']['macro_f1']:.4f} |\n"
f"| 默认映射健康分 MAE | {result['simulated_feedback']['default_score_test_mae']:.3f} |\n"
f"| 模拟校正健康分 MAE | {result['simulated_feedback']['corrected_score_test_mae']:.3f} |\n\n"
f"两级 MSF-MOFS 从 400 维分别选取 {len(s1)} 和 {len(s2)} 维特征，"
f"去重后 {len(set(s1)|set(s2))} 维。"
f"训练时正常健康状态与退化/故障状态由两个独立二分类器建模。\n\n"
f"测试概率和指标可以在 metrics.json 和 demo_predictions.csv 中逐条核对。"
f"自动人工校正只更新 0-100 健康分数的有序映射，不修改概率模型。"
f"全部测试来自同一合成生成器的未参与模型训练任务段，不代表真实应用效果。\n")
    (out/"RESULTS.md").write_text(summary,encoding="utf-8")
    print(summary)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/mvp")
    args=parser.parse_args()
    run(args.out)
