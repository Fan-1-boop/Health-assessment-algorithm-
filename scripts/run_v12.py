#!/usr/bin/env python3
"""V1.2 independent branch evaluation: NO late fusion, NO health score optimization.

- State branch: two INDEPENDENT MSF-MOFS binary selectors -> two-stage state
  classification -> discriminative HMM, with explicit Gaussian-emission HMM baseline.
- Anomaly branch: supervised raw-window anomaly model selection on validation only.
- Real source labels only mark anomaly intervals. State labels are preregistered proxies.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_v11 import load,load_frozen_labels
from highdim_features import highdim_features,feature_names
from msf_mofs import select_features
from health_v1 import causal_windows,split_points
from independent_branches import (train_two_stage,two_stage_probability,transition_matrix,
    posterior_as_likelihood,filtering_from_log_evidence,fit_gaussian_emissions,emission_loglik,
    state_metrics,anomaly_raw_features,train_anomaly_candidates,anomaly_score,anomaly_metrics,
    tune_hmm,choose_anomaly_threshold)


def jsondump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--data-dir',default=str(ROOT/'data/raw'))
    arg.add_argument('--out',default=str(ROOT/'results/v12'))
    arg.add_argument('--population',type=int,default=12)
    arg.add_argument('--generations',type=int,default=5)
    arg.add_argument('--lookback',type=int,default=64)
    a=arg.parse_args();start=time.perf_counter();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    directory=Path(a.data_dir)
    normal=load(directory/'P-1-train.csv');values=load(directory/'P-1-test.csv')
    meta=pd.read_csv(directory/'labeled_anomalies.csv')
    states_full,anomaly_full,intervals,rule_sha,rule=load_frozen_labels(
        len(values),meta,ROOT/'configs/proxy_label_rules_v1.json')
    windows,ends=causal_windows(values,a.lookback)
    states=states_full[ends]; y_anomaly=anomaly_full[ends]
    tr,va,te=split_points(ends)
    assert np.all(np.diff(ends)>0)
    assert set(np.unique(states[tr]))=={0,1,2}
    assert set(np.unique(states[va]))=={0,1,2}
    assert set(np.unique(states[te]))=={0,1,2}
    high=highdim_features(windows);names=feature_names(a.lookback)
    assert high.shape[1]==510
    tic=time.perf_counter()
    # ONLY TRAINING samples are used to fit Fisher rank, wrapper and final model.
    s1,info1=select_features(high[tr],(states[tr]>0).astype(int),
        population=a.population,generations=a.generations,seed=42,tolerance=.02)
    unwell=tr & (states>0)
    s2,info2=select_features(high[unwell],(states[unwell]==2).astype(int),
        population=a.population,generations=a.generations,seed=43,tolerance=.02)
    fs_runtime=time.perf_counter()-tic
    b=train_two_stage(high,states,tr,s1,s2)
    base=two_stage_probability(high,b,prior_correction=True)
    base_balanced=two_stage_probability(high,b,prior_correction=False)
    train_prior=np.bincount(states[tr],minlength=3).astype(float)
    train_prior/=train_prior.sum()
    T=transition_matrix(states[tr],alpha=.4)
    # The HMM applies estimated transition probabilities and observational evidence.
    # The 'posterior/prior' construction is an approximation, labelled accordingly.
    logev=posterior_as_likelihood(base,train_prior)
    selected_tune=tune_hmm(logev[va],states[va],T,train_prior)
    d_hmm=np.zeros_like(base)
    for mask in (tr,va,te):
        d_hmm[mask]=filtering_from_log_evidence(logev[mask],T,train_prior,
            gamma=selected_tune['gamma'],memory=selected_tune['memory'])
    # Independent supervised generative emission model (actual class-conditional
    # Gaussian densities), projected to compact train-only PCA component space.
    union=np.union1d(s1,s2)
    gen=fit_gaussian_emissions(high[tr][:,union],states[tr],n_components=4)
    gauss_ll=emission_loglik(gen,high[:,union])
    gauss_tune=tune_hmm(gauss_ll[va],states[va],T,train_prior)
    g_hmm=np.zeros_like(base)
    for mask in (tr,va,te):
        g_hmm[mask]=filtering_from_log_evidence(gauss_ll[mask],T,train_prior,
            gamma=gauss_tune['gamma'],memory=gauss_tune['memory'])
    # A classifier without feature selection shows the full-feature ceiling.
    full_train=train_two_stage(high,states,tr,np.arange(high.shape[1]),np.arange(high.shape[1]))
    full_probs=two_stage_probability(high,full_train,prior_correction=True)
    state_probs={
        'two_stage_full_510d':full_probs,
        'two_stage_MSF_static':base,
        'two_stage_MSF_balanced_uncalibrated':base_balanced,
        'two_stage_MSF_discriminative_HMM':d_hmm,
        'two_stage_MSF_gaussian_HMM':g_hmm,
    }
    state_tests={k:state_metrics(states[te],v[te]) for k,v in state_probs.items()}
    state_validation={k:state_metrics(states[va],v[va]) for k,v in state_probs.items()}
    # The model family below is trained on RAW causal sequences only.
    # Derivative/rolling augmentations are internal encoding of the raw sequence.
    n_w,_=causal_windows(normal,a.lookback)
    A=anomaly_raw_features(windows)
    A_normal=anomaly_raw_features(n_w[::3])
    Atr=np.vstack([A[tr],A_normal])
    ytr=np.concatenate([y_anomaly[tr],np.zeros(len(A_normal),dtype=int)])
    assert len(set(ytr))==2
    chosen,ranking=train_anomaly_candidates(Atr,ytr,A[va],y_anomaly[va])
    best_name,best_model,threshold=chosen
    test_score=anomaly_score(best_model,A[te])
    anomaly_evaluation=anomaly_metrics(y_anomaly[te],test_score,threshold)
    # Also report every alternative model on test. This is diagnostics ONLY: the
    # recommended model was selected exclusively with the validation interval.
    # Refit to obtain the same model family predictions; held-out test NEVER selects.
    from independent_branches import candidate_anomaly_models
    all_diagnostics={}
    for name,model in candidate_anomaly_models().items():
        model.fit(Atr,ytr)
        val_s=anomaly_score(model,A[va])
        th,_=choose_anomaly_threshold(y_anomaly[va],val_s)
        te_s=anomaly_score(model,A[te])
        all_diagnostics[name]=anomaly_metrics(y_anomaly[te],te_s,th)
    # Unsupervised novelty comparator trained on original published normal pretrain.
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(A_normal)
    iso=IsolationForest(n_estimators=170,max_samples=min(len(A_normal),256),
                        random_state=42,n_jobs=-1).fit(scaler.transform(A_normal))
    v_iso=-iso.score_samples(scaler.transform(A[va]))
    th_iso,_=choose_anomaly_threshold(y_anomaly[va],v_iso)
    t_iso=-iso.score_samples(scaler.transform(A[te]))
    all_diagnostics['isolation_forest_novelty']=anomaly_metrics(y_anomaly[te],t_iso,th_iso)
    ranking['isolation_forest_novelty']={'validation_ap':round(float(average_precision_score(y_anomaly[va],v_iso)),6),
       'validation_auc':round(float(roc_auc_score(y_anomaly[va],v_iso)),6),
       'validation_threshold':th_iso,'used_for_supervised_selection':False}
    info={
      'study':'V1.2 independent branch validation; no fusion; no automatic increment',
      'data':'NASA SMAP P-1 test and normal train telemetry, public binary anomaly intervals',
      'source_labels':{'real':'binary anomaly interval','three_state':'frozen synthetic position proxy',
                       'rule_id':rule['rule_id'],'rule_sha256':rule_sha,
                       'anomaly_intervals':intervals},
      'samples':{'train':int(tr.sum()),'validation':int(va.sum()),'test':int(te.sum()),
                 'test_positive':int(y_anomaly[te].sum())},
      'features':{'candidate_dim':len(names),'first_stage_selected_dim':len(s1),
                  'second_stage_selected_dim':len(s2),
                  'online_feature_union_dim':len(union),
                  'first_stage_names':[names[i] for i in s1],
                  'second_stage_names':[names[i] for i in s2],
                  'first_stage_search':info1,'second_stage_search':info2,
                  'dual_stage_search_seconds':round(fs_runtime,2)},
      'state_branch':{'transition_matrix':T.tolist(),'class_prior':train_prior.tolist(),
         'discriminative_hmm':selected_tune,'gaussian_hmm':gauss_tune,
         'validation':state_validation,'heldout_test':state_tests},
      'anomaly_branch':{'chosen_by_validation_ap':best_name,
         'validation_ranking':ranking,
         'chosen_test':anomaly_evaluation,
         'all_test_diagnostics_NOT_for_model_selection':all_diagnostics},
      'runtime_seconds':round(time.perf_counter()-start,2),
      'limitations':[
        'Only one spacecraft telemetry target channel; no physical spacecraft operation-mode labels.',
        'Labels degraded_proxy/fault_proxy based solely on frozen anomaly interval position, NOT real faults.',
        'A single anomaly episode in each of train/validation/test; event-generalization is weakly identified.',
        'The MSF-MOFS wrapper cross validation uses stratified folds and overlapping time windows; CV gains may be optimistic.',
        'S1 and S2 select different subsets: MSF method adapted to each binary task.',
        'Bayesian filtering is temporal and online-causal. Discriminative posterior/prior is only approximate; Gaussian HMM explicitly estimates class-conditional emissions.',
        'No fusion, health score mapping, human simulation, or incremental update is performed in V1.2.',
        'Anomaly and state branches report independent test criteria; test is never used to choose models or thresholds.',
      ]}
    jsondump(out/'metrics.json',info)
    pred=pd.DataFrame({'time_index':ends[te],
         'public_binary_anomaly':y_anomaly[te],
         'proxy_state_not_physical_fault':states[te],
         'anomaly_selected_score':test_score,
         'anomaly_detected':(test_score>=threshold).astype(int),
         'base_healthy':base[te,0],'base_degraded_proxy':base[te,1],
         'base_fault_proxy':base[te,2],
         'discriminative_hmm_healthy':d_hmm[te,0],
         'gaussian_hmm_healthy':g_hmm[te,0]})
    pred.to_csv(out/'heldout_predictions.csv',index=False)
    lines=['# V1.2 两条分支独立实验结果','',
      '> 尚未执行概率融合；健康/退化/故障全部是**冻结的代理标签**，并非真实物理故障真值。','',
      f'- 公开数据 NASA SMAP P-1，窗口 {a.lookback}，训练 {tr.sum()} / 验证 {va.sum()} / 测试 {te.sum()}。',
      f'- 原始 510 维高维工程特征，健康/不健康 MSF-MOFS 选择 {len(s1)} 维；退化/故障选择 {len(s2)} 维；上线计算并集 {len(union)} 维。',
      f'- MSF 包装评价：第一阶段 {info1["unique_wrapper_evaluations"]} 次；第二阶段 {info2["unique_wrapper_evaluations"]} 次。',
      '','## 一、基础动态贝叶斯分支（三级代理状态；测试集）','','| 模型 | Macro-F1 | Balanced accuracy | Brier |','|---|---:|---:|---:|']
    for k,v in state_tests.items():lines.append(f'| {k} | {v["macro_f1"]} | {v["balanced_accuracy"]} | {v["multi_brier"]} |')
    lines+=['','## 二、异常检测分支（真实公开异常标签；测试集）',
      '',f'**模型选择只依据验证集 AP**：`{best_name}`，测试 AP={anomaly_evaluation["average_precision"]}，测试异常基线发生率={anomaly_evaluation["base_rate"]}。',
      '','| 模型 | 验证 AP | 测试 AP | 测试 ROC-AUC | 测试 F1 | 误报数 |',
      '|---|---:|---:|---:|---:|---:|']
    for k,v in all_diagnostics.items():
        lines.append(f'| {k} | {ranking[k]["validation_ap"]} | {v["average_precision"]} | {v["roc_auc"]} | {v["f1"]} | {v["false_positive_count"]} |')
    lines+=['','## 三、结论限制','',
      '1. 验证集仅含一个异常事件，所选模型可能无法跨事件泛化；测试表现如实报告，不据测试重新调参。',
      '2. 三状态真值是先于实验冻结的区间位置代理，不能证明故障诊断、维修指导或健康度算法有效。',
      '3. HMM 高斯观测模型和判别式证据滤波均为完整运行的独立模型；如果不能提升验证/测试指标，不应在融合前默认采用。',
      '4. 两级特征选择不等于两级状态概率已经可靠；概率校准和真实物理状态标签是后续工作。',
      '5. 本版没有跨分支融合，也没有进行自动增量更新，符合分支分别验收的阶段目标。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'state_test':state_tests,'anomaly_test':anomaly_evaluation,
      'selected_anomaly_model':best_name,'runtime_seconds':info['runtime_seconds']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()