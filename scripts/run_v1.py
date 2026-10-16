#!/usr/bin/env python3
"""Reproducible NASA SMAP P-1 V1 proof of concept (real anomaly labels, proxy states)."""
import argparse,ast,json,time,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from msf_mofs import select_features
from health_v1 import (causal_windows,features,proxy_labels,split_points,train_base,base_proba,
    markov_transitions,hmm_filter,train_anomaly,calibrate_fusion,fusion_proba,
    health_score,fit_user_mapping,metrics,binary_metrics)


def load(path):
    df=pd.read_csv(path)
    if 'value' not in df.columns: raise ValueError(f'Expected value column in {path}')
    arr=df['value'].to_numpy(dtype=float)
    if not np.isfinite(arr).all():raise ValueError('Nonfinite samples')
    return arr


def pick_threshold(y,scores):
    grid=np.r_[.01,np.quantile(scores,np.linspace(.02,.99,150)),.99]
    vals=[f1_score(y,scores>=thr,zero_division=0) for thr in grid]
    return float(grid[np.argmax(vals)])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default=str(ROOT/'data/raw'))
    ap.add_argument('--output-dir',default=str(ROOT/'results'))
    ap.add_argument('--generations',type=int,default=5)
    ap.add_argument('--lookback',type=int,default=64)
    a=ap.parse_args(); p=Path(a.data_dir);out=Path(a.output_dir);out.mkdir(exist_ok=True,parents=True)
    rng=np.random.default_rng(20261016); start=time.perf_counter()
    pretrain=load(p/'P-1-train.csv');seq=load(p/'P-1-test.csv')
    md=pd.read_csv(p/'labeled_anomalies.csv');entry=md[(md.chan_id=='P-1')&(md.spacecraft=='SMAP')]
    if len(entry)!=1:raise ValueError('Expected unique P-1 metadata row')
    intervals=ast.literal_eval(entry.iloc[0]['anomaly_sequences'])
    if len(seq)!=int(entry.iloc[0]['num_values']):raise ValueError('P-1 length metadata mismatch')
    y_all,b_all=proxy_labels(len(seq),intervals)
    raw,ends=causal_windows(seq,a.lookback);feat=features(raw)
    y=y_all[ends];binary=b_all[ends]
    tr,va,te=split_points(ends)
    assert y[tr].min()==0 and set(np.unique(y[tr]))=={0,1,2}
    assert set(np.unique(y[va]))=={0,1,2} and set(np.unique(y[te]))=={0,1,2}
    selected,fsmeta=select_features(feat[tr],(y[tr]>0).astype(int),generations=a.generations)
    B=train_base(feat[:,selected],y,tr)
    p_static=base_proba(feat[:,selected],B)
    T=markov_transitions(y[tr]);state_prior=np.bincount(y[tr],minlength=3)+1
    state_prior=state_prior/state_prior.sum()
    p_hmm=np.empty_like(p_static)
    for mask in (tr,va,te):
        p_hmm[mask]=hmm_filter(p_static[mask],T,state_prior)
    # Supervised anomaly detector receives directly causal raw value sequences.
    # The separately provided original normal training sequence supplies extra negatives.
    raw_normal,_=causal_windows(pretrain,a.lookback)
    raw_fit=np.concatenate([raw[tr],raw_normal[::2]],axis=0)
    y_fit=np.r_[binary[tr],np.zeros(len(raw_normal[::2]),int)]
    model_a=train_anomaly(raw_fit,y_fit,np.ones(len(y_fit),bool))
    anomaly=model_a.predict_proba(raw)[:,1]
    threshold=pick_threshold(binary[va],anomaly[va])
    # Fusion is trained on validation ONLY; test period is never used for selection/training.
    model_c=calibrate_fusion(p_hmm,anomaly,y,va)
    p_fused=fusion_proba(p_hmm,anomaly,model_c)
    # User marks REAL signal segments, but for this proof of concept the scores below
    # are GENERATED from proxy labels. They are not human judgements.
    simulated_user_weights=np.array([96.,74.,12.])
    synthetic_score=simulated_user_weights[y]+rng.normal(0,2.5,size=len(y))
    synthetic_score=np.clip(synthetic_score,0,100)
    # Simulate expert uploading a labeled validation subset with stratification.
    candidate=np.flatnonzero(va)
    chosen=[]
    for k in range(3):
        ids=candidate[y[candidate]==k]
        chosen.extend(rng.choice(ids,size=min(60,len(ids)),replace=False))
    chosen=np.asarray(sorted(chosen),int)
    weights_user=fit_user_mapping(p_fused[chosen],synthetic_score[chosen],strength=.15)
    def score_stats(p):
        return {'default_mae_vs_synthetic_user':round(float(mean_absolute_error(synthetic_score[te],health_score(p[te]))),3),
          'adapted_mae_vs_synthetic_user':round(float(mean_absolute_error(synthetic_score[te],health_score(p[te],weights_user))),3)}
    results={
      'experiment':'SMAP_P-1_univariate_real_anomaly_proxy_three_state_v1',
      'source':'NASA SMAP/Telemanom; converted CSV by aminmalekisadr/ICC2023',
      'ground_truth_note':'Only binary anomaly intervals are externally verified. Three states and user scores are constructed proxies, never physical failure labels.',
      'data':{'train_normal_original_points':len(pretrain),'labeled_test_points':len(seq),
        'labeled_intervals':intervals,'window':a.lookback,'train_end_exclusive':2600,'val_end_exclusive':4000,
        'window_counts':{'train':int(tr.sum()),'val':int(va.sum()),'test':int(te.sum())},
        'binary_positive_counts':{'train':int(binary[tr].sum()),'val':int(binary[va].sum()),'test':int(binary[te].sum())},
        'proxy_class_counts':{name:np.bincount(y[mask],minlength=3).tolist() for name,mask in [('train',tr),('val',va),('test',te)]}},
      'feature_selection':fsmeta,'selected_indices':selected.tolist(),
      'anomaly_detector':{'classifier':'supervised_random_forest_raw_lag_window','threshold_from_validation':round(threshold,5),
        'validation':binary_metrics(binary[va],anomaly[va],threshold),
        'test':binary_metrics(binary[te],anomaly[te],threshold)},
      'transition_matrix':np.around(T,5).tolist(),
      'three_state_proxy_test':{'hierarchical_logreg':metrics(y[te],p_static[te]),
        'hierarchical_logreg_plus_HMM':metrics(y[te],p_hmm[te]),
        'late_fusion':metrics(y[te],p_fused[te])},
      'score_mapping_simulation':{'default_weights':[100,60,0],
        'synthetic_user_target_weights':simulated_user_weights.tolist(),
        'fitted_user_weights':np.around(weights_user,3).tolist(),
        'calibration_sample_n':int(len(chosen)),**score_stats(p_fused)},
      'runtime_seconds':round(float(time.perf_counter()-start),2),
      'limits':['Single satellite channel P-1; no multichannel command context in extracted CSV.',
        'Temporal split has one anomalous interval per train/val/test, not independent device-level replication.',
        'Proxy degraded/fault labels are onset/core partitions of annotated anomaly intervals, not actual equipment states.',
        'No actual maintenance labels, no true human scores, no operational incremental learning or drift gate.',
        'No proper HMM emission likelihood learned; posterior/prior discriminative approximation is used.']}
    (out/'metrics.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    rows=pd.DataFrame({'test_timestep':ends[te],'anomaly_label_real':binary[te],
        'three_state_proxy_label':y[te], 'anomaly_score':anomaly[te],
        'base_healthy':p_hmm[te,0],'base_degraded_proxy':p_hmm[te,1],
        'base_fault_proxy':p_hmm[te,2],
        'fused_healthy':p_fused[te,0],'fused_degraded_proxy':p_fused[te,1],
        'fused_fault_proxy':p_fused[te,2],
        'default_health_score':health_score(p_fused[te]),
        'user_calibrated_score':health_score(p_fused[te],weights_user),
        'synthetic_user_label':synthetic_score[te]})
    rows.to_csv(out/'heldout_predictions.csv',index=False)
    # Representative simulated feedback records are explicitly marked synthetic.
    feedback=[]
    for idx in chosen[:12]:
        feedback.append({'sample_id':f'P-1-test-{int(ends[idx])}',
            'label_source':'synthetic_simulation', 'human_verified':False,
            'model_state_proba':np.around(p_fused[idx],5).tolist(),
            'user_score_label':round(float(synthetic_score[idx]),2),
            'repair_feedback':{'confirmation_status':'not_inspected',
                'true_anomaly_from_public_annotation':bool(binary[idx])}})
    (out/'simulated_feedback.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in feedback)+'\n',encoding='utf-8')
    text='''# SMAP P-1 V1 实验结果\n\n> **限定**：真实标签仅包含遥测异常区间；三级健康状态与用户打分均为代理/模拟，**不代表真实卫星故障诊断准确率**。\n\n'''
    text+=f"- 数据：原始正常训练时序 {len(pretrain)} 点；带异常标签测试时序 {len(seq)} 点。使用测试时序前 {2600} 点做监督训练，2600–3999 做验证，4000–末尾做严格保留的时间测试。\n"
    text+=f"- MSF-MOFS：{fsmeta['candidate_dim']} 个候选特征选中 {fsmeta['selected_dim']} 个，包装器独立评估 {fsmeta['unique_wrapper_evaluations']} 次。\n"
    text+=f"- 异常检测真实区间标签（测试段）：AP={results['anomaly_detector']['test']['average_precision']}，ROC-AUC={results['anomaly_detector']['test']['roc_auc']}，F1={results['anomaly_detector']['test']['f1']}。阈值只使用验证集选择。\n"
    for name,met in results['three_state_proxy_test'].items():
        text+=f"- {name}：三级**代理**标签 macro-F1={met['macro_f1']}，Brier={met['multi_brier']}。\n"
    z=results['score_mapping_simulation']
    text+=f"- 模拟用户评分校正：默认映射 MAE={z['default_mae_vs_synthetic_user']}；用户映射 MAE={z['adapted_mae_vs_synthetic_user']}，训练样本 {z['calibration_sample_n']} 条。此 MAE 仅相对于人为合成的目标分数。\n"
    text+='''\n## 结果解读\n\n1. 异常检测指标只对应真实 SMAP 异常标注。P-1 仅单通道，属于概念验证而非通用卫星基准。\n2. 三级代理状态人为按异常区间前 20% 为退化代理、后 80% 为故障代理构造；其指标不能支持真实退化、故障能力的结论。\n3. HMM 使用从代理标签估计的转移矩阵，并将分类概率除以训练先验近似生成观测证据；它不是严格训练的生成式 HMM。\n4. 融合模型从**验证集**学习，因此测试结果需与基础模型比较；提升不具必然性。\n5. 用户评分均为合成标签。无专家健康度真值时只能验证校正机制的可运行性。\n6. 在线漂移门控、自动增量学习和维修人员真实诊断反馈属于 V2 研究，不在本次声称范围内。\n\n## 下一步\n\n扩展至少 5–10 个 SMAP 通道并按通道/任务分组验证；获得真实工况和维修记录，替代故障代理标签；复现 M2N2/AnDri 并开展正常工况漂移与退化污染实验。\n'''
    (out/'RESULTS.md').write_text(text,encoding='utf-8')
    print(json.dumps({'test_anomaly':results['anomaly_detector']['test'],
        'selected':fsmeta['selected_dim'],'three_state_proxy':results['three_state_proxy_test'],
        'user_mapping':z},indent=2))

if __name__=='__main__':main()
