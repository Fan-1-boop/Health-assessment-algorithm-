#!/usr/bin/env python3
"""V1.1: preregistered proxy labels + 510 multiscale telemetry features + MSF-MOFS.

Ground truth = binary NASA anomaly mask ONLY. Do NOT interpret proxy states as faults.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,f1_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from highdim_features import highdim_features, feature_names
from msf_mofs import select_features
from health_v1 import (causal_windows,features,proxy_labels,split_points,train_base,base_proba,
                        markov_transitions,hmm_filter,train_anomaly,calibrate_fusion,
                        fusion_proba,health_score,fit_user_mapping,metrics,binary_metrics)


def load(path):
    x=pd.read_csv(path)['value'].to_numpy(float)
    if not np.all(np.isfinite(x)):raise ValueError('Nonfinite values in '+str(path))
    return x


def load_frozen_labels(n,metadata,rule_path):
    content=Path(rule_path).read_bytes()
    rule=json.loads(content)
    if rule['rule_id']!='smap_p1_interval_position_proxy_v1_frozen':
        raise ValueError('Unexpected proxy rule id')
    if 'floor((b-a+1)*0.20)' not in rule['degraded_proxy']:
        raise ValueError('Cannot change preregistered 20 percent onset definition')
    entry=metadata[(metadata.chan_id=='P-1')&(metadata.spacecraft=='SMAP')]
    if len(entry)!=1:raise ValueError('P-1 metadata missing or duplicated')
    if int(entry.iloc[0]['num_values'])!=n:raise ValueError('Label length mismatch')
    intervals=ast.literal_eval(entry.iloc[0]['anomaly_sequences'])
    ordered=sorted((int(a),int(b)) for a,b in intervals)
    if any(ordered[i][1]>=ordered[i+1][0] for i in range(len(ordered)-1)):
        raise ValueError('Overlapping public anomaly intervals are unsupported')
    # Existing V1 algorithm implements exactly the previously frozen convention.
    states,binary=proxy_labels(n,intervals,edge_fraction=.2)
    return states,binary,intervals,hashlib.sha256(content).hexdigest(),rule


def threshold_from_validation(y,s):
    grid=np.unique(np.r_[0.,1.,np.quantile(s,np.linspace(.02,.98,120))])
    f=[f1_score(y,s>=v,zero_division=0) for v in grid]
    return float(grid[np.argmax(f)])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',default=str(ROOT/'data/raw'))
    parser.add_argument('--out',default=str(ROOT/'results/v11'))
    parser.add_argument('--generations',type=int,default=5)
    parser.add_argument('--population',type=int,default=12)
    parser.add_argument('--lookback',type=int,default=64)
    a=parser.parse_args()
    start=time.perf_counter();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    source=Path(a.data_dir)
    original=load(source/'P-1-train.csv')
    values=load(source/'P-1-test.csv')
    meta=pd.read_csv(source/'labeled_anomalies.csv')
    yall,ball,intervals,rule_sha,rule=load_frozen_labels(len(values),meta,ROOT/'configs/proxy_label_rules_v1.json')
    raw,t=causal_windows(values,a.lookback)
    states=yall[t]; anomaly_binary=ball[t]
    tr,va,te=split_points(t)
    for m in (tr,va,te):
        if set(np.unique(states[m]))!={0,1,2}:raise ValueError('All proxy classes needed in each time partition')
    # Predefined temporal splits; train-only feature search, never a test-dependent label rule.
    hd=highdim_features(raw)
    names=feature_names(a.lookback)
    base25=features(raw)
    if hd.shape[1]<200:raise ValueError('V1.1 should use a high-dimensional candidate set')
    selected,fs=select_features(hd[tr],(states[tr]>0).astype(int),population=a.population,generations=a.generations)
    np.savetxt(out/'selected_feature_indices.csv',selected,fmt='%d',delimiter=',',header='index',comments='')
    (out/'selected_feature_names.json').write_text(json.dumps([names[i] for i in selected],ensure_ascii=False,indent=2)+'\n')
    (out/'all_feature_names.json').write_text(json.dumps(names,ensure_ascii=False,indent=2)+'\n')
    T=markov_transitions(states[tr]); prior=np.bincount(states[tr],minlength=3)+1
    prior=prior/prior.sum()
    def fit_three(X):
        model=train_base(X,states,tr)
        static=base_proba(X,model)
        filt=np.empty_like(static)
        for m in (tr,va,te):filt[m]=hmm_filter(static[m],T,prior)
        return static,filt
    # Same fixed data partitions: quantify the effect of feature engineering / feature selection.
    static25,hmm25=fit_three(base25)
    static_all,hmm_all=fit_three(hd)
    static_sel,hmm_sel=fit_three(hd[:,selected])
    # Raw-lag anomaly detector is kept fixed from V1 to isolate feature-selection changes.
    pretrain_windows,_=causal_windows(original,a.lookback)
    Xfit=np.concatenate([raw[tr],pretrain_windows[::2]])
    yfit=np.r_[anomaly_binary[tr],np.zeros(len(pretrain_windows[::2]),int)]
    detector=train_anomaly(Xfit,yfit,np.ones(len(yfit),bool))
    anomaly=detector.predict_proba(raw)[:,1]
    threshold=threshold_from_validation(anomaly_binary[va],anomaly[va])
    # Late fusion fitted on validation only (not on test). Same train/val/test boundaries.
    fusion=calibrate_fusion(hmm_sel,anomaly,states,va)
    fused=fusion_proba(hmm_sel,anomaly,fusion)
    # Simulation of HUMAN score inputs, not user score observations.
    rng=np.random.default_rng(20261016)
    synthetic_weights=np.array([96.,74.,12.])
    synthetic_score=np.clip(synthetic_weights[states]+rng.normal(0,2.5,len(states)),0,100)
    ids=np.flatnonzero(va)
    selected_user=np.sort(np.concatenate([rng.choice(ids[states[ids]==k],size=min(60,sum(states[ids]==k)),replace=False)
                for k in range(3)]))
    w=fit_user_mapping(fused[selected_user],synthetic_score[selected_user],strength=.15)
    methods={'baseline_25d_static':static25,'baseline_25d_hmm':hmm25,
             'all_highdim_static':static_all,'all_highdim_hmm':hmm_all,
             'msf_selected_static':static_sel,'msf_selected_hmm':hmm_sel,
             'msf_hmm_plus_anomaly_fusion':fused}
    tests={k:metrics(states[te],v[te]) for k,v in methods.items()}
    binary_metrics_test=binary_metrics(anomaly_binary[te],anomaly[te],threshold)
    score_stats={
        'default_mae':round(float(mean_absolute_error(synthetic_score[te],health_score(fused[te]))),4),
        'adapted_mae':round(float(mean_absolute_error(synthetic_score[te],health_score(fused[te],w))),4),
        'selected_calibration_n':int(len(selected_user)),
        'user_weights':[round(float(x),4) for x in w],
        'simulated_target_weights':synthetic_weights.tolist(),
        'is_synthetic':True}
    info={
        'experiment':'SMAP_P1_V1.1_frozen_proxy_highdim_MSF_MOFS',
        'data_source':'NASA Telemanom SMAP P-1, ICC2023 CSV reformat',
        'ground_truth':'Only published binary anomaly intervals are real annotation; 3-state classes and health score are synthetic.',
        'labels':{'rule_id':rule['rule_id'],'frozen_rule_sha256':rule_sha,
                  'frozen_rule_committed_before_experiments':'861c46bf12a6acd1002c5b88cf8119297d44c3ea',
                  'intervals':intervals, 'proxy_class_counts':{
                    name:np.bincount(states[m],minlength=3).tolist() for name,m in [('train',tr),('validation',va),('test',te)]}},
        'data':{'lookback':a.lookback,'pretrain_normal_points':len(original),'n_annotated_points':len(values),
                'temporal_boundaries':{'train_lt':2600,'validation_lt':4000,'test_gte':4000},
                'window_counts':{'train':int(tr.sum()),'validation':int(va.sum()),'test':int(te.sum())}},
        'feature_engineering':{'method':'causal multiscale statistical + derivative + spectrum + autocorrelation features',
            'base_candidate_count':int(base25.shape[1]),'highdim_candidate_count':int(hd.shape[1]),
            'selected_feature_names':[names[i] for i in selected],
            'selected_feature_count':int(len(selected)),
            'feature_reduction_ratio':round(float(1-len(selected)/hd.shape[1]),5)},
        'msf_mofs':fs,'anomaly_detector':{'algorithm':'supervised RandomForest on raw 64-lag window',
            'test':binary_metrics_test,'validation_threshold':threshold},
        'state_proxy_test':tests,'user_mapping_simulated':score_stats,
        'transition_matrix':T.tolist(),
        'runtime_seconds':round(float(time.perf_counter()-start),2),
        'caveats':['P-1 target channel only. NO actual spacecraft condition or operational mode metadata.',
            'Proxy interval-onset vs core labels are not real degradation/fault labels.',
            'No maintenance, confirmed failure, or expert health scores exist in this dataset.',
            'Labeled interval locations supplied publicly; experiment uses an altered supervised split and not official unsupervised protocol.',
            'Anomaly detector is fixed in V1.1. Incremental drift updating is NOT implemented.',
            'MSF-MOFS wrapper uses logistic regression and a reduced search budget, not an exact replication of paper 5-NN settings.',
            'HMM is discriminative posterior/prior approximation; not independently trained generative HMM.',
            'One anomaly interval in each temporal partition; proxy evaluation has high uncertainty.']}
    (out/'metrics.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pred=pd.DataFrame({'timestep':t[te],'binary_anomaly_truth':anomaly_binary[te],
        'proxy_label_not_real_fault':states[te], 'anomaly_score':anomaly[te],
        'baseline_25d_health':static25[te,0], 'msf_static_health':static_sel[te,0],
        'hmm_health':hmm_sel[te,0], 'fused_health':fused[te,0],
        'fused_degraded_proxy':fused[te,1],'fused_fault_proxy':fused[te,2],
        'default_health_score':health_score(fused[te]),
        'simulated_user_calibrated_score':health_score(fused[te],w),
        'simulated_user_score_truth':synthetic_score[te]})
    pred.to_csv(out/'heldout_predictions.csv',index=False)
    feedback=[]
    for j in selected_user[:15]:
        feedback.append({'sample_id':f'SMAP_P1_{int(t[j])}','feedback_source':'SIMULATED',
            'is_human_label':False,'health_score_label':float(synthetic_score[j]),
            'public_anomaly_label':int(anomaly_binary[j]),
            'proxy_state_label':int(states[j]),
            'confirmation_status':'not_inspected','model_version':'v1.1'})
    (out/'simulated_feedback.jsonl').write_text('\n'.join(json.dumps(v,ensure_ascii=False) for v in feedback)+'\n')
    lines=['# SMAP P-1 V1.1 实验结果', '',
      '> 唯一真实标签是公开遥测异常区间；三状态及健康分数均为代理/模拟，不是卫星故障诊断真值。','',
      f"- 已冻结标签规则 `{rule['rule_id']}`，SHA-256 `{rule_sha}`，冻结提交 `861c46bf12a6acd1002c5b88cf8119297d44c3ea`。",
      f"- 多尺度高维特征：{hd.shape[1]} 维；MSF-MOFS 选择 {len(selected)} 维（减少 {(1-len(selected)/hd.shape[1])*100:.1f}%）。",
      f"- MSF-MOFS 交叉验证 G-mean：{fs['selection_cv_gmean']}；包装式子集评价次数 {fs['unique_wrapper_evaluations']}。",
      f"- 真实异常区间测试 AP={binary_metrics_test['average_precision']}、ROC-AUC={binary_metrics_test['roc_auc']}、F1={binary_metrics_test['f1']}。",
      '', '## 三级代理标签模型对照（测试集）','','| 方法 | Macro-F1 | Brier（越低越好） |','|---|---:|---:|']
    for k,v in tests.items():lines.append(f"| {k} | {v['macro_f1']} | {v['multi_brier']} |")
    lines+=['','## 模拟用户评分','',
        f"- 默认映射 MAE={score_stats['default_mae']}；模拟校正后 MAE={score_stats['adapted_mae']}。",
        f"- 学得映射系数={score_stats['user_weights']}，目标规则={score_stats['simulated_target_weights']}，模拟校正样本数={score_stats['selected_calibration_n']}。",
        '','## 解释与限制','',
        '1. 先冻结区间位置规则，再进行特征工程和多目标特征搜索，不根据测试表现更改代理标签。',
        '2. 数据不含真实故障位置、退化机制或卫星控制命令，因此没有证据将代理类别结果外推为物理故障诊断性能。',
        '3. 高维特征是单通道历史窗口上的多尺度派生量，不是真正的多通道物理遥测。',
        '4. 仅一个 P-1 遥测通道，三个异常事件分别位于训练、验证、测试段；报告是探索性实验。',
        '5. 用户评分和维修反馈都用程序生成，不能作为真实专家与维修验证。',
        '6. 受控增量模型更新与真实工况漂移区分仍未实现。',
        '']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'features':info['feature_engineering'],'anomaly':binary_metrics_test,
          'proxy_test':tests,'simulated_score':score_stats,'runtime':info['runtime_seconds']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
