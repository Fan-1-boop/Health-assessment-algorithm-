#!/usr/bin/env python3
"""Real OPS-SAT-AD single-dataset anomaly classification MVP.

The sole physical real-world label supported here is segment-level anomaly
(0=nominal,1=anomaly). There is NO expert 3-class healthy/degraded/fault truth,
therefore we do not fit fictitious HMM/DBN fault-state probabilities.

Source: https://doi.org/10.5281/zenodo.12588359
Uses original 18 official engineered statistical features. Official test flag
remains untouched until after selecting one algorithm on a validation split.
"""
from __future__ import annotations
import argparse,hashlib,io,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_recall_curve,average_precision_score,roc_auc_score,
    accuracy_score,precision_score,recall_score,f1_score,
    matthews_corrcoef,confusion_matrix)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from msf_mofs import select_features

def load_spec():
    p=ROOT/'real_mvp/PROTOCOL_FROZEN.json'
    raw=p.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()

def acquire(out,config):
    """Official Zenodo dataset, not a simulated or relabeled copy."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dest=out/'dataset.csv'
    if not dest.is_file():
        r=requests.get(config['source_url'],timeout=100)
        r.raise_for_status()
        if len(r.content)<400000:raise ValueError('Unexpected truncated OPS-SAT data')
        dest.write_bytes(r.content)
    df=pd.read_csv(dest)
    cols=set(config['features'])|{'segment','train','anomaly'}
    if not cols.issubset(df.columns) or len(df)!=int(config['rows_expected']):
        raise ValueError('Unexpected official OPS-SAT feature schema or row count')
    if df.segment.duplicated().any():raise ValueError('Repeated OPS-SAT segment ID')
    if not set(df.train.unique()).issubset({0,1}) or not set(df.anomaly.unique()).issubset({0,1}):
        raise ValueError('Unexpected official train / anomaly labels')
    manifest={
       'source_doi':'10.5281/zenodo.12588359','source_url':config['source_url'],
       'source_csv_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
       'dataset_origin':'Genuine ESA OPS-SAT satellite telemetry segments with official labels',
       'official_train_test_preserved':True,
       'rows':len(df),'feature_count':len(config['features']),
       'split_counts':df.groupby(['train','anomaly']).size().rename('count').reset_index().to_dict('records'),
       'source_repo':'https://github.com/kplabs-pl/OPS-SAT-AD',
       'important_note':'Real binary anomaly labels; no true physical degradation/fault severity labels',
    }
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    return df,manifest


def prepare_raw(df,features):
    """Never use train/label/segment/channel IDs as predictive model features."""
    return (df[list(features)].apply(pd.to_numeric,errors='coerce')
            .replace([np.inf,-np.inf],np.nan)).to_numpy(dtype=float)


def median_imputer_fit(X):
    x=np.asarray(X,float)
    vals=np.nanmedian(x,axis=0)
    vals=np.nan_to_num(vals,nan=0,posinf=0,neginf=0)
    return vals

def median_imputer_apply(X,median):
    x=np.asarray(X,float)
    return np.where(np.isfinite(x),x,median[None,:])


def candidate(name):
    if name=='rf':
        return RandomForestClassifier(n_estimators=230,
                  min_samples_leaf=1,max_features='sqrt',
                  class_weight='balanced_subsample',random_state=42,n_jobs=2)
    if name=='extra_trees':
        return ExtraTreesClassifier(n_estimators=270,
                  min_samples_leaf=1,max_features=.9,
                  class_weight='balanced_subsample',random_state=42,n_jobs=2)
    if name=='rbf_svm':
        return make_pipeline(StandardScaler(),
                SVC(C=6.,gamma='scale',class_weight='balanced',
                probability=True,random_state=42))
    if name=='hist_gb':
        return HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=11,
                  learning_rate=.055,l2_regularization=.4,random_state=42)
    raise ValueError('Unknown registered model: '+str(name))


def threshold_f1(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float)
    p,r,t=precision_recall_curve(y,s)
    if len(t)==0:raise ValueError('No validation scores')
    f1=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-10)
    # high threshold on ties to avoid false alarm exaggeration
    i=max(np.flatnonzero(f1>=f1.max()-1e-12))
    return float(np.clip(t[i],.001,.999))


def classification_metrics(y,s,threshold):
    y=np.asarray(y,int);s=np.asarray(s,float)
    pred=(s>=threshold).astype(int)
    return {'n':int(len(y)),'positives':int(y.sum()),'prevalence':float(y.mean()),
        'accuracy':float(accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)),
        'average_precision':float(average_precision_score(y,s)),
        'roc_auc':float(roc_auc_score(y,s)),
        'mcc':float(matthews_corrcoef(y,pred)),
        'false_positives':int(np.sum((pred==1)&(y==0))),
        'false_negatives':int(np.sum((pred==0)&(y==1))),
        'confusion_matrix_TN_FP_FN_TP':confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'threshold':float(threshold)}


def execute(out):
    spec,spec_hash=load_spec()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df,manifest=acquire(out,spec)
    raw=prepare_raw(df,spec['features'])
    y=df.anomaly.to_numpy(dtype=int)
    official_train=np.flatnonzero(df.train.to_numpy(int)==1)
    official_test=np.flatnonzero(df.train.to_numpy(int)==0)
    fit_ix,val_ix=train_test_split(official_train,
            test_size=float(spec['split']['validation_fraction_of_official_train']),
            random_state=int(spec['split']['stratified_random_seed']),
            stratify=y[official_train])
    fit_ix=np.sort(fit_ix);val_ix=np.sort(val_ix)
    if len(set(fit_ix)|set(val_ix)|set(official_test))!=len(df):
        raise AssertionError('Split does not partition segments')
    median=median_imputer_fit(raw[fit_ix])
    xfit=median_imputer_apply(raw[fit_ix],median)
    xval=median_imputer_apply(raw[val_ix],median)
    xhold=median_imputer_apply(raw[official_test],median)
    yfit=y[fit_ix];yval=y[val_ix];yhold=y[official_test]
    search=spec['feature_selection']
    selected,search_metrics=select_features(xfit,yfit,
         population=int(search['population']),generations=int(search['generations']),
         seed=int(search['seed']),tolerance=float(search['tolerance']))
    if len(selected)==0:raise ValueError('MSF returned zero features')
    mode_selectors={'msf':selected,'all_18':np.arange(xfit.shape[1])}
    comparisons={};stored={}
    for mode in spec['feature_modes']:
        columns=mode_selectors[mode]
        for name in spec['candidate_models']:
            key=mode+'__'+name
            model=candidate(name)
            model.fit(xfit[:,columns],yfit)
            pval=model.predict_proba(xval[:,columns])[:,1]
            th=threshold_f1(yval,pval)
            performance=classification_metrics(yval,pval,th)
            comparisons[key]={'features':int(len(columns)),**performance}
            stored[key]=(model,columns.copy(),th)
    # No official test labels touched during this choice.
    chosen=max(sorted(comparisons),key=lambda k:
               (comparisons[k]['f1'],comparisons[k]['average_precision'],
                -comparisons[k]['features']))
    model,columns,threshold=stored[chosen]
    phold=model.predict_proba(xhold[:,columns])[:,1]
    test=classification_metrics(yhold,phold,threshold)
    resultdf=pd.DataFrame({
        'segment':df.segment.to_numpy()[official_test],
        'channel':df.channel.to_numpy()[official_test],
        'official_anomaly_label':yhold,
        'anomaly_probability':phold,
        'predicted_anomaly':(phold>=threshold).astype(int),
        'demonstration_normality_index_not_physical_health':100*(1-phold),
    })
    resultdf.to_csv(out/'test_predictions.csv',index=False)
    artifact={'model':model,'selected_feature_indices':columns,
         'all_feature_names':spec['features'],'median_imputer':median,
         'threshold':threshold,'candidate_key':chosen,
         'task':'real OPS-SAT-AD 0/1 anomaly only',
         'protocol_sha256':spec_hash,
         'source_sha256':manifest['source_csv_sha256']}
    joblib.dump(artifact,out/'model_bundle.joblib',compress=3)
    metrics={
      'study':'Real ESA OPS-SAT-AD satellite telemetry anomaly detection',
      'not_simulated_data':True,
      'label_semantics':'Official segment-level 0=normal,1=anomaly; no verified 3-class physical health state',
      'dataset':manifest,
      'split':{'fit_segments':len(fit_ix),'validation_segments':len(val_ix),
             'official_heldout_test_segments':len(official_test),
             'fit_anomalies':int(yfit.sum()),'validation_anomalies':int(yval.sum()),
             'test_anomalies':int(yhold.sum())},
      'feature_selection':{'method':'MSF-MOFS (project implementation)',
               'candidate_feature_count':len(spec['features']),
               'selected_feature_count':len(selected),
               'selected_feature_names':[spec['features'][i] for i in selected],
               'inner_search':search_metrics},
      'model_selection':'best validation F1 with frozen candidate set and validation-only threshold',
      'chosen_by_validation':chosen,'validation_candidates':comparisons,
      'final_official_heldout_test':test,
      'source_protocol_sha256':spec_hash,
      'limitations':[
         'Real binary spacecraft telemetry anomaly task, NOT real healthy/degraded/fault detection',
         'Official precomputed statistical features describe entire labeled segments; this is retrospective segment classification, NOT causal early anomaly onset detection',
         'MSF-MOFS feature selection uses its internal wrapper CV for optimization, but only the official train subset; reported final test is untouched until model selection',
         'No HMM across independent segments; never interpret independent segment order as time series',
         'No artificial relabeling, no autonomous online model update, no cross-dataset generalization claim',
         'The normality index is a visualization based on binary score, NOT a physically calibrated device health score'
      ]}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False,default=lambda x:int(x) if isinstance(x,np.integer) else float(x))+'\n')
    (out/'RESULTS.md').write_text(
       '# OPS-SAT-AD 真实卫星数据异常检测 MVP\n\n'
       '**数据：真实 ESA OPS-SAT 遥测，官方正常/异常二分类标签；不是物理健康/退化/故障三级真值。**\n\n'
       f"- 官方片段总数：{len(df)}；模型训练 {len(fit_ix)}，模型选择 {len(val_ix)}，官方测试 {len(official_test)}。\n"
       f"- MSF-MOFS：从 {len(spec['features'])} 个官方统计特征中筛选 {len(selected)} 个。\n"
       f"- 验证集选择模型：\`{chosen}\`；验证 F1 = {comparisons[chosen]['f1']:.4f}。\n"
       f"- **官方测试**：Precision={test['precision']:.4f}，Recall={test['recall']:.4f}，F1={test['f1']:.4f}，"
       f"AP={test['average_precision']:.4f}，AUROC={test['roc_auc']:.4f}，Accuracy={test['accuracy']:.4f}。\n"
       f"- 混淆矩阵（真实正常/异常 × 预测正常/异常）：\`{test['confusion_matrix_TN_FP_FN_TP']}\`。\n"
       '- 所有测试结果均由脚本计算；阈值、模型类别和特征子集只使用原始训练数据内部的拟合和验证片段决定。\n'
       '- 完整数据来源和 SHA-256 见 \`source_manifest.json\`。真实数据不包含经确认的三级物理健康状态，因此不对其生成虚假的故障概率或 HMM 结果。\n'
       '- 本模型只接受高质量人工标注数据用于未来**人工审批的受控更新**，不进行无标签自训练。\n',
       encoding='utf-8')
    print(json.dumps({'chosen_by_validation':chosen,
         'feature_selected':len(selected),
         'validation_F1':comparisons[chosen]['f1'],
         'test':test},indent=2))
    return metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results/real_ops_sat')
    a=p.parse_args()
    execute(a.out)

if __name__=='__main__':main()
