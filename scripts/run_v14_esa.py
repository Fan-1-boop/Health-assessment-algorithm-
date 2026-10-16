#!/usr/bin/env python3
"""V1.4 ESA anomaly experiment: official-ID-aware, cross-channel exploratory test.

This run is not the official ESA-ADB benchmark; it uses one channel, 10-minute
causal sampling, and a new predetermined alerting policy. Do not merge its
anomaly labels with any synthetic 3-state physical fault interpretation.
"""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from v13_protocol import read_esa_preprocessed, windows_in_partition
from v13_anomaly import RobustLevelDeviation,WindowedIForest,ForecastRidge,score_metrics
from v14_event_ids import sampled_events,make_id_safe_partitions,id_event_metrics
from v14_alarm import sustained_alert,validation_nominal_threshold,alert_statistics

MODEL_NAMES=('robust_level_deviation','windowed_iforest','autoregressive_ridge')
BUDGET=.002      # frozen before new channel_21 performance experiment
PERSIST=3        # frozen before new channel_21 performance experiment

def run(input_csv,catalog_csv,channel,out,lookback=16,stride=4,minimum=2):
    X,codes,times,names=read_esa_preprocessed(input_csv,channel)
    cat=pd.read_csv(catalog_csv)
    records=sampled_events(cat,channel,times,codes)
    parts,audit=make_id_safe_partitions(records,len(codes),lookback=lookback,
                                         min_anomaly_ids=minimum,channel=channel)
    if len({x for ids in audit['official_anomaly_event_ids'].values() for x in ids}) != sum(
        len(ids) for ids in audit['official_anomaly_event_ids'].values()):
        raise AssertionError('Official anomaly IDs leak across splits')
    x={};ends={};y={};rare={}
    for p in parts:
        x[p.name],ends[p.name],y[p.name],rare[p.name]=windows_in_partition(
            X,codes,p,lookback=lookback,stride=stride)
        if len(np.unique(y[p.name]))<2:
            raise ValueError(f'BLOCKED: {p.name} has no positive/negative sampled evaluation windows')
    # No train labels enter unsupervised nominal-reference model.
    e=ends['train']
    csum=np.r_[0,(codes!=0).astype(int).cumsum()]
    purely_nominal=(csum[e+1]-csum[e+1-lookback]==0)
    xt=x['train'][purely_nominal]
    if len(xt)<200:raise ValueError('BLOCKED: insufficient purely nominal training windows')
    if len(xt)>20000:xt=xt[np.linspace(0,len(xt)-1,20000,dtype=int)]
    models=dict(zip(MODEL_NAMES,(RobustLevelDeviation(),WindowedIForest(),ForecastRidge())))
    val={};test_scores={}
    for name,model in models.items():
        model.fit(xt)
        sv=model.score(x['validation'])
        st=model.score(x['test'])
        threshold=validation_nominal_threshold(sv,ends['validation'],codes,BUDGET)
        val[name]={**score_metrics(y['validation'],sv),'threshold':threshold,
                    'nominal_point_budget':BUDGET,
                    'n_validation_nominal':int((codes[ends['validation']]==0).sum())}
        test_scores[name]=st
    # Only validation AP selects model. Test metrics of every model diagnostic.
    chosen=max(MODEL_NAMES,key=lambda name:val[name]['point_ap'])
    comparison={}
    for name,scores in test_scores.items():
        th=val[name]['threshold']
        raw=sustained_alert(scores,th,ends['test'],min_run=1,expected_stride=stride)
        persistent=sustained_alert(scores,th,ends['test'],min_run=PERSIST,expected_stride=stride)
        comparison[name]={'point_scores':score_metrics(y['test'],scores)}
        for title,pred in [('raw_threshold',raw),('persistence_3',persistent)]:
            comparison[name][title]={
               **alert_statistics(ends['test'],pred,codes,stride),
               **id_event_metrics(records,parts[2],ends['test'],pred,codes)}
    res={'version':'1.4.0','data_provenance':'ESA-ADB Mission1 v2, official event IDs; 10-minute remote subset',
         'evaluation_nature':'exploratory; this channel has no prior published model performance in our project'
                                if channel!='channel_41' else 'diagnostic reuse of exposed channel_41',
         'target_channel':channel,'raw_telemetry_input_columns':names,
         'sample_rate':'10min','lookback':lookback,'stride':stride,
         'source_sha256':hashlib.sha256(Path(input_csv).read_bytes()).hexdigest(),
         'catalog_sha256':hashlib.sha256(Path(catalog_csv).read_bytes()).hexdigest(),
         'official_ids':audit,'official_event_id_counts':{
           p.name:{kind:len(audit[f'official_{kind}_event_ids'][p.name]) for kind in ('anomaly','rare')}
           for p in parts},
         'note':'Official IDs are not equivalent to statistically independent physical failure mechanisms.',
         'evaluated_windows':{p.name:int(len(ends[p.name])) for p in parts},
         'train_pure_nominal_windows_used':int(len(xt)),
         'validation':val,
         'selection_rule':'fixed model set; validation AP only',
         'selected_by_validation':chosen,
         'alarm_policy_predeclared':{'nominal_FPR_budget_on_validation':BUDGET,
                                     'minimum_consecutive_evaluation_windows':PERSIST,
                                     'legacy_raw_for_comparison':True},
         'test_all_candidates_diagnostic_only':comparison,
         'selected_test':comparison[chosen],
         'limitations':[
            '10-minute as-of resampling changes ESA benchmark and misses short events',
            'No telecommands/cross-channel inputs',
            'Some official events may be unevaluable on 10-minute grid or chosen stride',
            'Official IDs may represent multichannel correlated events, not independent failures',
            'No physical health/degradation/fault truth; no HMM or fusion inference',
            'Not the official Telemanom-ESA LSTM or ESA corrected F0.5 benchmark',
            'Reusing previous channel_41 holdout provides diagnostic analysis only',
            'Once seen, these test metrics cannot be used for new confirmatory tuning']}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    small={'selected':chosen,'channel':channel,'validation_AP':val[chosen]['point_ap'],
        'test_AP':comparison[chosen]['point_scores']['point_ap'],
        'official_ID_recall':comparison[chosen]['persistence_3']['official_id_event_recall'],
        'false_alarm_episodes':comparison[chosen]['persistence_3']['false_alarm_episodes']}
    print(json.dumps(small,indent=2))
    return res

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True)
    p.add_argument('--catalog',required=True)
    p.add_argument('--channel',default='channel_21')
    p.add_argument('--out',default='results/v14/esa_channel_21')
    p.add_argument('--lookback',type=int,default=16)
    p.add_argument('--stride',type=int,default=4)
    p.add_argument('--min-official-ids-per-split',type=int,default=2)
    args=p.parse_args()
    run(args.csv,args.catalog,args.channel,args.out,
        args.lookback,args.stride,args.min_official_ids_per_split)
