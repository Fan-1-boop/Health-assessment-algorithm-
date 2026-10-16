#!/usr/bin/env python3
"""V1.5 ESA-Mission1 event-ID-pure 3-channel anomaly ablation.

Frozen protocol in configs/v15_esa_protocol_frozen.json must predate all test
metrics. This is NOT the official ESA-ADB 30s/TimeEval benchmark. The entire
ESA mission has been explored before on other channels, so claim exploratory
validation rather than strict mission-wide untouched holdout.
"""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from v13_protocol import windows_in_partition
from v13_anomaly import RobustLevelDeviation,WindowedIForest,ForecastRidge,score_metrics
from v14_alarm import sustained_alert,validation_nominal_threshold,alert_statistics
from v15_global_events import CHANNELS,TARGET,get_global_event_records,global_id_partitions,event_alarm_metrics

def _model(name):
    if name=='robust_level_deviation': return RobustLevelDeviation()
    if name=='windowed_iforest':return WindowedIForest(seed=42)
    if name=='autoregressive_ridge':return ForecastRidge()
    raise ValueError('Unregistered model: '+name)


def run(source,catalog_path,out,protocol_path):
    protocol=json.loads(Path(protocol_path).read_text())
    if tuple(protocol['context_channels'])!=CHANNELS[1:] or protocol['primary_target']!=TARGET:
        raise ValueError('Frozen cohort does not match script')
    df=pd.read_csv(source)
    required={'timestamp','eval_label',*(list(CHANNELS)),*(f'label_{c}' for c in CHANNELS)}
    if not required.issubset(df.columns):raise ValueError('Missing ESA source columns: '+str(required-set(df.columns)))
    times=pd.to_datetime(df.timestamp,utc=True)
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError('Invalid chronological source')
    codes=df.eval_label.to_numpy(int)
    if not np.isin(codes,[0,1,2,3,4]).all():raise ValueError('Invalid evaluation codes')
    signals=df[list(CHANNELS)].to_numpy(float)
    if not np.isfinite(signals).all():raise ValueError('Bad telemetry values')
    annotation=pd.read_csv(catalog_path)
    events=get_global_event_records(annotation,times,codes)
    lookback=int(protocol['lookback']);stride=int(protocol['stride'])
    parts,audit=global_id_partitions(events,len(df),lookback,
                        int(protocol['min_target_anomaly_ids_per_partition']))
    windows={};end={};y={}
    for p in parts:
        windows[p.name],end[p.name],y[p.name],_=windows_in_partition(
             signals,codes,p,lookback=lookback,stride=stride)
        if not (np.any(y[p.name]==1) and np.any(y[p.name]==0)):
            raise ValueError(f'BLOCKED: {p.name} no positive/negative evaluation windows')
    e=end['train'];cs=np.r_[0,(codes!=0).astype(int).cumsum()]
    good=(cs[e+1]-cs[e+1-lookback])==0
    train=windows['train'][good]
    if len(train)<200:raise ValueError('BLOCKED: too few fully nominal train windows')
    if len(train)>20000:
        train=train[np.linspace(0,len(train)-1,20000,dtype=int)]
    names=list(protocol['models'])
    variants=list(protocol['variants'])
    if set(names)!={'robust_level_deviation','windowed_iforest','autoregressive_ridge'}:
        raise ValueError('Unrecognized model set')
    if variants!=['target_only','target_plus_context']:
        raise ValueError('Unrecognized context variants')
    results={};val={}
    for variant in variants:
        slice_x=(lambda a:a[:,:,:1]) if variant=='target_only' else (lambda a:a)
        for name in names:
            key=name+'__'+variant
            m=_model(name).fit(slice_x(train))
            s_val=m.score(slice_x(windows['validation']))
            s_test=m.score(slice_x(windows['test']))
            threshold=validation_nominal_threshold(s_val,end['validation'],codes,
                                    protocol['alarm_policy']['nominal_FPR_budget_on_validation']
                                    if 'nominal_FPR_budget_on_validation' in protocol['alarm_policy']
                                    else .002)
            a=sustained_alert(s_test,threshold,end['test'],
                       min_run=int(protocol['alarm_policy']['min_consecutive_above_threshold']),
                       expected_stride=stride)
            val[key]={**score_metrics(y['validation'],s_val),
                      'threshold':threshold,
                      'validation_nominal_samples':int(np.sum(codes[end['validation']]==0))}
            results[key]={'score_quality':score_metrics(y['test'],s_test),
                  'alert_occupancy':alert_statistics(end['test'],a,codes,stride),
                  'global_event_metrics':event_alarm_metrics(events,parts[-1],end['test'],codes,a,
                                         stride,grid_step_minutes=10)}
    selected=max(sorted(val),key=lambda name:val[name]['point_ap'])
    metrics={'version':'1.5.0',
      'study':'ESA Mission1 3-channel cross-channel event-ID-pure split',
      'research_scope':'exploratory, not independent entire-mission or ESA official benchmark',
      'source':str(source),'target':TARGET,'signal_inputs':list(CHANNELS),
      'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
      'catalog_sha256':hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(),
      'frozen_protocol_sha256':hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest(),
      'split':audit,'window_counts':{k:len(v) for k,v in windows.items()},
      'nominal_training_windows_used':len(train),
      'sampled_test_positive_prevalence':float(y['test'].mean()),
      'selection_rule':'validation AP over frozen model x context grid',
      'validation':val,'chosen_by_validation':selected,
      'selected_test':results[selected],
      'all_test_diagnostics_NOT_for_model_selection':results,
      'limitations':['No physical 3-state fault labels; no fusion',
         '10-minute as-of is nonofficial and may miss short events',
         'No telecommands; three channels in the same official subsystem',
         'Some events IDs may overlap with previously explored channel 21/41 IDs',
         'Global ID split does not guarantee independent physical fault mechanism',
         'Test score may not be used to change models, windows, thresholds or alarm policy']}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False,default=int)+'\n')
    def summarize(result):
        e=result['global_event_metrics'];v=result['score_quality']
        a=result['alert_occupancy']
        return {'AP':v['point_ap'],'ROC_AUC':v['point_roc_auc'],
             'coverage_recall':e['coverage_ID_recall'],
             'onset_recall':e['strict_new_onset_ID_recall'],
             'false_alarm_onsets_per_30d':e['false_alarm_new_onsets_per_30_elapsed_days'],
             'normal_alert_rate':a['normal_alert_rate'],
             'rare_alert_rate':a['rare_nominal_alert_rate']}
    print(json.dumps({'chosen':selected,'global_event_counts':{
        n:len(audit['global_id_splits'][n]['target_evaluable_anomaly']) for n in ('train','validation','test')},
        'validation_AP':val[selected]['point_ap'],
        'selected_test':summarize(results[selected]),
        'all_test_diagnostics':{k:summarize(v) for k,v in results.items()}},
        ensure_ascii=False,indent=2))
    return metrics


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--csv',default='data/esa_v15_multichannel/three_channels.csv.gz')
    a.add_argument('--catalog',default='data/esa_v15_multichannel/mission1_event_catalog.csv')
    a.add_argument('--config',default='configs/v15_esa_protocol_frozen.json')
    a.add_argument('--out',default='results/v15/esa_multichannel')
    x=a.parse_args()
    run(x.csv,x.catalog,x.out,x.config)

if __name__=='__main__':main()
