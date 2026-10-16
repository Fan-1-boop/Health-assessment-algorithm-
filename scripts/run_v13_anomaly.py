#!/usr/bin/env python3
"""V1.3 event-blocked spacecraft anomaly baseline on official ESA preprocessed CSV.

Not the full ESA-ADB benchmark nor its official corrected F0.5 implementation.
Requires an independently downloaded Mission1 dataset. Does not download 3.8 GB zip.
"""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from v13_protocol import read_esa_preprocessed,event_first_partition,windows_in_partition,diagnostic_manifest
from v13_anomaly import (RobustLevelDeviation,WindowedIForest,ForecastRidge,
    score_metrics,episode_metrics,threshold_at_normal_false_positive_budget)


def run(input_csv, channel, out, *, lookback=64,stride=8,max_rows=None,
        min_events_per_split=2,false_alarm_budget=.02):
    X,codes,times,names=read_esa_preprocessed(input_csv,channel,max_rows)
    partitions=event_first_partition(codes,min_events_per_split)
    x={};index={};y={};rare={}
    for part in partitions:
        x[part.name],index[part.name],y[part.name],rare[part.name]=windows_in_partition(
            X,codes,part,lookback=lookback,stride=stride)
    if not (len(np.unique(y['validation']))==2 and len(np.unique(y['test']))==2):
        raise ValueError('Not enough sampled true anomalies after window construction')
    # Train unsupervised / semi-supervised detector on strictly nominal *entire* windows.
    e=index['train'];nominal=(codes[e]==0)
    not_nominal=(codes!=0).astype(int);cs=np.r_[0,not_nominal.cumsum()]
    nominal &= (cs[e+1]-cs[e+1-lookback]==0)
    xn=x['train'][nominal]
    if len(xn)<50:raise ValueError('Insufficient purely normal training windows')
    if len(xn)>20000:
        xn=xn[np.linspace(0,len(xn)-1,20000,dtype=int)]
    models={'robust_level_deviation':RobustLevelDeviation(),
            'windowed_iforest':WindowedIForest(),
            'autoregressive_ridge':ForecastRidge()}
    val={};scores={}
    for name,model in models.items():
        model.fit(xn)
        sv=model.score(x['validation']);st=model.score(x['test'])
        threshold=threshold_at_normal_false_positive_budget(sv,y['validation'],false_alarm_budget)
        val[name]={**score_metrics(y['validation'],sv),'threshold_from_validation':threshold}
        scores[name]=st
    # Select only by validation AP: no test peeking.
    best=max(val,key=lambda name:val[name]['point_ap'])
    results={}
    for name,score in scores.items():
        th=val[name]['threshold_from_validation']
        test_p=partitions[-1]
        results[name]={**score_metrics(y['test'],score),
            **episode_metrics(index['test'],y['test'],score,th,test_p.events),
            'false_positive_points_rare_nominal':int(np.sum((score>=th)&(rare['test']==1))),
            'rare_nominal_evaluated_points':int(np.sum(rare['test']==1)),
            'normal_false_positive_points':int(np.sum((score>=th)&(codes[index['test']]==0))),
            'test_points':int(len(score))}
    manifest={'version':'1.3.0','research_status':'REAL_DATA_IF_RUN_WITH_OFFICIAL_ESA_CSV',
         'selection_rule':'maximize validation point AP over 3 fixed baselines only',
         'labels':'ESA official per-channel codes; positive=1, rare nominal code 2 is NEGATIVE',
         'source':str(input_csv),'target_channel':channel,
         'source_files_not_in_repository':True,
         'source_checksum_sha256':hashlib.sha256(Path(input_csv).read_bytes()).hexdigest() if Path(input_csv).stat().st_size<40_000_000 else 'omit_large_file_hash_record_externally',
         'n_signals':len(names),'input_signal_columns':names,
         'window_length':lookback,'window_stride':stride,
         'false_alarm_budget_on_validation_nominal':false_alarm_budget,
         'event_protocol':diagnostic_manifest(partitions,codes),
         'samples_per_split':{n:len(y[n]) for n in ('train','validation','test')},
         'pure_normal_training_windows':len(xn),
         'validation_metrics':val,
         'selected_by_validation':best,
         'heldout_test_all_candidates_DIAGNOSTIC_ONLY':results,
         'selected_heldout_test':results[best],
         'limitations':['This is a subset research baseline, NOT official ESA-ADB metric reproduction.',
                        'Sampled windows have stride >1; detection delay measured in raw time steps.',
                        'Per-channel event codes may group overlapping annotated events; prefer official ID table for event-level external reporting.',
                        'Repeated use of the same holdout during development invalidates confirmatory inference.',
                        'Telemanom-ESA is NOT implemented or evaluated by this script.']}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=int)+'\n',encoding='utf-8')
    print(json.dumps({'selected':best,'test':results[best],
          'anomaly_episodes':manifest['event_protocol']['splits']},indent=2,ensure_ascii=False))
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--esa-preprocessed-csv',required=True,
                    help='Output CSV of official Mission1_semisupervised_prep_from_raw.py')
    ap.add_argument('--channel',required=True)
    ap.add_argument('--out',default=str(ROOT/'results/v13/esa_mission1'))
    ap.add_argument('--lookback',type=int,default=64)
    ap.add_argument('--stride',type=int,default=8)
    ap.add_argument('--min-events-per-split',type=int,default=2)
    ap.add_argument('--max-rows',type=int,default=None,
                    help='Only for exploratory data validation, never for final performance claims')
    args=ap.parse_args()
    run(args.esa_preprocessed_csv,args.channel,args.out,lookback=args.lookback,
        stride=args.stride,max_rows=args.max_rows,min_events_per_split=args.min_events_per_split)
