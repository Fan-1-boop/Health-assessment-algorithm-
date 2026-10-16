#!/usr/bin/env python3
"""ESA Mission1 single-channel telecommand context ablation (EXPLORATORY).

Our channel_21 test metrics have already been inspected in V1.4. Consequently
this post-hoc command-context study is strictly diagnostic: NO new unbiased
generalization claim may be made on channel_21, even with validation-only
calibration of the two candidate models.
"""
import sys,argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from v13_protocol import read_esa_preprocessed,windows_in_partition
from v13_anomaly import ForecastRidge,score_metrics
from v14_context import TargetRidgeWithCommands
from v14_event_ids import sampled_events,make_id_safe_partitions,id_event_metrics
from v14_alarm import sustained_alert,validation_nominal_threshold,alert_statistics


def run(telemetry,catalog,commands,channel,out,lookback=16,stride=4):
    raw,codes,timestamps,names=read_esa_preprocessed(telemetry,channel)
    df=pd.read_csv(commands)
    if len(df)!=len(raw) or not np.array_equal(df.timestamp.astype(str).to_numpy(),np.asarray(timestamps).astype(str)):
        raise ValueError('Command and telemetry timestamps are not aligned: no positional joining')
    cmd_names=[c for c in df.columns if c!='timestamp']
    if len(cmd_names)<1:raise ValueError('No command columns')
    ctx=df[cmd_names].to_numpy(dtype=float)
    if not np.isfinite(ctx).all() or np.any(ctx<0):raise ValueError('Invalid command context')
    enriched=np.column_stack([raw[:,0],ctx])
    events=sampled_events(pd.read_csv(catalog),channel,timestamps,codes)
    parts,audit=make_id_safe_partitions(events,len(codes),lookback=lookback,channel=channel)
    ends={};wx={};labels={}
    for p in parts:
        wx[p.name],ends[p.name],labels[p.name],_=windows_in_partition(
            enriched,codes,p,lookback=lookback,stride=stride)
    e=ends['train']; bad=(codes!=0).astype(int);cum=np.r_[0,bad.cumsum()]
    good=cum[e+1]-cum[e+1-lookback]==0
    train=wx['train'][good]
    if len(train)<200:raise ValueError('Not enough clean nominal windows')
    if len(train)>20000:
        train=train[np.linspace(0,len(train)-1,20000,dtype=int)]
    models={'ridge_target_only':ForecastRidge(),
            'ridge_target_plus_commands':TargetRidgeWithCommands()}
    val={};test={}
    for name,model in models.items():
        sliced=lambda a:a[:,:,:1] if name=='ridge_target_only' else a
        model.fit(sliced(train))
        sv=model.score(sliced(wx['validation']))
        st=model.score(sliced(wx['test']))
        th=validation_nominal_threshold(sv,ends['validation'],codes,.002)
        alert=sustained_alert(st,th,ends['test'],min_run=3,expected_stride=stride)
        val[name]=score_metrics(labels['validation'],sv)
        test[name]={**score_metrics(labels['test'],st),
                    **alert_statistics(ends['test'],alert,codes,stride),
                    **id_event_metrics(events,parts[2],ends['test'],alert,codes)}
    # Validation ranking is given for transparency; do not declare either as
    # a new confirmatory winner on this previously seen channel.
    winner=max(val,key=lambda k:val[k]['point_ap'])
    result={'study':'command context diagnostic on previously exposed ESA channel_21 holdout',
            'status':'EXPLORATORY_ONLY; NO CONFIRMATORY TEST CLAIM',
            'telemetry':str(telemetry),'channel':channel,
            'lookback':lookback,'stride':stride,'context_columns':cmd_names,
            'validation':val,'validation_preferred':winner,
            'exposed_test_diagnostic':test,'split_id_audit':audit,
            'command_file_sha256':hashlib.sha256(Path(commands).read_bytes()).hexdigest(),
            'anomaly_definition':'official Anomaly=positive, Rare Event=negative, Gap=excluded',
            'limitations':['Event IDs may share one physical cause',
                           'Anonymized command channels not proven relevant to this sensor',
                           '10-minute sampled command pulses may miss short command sequences',
                           'Already-exposed channel_21 test results invalidate confirmatory claims',
                           'No physical three-state fault labels and no late fusion']}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'validation':val,'test_diagnostic':{k:{
       'AP':v['point_ap'],'ID_recall':v['official_id_event_recall'],
       'normal_alerts':v['normal_alert_windows'],
       'rare_alerts':v['rare_nominal_alert_windows']} for k,v in test.items()}},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True)
    p.add_argument('--catalog',required=True)
    p.add_argument('--commands',required=True)
    p.add_argument('--channel',default='channel_21')
    p.add_argument('--out',default='results/v14/command_context')
    args=p.parse_args()
    run(args.csv,args.catalog,args.commands,args.channel,args.out)
