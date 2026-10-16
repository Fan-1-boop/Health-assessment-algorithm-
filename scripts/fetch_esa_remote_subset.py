#!/usr/bin/env python3
"""ESA-ADB Mission1 v2 selective remote-channel extractor (exploratory, not official preprocessor).

Source: https://doi.org/10.5281/zenodo.15237121
Never interpret published anomaly labels as physical fault truth.
"""
from __future__ import annotations
import argparse, hashlib, io, json
from pathlib import Path
import numpy as np
import pandas as pd

ESA_URL='https://zenodo.org/records/15237121/files/ESA-Mission1.zip?download=1'
PREFIX='ESA-Mission1/'

def source_metadata(z):
    labels=pd.read_csv(io.BytesIO(z.read(PREFIX+'labels.csv')))
    types=pd.read_csv(io.BytesIO(z.read(PREFIX+'anomaly_types.csv')))
    meta=labels.merge(types[['ID','Category','Class','Subclass']],on='ID',how='left',validate='many_to_one')
    if meta.Category.isna().any():raise ValueError('Unknown official anomaly ID')
    meta['StartTime']=pd.to_datetime(meta.StartTime,utc=True)
    meta['EndTime']=pd.to_datetime(meta.EndTime,utc=True)
    if (meta.EndTime<meta.StartTime).any():raise ValueError('Invalid event interval')
    return meta

def available_channels(z,meta,minimum_anomalies=6,selected=None):
    names={s.split('/')[-1][:-4] for s in z.namelist() if s.startswith(PREFIX+'channels/') and s.endswith('.zip')}
    stats=meta[meta.Category=='Anomaly'].groupby('Channel').ID.nunique().sort_values(ascending=False)
    if selected is not None:
        if selected not in names:raise ValueError('Unavailable selected channel '+selected)
        if int(stats.get(selected,0))<minimum_anomalies:raise ValueError('Too few distinct anomaly IDs for selected channel')
        return selected,stats
    candidates=[x for x in stats.index if x in names and stats[x]>=minimum_anomalies]
    if not candidates:raise ValueError('No channel has sufficient unique anomaly IDs')
    return candidates[0],stats

def extract_channel(z,meta,name,resample='10min',max_staleness='60min'):
    """Fixed 10-minute chronological sampling; use only past observations, no future fills."""
    member=PREFIX+'channels/'+name+'.zip'
    raw=z.read(member)
    frame=pd.read_pickle(io.BytesIO(raw),compression='zip')
    if frame.empty:raise ValueError('Empty channel telemetry')
    frame=frame.copy()
    frame.index=pd.to_datetime(frame.index,utc=True)
    frame=frame.sort_index()
    frame=frame[~frame.index.duplicated(keep='first')]
    column=name if name in frame.columns else ('value' if 'value' in frame.columns else None)
    if column is None:raise ValueError('Unexpected channel columns')
    signal=pd.to_numeric(frame[column],errors='coerce').to_numpy(float)
    observed=frame.index.as_unit('ns').asi8
    target=pd.date_range(frame.index[0].ceil(resample),frame.index[-1].floor(resample),freq=resample)
    times=target.as_unit('ns').asi8
    locations=np.searchsorted(observed,times,side='right')-1
    nearest=locations.clip(0,len(signal)-1)
    values=signal[nearest].copy()
    bad=(locations<0)|(times-observed[nearest]>pd.Timedelta(max_staleness).value)|~np.isfinite(values)
    code=np.zeros(len(times),dtype=np.uint8)
    kinds={'Anomaly':1,'Rare Event':2,'Gap':3,'Communication Gap':3,'Invalid':4}
    order={0:0,2:1,1:2,3:3,4:4}
    events=meta[meta.Channel==name].sort_values('StartTime')
    overlap=0;matching=0
    for _,e in events.iterrows():
        a=int(np.searchsorted(times,int(e.StartTime.value),'left'))
        b=int(np.searchsorted(times,int(e.EndTime.value),'right'))
        if b<=a:continue
        matching+=1
        value=kinds.get(e.Category,3)
        current=code[a:b]
        overlap+=int(np.any((current!=0)&(current!=value)))
        code[a:b]=np.where(np.array([order[value]>order[int(v)] for v in current]),value,current)
    code[bad]=3
    values[bad]=0
    if len(events[events.Category=='Anomaly']) and not np.any(code==1):
        raise ValueError('No surviving true-anomaly points on the fixed grid; check timestamp units, coverage and resolution')
    out=pd.DataFrame({'timestamp':target.strftime('%Y-%m-%d %H:%M:%S.%f%z'),name:values,'is_anomaly_'+name:code})
    audit={'target_channel':name,'source_dataset_doi':'10.5281/zenodo.15237121',
           'nested_channel_archive':member,'source_channel_sha256':hashlib.sha256(raw).hexdigest(),
           'original_sample_count':len(frame),'resampled_sample_count':len(out),
           'resample_rule':resample,'max_staleness':max_staleness,
           'missing_or_stale_points':int(bad.sum()),
           'unique_true_anomaly_event_ids':int(events[events.Category=='Anomaly'].ID.nunique()),
           'unique_rare_nominal_event_ids':int(events[events.Category=='Rare Event'].ID.nunique()),
           'annotation_rows':len(events),'annotations_touching_resample_grid':matching,
           'overlapping_regions':overlap,
           'label_distribution':{str(int(k)):int(v) for k,v in zip(*np.unique(code,return_counts=True))},
           'official_esa_benchmark_preprocessing':False,'includes_telecommands':False,
           'limitations':['Fixed 10-minute grid can miss short events','Exploratory research, NOT official ESA-ADB scores']}
    return out,audit

def fetch_and_write(z,outdir,channel=None,min_events=6,resample='10min'):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    meta=source_metadata(z)
    name,stats=available_channels(z,meta,minimum_anomalies=min_events,selected=channel)
    data,audit=extract_channel(z,meta,name,resample=resample)
    output=outdir/(name+'.csv')
    data.to_csv(output,index=False)
    audit['derived_csv_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    audit['source_url']=ESA_URL
    audit['top_channel_event_counts']={k:int(v) for k,v in stats.head(30).items()}
    (outdir/'extraction_manifest.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'path':str(output),**audit},indent=2,ensure_ascii=False))
    return output,audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default=ESA_URL)
    p.add_argument('--out',default='data/esa_lightweight')
    p.add_argument('--channel',default=None)
    p.add_argument('--min-events',type=int,default=6)
    p.add_argument('--resample',default='10min')
    a=p.parse_args()
    from remotezip import RemoteZip
    with RemoteZip(a.url,timeout=100,initial_buffer_size=1024*512) as archive:
        fetch_and_write(archive,a.out,a.channel,a.min_events,a.resample)

if __name__=='__main__':main()
