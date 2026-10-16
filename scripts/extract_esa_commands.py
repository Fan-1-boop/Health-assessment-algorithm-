#!/usr/bin/env python3
"""Extract bounded-cost ESA telecommand pulses from official Mission1 v2 ZIP.

Command selection is deterministic FROM command priority and archive sizes
only, not spacecraft anomaly or rare-event labels, and not test metrics.
Used for exploratory context ablation; not the ESA official preprocessor.
"""
from __future__ import annotations
import argparse,hashlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd

ESA_URL='https://zenodo.org/records/15237121/files/ESA-Mission1.zip?download=1'
PREFIX='ESA-Mission1/telecommands/'


def select_commands(catalog,members,priority_at_least=3,count=8,max_archive_bytes=250000):
    """Fixed, label-blind selection of recurrent command candidates by archive size."""
    names={n[len(PREFIX):-4]:size for n,size in members.items()
           if n.startswith(PREFIX) and n.endswith('.zip')}
    rows=catalog[(catalog['Priority']>=priority_at_least)&
                 catalog['Telecommand'].isin(names)].copy()
    rows['compressed_size']=rows.Telecommand.map(names)
    rows=rows[rows.compressed_size<=max_archive_bytes]
    # Largest compressed archives within budget more likely contain repeated
    # commands; this is a heuristic and NOT learned from target labels.
    rows=rows.sort_values(['compressed_size','Telecommand'],ascending=[False,True])
    if len(rows)<count:raise ValueError('Too few command archives under frozen size cap')
    return rows.iloc[:count][['Telecommand','Priority','compressed_size']].to_dict('records')


def pulses_asof(target_times,positive_command_timestamps,window='10min'):
    ts=pd.DatetimeIndex(pd.to_datetime(target_times,utc=True))
    ev=pd.DatetimeIndex(pd.to_datetime(positive_command_timestamps,utc=True)).sort_values()
    if ts.hasnans or not ts.is_monotonic_increasing or ts.has_duplicates:
        raise ValueError('Bad target chronology')
    if len(ev)==0:return np.zeros(len(ts),dtype=np.uint8)
    t=ts.as_unit('ns').asi8; e=ev.as_unit('ns').asi8
    width=pd.Timedelta(window).value
    count=np.searchsorted(e,t,side='right')-np.searchsorted(e,t-width,side='right')
    return (count>0).astype(np.uint8)


def read_command_events(archive,command_name):
    raw=archive.read(PREFIX+command_name+'.zip')
    df=pd.read_pickle(io.BytesIO(raw),compression='zip')
    if df.empty:return pd.DatetimeIndex([],tz='UTC'),hashlib.sha256(raw).hexdigest()
    df.index=pd.to_datetime(df.index,utc=True)
    vals=df.iloc[:,0]
    positive=pd.to_numeric(vals,errors='coerce').fillna(0).to_numpy()!=0
    return pd.DatetimeIndex(df.index[positive]),hashlib.sha256(raw).hexdigest()


def extract_commands(archive,target_times,outdir,channel,
                     count=8,max_archive_bytes=250000,window='10min'):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    info=pd.read_csv(io.BytesIO(archive.read('ESA-Mission1/telecommands.csv')))
    sizes={n:archive.getinfo(n).compress_size for n in archive.namelist()
           if n.startswith(PREFIX) and n.endswith('.zip')}
    selected=select_commands(info,sizes,count=count,max_archive_bytes=max_archive_bytes)
    t=pd.DatetimeIndex(pd.to_datetime(target_times,utc=True))
    output=pd.DataFrame({'timestamp':t.strftime('%Y-%m-%d %H:%M:%S.%f%z')})
    for row in selected:
        cmd=row['Telecommand']
        events,sha=read_command_events(archive,cmd)
        output[cmd]=pulses_asof(t,events,window=window)
        row['source_sha256']=sha
        row['observed_positive_events']=int(len(events))
        row['context_positive_grid_rows']=int(output[cmd].sum())
    output_path=outdir/'command_pulses.csv';output.to_csv(output_path,index=False)
    manifest={'study':'ESA Mission1 single-channel command-context exploratory ablation',
              'source_url':ESA_URL,'target_channel':channel,
              'selection':'Priority >=3 then largest 8 archives <=250000 compressed bytes, deterministic, no labels',
              'lookback_for_pulses':window,
              'n_grid_points':int(len(output)),
              'commands':selected,
              'context_csv_sha256':hashlib.sha256(output_path.read_bytes()).hexdigest(),
              'label_usage_in_command_selection':False,
              'not_official_esa_preprocessing':True,
              'warning':'Command streams are anonymized; no claim of physical relevance to target sensor.'}
    (outdir/'command_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return output_path,manifest


def main():
    import zipfile64.zipfile as _patched_zipfile  # patch Deflate64 in zipfile
    from remotezip import RemoteZip
    arg=argparse.ArgumentParser()
    arg.add_argument('--telemetry',required=True)
    arg.add_argument('--channel',required=True)
    arg.add_argument('--out',required=True)
    arg.add_argument('--count',type=int,default=8)
    a=arg.parse_args()
    ts=pd.read_csv(a.telemetry,usecols=['timestamp']).timestamp
    with RemoteZip(ESA_URL,timeout=110,initial_buffer_size=1024*512) as z:
        path,m=extract_commands(z,ts,a.out,a.channel,count=a.count)
        print(json.dumps({'output':str(path),'commands':m['commands']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
