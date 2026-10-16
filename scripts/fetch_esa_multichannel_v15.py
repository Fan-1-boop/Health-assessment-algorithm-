#!/usr/bin/env python3
"""Reproducible ESA Mission1 3-channel as-of extraction (NOT official ESA preprocessing).

A label-blind pre-registered cohort is used (v15 protocol). Download only three
nested channel members from the large Zenodo archive using HTTP Range requests.
"""
from __future__ import annotations
import sys,json,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fetch_esa_remote_subset import ESA_URL,source_metadata,extract_channel

COHORT=('channel_29','channel_30','channel_31')
TARGET='channel_29'

def extract(z,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    meta=source_metadata(z)
    catalog_file=output/'mission1_event_catalog.csv'
    meta.sort_values(['StartTime','ID','Channel']).to_csv(catalog_file,index=False)
    frames=[];audits={}
    for name in COHORT:
        f,audit=extract_channel(z,meta,name,resample='10min',max_staleness='60min')
        f=f.rename(columns={'is_anomaly_'+name:'label_'+name})
        f['timestamp']=pd.to_datetime(f['timestamp'],utc=True)
        if f['timestamp'].duplicated().any():raise ValueError('Duplicate time samples')
        frames.append(f);audits[name]=audit
    combined=frames[0]
    for f in frames[1:]:
        combined=combined.merge(f,on='timestamp',how='inner',validate='one_to_one',sort=True)
    if not combined.timestamp.is_monotonic_increasing:raise AssertionError('Unsynchronised timestamps')
    raw_labels=combined[[f'label_{c}' for c in COHORT]].to_numpy(dtype=int)
    target=raw_labels[:,0]
    others=raw_labels[:,1:]
    # Exclude any unusable channel and non-target anomalies, while preserving
    # target anomalies when another channel is also positive.
    invalid=np.any(raw_labels>=3,axis=1)
    other_anomaly=np.any(others==1,axis=1)
    target_anomaly=(target==1)
    rare=np.any(raw_labels==2,axis=1)
    code=np.where(invalid,3,np.where(target_anomaly,1,
          np.where(other_anomaly,3,np.where(rare,2,0)))).astype('uint8')
    combined['eval_label']=code
    # Missing/stale raw readings are placeholders, never seen by model:
    # our later window builder excludes all windows crossing eval_label>=3.
    combined['timestamp']=combined['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f%z')
    dest=output/'three_channels.csv.gz'
    combined.to_csv(dest,index=False,compression={'method':'gzip','compresslevel':5,'mtime':0})
    manifest={'status':'EXPLORATORY_MULTI_CHANNEL_NOT_ESA_OFFICIAL',
        'source':ESA_URL,'channels':COHORT,'primary_target':TARGET,
        'resampling':'10min strictly historical-asof, max_staleness=60min',
        'source_channel_audits':audits,
        'rows_common_grid':int(len(combined)),
        'evaluation_label_counts':{str(k):int(v) for k,v in zip(*np.unique(code,return_counts=True))},
        'target_anomaly_points_masked_by_context_gap':int(np.sum(target_anomaly & invalid)),
        'context_other_anomaly_points_excluded':int(np.sum((~target_anomaly)&other_anomaly&(~invalid))),
        'csv_gzip_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'catalog_sha256':hashlib.sha256(catalog_file.read_bytes()).hexdigest(),
        'warnings':['No physical health-state labels',
                    'Target anomaly only; overlapping context-only anomalies excluded for fair target comparison',
                    '10min downsample may eliminate short events',
                    'Some official IDs may span channels and previous explored cohorts']}
    (output/'extraction_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'grid_rows':len(combined),'label_counts':manifest['evaluation_label_counts'],
         'source_audits':{k:v['original_sample_count'] for k,v in audits.items()}},indent=2))
    return dest,catalog_file,manifest

def main():
    import argparse
    from remotezip import RemoteZip
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='data/esa_v15_multichannel')
    args=p.parse_args()
    with RemoteZip(ESA_URL,timeout=120,initial_buffer_size=1024*512) as z:
        extract(z,args.out)

if __name__=='__main__':main()
