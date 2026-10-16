#!/usr/bin/env python3
"""Data adequacy gate and preflight for legacy P-1 proxy state target."""
import sys, csv,ast, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from v13_protocol import event_first_partition
import numpy as np


def smap_p1_preflight(labels_csv):
    rows=list(csv.DictReader(open(labels_csv,encoding='utf-8')))
    matched=[r for r in rows if r.get('chan_id',r.get('channel_id'))=='P-1']
    if not matched:raise ValueError('P-1 channel not found')
    intervals=ast.literal_eval(matched[0]['anomaly_sequences'])
    end=max(int(b) for a,b in intervals)+200
    c=np.zeros(end,dtype=int)
    for start,stop in intervals:
        c[int(start):int(stop)+1]=1
    from v13_protocol import find_events
    n=len([e for e in find_events(c) if e.label==1])
    try:
        event_first_partition(c,min_anomalies_per_split=2)
        status='PASS'
    except ValueError as exc:
        status='BLOCKED: '+str(exc)
    return {'dataset':'SMAP P-1','anomaly_episodes':n,
            'protocol':'2 independent anomalies per train/validation/test split',
            'status':status,'proxy_labels_are_real_faults':False,
            'guidance':'retain V1.2 as legacy; do not random-shuffle overlapping windows to bypass gate'}

if __name__=='__main__':
    p=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'data/raw/labeled_anomalies.csv'
    print(json.dumps(smap_p1_preflight(p),indent=2,ensure_ascii=False))
