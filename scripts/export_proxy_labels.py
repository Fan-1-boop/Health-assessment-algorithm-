#!/usr/bin/env python3
"""Export preregistered NASA anomaly-interval-position proxy labels.

NOT verified degradation/fault labels. No telemetry amplitudes used.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_v11 import load_frozen_labels


def build_label_table(n, intervals, states, anomaly, rule_id, rule_hash):
    event_id=np.full(n,-1,dtype=int)
    offset=np.full(n,-1,dtype=int)
    for k,(start,end) in enumerate(intervals):
        event_id[start:end+1]=k
        offset[start:end+1]=np.arange(end-start+1)
    return pd.DataFrame({
        'channel_id':'P-1',
        'timestep':np.arange(n),
        'nasa_binary_anomaly_annotation':anomaly,
        'proxy_state_code':states,
        'proxy_state_name':np.array(['healthy_proxy','degraded_proxy','fault_proxy'])[states],
        'published_anomaly_event_index':event_id,
        'offset_within_event':offset,
        'label_source':'deterministic_anomaly_interval_position',
        'physical_fault_confirmed':False,
        'frozen_rule_id':rule_id,
        'frozen_rule_sha256':rule_hash})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',default=str(ROOT/'data/raw/labeled_anomalies.csv'))
    p.add_argument('--output',default=str(ROOT/'data/derived/P-1_proxy_labels_v1.csv'))
    a=p.parse_args()
    metadata=pd.read_csv(a.metadata)
    entry=metadata[(metadata.chan_id=='P-1')&(metadata.spacecraft=='SMAP')]
    if len(entry)!=1:raise ValueError('Expected exactly one P-1 entry')
    n=int(entry.iloc[0]['num_values'])
    states,anomaly,intervals,sha,rule=load_frozen_labels(n,metadata,ROOT/'configs/proxy_label_rules_v1.json')
    table=build_label_table(n,intervals,states,anomaly,rule['rule_id'],sha)
    destination=Path(a.output);destination.parent.mkdir(exist_ok=True,parents=True)
    table.to_csv(destination,index=False)
    print(f'Exported {n} rows to {destination}; proxy counts={table.proxy_state_name.value_counts().to_dict()}')

if __name__=='__main__':main()
