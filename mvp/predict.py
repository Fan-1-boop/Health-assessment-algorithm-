#!/usr/bin/env python3
"""Load a previously trained synthetic health MVP and predict without retraining.

Accepted input is the simulator's column schema: mission_id, cycle_id,
the 8 named raw channels and optionally sample_in_mission. No physical
fault labels, test labels or human corrections are required for inference.

Only load model_bundle.joblib from a trusted source.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'mvp'))
from synthetic import CHANNELS,make_windows,highdim_features
from v13_state import predict_hierarchical,filter_groups
from run import stacking_inputs

def infer(input_csv,model_file,output_csv,mission=None):
    frame=pd.read_csv(input_csv)
    required={'mission_id','cycle_id',*CHANNELS}
    if not required.issubset(frame.columns):
        raise ValueError('Required columns missing: '+str(required-set(frame.columns)))
    # Chronological order is an input contract. Never re-sort inside
    # this online causal prediction path.
    if frame[['mission_id','cycle_id']].isna().any().any():
        raise ValueError('Missing sequence identity')
    if any(np.diff(p.sample_in_mission.to_numpy())<0
           for _,p in frame.groupby('mission_id',sort=False)) if 'sample_in_mission' in frame else False:
        raise ValueError('Mission records must be chronological')
    pack=joblib.load(model_file)
    if tuple(pack['raw_sensor_channels'])!=CHANNELS:
        raise ValueError('Model expects a different telemetry schema')
    windows,ends=make_windows(frame)
    if mission is None:
        chosen=np.ones(len(ends),bool)
    else:
        chosen=(ends.mission_id.to_numpy()==int(mission))
    if not np.any(chosen):raise ValueError('No windows for requested mission')
    x,fn=highdim_features(windows[chosen])
    if fn!=pack['feature_names']:
        raise ValueError('Feature schema changed since training')
    groups=ends.loc[chosen,'cycle_id'].to_numpy(int)
    h=pack['hierarchical']
    post=predict_hierarchical(h,x)
    base=filter_groups(h,post,groups,gamma=.6,memory=.5)
    raw=windows[chosen].reshape(int(chosen.sum()),-1)
    d=pack['anomaly_detector'].predict_proba(raw)[:,1]
    final=pack['fusion'].predict_proba(stacking_inputs(base,d))
    scores=final@pack['corrected_score_mapping']
    output=ends.loc[chosen,['mission_id','cycle_id']].reset_index(drop=True).copy()
    if 'sample_in_mission' in ends:
        output['sample_in_mission']=ends.loc[chosen,'sample_in_mission'].to_numpy(int)
    output['p_healthy']=final[:,0]
    output['p_degraded']=final[:,1]
    output['p_fault']=final[:,2]
    output['p_anomaly']=d
    output['anomaly_flag']=(d>=pack['anomaly_threshold']).astype(int)
    output['health_score_predicted']=scores
    output['state_predicted']=np.argmax(final,axis=1)
    path=Path(output_csv);path.parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(path,index=False)
    print('Predicted',len(output),'causal windows without retraining ->',path)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/mvp/synthetic_satellite_telemetry.csv')
    p.add_argument('--model',default='results/mvp/model_bundle.joblib')
    p.add_argument('--mission',type=int,default=3)
    p.add_argument('--out',default='results/mvp/inference_reloaded.csv')
    a=p.parse_args()
    infer(a.input,a.model,a.out,a.mission)
