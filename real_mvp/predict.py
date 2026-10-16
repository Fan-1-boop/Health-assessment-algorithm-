#!/usr/bin/env python3
"""Predict anomaly probability from official OPS-SAT-AD statistical features.

Does not need anomaly labels for inference; 'train' is used ONLY for the
--official-test demonstration filter and never as a model input.
Do not load untrusted joblib files.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import prepare_raw,median_imputer_apply

def predict(dataset_csv,model_path,output_csv,official_test_only=False):
    df=pd.read_csv(dataset_csv)
    bundle=joblib.load(model_path)  # Only trusted, project-created artifacts.
    feats=bundle['all_feature_names']
    if any(k not in df.columns for k in feats):
        raise ValueError('Missing the required OPS-SAT segment-level features')
    raw=prepare_raw(df,feats)
    clean=median_imputer_apply(raw,np.asarray(bundle['median_imputer']))
    selection=bundle['selected_feature_indices']
    probs=bundle['model'].predict_proba(clean[:,selection])[:,1]
    out=pd.DataFrame({
      'segment':df.segment if 'segment' in df else np.arange(len(df)),
      'anomaly_probability':probs,
      'predicted_anomaly':(probs>=bundle['threshold']).astype(int),
      'demonstration_normality_index_not_physical_health':100*(1-probs)})
    if official_test_only:
        if 'train' not in df:raise ValueError('No official train/test indicator')
        out=out.loc[df.train==0].copy()
    out=out.reset_index(drop=True)
    path=Path(output_csv);path.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(path,index=False)
    print(f'Reloaded model and predicted {len(out)} OPS-SAT segments without retraining.')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/real_ops_sat/dataset.csv')
    p.add_argument('--model',default='results/real_ops_sat/model_bundle.joblib')
    p.add_argument('--out',default='results/real_ops_sat/reload_predictions.csv')
    p.add_argument('--official-test-only',action='store_true')
    a=p.parse_args()
    predict(a.input,a.model,a.out,a.official_test_only)
