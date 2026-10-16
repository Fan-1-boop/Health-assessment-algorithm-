"""Deterministic single-dataset spacecraft telemetry simulator for MVP demonstrations.

Labels H/D/F, anomalous transient intervals and operator scores are GENERATED,
not sourced from real satellite failures or human diagnosis. Frozen simulator
parameters must not be modified after observing the test metrics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SEED=20261016
CHANNELS=("bus_voltage","load_current","panel_temperature","battery_temperature",
          "vibration_rms","reaction_wheel_current","thermal_gradient","sensor_reference")
WINDOW=32
STRIDE=4
CHANNEL_COUNT=len(CHANNELS)
STATE_NAMES=("Healthy_simulated","Degraded_simulated","Fault_simulated")

def generate(seed: int=SEED, missions: int=4, cycles: int=4, cycle_len: int=300):
    rng=np.random.default_rng(seed)
    out=[]
    for mission in range(missions):
        n=cycles*cycle_len
        t=np.arange(n)
        phase=t%cycle_len
        state=np.where(phase<115,0,np.where(phase<210,1,np.where(phase<265,2,0)))
        wear=np.clip((phase-115)/95,0,1)*(state==1)
        severity=wear+2.3*(state==2)
        regime=((t//160+mission)%2).astype(float)
        slow=np.sin(2*np.pi*t/96)
        x=rng.normal(0,.22,size=(n,CHANNEL_COUNT))
        # Explicit causal operating-regime changes are NOT failures.
        x[:,0] += 12.0+0.5*regime-0.18*severity+0.15*slow
        x[:,1] += 4.0+0.8*regime+0.95*severity+0.12*slow
        x[:,2] += 24+3.0*regime+1.25*severity+0.45*slow
        x[:,3] += 22+1.3*regime+1.8*severity+0.25*slow
        x[:,4] += 1.0+0.08*regime+1.15*severity+0.15*np.sin(t/4)
        x[:,5] += 2.0+0.6*regime+1.05*severity+0.07*slow
        x[:,6] += 2.0+0.7*regime+0.9*severity+0.2*slow
        x[:,7] += 5.0+0.6*regime+0.08*slow
        # Transient sensor excursions: anomaly without necessarily being a fault.
        transient=np.zeros(n,dtype=int)
        for cycle in range(cycles):
            for offset in (38,280):
                a=cycle*cycle_len+offset
                b=min(a+10,n)
                transient[a:b]=1
                x[a:b,4]+=2.0
                x[a:b,5]+=1.35
        # Rare but benign operations: high signal change WITHOUT anomaly label.
        benign=np.zeros(n,dtype=int)
        for cycle in range(cycles):
            a=cycle*cycle_len+82
            benign[a:a+12]=1
            x[a:a+12,7]+=2.3
        anomaly=((state==2)|(transient==1)).astype(int)
        score_truth=np.array([96.,68.,16.])[state]+rng.normal(0,2.2,size=n)
        score_truth=np.clip(score_truth,0,100)
        f=pd.DataFrame(x,columns=CHANNELS)
        f.insert(0,"timestamp",pd.date_range("2025-01-01",periods=n,freq="min")
                 +pd.Timedelta(days=mission*30))
        f.insert(0,"sample_in_mission",t)
        f.insert(0,"mission_id",mission)
        f["cycle_id"]=mission*cycles+(t//cycle_len)
        f["operating_mode"]=regime.astype(int)
        f["health_state_simulated"]=state
        f["anomaly_simulated"]=anomaly
        f["benign_regime_event"]=benign
        f["operator_score_simulated"]=score_truth
        out.append(f)
    return pd.concat(out,ignore_index=True)


def make_windows(frame:pd.DataFrame,window:int=WINDOW,stride:int=STRIDE):
    """Causal windows cannot cross mission or cycle boundaries."""
    arrays=[];idx=[]
    for (_, _), part in frame.groupby(["mission_id","cycle_id"],sort=True):
        arr=part[list(CHANNELS)].to_numpy(dtype=float)
        rows=part.index.to_numpy()
        if len(rows)<window:continue
        ends=np.arange(window-1,len(part),stride)
        gather=ends[:,None]-np.arange(window-1,-1,-1)[None,:]
        arrays.append(arr[gather]);idx.extend(rows[ends])
    return np.concatenate(arrays,axis=0),frame.loc[idx].reset_index(drop=True)


def _slope(w:np.ndarray):
    n=w.shape[1]
    tm=np.arange(n)-(n-1)/2
    return np.einsum("ntd,t->nd",w,tm)/max(np.dot(tm,tm),1)


def highdim_features(windows:np.ndarray):
    """5 causal scales x 10 descriptors x 8 original telemetry channels=400 features.

    Window-level mean/std/percentiles are computed without future samples.
    The MSF selector is then applied to the 400-dimensional engineered matrix.
    """
    w=np.asarray(windows,float)
    if w.ndim!=3 or w.shape[1]!=WINDOW or w.shape[2]!=CHANNEL_COUNT:
        raise ValueError(f"expected (N,{WINDOW},{CHANNEL_COUNT})")
    mats=[];names=[]
    for width in (6,10,16,24,32):
        z=w[:,-width:,:];diff=np.diff(z,axis=1)
        descriptors=[
            ("mean",z.mean(axis=1)),("std",z.std(axis=1)),
            ("min",z.min(axis=1)),("max",z.max(axis=1)),
            ("p10",np.percentile(z,10,axis=1)),
            ("p90",np.percentile(z,90,axis=1)),
            ("rms",np.sqrt(np.mean(z*z,axis=1))),
            ("slope",_slope(z)),
            ("trend",z[:,-1,:]-z[:,0,:]),
            ("change",np.abs(diff).mean(axis=1))]
        for kind,vals in descriptors:
            mats.append(vals)
            names.extend(f"{c}_scale{width}_{kind}" for c in CHANNELS)
    feats=np.column_stack(mats)
    if not np.isfinite(feats).all():raise ValueError("Nonfinite engineered features")
    return feats,names
