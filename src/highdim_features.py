"""Causal, multiscale feature engineering for ONE telemetry signal.

This is high-dimensional univariate feature construction, NOT multichannel telemetry.
All windows are [t-L+1,t]; no future measurements are used.
"""
from __future__ import annotations
import numpy as np


def feature_names(window_length: int = 64) -> list[str]:
    names=[]
    for length in (8,16,24,32,48,64):
        if length > window_length:
            continue
        for offset in (0,8,16,24,32,48):
            if length + offset > window_length:
                continue
            base=f"len{length}_lag{offset}_"
            names.extend(base+n for n in (
                'mean','std','min','max','median','q10','q90','rms','mean_abs',
                'diff_abs_mean','diff_std','slope','skew','kurt','acf1','range',
                'start_end_delta','outlier_rate'))
    names.extend(f'global_fft_power_{i:02d}' for i in range(1,window_length//2+1))
    names.extend(f'global_acf_{lag:02d}' for lag in (1,2,3,4,6,8,12,16,24,32) if lag < window_length)
    return names


def highdim_features(w: np.ndarray) -> np.ndarray:
    w=np.asarray(w,dtype=np.float64)
    if w.ndim!=2 or w.shape[1]<8:
        raise ValueError('expected [num_windows, >=8] causal windows')
    n,L=w.shape
    columns=[]
    for length in (8,16,24,32,48,64):
        if length>L: continue
        for offset in (0,8,16,24,32,48):
            if length+offset>L: continue
            seg=w[:,L-length-offset:L-offset]
            mu=seg.mean(1)
            sd=seg.std(1)
            norm=(seg-mu[:,None])/(sd[:,None]+1e-5)
            dx=np.diff(seg,axis=1)
            x=np.arange(length,dtype=float)-(length-1)/2
            slope=((seg-mu[:,None])@x)/(x@x)
            acf=np.sum((seg[:,:-1]-mu[:,None])*(seg[:,1:]-mu[:,None]),axis=1)/(
                np.sum((seg-mu[:,None])**2,axis=1)+1e-9)
            columns.extend([
                mu, sd, seg.min(1),seg.max(1),np.median(seg,axis=1),
                np.quantile(seg,.1,axis=1),np.quantile(seg,.9,axis=1),
                np.sqrt(np.mean(seg**2,axis=1)),np.mean(np.abs(seg),axis=1),
                np.mean(np.abs(dx),axis=1),np.std(dx,axis=1),slope,
                np.mean(norm**3,axis=1),np.mean(norm**4,axis=1)-3,
                acf,np.ptp(seg,axis=1),seg[:,-1]-seg[:,0],
                np.mean(np.abs(norm)>2,axis=1)])
    centered=w-w.mean(1,keepdims=True)
    spectrum=np.abs(np.fft.rfft(centered,axis=1))[:,1:]**2
    spectrum=spectrum/(spectrum.sum(1,keepdims=True)+1e-10)
    columns.extend([spectrum[:,i] for i in range(spectrum.shape[1])])
    denom=np.sum(centered*centered,axis=1)+1e-9
    for lag in (1,2,3,4,6,8,12,16,24,32):
        if lag<L:
            columns.append(np.sum(centered[:,:-lag]*centered[:,lag:],axis=1)/denom)
    result=np.column_stack(columns)
    assert result.shape[1]==len(feature_names(L)),(result.shape,len(feature_names(L)))
    return np.nan_to_num(result,nan=0,posinf=0,neginf=0)
