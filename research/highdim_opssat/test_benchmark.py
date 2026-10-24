"""Fast checks for raw-OPS-SAT high-dimensional feature engineering."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from benchmark import (
    features_for_one_segment,get_partition,choose_threshold,quality,standardize_fitted)


WIDTHS=[8,12,16,24,32,48,64,96]


def test_genuine_signal_generates_exact_2560_deterministic_finite_features():
    raw=3+np.sin(np.linspace(0,7,11))
    a,desc=features_for_one_segment(raw,WIDTHS)
    b,_=features_for_one_segment(raw,WIDTHS)
    assert a.shape==(2560,)
    assert len(desc)==20
    assert np.isfinite(a).all()
    assert np.array_equal(a,b)
    assert np.count_nonzero(a)>400


def test_constant_and_short_segments_no_nan():
    x,_=features_for_one_segment(np.ones(8)*17,WIDTHS)
    assert len(x)==2560
    assert np.isfinite(x).all()


def test_full_segment_temporal_reverse_not_same_features():
    signal=np.linspace(0,10,37)**2
    a,_=features_for_one_segment(signal,WIDTHS)
    b,_=features_for_one_segment(signal[::-1],WIDTHS)
    assert not np.allclose(a,b)
    # Separate physical segment "future" must not be mixed in this segment.
    c,_=features_for_one_segment(signal,WIDTHS)
    assert np.array_equal(a,c)


def test_partition_excludes_official_test():
    y=np.array([0,1]*60)
    train=np.ones(120,int);train[100:]=0
    df=pd.DataFrame({'anomaly':y,'train':train})
    fit,cal,audit,excluded=get_partition(df,42)
    assert set(np.r_[fit,cal,audit])==set(range(100))
    assert set(excluded)==set(range(100,120))
    assert len(set(np.r_[fit,cal,audit]))==100


def test_train_only_imputation_and_calibration():
    x=np.array([[10.,np.nan],[30.,4.],[90.,8.]],float)
    a,med=standardize_fitted(x,np.array([0,1]))
    assert med[0]==20 and med[1]==4
    assert a[0,1]==4
    y=np.array([0,0,1,1])
    p=np.array([.1,.2,.85,.92])
    threshold=choose_threshold(y,p)
    q=quality(y,p,threshold)
    assert q["f1"]==1. and q["false_positives"]==0
