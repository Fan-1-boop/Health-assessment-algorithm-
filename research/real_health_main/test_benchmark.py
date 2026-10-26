"""Mathematical invariants only; no generated observations used as experimental evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from benchmark import (features_last16,static_and_causal_hmm,
     split_groups,threshold_for_f1,report)


def test_16_observed_values_do_not_see_future_suffix():
    original=np.linspace(1,3,16)
    a=features_last16(original)
    b=features_last16(np.r_[original,np.ones(8)*1e5][:16])
    assert np.array_equal(a,b)


def test_hmm_outputs_normalized_and_causal_without_looking_ahead():
    demo=np.array([.3,.5,1.0,.5,2.,1.5])
    hmm={"mean":np.array([0.,.5,1.]),
         "variance":np.array([.2,.2,.2]),
         "transition":np.array([[.8,.18,.02],[.2,.65,.15],[.1,.2,.7]]),
         "prior":np.array([.6,.3,.1])}
    s,p,e=static_and_causal_hmm(demo,hmm)
    s2,p2,e2=static_and_causal_hmm(np.r_[demo[:4],1000,1000],hmm)
    assert np.allclose(s.sum(axis=1),1)
    assert np.allclose(p.sum(axis=1),1)
    assert np.allclose(e.sum(axis=1),1)
    assert np.array_equal(p[:4],p2[:4])


def test_grouped_segments_never_cross_phases():
    keys=np.arange(160)
    groups=np.repeat(np.arange(20),8)
    old,other=split_groups(keys,groups,.4,20261026)
    assert len(old)+len(other)==160
    assert set(groups[old]).isdisjoint(set(groups[other]))


def test_threshold_is_calibration_only():
    th=threshold_for_f1(np.array([0,0,1,1]),
                        np.array([.1,.3,.7,.9]))
    assert .3<th<=.7


def test_real_segments_evaluated_once_not_each_inherited_label_window():
    y=np.array([0,0,1,1]);s=np.array([.1,.2,.7,.9])
    r=report(y,s,.5,calibrated=True)
    assert r["segments"]==4 and r["FP"]==0 and r["FN"]==0
