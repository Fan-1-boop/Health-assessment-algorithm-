"""Smoke/regression tests of frozen synthetic spacecraft MVP."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from synthetic import generate,make_windows,highdim_features,CHANNELS
from run import learn_mapping,stacking_inputs


def test_reproducible_and_labeled():
    a=generate(missions=2,cycles=2)
    b=generate(missions=2,cycles=2)
    assert a.equals(b)
    assert a.health_state_simulated.nunique()==3
    assert set(a.anomaly_simulated.unique())=={0,1}
    assert len(CHANNELS)==8
    # Fault periods always marked as anomalies; benign operational events not.
    assert (a.loc[a.health_state_simulated==2,"anomaly_simulated"]==1).all()
    assert (a.loc[a.benign_regime_event==1,"anomaly_simulated"]==0).all()


def test_causal_window_features_only_from_one_cycle():
    df=generate(missions=2,cycles=2)
    w,end=make_windows(df)
    f,names=highdim_features(w)
    assert f.shape==(len(end),400)
    assert len(names)==len(set(names))==400
    assert np.isfinite(f).all()
    assert w.shape[1:]==(32,8)
    assert len(end)==2*2*68


def test_ordered_operator_score_mapping():
    p=np.array([[.98,.01,.01],[.1,.8,.1],[.03,.02,.95]]*20)
    true=p@np.array([96.,68.,16.])
    w=learn_mapping(p,true)
    assert 0<=w[2]<=w[1]<=w[0]<=100
    assert np.mean(np.abs(p@w-true))<3


def test_fusion_inputs_finite_even_at_probability_extremes():
    p=np.array([[1.,0.,0.],[0.,0.,1.]])
    d=np.array([0.,1.])
    x=stacking_inputs(p,d)
    assert x.shape==(2,4)
    assert np.isfinite(x).all()
