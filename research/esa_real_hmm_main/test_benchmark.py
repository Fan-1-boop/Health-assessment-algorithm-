"""Numerical unit tests only. Fixtures are NOT satellite experiment data."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark import nominal_three_tier_fit,static_ema_hmm


def make_weak_state():
    r=np.linspace(0.01,7,300)
    ends=np.arange(300)*4
    return nominal_three_tier_fit(r,ends,{"stride":4})


def test_training_tiers_come_only_from_supplied_nominal_reference():
    p=make_weak_state()
    assert p["quantile70"]<p["quantile95"]
    assert p["consecutive_nominal_transition_pairs"]==299
    assert np.allclose(p["transition"].sum(axis=1),1)
    assert np.allclose(p["prior"].sum(),1)


def test_hmm_ignores_future_observations():
    p=make_weak_state()
    x=np.linspace(.1,4,75)
    ends=np.arange(len(x))*4
    a,b,c=static_ema_hmm(x,ends,p)
    corrupted=x.copy();corrupted[55:]*=100
    d,e,f=static_ema_hmm(corrupted,ends,p)
    assert np.array_equal(c[:55],f[:55])
    assert np.allclose(a.sum(axis=1),1)
    assert np.allclose(b.sum(axis=1),1)
    assert np.allclose(c.sum(axis=1),1)


def test_zero_hmm_memory_matches_static_classification():
    p=make_weak_state()
    x=np.linspace(.1,4,120)
    e=np.arange(len(x))*4
    s,ema,h=static_ema_hmm(x,e,p,memory=0)
    assert np.allclose(h,s,atol=1e-12)


def test_gapped_windows_do_not_carry_hmm_memory():
    p=make_weak_state()
    a=np.r_[np.linspace(.1,2,20),np.linspace(2,4,20)]
    e=np.r_[np.arange(20)*4,np.arange(20)*4+200]
    _,_,all_h=static_ema_hmm(a,e,p)
    _,_,second_h=static_ema_hmm(a[20:],e[20:],p)
    assert np.array_equal(all_h[20:],second_h)


def test_no_fake_physical_state_assumptions_in_config():
    import json
    q=json.loads((Path(__file__).resolve().parent/"PROTOCOL_FROZEN.json").read_text())
    assert q["physical_fault_state_claim"] is False
    assert q["source_data_untouched"] is True
    assert len(q["selected_channels"])==3
    assert q["alarm"]["nominal_validation_false_positive_quantile"]==.998
