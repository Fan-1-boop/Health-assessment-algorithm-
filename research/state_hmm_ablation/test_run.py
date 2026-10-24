"""Fast unit tests for synthetic HMM/fusion research diagnostics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from run import causal_rolling_mean,true_transition_delays,quality


def test_causal_history_never_uses_future_evidence():
    p=np.array([[.8,.1,.1],[.7,.2,.1],[.1,.8,.1],[.1,.1,.8]])
    g=np.array([11,11,11,11])
    a=causal_rolling_mean(p,g,k=3)
    q=p.copy()
    q[3]=[.9,.05,.05] # Future evidence must not alter previous outputs.
    b=causal_rolling_mean(q,g,k=3)
    assert np.allclose(a[:3],b[:3])
    assert np.allclose(a[0],p[0])
    assert np.allclose(a[1],np.mean(p[:2],axis=0))


def test_group_resets_no_cross_cycle_history():
    p=np.array([[0,1,0],[0,1,0],[1,0,0],[1,0,0]],float)
    g=np.array([3,3,4,4])
    a=causal_rolling_mean(p,g,k=3)
    assert np.allclose(a[2],p[2])


def test_transition_delay_detected_and_one_missed():
    y=np.array([0,0,1,1,2,2,0,0])
    pred=np.array([0,0,0,1,1,1,0,0])
    grp=np.array([1]*6+[2]*2)
    t=np.array([0,4,8,12,16,20,0,4])
    res=true_transition_delays(y,pred,grp,t)
    assert res["true_change_events"]==2
    assert res["transition_detected"]==1
    assert res["transition_missed"]==1
    assert res["median_detected_delay_minutes"]==4
    assert res["by_target"]["fault"]["missed"]==1


def test_chatter_is_counted_and_probabilities_validated():
    y=np.array([0,0,1,1])
    g=np.array([1]*4)
    t=np.array([0,4,8,12])
    probs=np.array([[.7,.2,.1],[.1,.8,.1],[.2,.7,.1],[.2,.7,.1]])
    q=quality(y,probs,g,t)
    assert q["temporal"]["spurious_prediction_state_switches"]==1
    assert q["temporal"]["transition_detected"]==1
    assert q["macro_f1"]<1


def test_noncontiguous_groups_raise_not_silently_link():
    p=np.ones((4,3))/3
    with pytest.raises(ValueError,match="contiguous"):
        causal_rolling_mean(p,np.array([1,2,1,2]))
