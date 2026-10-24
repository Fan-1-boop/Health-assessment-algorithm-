"""Unit tests for direct-3class vs two-stage probability study."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.metrics import log_loss
from benchmark import make_direct,apply_temperature,temperature_calibration,quality


def test_temperature_produces_valid_three_class_probabilities():
    p=np.array([[.90,.08,.02],[.15,.75,.10],[.09,.11,.80],[.92,.05,.03]])
    for t in (.3,1.,2.,8.):
        out=apply_temperature(p,t)
        assert out.shape==p.shape
        assert np.isfinite(out).all()
        assert np.allclose(out.sum(axis=1),1)
    assert np.allclose(apply_temperature(p,1),p,atol=1e-12)


def test_temperature_is_fit_only_to_supplied_calibration_labels():
    p=np.array([[.85,.08,.07],[.75,.2,.05],[.1,.80,.1],
               [.1,.70,.2],[.1,.2,.7],[.12,.18,.7]])
    y=np.array([0,0,1,1,2,2])
    t=temperature_calibration(p,y)
    assert .2<=t<=8.
    old=float(log_loss(y,p,labels=[0,1,2]))
    new=float(log_loss(y,apply_temperature(p,t),labels=[0,1,2]))
    assert new<=old+1e-6


def test_direct_classifier_supports_simulated_three_states():
    rg=np.random.default_rng(38)
    x=rg.normal(size=(120,9))
    y=np.array([0,1,2]*40)
    x[y==2,:2]+=1.6
    m=make_direct().fit(x,y)
    p=m.predict_proba(x[:9])
    assert p.shape==(9,3)
    assert list(m.classes_)==[0,1,2]


def test_transition_event_metrics_and_recall_not_mislabeled_real():
    y=np.array([0,0,1,1,2,2])
    groups=np.array([4]*6)
    t=np.arange(6)*4
    p=np.eye(3)[y]*.9+np.ones((6,3))*(.1/3)
    q=quality(y,p,groups,t)
    assert q["macro_f1"]==1
    assert q["transition"]["true_change_events"]==2
    assert q["transition"]["transition_missed"]==0
    assert np.allclose(q["recalls"],1)
