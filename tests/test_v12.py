import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from independent_branches import (filtering_from_log_evidence,transition_matrix,
    fit_gaussian_emissions,emission_loglik,train_two_stage,two_stage_probability,
    anomaly_raw_features,choose_anomaly_threshold,anomaly_metrics,train_anomaly_candidates)

def test_filter_normalizes_probabilities_and_is_causal():
    T=np.array([[.95,.04,.01],[.03,.93,.04],[.01,.03,.96]])
    prior=np.array([.8,.15,.05]);e=np.zeros((10,3));e[3,2]=4
    a=filtering_from_log_evidence(e,T,prior)
    b=filtering_from_log_evidence(np.vstack([e,[50,0,0]]),T,prior)
    assert np.allclose(a.sum(1),1.)
    assert np.allclose(a,b[:len(a)])

def test_gaussian_emission_fit():
    rng=np.random.default_rng(3);X=np.vstack([rng.normal(k,.5,(25,8)) for k in (0,1,2)])
    y=np.repeat(np.arange(3),25)
    model=fit_gaussian_emissions(X,y)
    e=emission_loglik(model,X)
    assert e.shape==(75,3) and np.isfinite(e).all()

def test_two_separate_feature_sets():
    rng=np.random.default_rng(0);y=np.tile(np.array([0,1,2]),30)
    X=rng.normal(size=(90,6))
    X[:,0]+=y>0;X[:,3]+=y==2
    model=train_two_stage(X,y,np.ones(len(y),bool),np.array([0,1]),np.array([3,4]))
    p=two_stage_probability(X,model)
    assert p.shape==(90,3) and np.allclose(p.sum(1),1.)

def test_anomaly_threshold_no_test_labels():
    scores=np.array([.1,.2,.3,.6,.8,.9])
    y=np.array([0,0,0,1,1,1]);th,_=choose_anomaly_threshold(y,scores)
    assert .3<th<=.6
    m=anomaly_metrics(y,scores,th)
    assert m['f1']==1 and m['false_positive_count']==0

def test_raw_features_shape():
    a=np.arange(256.).reshape(4,64)
    f=anomaly_raw_features(a)
    assert f.shape==(4,133) and np.isfinite(f).all()
