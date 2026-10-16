import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from msf_mofs import sizes,fisher_scores,select_features
from health_v1 import (causal_windows,features,proxy_labels,markov_transitions,
    hmm_filter,health_score,fit_user_mapping)

def test_causal():
    w,t=causal_windows(np.arange(100),16)
    assert w.shape==(85,16) and np.all(w[:,-1]==t)
    assert features(w).shape[0]==85

def test_msf():
    rng=np.random.default_rng(10)
    X=rng.normal(size=(160,20)); y=(X[:,0]+.7*X[:,1]>0).astype(int)
    s,info=select_features(X,y,population=8,generations=2)
    assert 0<len(s)<=20 and info['selected_dim']==len(s)
    assert info['guided_sizes']==sizes(20)
    assert np.argmax(fisher_scores(X,y)) in [0,1]

def test_proxy():
    y,b=proxy_labels(50,[(10,29)])
    assert (b[10:30]==1).all() and (y[10:14]==1).all() and (y[14:30]==2).all()

def test_hmm():
    y=np.array([0,0,1,1,2,2,0]);T=markov_transitions(y)
    p=hmm_filter(np.tile([.7,.2,.1],(10,1)),T,np.array([.5,.25,.25]))
    assert np.allclose(p.sum(axis=1),1)
    assert np.all((p>=0)&(p<=1))

def test_mapping():
    rng=np.random.default_rng(9);p=rng.dirichlet([1,1,1],size=200)
    truth=np.array([95,74,12]); y=p@truth
    w=fit_user_mapping(p,y,strength=.001)
    assert np.all(np.diff(w)<=0) and np.all((w>=0)&(w<=100))
    assert np.mean(np.abs(p@w-y))<1
    assert health_score(p,w).shape==(200,)
