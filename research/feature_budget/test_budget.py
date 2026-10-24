"""Correctness and leakage tests for feature budget MSF research."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from run_budget import selection_with_budget,pareto


def test_budget_selector_returns_reproducible_bounded_subset():
    rg=np.random.default_rng(28)
    X=rg.normal(size=(80,36))
    y=np.array([0,1]*40)
    X[y==1,:4]+=1.2
    rank=np.arange(X.shape[1])
    a,ai=selection_with_budget(X,y,rank,10,seed=19,
        population=6,generations=1,folds=2)
    b,bi=selection_with_budget(X,y,rank,10,seed=19,
        population=6,generations=1,folds=2)
    assert np.array_equal(a,b)
    assert 5<=len(a)<=10
    assert ai["unique_cv_wrapper_evaluations"]>=6
    assert ai["chosen_gmean"]==bi["chosen_gmean"]


def test_multiple_budgets_not_larger_than_cap():
    rg=np.random.default_rng(29)
    X=rg.normal(size=(48,20));y=np.array([0,1]*24)
    for k in (4,8,12):
        select,meta=selection_with_budget(X,y,np.arange(20),k,
               seed=11+k,population=6,generations=1,folds=2)
        assert (k+1)//2<=len(select)<=k
        assert meta["budget"]==k


def test_pareto_recognizes_efficiency_accuracy_tradeoff():
    data=[
      {"method":"a","fit_median_ms":5,"audit_F1":.80},
      {"method":"b","fit_median_ms":15,"audit_F1":.90},
      {"method":"c","fit_median_ms":25,"audit_F1":.83},
    ]
    assert set(pareto(data,"fit_median_ms","audit_F1"))=={"a","b"}
