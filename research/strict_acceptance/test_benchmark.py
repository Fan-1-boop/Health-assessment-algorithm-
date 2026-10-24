"""Regression tests: disjoint acceptance, old-version identity, no silent update."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from benchmark import guard_quality,alpha_from_calibration,decision_output,make_metrics
CONSTRAINT={
 "max_f1_drop":.02,"max_AP_drop":.03,
 "max_recall_drop":.06,"max_FPR_increase":.02,
 "max_brier_increase":0.0,"max_NLL_increase":0.0
}

def specimen(**kw):
    return {"f1":.84,"AP":.90,"recall":.85,"FPR":.03,
            "brier":.06,"nll":.14}|kw


def test_brier_regression_passes_old_gate_but_fails_quality_gate():
    a=specimen()
    b=specimen(brier=.095,nll=.21)
    _,old=guard_quality(a,b,CONSTRAINT,False)
    check,new=guard_quality(a,b,CONSTRAINT,True)
    assert old and not new and not check["Brier"] and not check["NLL"]


def test_good_candidate_satisfies_proper_scoring_guard():
    a=specimen()
    b=specimen(f1=.85,AP=.92,recall=.89,
               brier=.05,nll=.13)
    _,allowed=guard_quality(a,b,CONSTRAINT,True)
    assert allowed


def test_alpha_zero_explicitly_means_no_adaptation():
    y=np.array([0,1,0,1,0,1],int)
    old=np.array([.01,.99,.10,.91,.07,.95])
    bad=np.array([.4,.6,.45,.55,.45,.52])
    a,loss,selected=alpha_from_calibration(y,old,bad,[0,.25,.5,.75,1],.0001)
    assert a==0 and selected==loss["0"]
    assert np.array_equal(decision_output(old,bad,False),old)


def test_calibration_chooses_nonzero_when_candidate_is_better():
    y=np.array([0,1,0,1,0,1],int)
    old=np.array([.42,.59,.38,.53,.40,.52])
    good=np.array([.03,.96,.05,.96,.06,.95])
    a,_,_=alpha_from_calibration(y,old,good,[0,.25,.5,.75,1],.0001)
    assert a==1


def test_invalid_probability_and_shapes_fail_closed():
    try:decision_output(np.array([.2,.5]),np.array([.2]),False)
    except ValueError:pass
    else:raise AssertionError("Wrong model version shapes accepted")
    y=np.array([0,0,1,1]);p=np.array([.01,.2,.75,.9])
    m=make_metrics(y,p,.5)
    assert m["FPR"]==0 and m["FN"]==0 and m["nll"]>0
