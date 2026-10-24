"""Regression tests for one-dataset feature cost-vs-performance benchmark."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.model_selection import train_test_split
from benchmark import split_ids, calibration_threshold, audit_metrics, pareto_front

def test_split_disjoint_stratified_and_source_test_excluded():
    y=np.array([0]*800+[1]*200+[0]*400+[1]*100)
    eligible=np.arange(1000)
    fit,cal,audit=split_ids(y,eligible,20261024)
    assert (len(fit),len(cal),len(audit))==(600,200,200)
    assert set(fit).isdisjoint(cal)
    assert set(cal).isdisjoint(audit)
    assert set(fit).isdisjoint(audit)
    assert len(set(fit)|set(cal)|set(audit))==1000
    assert not set(np.r_[fit,cal,audit])&set(np.arange(1000,len(y)))
    assert np.mean(y[fit])==.2
    assert np.mean(y[cal])==.2

def test_threshold_uses_calibration_not_internal_audit():
    y=np.array([0,0,0,1,1],int)
    s=np.array([.02,.25,.40,.70,.90])
    t=calibration_threshold(y,s)
    q=audit_metrics(y,s,t)
    assert t>.4 and t<=.7 and q["f1"]==1
    assert q["false_positives"]==q["false_negatives"]==0

def test_pareto_requires_not_dominated():
    rows=[
      dict(name="short",model_fit_median_ms=6,audit_f1=.78),
      dict(name="long",model_fit_median_ms=9,audit_f1=.96),
      dict(name="dominated",model_fit_median_ms=14,audit_f1=.80),
    ]
    assert set(pareto_front(rows))=={"short","long"}

def test_audit_does_not_ignore_normal_false_alerts():
    y=np.array([0,0,0,1,1],int)
    s=np.array([.1,.9,.2,.6,.8])
    q=audit_metrics(y,s,threshold=.5)
    assert q["false_positives"]==1
    assert q["fpr"]==1/3
    assert q["recall"]==1
