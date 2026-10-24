"""Prevent leakage and unsafe interpretations in one real OPS-SAT update pilot."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from benchmark import choose_threshold,metrics,gate,estimator

G={"min_f1_relative":-.02,"min_recall_relative":-.06,
   "min_AP_relative":-.03,"max_FPR_relative":.02}


def test_gate_rejects_recall_collapse_despite_high_precision():
    a={"f1":.82,"AP":.9,"recall":.85,"FPR":.02}
    b={"f1":.81,"AP":.9,"recall":.60,"FPR":0}
    checks,ok=gate(a,b,G)
    assert not ok and not checks["recall"]


def test_gate_rejects_excess_normal_false_alarm():
    a={"f1":.82,"AP":.9,"recall":.85,"FPR":.02}
    b={"f1":.84,"AP":.90,"recall":.88,"FPR":.10}
    checks,ok=gate(a,b,G)
    assert not ok and not checks["FPR"]


def test_test_metric_is_not_used_to_fit_threshold():
    y=np.array([0,0,0,1,1])
    s=np.array([.1,.2,.49,.60,.88])
    threshold=choose_threshold(y,s)
    assert .49<threshold<=.60
    report=metrics(y,s,threshold)
    assert report["FP"]==0 and report["FN"]==0


def test_deterministic_binary_clf_schema():
    r=np.random.default_rng(8)
    x=r.normal(size=(120,22))
    y=np.array([0,1]*60)
    m=estimator().fit(x,y)
    p=m.predict_proba(x)[:,1]
    assert p.shape==(120,)
    assert np.all(np.isfinite(p))


def test_high_confidence_old_model_disagreement_cannot_prove_label_source():
    # Model disagreement alone is ambiguous: both wrong label and new regime
    # can produce the very same old-model p and new observed label.
    old_model_probability=.02
    submitted_label=1
    disagreement=old_model_probability<.2 and submitted_label==1
    assert disagreement
    # The substantive truth is unknown WITHOUT external ground truth.
    # No automatic "wrong human label" classification occurs in public gate.
