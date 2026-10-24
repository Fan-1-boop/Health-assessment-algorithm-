"""Tests for ordinary supervised branch update / fusion compatibility pilot."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from benchmark import (transform_stress,js_categorical,_jensen_shannon_bernoulli,
                       qualify_gate,metrics_binary,probability_shift)


def test_probability_stress_monotone_preserves_rank_but_changes_scale():
    p=np.array([0.0001,0.04,.20,.48,.76,.93,.999])
    moved=transform_stress(p)
    assert np.all(np.isfinite(moved))
    assert np.all((moved>0)&(moved<1))
    assert np.array_equal(np.argsort(p),np.argsort(moved))
    assert not np.allclose(p,moved)
    assert _jensen_shannon_bernoulli(p,p)<1e-12
    assert _jensen_shannon_bernoulli(p,moved)>0


def test_divergence_zero_for_same_state_vector():
    p=np.array([[.5,.4,.1],[.3,.3,.4],[.98,.01,.01]])
    q=np.array([[.4,.5,.1],[.1,.2,.7],[.9,.08,.02]])
    assert js_categorical(p,p)<1e-10
    assert js_categorical(p,q)>0


def _gate_case(brier,nll,recall,fpr,mae,macro=.90):
    return {"state":{"macro_f1":macro,"brier":brier,"nll":nll},
            "anomaly":{"recall":recall,"fpr":fpr},"health_score_MAE":mae}


def test_fail_closed_rejects_probability_degradation():
    base=_gate_case(.05,.10,.90,.01,5.0)
    new=_gate_case(.09,.23,.93,.02,5.1)
    rule={"max_macro_f1_drop":.02,"max_brier_increase":.01,
          "max_state_nll_increase":.02,"max_anomaly_recall_drop":.05,
          "max_anomaly_fpr_increase":.02,"max_health_MAE_increase":1}
    tests,accepted=qualify_gate(base,new,rule)
    assert not accepted
    assert not tests["state_brier"]
    assert not tests["state_nll"]


def test_gate_rejects_excess_anomaly_false_positives_even_if_state_f1_good():
    base=_gate_case(.05,.1,.9,.02,5)
    new=_gate_case(.04,.09,.9,.1,4.8,.91)
    rule={"max_macro_f1_drop":.02,"max_brier_increase":.01,
          "max_state_nll_increase":.02,"max_anomaly_recall_drop":.05,
          "max_anomaly_fpr_increase":.02,"max_health_MAE_increase":1}
    checks,allowed=qualify_gate(base,new,rule)
    assert not allowed and not checks["anomaly_fpr"]


def test_binary_metrics_stress_probabilities_not_just_classification():
    y=np.array([0,0,1,1],int)
    p=np.array([.15,.2,.8,.85])
    changed=transform_stress(p)
    a=metrics_binary(y,p,.5)
    b=metrics_binary(y,changed,.5)
    assert a["f1"]==b["f1"]
    assert abs(a["prob_brier"]-b["prob_brier"])>1e-5


def test_shift_metrics_reflect_interface_change_even_if_ranking_unchanged():
    old=np.array([.08,.16,.72,.94])
    new=transform_stress(old)
    shift=probability_shift(old,new)
    assert shift["anomaly_binary_rank_preserved"]
    assert shift["anomaly_probability_mean_abs_change"]>.01
    assert shift["anomaly_bernoulli_JS_nats"]>0
