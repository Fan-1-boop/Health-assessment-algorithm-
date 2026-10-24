"""Phase-independent selection and in-memory rollback correctness."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from run import (evaluate_probs,stage_accept,stage_select,probability_js,stacking_inputs)

RULES={
  "max_state_macro_f1_drop":.02,
  "max_state_Brier_increase":.01,
  "max_state_NLL_increase":.02,
  "max_anomaly_recall_drop":.05,
  "max_anomaly_FPR_increase":.02,
  "max_health_MAE_increase":1.0
}


def m(f1=.9,brier=.1,nll=.2,recall=.9,fpr=.03,mae=6.):
    return {
      "state_macro_f1":f1,"state_brier":brier,"state_nll":nll,
      "anomaly_recall":recall,"anomaly_normal_fpr":fpr,
      "score_default_weights_MAE":mae
    }


def test_good_update_can_be_staged_but_never_auto_deployed():
    base=m()
    new=m(f1=.91,brier=.08,nll=.18,recall=.92,fpr=.025,mae=5.5)
    checks,ok=stage_accept(base,new,RULES)
    assert ok and all(checks.values())


def test_bad_state_semantics_must_trigger_rollback():
    original=np.array([[.85,.1,.05],[.15,.7,.15],[.05,.15,.8]])
    wrong=original[:,[2,1,0]]
    y=np.array([0,1,2])
    anomaly=np.array([0,0,1])
    healthy_score=np.array([98.,64.,14.])
    old=evaluate_probs(y,anomaly,healthy_score,np.array([.02,.05,.92]),.5,original)
    new=evaluate_probs(y,anomaly,healthy_score,np.array([.02,.05,.92]),.5,wrong)
    checks,ok=stage_accept(old,new,RULES)
    assert not ok
    assert not checks["state_macro_f1"] and not checks["state_brier"]
    final=stage_select(original,wrong,ok)
    assert np.array_equal(final,original)


def test_rollback_bitwise_preserves_exact_distribution():
    old=np.array([[.77,.20,.03],[.11,.79,.10]],float)
    candidate=np.array([[.2,.2,.6],[.4,.1,.5]],float)
    assert np.array_equal(stage_select(old,candidate,False),old)
    assert np.array_equal(stage_select(old,candidate,True),candidate)


def test_candidate_probability_semantics_distinct_from_js_closeness():
    p=np.array([[.8,.15,.05],[.1,.2,.7]])
    q=np.array([[.7,.2,.1],[.3,.2,.5]])
    assert probability_js(p,p)<1e-12
    assert probability_js(p,q)>0
    assert np.allclose(stacking_inputs(p,np.array([.04,.8]))[:,3],
                       np.log(np.array([.04/.96,.8/.2])))


def test_invalid_probability_vector_fails_closed():
    try:
        evaluate_probs(np.array([0,1,2]),np.array([0,1,1]),
            np.array([90,60,20]),np.array([.1,.5,.7]),.5,
            np.array([[1.,0,0],[.1,.9,0],[.3,.2,.3]]))
    except ValueError as exc:
        assert "Invalid three-state" in str(exc)
    else:raise AssertionError("Invalid probabilities wrongly accepted")
