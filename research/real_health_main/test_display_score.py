"""API safety tests only; DO NOT interpret these as expert score evidence."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from display_score import (DisplayScorePolicy,fit_candidate_from_real_expert_ratings)

def test_post_fusion_score_mapping_cannot_change_objective_alarm_input():
    p=np.array([[.7,.2,.1],[.2,.2,.6]])
    a=np.array([.8,.1])
    copy_a=a.copy()
    policy=DisplayScorePolicy(compatible_fusion_version="fixed-v1")
    r=policy.evaluate(p,a,fusion_version="fixed-v1")
    assert np.all((r>=0)&(r<=100))
    assert np.array_equal(copy_a,a)
    assert r[0]==pytest.approx(70+12-15*.8)
    assert policy.version_digest()==policy.version_digest()

def test_invalid_score_order_is_rejected():
    with pytest.raises(ValueError):
        DisplayScorePolicy(w_healthy=20,w_deviation=60,w_high_risk=0)

def test_new_fusion_version_requires_review_of_score_policy():
    p=np.array([[1.,0.,0.]])
    policy=DisplayScorePolicy(compatible_fusion_version="v-old")
    with pytest.raises(ValueError):
        policy.evaluate(p,np.array([.1]),fusion_version="v-new")

def test_cannot_claim_expert_calibration_without_expert_anchors():
    policy=DisplayScorePolicy(compatible_fusion_version="v-old")
    p=np.array([[1.,0.,0.]])
    with pytest.raises(ValueError):
        fit_candidate_from_real_expert_ratings(
          p,np.array([.1]),np.array([90.]),operator_review_reference="",
          compatible_fusion_version="v-old",version="v2",
          prior_policy=policy,rated_segment_ids=[123])

def test_score_version_cannot_approve_itself():
    policy=DisplayScorePolicy(compatible_fusion_version="v-old")
    assert policy.expert_mapping_version.startswith("INIT")
