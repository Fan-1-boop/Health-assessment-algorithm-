"""Ordinary post-fusion score-policy engine (NOT confidential patent innovation).

Not a fault diagnosis. Preserves HMM/fusion probabilities and objective alarm.
No expert-rated groundtruth is supplied by current OPS-SAT/ESA datasets.
Fitting scores REQUIRES genuine external expert judgments; untrusted system
predictions are not acceptable as "human score" supervision.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from typing import Sequence
import numpy as np
from scipy.optimize import minimize

DEFAULT_WEIGHTS=np.array([100.,60.,0.],float)
DEFAULT_ANOMALY_PENALTY=15.

@dataclass(frozen=True)
class DisplayScorePolicy:
    """Only defines DISPLAY score; it cannot operate the anomaly detector."""
    w_healthy:float=100.
    w_deviation:float=60.
    w_high_risk:float=0.
    anomaly_penalty:float=15.
    expert_mapping_version:str="INIT_POLICY_NO_EXPERT_CALIBRATION"
    compatible_fusion_version:str="unvalidated_fusion_version"

    def __post_init__(self):
        w=np.array([self.w_healthy,self.w_deviation,self.w_high_risk])
        if not(np.isfinite(w).all() and
          100>=w[0]>=w[1]>=w[2]>=0 and
          np.isfinite(self.anomaly_penalty) and 0<=self.anomaly_penalty<=40):
            raise ValueError("Score weights must be bounded and ordinal")
        if not self.expert_mapping_version or not self.compatible_fusion_version:
            raise ValueError("Every policy needs version provenance")

    def evaluate(self,p_states,p_fused_anomaly,*,fusion_version):
        if fusion_version!=self.compatible_fusion_version:
            raise ValueError("Fusion version mismatch: score calibration needs revalidation")
        p=np.asarray(p_states,float)
        a=np.asarray(p_fused_anomaly,float)
        if p.ndim!=2 or p.shape[1]!=3 or a.shape!=(len(p),):
            raise ValueError("Invalid probability interface")
        if not(np.isfinite(p).all() and np.isfinite(a).all() and
               np.all(p>=0) and np.all(a>=0) and np.all(a<=1) and
               np.allclose(p.sum(axis=1),1,atol=1e-6)):
            raise ValueError("Probability inputs must be normalized")
        w=np.array([self.w_healthy,self.w_deviation,self.w_high_risk])
        return np.clip(p@w-self.anomaly_penalty*a,0,100)

    def version_digest(self):
        payload=json.dumps({
         "w":[self.w_healthy,self.w_deviation,self.w_high_risk],
         "anomaly_penalty":self.anomaly_penalty,
         "expert_mapping_version":self.expert_mapping_version,
         "fusion_version":self.compatible_fusion_version},sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def fit_candidate_from_real_expert_ratings(
    p_states,
    p_fused_anomaly,
    expert_scores,
    *,
    operator_review_reference,
    compatible_fusion_version,
    version,
    prior_policy:DisplayScorePolicy,
    independent_audit_segment_ids:Sequence[int]=(),
    rated_segment_ids:Sequence[int]=(),
    minimum_distinct_ratings=8,
):
    """Return a *candidate* score policy, NEVER approve or deploy.

    A caller must separately authenticate genuine human reviewer credentials,
    obtain declared score semantics, and assess on untouched expert-rated
    segments. This function does NOT confirm physical state labels.
    """
    if not operator_review_reference or not str(operator_review_reference).strip():
        raise ValueError("A real human review source must be supplied")
    if not compatible_fusion_version or not version:
        raise ValueError("Missing scoring/fusion model versions")
    if version.startswith("INIT") or version.startswith("SIMULATED"):
        raise ValueError("Simulated scoring must not masquerade as expert calibration")
    p=np.asarray(p_states,float);a=np.asarray(p_fused_anomaly,float)
    y=np.asarray(expert_scores,float)
    ids=list(map(str,rated_segment_ids))
    blocked=set(map(str,independent_audit_segment_ids))
    if len(p)<minimum_distinct_ratings or len(y)!=len(p) or len(a)!=len(p):
        raise ValueError("Too few human review rating anchors")
    if not ids or len(ids)!=len(p) or len(set(ids))!=len(ids):
        raise ValueError("Rating anchor IDs must be unique and complete")
    if blocked.intersection(ids):
        raise ValueError("Independent expert-audit cases cannot enter calibration")
    if not np.isfinite(y).all() or not np.all((y>=0)&(y<=100)):
        raise ValueError("Expert score labels must be valid 0..100 numbers")
    # Schema/probability checks using *prior* model before fitting new mapping.
    prior_policy.evaluate(p,a,fusion_version=compatible_fusion_version)
    if prior_policy.compatible_fusion_version!=compatible_fusion_version:
        raise ValueError("Recalibrate on matching underlying fusion model only")
    initial=np.array([
      prior_policy.w_healthy,prior_policy.w_deviation,
      prior_policy.w_high_risk,prior_policy.anomaly_penalty],float)
    def obj(v):
        yp=np.clip(p@v[:3]-v[3]*a,0,100)
        return float(np.mean((yp-y)**2)+.01*np.sum((v-initial)**2))
    cons=[{"type":"ineq","fun":lambda x:x[0]-x[1]},
          {"type":"ineq","fun":lambda x:x[1]-x[2]}]
    sol=minimize(obj,initial,bounds=[(0,100)]*3+[(0,40)],
           constraints=cons,method="SLSQP",options={"maxiter":1000,"ftol":1e-10})
    if not sol.success:
        raise RuntimeError("Constrained expert score mapping could not be fitted")
    w=sol.x
    policy=DisplayScorePolicy(float(w[0]),float(w[1]),float(w[2]),
               float(w[3]),version,compatible_fusion_version)
    return {
      "candidate_mapping":policy,
      "input_ratings":len(y),
      "review_reference":str(operator_review_reference),
      "anchor_data_hash":hashlib.sha256(
          np.column_stack([p,a,y]).tobytes()).hexdigest(),
      "training_MAE_DESCRIPTIVE_NOT_HELDOUT":float(np.mean(
            np.abs(policy.evaluate(p,a,fusion_version=compatible_fusion_version)-y))),
      "release_allowed":False,
      "requires_true_review_authentication":True,
      "requires_independent_expert_score_audit":True,
      "anomaly_detector_parameters_untouched":True,
      "hmm_parameters_untouched":True,
      "anomaly_alarm_untouched":True,
    }
