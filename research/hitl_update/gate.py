"""Conventional input contract for human-reviewed supervised update batches.

PUBLIC RESEARCH INFRASTRUCTURE ONLY; no novel patent algorithm is implemented.
Passing structural checks is NOT proof an actual reviewer is authentic.
Authenticated approval must be enforced out of band.

A demo with a simulated review is opt-in and can NEVER deploy.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib
import re
import numpy as np
import pandas as pd

ALLOWED_TARGETS={"anomaly_binary":{0,1},"physical_state_3class":{0,1,2}}
ALLOWED_ROUTES={
    "anomaly_binary":"anomaly_detector_candidate",
    "physical_state_3class":"two_stage_state_candidate",
    "health_score":"score_mapping_candidate",
}
REQUIRED_META={
    "batch_id","target_task","label_source","review_status","reviewer_id",
    "reviewed_at","parent_model_version","data_sha256","approval_status",
    "data_unit","demo_only"
}
HEX_SHA=re.compile(r"^[0-9a-f]{64}$")


class LabelQualityError(ValueError):
    """Fail closed for untrusted or incorrectly typed supervised data."""


@dataclass(frozen=True)
class ReviewResult:
    candidate_allowed:bool
    release_allowed:bool
    route:str
    rows:int
    positive:int | None
    incomplete_fraction:float
    digest:str
    demo_only:bool
    issues:tuple[str,...]


def _utc_time(text):
    try:
        d=datetime.fromisoformat(str(text).replace("Z","+00:00"))
    except (TypeError,ValueError) as e:
        raise LabelQualityError("Unparseable review timestamp") from e
    if d.tzinfo is None or d.utcoffset() != timezone.utc.utcoffset(d):
        raise LabelQualityError("Review timestamp must be explicit UTC")
    if d>datetime.now(timezone.utc):
        raise LabelQualityError("Review timestamp cannot be in the future")
    return d


def canonical_csv(frame):
    """Stable byte encoding for exact checked batch; must match validator input."""
    return frame.to_csv(index=False,lineterminator="\n",float_format="%.12g").encode("utf-8")


def validate_supervised_upload(
    frame:pd.DataFrame,
    manifest:dict,
    feature_columns:list[str],
    *,
    blocked_sample_ids=(),
    expected_parent_model=None,
    allow_simulated_review=False,
    max_missing_fraction=.02,
)->ReviewResult:
    """Validate before staging. No unreviewed/pseudo-labeled updates.

    Approval status pending permits evaluation of a candidate, not activation.
    Even if the manifest claims approval, release_allowed remains False until
    an independent authenticated approval action and independent validation.
    """
    miss=REQUIRED_META-set(manifest)
    if miss:
        raise LabelQualityError(f"Missing manifest fields: {sorted(miss)}")
    task=str(manifest["target_task"])
    if task not in ALLOWED_ROUTES:
        raise LabelQualityError("Unknown label type; no model update authorized")
    if manifest["data_unit"] not in ("annotated_segment","timestamped_telemetry"):
        raise LabelQualityError("Unsupported measurement unit")
    if manifest["label_source"] in ("model_prediction","pseudo_label","unreviewed_stream",""):
        raise LabelQualityError("Model predictions or unlabeled streams cannot become ground truth")
    if manifest["approval_status"] not in ("pending","approved"):
        raise LabelQualityError("Approval state must be pending or approved")
    if not str(manifest["reviewer_id"]).strip():
        raise LabelQualityError("Reviewer identity is missing")
    _utc_time(manifest["reviewed_at"])
    is_demo=manifest["demo_only"] is True
    if is_demo:
        if not allow_simulated_review or manifest["review_status"]!="simulated_review":
            raise LabelQualityError("Demo-only review is prohibited outside explicit tests")
        if manifest["approval_status"]!="pending":
            raise LabelQualityError("Simulated reviewer cannot approve production release")
    elif manifest["review_status"]!="verified_human":
        raise LabelQualityError("Independent human verification required")
    if expected_parent_model is not None and str(manifest["parent_model_version"])!=str(expected_parent_model):
        raise LabelQualityError("Stale or incompatible parent model")
    if not HEX_SHA.fullmatch(str(manifest["data_sha256"])):
        raise LabelQualityError("Invalid data checksum format")
    if not feature_columns or len(set(feature_columns))!=len(feature_columns):
        raise LabelQualityError("Invalid feature schema")
    if frame.empty:
        raise LabelQualityError("No reviewed examples")
    required={"sample_id","label",*feature_columns}
    if manifest["data_unit"]=="timestamped_telemetry":
        required.add("timestamp")
    if not required.issubset(frame.columns):
        raise LabelQualityError(f"Missing mandatory sample fields: {sorted(required-set(frame.columns))}")
    if frame["sample_id"].isna().any() or frame["sample_id"].astype(str).str.strip().eq("").any():
        raise LabelQualityError("Missing sample identity")
    if frame.sample_id.astype(str).duplicated().any():
        raise LabelQualityError("Duplicate sample IDs")
    blocked=set(map(str,blocked_sample_ids))
    if blocked.intersection(set(map(str,frame.sample_id))):
        raise LabelQualityError("Supervised batch overlaps reserved reference/audit set")
    if manifest["data_unit"]=="timestamped_telemetry":
        t=pd.to_datetime(frame.timestamp,utc=True,errors="coerce")
        if t.isna().any() or not t.is_monotonic_increasing or t.duplicated().any():
            raise LabelQualityError("Invalid chronological timestamps")
    numeric=frame.loc[:,feature_columns].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    incomplete=float(np.mean(~np.isfinite(numeric)))
    if incomplete>max_missing_fraction:
        raise LabelQualityError(f"Missing/bad sensor features exceed threshold: {incomplete:.3%}")
    y=pd.to_numeric(frame.label,errors="coerce").to_numpy(float)
    if not np.isfinite(y).all():
        raise LabelQualityError("Missing review labels")
    if task in ALLOWED_TARGETS:
        if not np.isin(y,list(ALLOWED_TARGETS[task])).all():
            raise LabelQualityError("Invalid label for this supervision target")
    elif task=="health_score":
        if not np.all((y>=0)&(y<=100)):
            raise LabelQualityError("Health scores must be in [0,100]")
    if task=="physical_state_3class" and manifest["label_source"]=="official_anomaly_interval":
        raise LabelQualityError("Anomaly intervals do not establish physical states")
    digest=hashlib.sha256(canonical_csv(frame)).hexdigest()
    if digest!=manifest["data_sha256"]:
        raise LabelQualityError("Upload bytes do not match reviewed SHA-256")
    return ReviewResult(
        candidate_allowed=True,release_allowed=False,
        route=ALLOWED_ROUTES[task],rows=len(frame),
        positive=int((y==1).sum()) if task=="anomaly_binary" else None,
        incomplete_fraction=incomplete,digest=digest,demo_only=is_demo,
        issues=("Manifest is not cryptographic reviewer authentication.",
                "Candidate is not approved or deployed."))
