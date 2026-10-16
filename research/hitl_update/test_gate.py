"""Fail-closed gate unit tests. DEMO review is not genuine human approval."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hashlib
import numpy as np
import pandas as pd
import pytest
from gate import canonical_csv, validate_supervised_upload, LabelQualityError


def fixtures(task="anomaly_binary"):
    labels=[0,1,0,1] if task=="anomaly_binary" else ([0,1,2,0] if task=="physical_state_3class" else [80,90,25,55])
    df=pd.DataFrame({"sample_id":["t1","t2","t3","t4"],
                     "label":labels,"voltage":[10.,11.,9.,12.],
                     "current":[2.,3.,2.,4.]})
    manifest={"batch_id":"DEMO_ONLY",
              "target_task":task,
              "label_source":"external_annotated_public_data",
              "review_status":"simulated_review",
              "reviewer_id":"SIMULATED_DO_NOT_USE",
              "reviewed_at":"2026-10-16T00:00:00Z",
              "parent_model_version":"v1",
              "data_sha256":hashlib.sha256(canonical_csv(df)).hexdigest(),
              "approval_status":"pending",
              "data_unit":"annotated_segment",
              "demo_only":True}
    return df,manifest


def validate_demo(df,manifest,**kw):
    return validate_supervised_upload(df,manifest,["voltage","current"],
          expected_parent_model="v1",allow_simulated_review=True,**kw)


def test_simulated_review_rejected_by_default_and_never_releases():
    df,manifest=fixtures()
    with pytest.raises(LabelQualityError,match="Demo-only"):
        validate_supervised_upload(df,manifest,["voltage","current"])
    result=validate_demo(df,manifest)
    assert result.candidate_allowed
    assert not result.release_allowed
    assert result.route=="anomaly_detector_candidate"
    assert result.rows==4 and result.positive==2 and result.demo_only


def test_real_review_schema_never_authorizes_release_without_authenticated_approval():
    df,manifest=fixtures()
    manifest.update(demo_only=False,review_status="verified_human",
                    reviewer_id="operator_123",approval_status="approved")
    result=validate_supervised_upload(df,manifest,["voltage","current"])
    assert result.candidate_allowed and not result.release_allowed


@pytest.mark.parametrize("task,route",[("physical_state_3class","two_stage_state_candidate"),
                                       ("health_score","score_mapping_candidate")])
def test_task_directed_routing(task,route):
    df,manifest=fixtures(task)
    assert validate_demo(df,manifest).route==route


def test_abnormal_binary_label_cannot_train_three_state():
    df,manifest=fixtures("physical_state_3class")
    df.loc[0,"label"]=5
    manifest["data_sha256"]=hashlib.sha256(canonical_csv(df)).hexdigest()
    with pytest.raises(LabelQualityError,match="Invalid label"):
        validate_demo(df,manifest)


def test_pseudolabels_cannot_train():
    df,manifest=fixtures()
    manifest["label_source"]="model_prediction"
    with pytest.raises(LabelQualityError,match="Model predictions"):
        validate_demo(df,manifest)


def test_digest_detects_tampering():
    df,manifest=fixtures()
    df.loc[0,"voltage"]=19
    with pytest.raises(LabelQualityError,match="SHA-256"):
        validate_demo(df,manifest)


def test_no_train_on_reserved_validation_or_test_ids():
    df,manifest=fixtures()
    with pytest.raises(LabelQualityError,match="overlaps"):
        validate_demo(df,manifest,blocked_sample_ids={"t3"})


def test_duplicate_ids_and_sparse_features_blocked():
    df,manifest=fixtures()
    df.loc[1,"sample_id"]="t1"
    with pytest.raises(LabelQualityError,match="Duplicate"):
        validate_demo(df,manifest)
    df,manifest=fixtures()
    df.loc[0,"current"]=np.nan
    with pytest.raises(LabelQualityError,match="exceed"):
        validate_demo(df,manifest)


def test_stale_parent_rejected():
    df,manifest=fixtures()
    manifest["parent_model_version"]="v0"
    with pytest.raises(LabelQualityError,match="parent model"):
        validate_demo(df,manifest)


def test_simulated_review_cannot_fake_production_approval():
    df,manifest=fixtures()
    manifest["approval_status"]="approved"
    with pytest.raises(LabelQualityError,match="Simulated reviewer"):
        validate_demo(df,manifest)


def test_missing_timestamp_blocks_telemetry_task():
    df,manifest=fixtures()
    manifest["data_unit"]="timestamped_telemetry"
    with pytest.raises(LabelQualityError,match="timestamp"):
        validate_demo(df,manifest)


def test_state_labels_not_inferred_from_anomaly_intervals():
    df,manifest=fixtures("physical_state_3class")
    manifest["label_source"]="official_anomaly_interval"
    with pytest.raises(LabelQualityError,match="Anomaly intervals"):
        validate_demo(df,manifest)
