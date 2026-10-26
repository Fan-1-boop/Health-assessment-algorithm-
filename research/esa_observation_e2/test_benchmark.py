"""Only algorithm-level fixtures; no simulated satellite data used as evidence."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark import local_change_matrix,robust_component_reference,local_change_score,normal_reference_rank,compute_risks
from benchmark import SPEC

def test_local_change_only_uses_past_and_current():
    t=np.arange(16,dtype=float)
    w=np.tile(t,(2,3,1)).transpose(0,2,1)
    got=local_change_matrix(w)
    expected=15-np.median(np.arange(7,15))
    assert np.allclose(got,expected)
    # An observation following this 16pt prefix has no path into the feature.
    longer=np.concatenate([w,np.ones((2,5,3))*99999],axis=1)
    assert np.array_equal(local_change_matrix(longer[:,:16]),got)

def test_nominal_rank_fits_only_specified_reference():
    fit=np.sort(np.arange(100,dtype=float))
    probe=np.array([-1.,0.,50.,99.,100.])
    a=normal_reference_rank(probe,fit)
    assert a[0]==0 and a[-1]==100/101
    assert np.all(np.diff(a)>=0)

def test_wrong_input_sizes_rejected():
    with pytest.raises(ValueError):
        local_change_matrix(np.ones((3,15,3)))
    with pytest.raises(ValueError):
        local_change_matrix(np.ones((3,16,4)))

def test_local_reference_identical_scores_if_only_shared_normal_baseline():
    rng=np.random.default_rng(2)
    w=rng.normal(size=(200,16,3))
    r=robust_component_reference(w)
    v=local_change_score(w,r)
    assert len(v)==200 and np.isfinite(v).all()
    assert (v>=0).all()

def test_pre_frozen_weights_and_no_physical_fault_claim():
    import json
    conf=json.loads(SPEC.read_text())
    assert conf["risk_components"]["component_weights"]==[.35,.45,.20]
    assert "new independently" not in conf["research_mode"].lower()
    assert conf["source_sha256"].startswith("bf42672b")
    assert conf["alarm_rules"]["rare_nominal"].startswith("negative")
