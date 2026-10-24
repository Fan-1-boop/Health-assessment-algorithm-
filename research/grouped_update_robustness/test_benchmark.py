"""Grouping integrity, source label semantics, private replay interface tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from benchmark import (channel_order_blocks,holdout_by_group,
  validate_fold,label_conflicts,blocked_fold,channel_shift_fold)


def fake():
    channels=np.repeat(["a","b","c","d"],120)
    seg=np.arange(1,len(channels)+1)
    y=np.tile(np.r_[np.ones(22,int),np.zeros(98,int)],4)
    return pd.DataFrame({"channel":channels,"segment":seg,
              "train":np.ones(len(seg),int),"anomaly":y})


def test_group_blocks_keep_same_channel_and_consecutive_segment_order():
    f=fake()
    g=channel_order_blocks(f,16)
    assert len(g)==len(f)
    assert g[0]==g[15] and g[15]!=g[16]
    assert g[16]!=g[120]


def test_one_group_holdout_has_no_overlap():
    df=fake()
    grp=channel_order_blocks(df,16)
    a,b=holdout_by_group(np.arange(len(df)),grp,.25,105)
    assert set(a).isdisjoint(set(b))
    assert set(grp[a]).isdisjoint(set(grp[b]))
    assert len(a)+len(b)==len(df)


def test_end_to_end_blocked_phase_partition_has_disjoint_segment_and_groups():
    f=fake()
    g=channel_order_blocks(f,16)
    phases=blocked_fold(f,g,105)
    pts=[x for x in phases.values()]
    assert len(np.concatenate(pts))==len(f)
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            assert not set(g[pts[i]]) & set(g[pts[j]])


def test_new_channel_is_never_in_historical_fit_or_calibration():
    f=fake()
    group=channel_order_blocks(f,16)
    phase=channel_shift_fold(f,group,"a",120)
    assert not (set(f.iloc[phase["old_fit"]].channel)|set(f.iloc[phase["cal"]].channel)) & {"a"}
    assert (f.iloc[phase["batch"]].channel=="a").all()
    assert (f.iloc[phase["accept"]].channel=="a").all()
    assert (f.iloc[phase["audit"]].channel=="a").all()


def test_tampered_labels_raise_conflicts_but_do_not_prove_wrong_label():
    y=np.array([0,0,1,1,0,1])
    p=np.array([.04,.99,.02,.98,.16,.99])
    assert label_conflicts(y,p)==2
    changed=1-y
    assert label_conflicts(changed,p)>2


def test_minimum_class_failure_is_recorded_without_relaxing_gate():
    f=fake()
    g=channel_order_blocks(f,16)
    phase=blocked_fold(f,g,105)
    phase["audit"]=phase["audit"][:1]
    ok,reason,count=validate_fold(phase,g,f.anomaly.to_numpy(),np.array([],int),
                   {"audit":3})
    assert not ok and "audit" in reason


def test_official_test_never_eligible_in_phase():
    f=fake()
    g=channel_order_blocks(f,16)
    p=blocked_fold(f,g,105)
    official=p["audit"][:2]
    # Even if abnormal coverage is too small, no silent approval of overlap.
    try:
        validate_fold(p,g,f.anomaly.to_numpy(),official,{"old_fit":0,
                     "batch":0,"cal":0,"accept":0,"audit":0})
    except AssertionError as exc:
        assert "official" in str(exc).lower()
    else:raise AssertionError("Test partition leakage not detected")
