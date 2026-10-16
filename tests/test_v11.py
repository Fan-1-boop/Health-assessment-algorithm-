import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from highdim_features import highdim_features,feature_names
from run_v11 import load_frozen_labels

def test_highdim_shape_and_finiteness():
    rng=np.random.default_rng(3)
    w=rng.normal(size=(30,64))
    X=highdim_features(w)
    assert X.shape==(30,len(feature_names()))
    assert X.shape[1]>=300
    assert np.isfinite(X).all()

def test_features_are_causal():
    rng=np.random.default_rng(1)
    w=rng.normal(size=(2,64))
    np.testing.assert_allclose(highdim_features(w[:1]),highdim_features(w)[:1])
    assert len(set(feature_names()))==len(feature_names())

def test_frozen_label_boundaries(tmp_path):
    src=Path(__file__).resolve().parents[1]/'configs/proxy_label_rules_v1.json'
    metadata=pd.DataFrame([{'chan_id':'P-1','spacecraft':'SMAP','num_values':100,
                            'anomaly_sequences':'[[10,29],[50,59]]'}])
    state,binary,intervals,sha,rule=load_frozen_labels(100,metadata,src)
    assert state[10:14].tolist()==[1]*4
    assert state[14:30].tolist()==[2]*16
    assert state[50:52].tolist()==[1]*2
    assert state[52:60].tolist()==[2]*8
    assert (state[:10]==0).all()
    assert len(sha)==64
    tampered=json.loads(src.read_text())
    tampered['degraded_proxy']='modified'
    dest=tmp_path/'bad.json';dest.write_text(json.dumps(tampered))
    import pytest
    with pytest.raises(ValueError):load_frozen_labels(100,metadata,dest)

def test_proxy_csv_export_provenance():
    from export_proxy_labels import build_label_table
    from health_v1 import proxy_labels
    states,binary=proxy_labels(40,[(10,19)])
    df=build_label_table(40,[(10,19)],states,binary,'frozen','deadbeef')
    assert len(df)==40
    assert df.loc[10,'proxy_state_name']=='degraded_proxy'
    assert df.loc[12,'proxy_state_name']=='fault_proxy'
    assert df.loc[10,'published_anomaly_event_index']==0
    assert df.loc[5,'published_anomaly_event_index']==-1
    assert not df['physical_fault_confirmed'].any()
