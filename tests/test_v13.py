"""No claims about real spacecraft performance can be drawn from these tests."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from v13_protocol import (read_esa_preprocessed,find_events,event_first_partition,
                          windows_in_partition)
from v13_anomaly import RobustLevelDeviation,WindowedIForest,ForecastRidge,episode_metrics
from v13_state import (select_grouped_msf,fit_hierarchical,predict_hierarchical,
                       filter_groups,evaluation,grouped_transitions)
from run_v13_anomaly import run


def toy_stream(n_events=9,step=140,seed=21):
    rng=np.random.default_rng(seed);n=(n_events+1)*step
    X=rng.normal(size=(n,2))*.2;c=np.zeros(n,dtype=int)
    for k in range(n_events):
        t=(k+1)*step-45
        c[t:t+20]=1
        X[t:t+20,0]+=3
        if k%2==0:
            c[t+38:t+52]=2
            X[t+38:t+52,0]+=2 # rare nominal looks like anomaly
    return X,c


def test_esa_codes_read_and_no_label_features(tmp_path):
    X,c=toy_stream(6)
    df=pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=len(X),freq='min'),
                     'TM_1':X[:,0],'CMD_1':X[:,1], 'is_anomaly_TM_1':c,
                     'is_anomaly_CMD_1':np.zeros(len(c),int)})
    path=tmp_path/'input.csv';df.to_csv(path,index=False)
    values,codes,ts,names=read_esa_preprocessed(path,'TM_1')
    assert names==['TM_1','CMD_1']
    assert values.shape[1]==2 and np.array_equal(codes,c)
    assert not any('is_anomaly' in n for n in names)
    parts=event_first_partition(c,min_anomalies_per_split=2)
    for i,p in enumerate(parts):
        w,end,y,rare=windows_in_partition(values,codes,p,lookback=16,stride=4)
        assert w.shape[-1]==2
        assert np.min(end-15)>=p.start and np.max(end)<p.stop
        assert y.sum()>0
        if i:assert parts[i-1].stop==p.start


def test_insufficient_anomaly_episodes_blocks_not_random_split():
    _,c=toy_stream(3)
    with pytest.raises(ValueError,match='Insufficient independent anomaly'):
        event_first_partition(c,2)


def test_neither_invalid_nor_gap_used():
    X,c=toy_stream(6)
    c[10:20]=3;c[20:29]=4
    p=event_first_partition(c,2)[0]
    w,ends,_,_=windows_in_partition(X,c,p,lookback=16,stride=1)
    assert not np.any((ends>=10)&(ends<=43))


def test_chronological_partition_contains_no_label_crossing():
    X,c=toy_stream(9)
    ps=event_first_partition(c,2)
    assert [sum(e.label==1 for e in p.events) for p in ps]==[5,2,2]
    assert sum(p.stop-p.start for p in ps)==len(c)


def test_anomaly_models_and_event_recall():
    X,c=toy_stream(9)
    ps=event_first_partition(c,2)
    ww,ee,yy,rr={}, {}, {}, {}
    for p in ps:
        ww[p.name],ee[p.name],yy[p.name],rr[p.name]=windows_in_partition(X,c,p,16,4)
    train=ww['train'][(yy['train']==0)&(rr['train']==0)]
    for m in (RobustLevelDeviation(),WindowedIForest(),ForecastRidge()):
        m.fit(train);s=m.score(ww['test'])
        assert s.shape==(len(ww['test']),) and np.all(np.isfinite(s))
        r=episode_metrics(ee['test'],yy['test'],s,np.median(s),ps[-1].events)
        assert r['n_anomaly_events']==2


def _state_data(seed=42):
    rng=np.random.default_rng(seed)
    groups=np.repeat(np.arange(16),75)
    y=np.tile(np.repeat([0,1,2],25),16)
    X=rng.normal(size=(len(y),30))*.75
    X[:,0]+=y*1.1
    X[:,1]+=(y>0)*1.3
    X[:,2]+=(y==2)*.8
    return X,y,groups


def test_group_fisher_selection_and_calibration():
    X,y,g=_state_data()
    mask=g<=7
    s1,info=select_grouped_msf(X[mask],(y[mask]!=0).astype(int),g[mask],population=6,generations=1)
    sick=mask&(y!=0)
    s2,info2=select_grouped_msf(X[sick],(y[sick]==2).astype(int),g[sick],population=6,generations=1,seed=43)
    assert info['wrapper_group_folds']>=2
    assert info['fisher_seed_group'] not in set()
    fit=g<=7;cal=(g>=8)&(g<=10)
    model=fit_hierarchical(X[fit],y[fit],g[fit],X[cal],y[cal],s1,s2)
    test=g>=11
    prob=predict_hierarchical(model,X[test])
    assert prob.shape==(test.sum(),3)
    filtered=filter_groups(model,prob,g[test],gamma=.5,memory=.5)
    assert np.allclose(filtered.sum(1),1)
    assert set(evaluation(y[test],filtered))=={'macro_f1','recalls','brier','nll'}
    assert np.allclose(model.transition.sum(axis=1),1)


def test_group_msf_rejects_too_few_independent_groups():
    X,y,g=_state_data()
    idx=g<3
    with pytest.raises(ValueError,match='independent training groups'):
        select_grouped_msf(X[idx],(y[idx]!=0).astype(int),g[idx])


def test_run_complete_synthetic_smoke(tmp_path):
    X,c=toy_stream(9)
    df=pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=len(X),freq='min'),
       'TM_1':X[:,0],'CMD_1':X[:,1],'is_anomaly_TM_1':c})
    p=tmp_path/'toy.csv';df.to_csv(p,index=False)
    out=run(p,'TM_1',tmp_path/'results',lookback=16,stride=4)
    assert (tmp_path/'results/metrics.json').exists()
    assert 'rare_nominal_evaluated_points' in out['selected_heldout_test']
    assert out['event_protocol']['splits']['test']['anomaly_events']==2
    assert out['source_files_not_in_repository']
