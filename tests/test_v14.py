import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from v14_event_ids import (catalog_from_official,sampled_events,
                            make_id_safe_partitions,id_event_metrics)
from v14_alarm import sustained_alert,validation_nominal_threshold,alert_statistics


def _make_timeline():
    n=1200
    ts=pd.date_range('2001-01-01',periods=n,freq='min',tz='UTC')
    code=np.zeros(n,dtype=int)
    rows=[];types=[]
    for k in range(9):
        a=40+k*115;b=a+17
        code[a:b+1]=1
        ident=f'event_{k}'
        rows.append({'ID':ident,'Channel':'channel_21','StartTime':ts[a],
                     'EndTime':ts[b]})
        types.append({'ID':ident,'Category':'Anomaly','Class':'x','Subclass':'sub'})
    for k in range(6):
        a=28+k*166;b=a+3
        if (code[a:b+1]==0).all():
            code[a:b+1]=2
            ident=f'rare_{k}'
            rows.append({'ID':ident,'Channel':'channel_21',
                         'StartTime':ts[a],'EndTime':ts[b]})
            types.append({'ID':ident,'Category':'Rare Event','Class':'x','Subclass':'sub'})
    cat=catalog_from_official(pd.DataFrame(rows),pd.DataFrame(types))
    return n,ts,code,cat


def test_official_event_splits_are_nonoverlapping():
    n,ts,c,cat=_make_timeline()
    records=sampled_events(cat,'channel_21',ts,c)
    parts,audit=make_id_safe_partitions(records,n,lookback=8,min_anomaly_ids=2,channel='channel_21')
    anomaly_lists=list(audit['official_anomaly_event_ids'].values())
    assert len(set(sum(anomaly_lists,[])))==9
    assert all(len(x)>=2 for x in anomaly_lists)
    for p in parts:
        assert not any(p.start<r['start']<p.stop<r['stop'] for r in records)
        assert not any(r['start']<p.start<r['stop'] for r in records if r['evaluable'])
    assert parts[0].stop==parts[1].start and parts[1].stop==parts[2].start


def test_no_pseudo_events_when_resampling_hides_short_interval():
    n,ts,c,cat=_make_timeline()
    c[40:58]=0
    r=sampled_events(cat,'channel_21',ts,c)
    e=next(x for x in r if x['id']=='event_0')
    assert e['evaluable']==False
    assert e['visible_start'] is None


def test_invalid_id_data_blocks():
    n,ts,c,cat=_make_timeline()
    with np.testing.assert_raises_regex(ValueError,'BLOCKED'):
        make_id_safe_partitions(sampled_events(cat,'channel_21',ts,c)[:3],n,
                                min_anomaly_ids=2)


def test_alert_requires_consecutive_high_scores_and_resets_across_gap():
    s=np.array([.0,3,3,3,3,3])
    e=np.array([0,1,2,3,8,9])
    a=sustained_alert(s,2,e,min_run=3,expected_stride=1)
    assert a.tolist()==[False,False,False,True,False,False]
    raw=sustained_alert(s,2,e,min_run=1,expected_stride=1)
    assert raw.sum()==5


def test_nominal_only_threshold_not_rare_normal():
    scores=np.array([1.,2.,3.,10.,11.])
    ends=np.arange(5)
    codes=np.array([0,0,0,2,1])
    th=validation_nominal_threshold(scores,ends,codes,.1)
    assert 2<th<3
    stats=alert_statistics(ends,np.array([0,0,1,1,1],bool),codes,1)
    assert stats['normal_alert_windows']==1
    assert stats['rare_nominal_alert_windows']==1


def test_id_detection_and_delay():
    n,ts,c,cat=_make_timeline()
    r=sampled_events(cat,'channel_21',ts,c)
    parts,_=make_id_safe_partitions(r,n,lookback=8,min_anomaly_ids=2)
    p=parts[2]
    ends=np.arange(p.start,p.stop)
    alerts=np.zeros(len(ends),bool)
    event=next(x for x in r if x['category']=='Anomaly' and p.start<=x['start'] and x['stop']<=p.stop)
    alerts[event['start']+3-p.start]=True
    m=id_event_metrics(r,p,ends,alerts,c)
    assert m['official_event_ids_detected']==1
    assert m['mean_first_detection_delay_grid_steps']==3
