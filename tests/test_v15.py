import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from v15_global_events import get_global_event_records,global_id_partitions,event_alarm_metrics
from v13_protocol import windows_in_partition


def artificial_catalog():
    n=2400
    ts=pd.date_range('2005-01-01',periods=n,freq='10min',tz='UTC')
    c=np.zeros(n,dtype=int)
    rows=[]
    for k in range(12):
        a=65+k*180
        c[a:a+13]=1
        rows.append({'ID':f'id_{k}','Channel':'channel_29','StartTime':ts[a],
                     'EndTime':ts[a+12],'Category':'Anomaly'})
        # Same official global ID affects a different channel at nearby time.
        rows.append({'ID':f'id_{k}','Channel':'channel_31','StartTime':ts[a-3],
                     'EndTime':ts[a+15],'Category':'Anomaly'})
    rows.append({'ID':'nominal_rare','Channel':'channel_30',
                 'StartTime':ts[1005],'EndTime':ts[1025],
                 'Category':'Rare Event'})
    return ts,c,pd.DataFrame(rows)


def test_global_ID_not_split_across_partitions():
    ts,c,cat=artificial_catalog()
    e=get_global_event_records(cat,ts,c)
    parts,audit=global_id_partitions(e,len(ts),lookback=16)
    allid=sum((audit['global_id_splits'][name]['Anomaly']
                for name in ('train','validation','test')),[])
    assert len(allid)==12 and len(set(allid))==12
    for p in parts:
        assert len(audit['global_id_splits'][p.name]['target_evaluable_anomaly'])>=2
        for r in e:
            assert not (r['global_start']<p.stop<r['global_stop'])
            assert not (r['global_start']<p.start<r['global_stop'])
        x,ends,y,_=windows_in_partition(np.zeros((len(ts),3)),c,p,lookback=16,stride=4)
        assert ends.min()-15>=p.start and ends.max()<p.stop


def test_event_coverage_does_not_imply_new_alarm_start():
    n=100
    c=np.zeros(n,dtype=int);c[20:30]=1
    r=[{'id':'id1','target_evaluable':True,'global_start':20,
        'global_stop':30,'target_start':20,'target_stop':30}]
    from v13_protocol import Partition
    part=Partition('test',0,n,tuple())
    on=np.zeros(n,dtype=bool)
    on[17:25]=True  # Alert was already active before anomalous event.
    s=event_alarm_metrics(r,part,np.arange(n),c,on,stride=1)
    assert s['coverage_ID_recall']==1
    assert s['strict_new_onset_ID_recall']==0
    on[:]=False;on[23:26]=True
    s=event_alarm_metrics(r,part,np.arange(n),c,on,stride=1)
    assert s['strict_new_onset_ID_recall']==1
    assert s['mean_strict_onset_delay_grid_steps']==3


def test_cannot_silently_shrink_independent_events():
    ts,c,cat=artificial_catalog()
    e=get_global_event_records(cat,ts,c)[:3]
    with pytest.raises(ValueError,match='BLOCKED'):
        global_id_partitions(e,len(ts),lookback=16,min_target_ids=2)


def test_duplicate_global_ID_mixed_category_rejected():
    ts,c,cat=artificial_catalog()
    cat.loc[cat.ID=='id_3','Category']='Rare Event'
    # Construct a single ID with conflicting labels (official metadata corruption).
    cat.loc[cat[(cat.ID=='id_3')&(cat.Channel=='channel_29')].index,'Category']='Anomaly'
    with pytest.raises(ValueError,match='multiple category'):
        get_global_event_records(cat,ts,c)
