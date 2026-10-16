"""Synthetic adapter tests; these are NOT ESA spacecraft performance experiments."""
import io,sys,zipfile,pickle
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fetch_esa_remote_subset import source_metadata,available_channels,fetch_and_write

def test_official_id_labels_and_causal_resampling(tmp_path):
    stamps=pd.date_range('2020-01-01',periods=250,freq='min',tz='UTC')
    frame=pd.DataFrame({'channel_41':np.arange(250,dtype=float)},index=stamps)
    inner=io.BytesIO()
    with zipfile.ZipFile(inner,'w',compression=zipfile.ZIP_DEFLATED) as zi:
        zi.writestr('values.pkl',pickle.dumps(frame))
    ids=[f'id_{j}' for j in range(9)]
    starts=[stamps[20+22*j].isoformat() for j in range(9)]
    stops=[stamps[25+22*j].isoformat() for j in range(9)]
    labels=pd.DataFrame({'ID':ids,'Channel':['channel_41']*9,'StartTime':starts,'EndTime':stops})
    classes=pd.DataFrame({'ID':ids,'Category':['Anomaly']*7+['Rare Event','Anomaly'],
                          'Class':['c']*9,'Subclass':['s']*9})
    outer=io.BytesIO()
    with zipfile.ZipFile(outer,'w') as z:
        z.writestr('ESA-Mission1/labels.csv',labels.to_csv(index=False))
        z.writestr('ESA-Mission1/anomaly_types.csv',classes.to_csv(index=False))
        z.writestr('ESA-Mission1/channels/channel_41.zip',inner.getvalue())
    outer.seek(0)
    with zipfile.ZipFile(outer) as z:
        meta=source_metadata(z)
        selected,counts=available_channels(z,meta,minimum_anomalies=6)
        assert selected=='channel_41' and counts['channel_41']==8
        path,audit=fetch_and_write(z,tmp_path,min_events=6,resample='1min')
        result=pd.read_csv(path)
        assert len(result)==250
        assert (result.is_anomaly_channel_41==1).sum()==48
        assert (result.is_anomaly_channel_41==2).sum()==6
        assert audit['unique_true_anomaly_event_ids']==8
        assert audit['official_esa_benchmark_preprocessing'] is False
