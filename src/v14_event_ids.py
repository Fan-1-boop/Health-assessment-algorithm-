"""V1.4: official ESA event-ID audit and no-cross-event chronological splitting.

Exploratory single-channel experiment only. Never construe anomaly IDs as verified
physical fault IDs. Splits are generated before causal windows. Normal/rareness
are different evaluation semantics.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from v13_protocol import Partition, Event


def catalog_from_official(labels: pd.DataFrame, types: pd.DataFrame) -> pd.DataFrame:
    need={'ID','Channel','StartTime','EndTime'}
    if not need.issubset(labels.columns) or not {'ID','Category'}.issubset(types.columns):
        raise ValueError('Official ESA labels/types schema mismatch')
    if types.ID.duplicated().any(): raise ValueError('Each event ID must have a unique type')
    cat=labels.merge(types[['ID','Category','Class','Subclass']].drop_duplicates('ID'),
                     on='ID',how='left',validate='many_to_one')
    if cat.Category.isna().any(): raise ValueError('Unclassified event ID')
    cat=cat[['ID','Channel','StartTime','EndTime','Category','Class','Subclass']].copy()
    cat['StartTime']=pd.to_datetime(cat.StartTime,utc=True)
    cat['EndTime']=pd.to_datetime(cat.EndTime,utc=True)
    if (cat.EndTime<cat.StartTime).any(): raise ValueError('Backwards event interval')
    return cat


def _time_index(raw_timestamps) -> pd.DatetimeIndex:
    ts=pd.DatetimeIndex(pd.to_datetime(raw_timestamps,utc=True))
    if ts.hasnans or not ts.is_monotonic_increasing or ts.has_duplicates:
        raise ValueError('timestamps must be strictly ordered')
    return ts


def sampled_events(catalog,channel,timestamps,codes):
    """Official event-ID episodes on fixed source grid.

    An event is evaluable if at least one grid sample within its annotation
    has the matching published category after priority/invalid masking.
    Preserve non-evaluable IDs in audit; don't pretend they were detected.
    """
    ts=_time_index(timestamps)
    c=np.asarray(codes,dtype=int)
    if len(ts)!=len(c):raise ValueError('timestamp/label count mismatch')
    target=catalog[catalog.Channel==channel]
    starts=ts.searchsorted(pd.DatetimeIndex(target.StartTime),'left')
    stops=ts.searchsorted(pd.DatetimeIndex(target.EndTime),'right')
    records=[]
    for row,start,stop in zip(target.itertuples(index=False),starts,stops):
        kind={'Anomaly':1,'Rare Event':2}.get(row.Category)
        if kind is None:continue
        sampled=np.flatnonzero(c[start:stop]==kind) + start
        records.append({'id':str(row.ID),'category':row.Category,'start':int(start),'stop':int(stop),
                        'evaluable':len(sampled)>0,'visible_start':int(sampled[0]) if len(sampled) else None,
                        'visible_stop':int(sampled[-1]+1) if len(sampled) else None})
    grouped={}
    for r in records:
        x=grouped.setdefault(r['id'],{'id':r['id'],'category':r['category'],'start':r['start'],
                                      'stop':r['stop'],'evaluable':False,
                                      'visible_start':None,'visible_stop':None,'annotation_rows':0})
        if x['category']!=r['category']:raise ValueError('Inconsistent category for same official ID')
        x['start']=min(x['start'],r['start']);x['stop']=max(x['stop'],r['stop'])
        x['annotation_rows']+=1
        if r['evaluable']:
            x['evaluable']=True
            x['visible_start']=r['visible_start'] if x['visible_start'] is None else min(x['visible_start'],r['visible_start'])
            x['visible_stop']=r['visible_stop'] if x['visible_stop'] is None else max(x['visible_stop'],r['visible_stop'])
    return sorted(grouped.values(),key=lambda a:(a['start'],a['stop'],a['id']))


def make_id_safe_partitions(records,n,lookback=16,min_anomaly_ids=2,channel='unknown'):
    """Locate chronological cuts which do not slice any official target-channel ID.

    Choose among safe nominal cut positions by anomaly-ID count ratio (60/20/20).
    This does not claim independent physical failure mechanisms across IDs.
    """
    usable=[r for r in records if r['evaluable'] and r['category']=='Anomaly']
    if len(usable)<3*min_anomaly_ids:raise ValueError('BLOCKED: insufficient evaluable official anomaly event IDs')
    spans=[r for r in records if r['evaluable'] and r['stop']>r['start']]
    candidates={lookback, n-lookback}
    # Cuts after complete anomaly IDs; reject if ANY target event (rare included) crosses.
    candidates|={int(r['stop'])+lookback for r in usable if int(r['stop'])+lookback<n-lookback}
    candidates|={int(r['start'])-lookback for r in usable if int(r['start'])-lookback>lookback}
    safe=sorted(p for p in candidates if lookback<p<n-lookback and
                all(not (r['start']<p<r['stop']) for r in spans))
    options=[]
    for b1 in safe:
        n1=sum(r['stop']<=b1 for r in usable)
        if n1<min_anomaly_ids:continue
        for b2 in safe:
            if b2-b1<lookback*2:continue
            n2=sum(b1<=r['start'] and r['stop']<=b2 for r in usable)
            n3=sum(r['start']>=b2 for r in usable)
            if min(n2,n3)<min_anomaly_ids:continue
            if n1+n2+n3!=len(usable):continue
            score=(n1/len(usable)-.6)**2+(n2/len(usable)-.2)**2+(n3/len(usable)-.2)**2
            # prefer time-spread as tie-break to retain nominal contexts
            score+=0.01*((b1/n-.6)**2+(b2/n-.8)**2)
            options.append((score,b1,b2))
    if not options:raise ValueError('BLOCKED: no clean ID-safe chronology with enough events per partition')
    _,b1,b2=min(options)
    bounds=[(0,b1),(b1,b2),(b2,n)]
    ans=[]
    for name,(a,b) in zip(('train','validation','test'),bounds):
        entries=[r for r in records if r['evaluable'] and a<=r['start'] and r['stop']<=b]
        events=tuple(Event(int(r['visible_start']),int(r['visible_stop']),
                           1 if r['category']=='Anomaly' else 2) for r in entries)
        ans.append(Partition(name,a,b,events))
    return tuple(ans),{'channel':channel,'split_indices':{'train_end':b1,'validation_end':b2},
      'official_anomaly_event_ids':{p.name:[r['id'] for r in records
                     if r['evaluable'] and r['category']=='Anomaly' and p.start<=r['start'] and r['stop']<=p.stop]
                 for p in ans},
      'official_rare_event_ids':{p.name:[r['id'] for r in records
                     if r['evaluable'] and r['category']=='Rare Event' and p.start<=r['start'] and r['stop']<=p.stop]
                 for p in ans},
      'non_evaluable_ids':[r['id'] for r in records if not r['evaluable']],
      'split_semantics':'No target-channel official ID spans a cut; no adjacent partitions share source observations.'}


def id_event_metrics(records,part,ends,alarms,codes):
    """Official-ID hit metric without point adjustment; report unevaluable IDs."""
    ends=np.asarray(ends,int);alarms=np.asarray(alarms,bool);codes=np.asarray(codes,int)
    filtered=[r for r in records if part.start<=r['start'] and r['stop']<=part.stop and
              r['category']=='Anomaly']
    evaluated=0;hit=0;delay=[];outcomes=[]
    for r in filtered:
        relevant=(ends>=r['start'])&(ends<r['stop'])&(codes[ends]==1)
        n=int(relevant.sum())
        if n:
            evaluated+=1
            where=ends[relevant&alarms]
            detected=len(where)>0
            if detected:
                hit+=1;delay.append(int(where[0]-r['start']))
        else:detected=False
        outcomes.append({'id':r['id'],'evaluable_at_selected_stride':bool(n),
                         'n_evaluation_windows':n,'detected':bool(detected)})
    if len({r['id'] for r in outcomes})!=len(outcomes):raise ValueError('Duplicate event IDs in reporting')
    return {'official_event_ids_in_partition':len(filtered),
        'official_event_ids_evaluable':evaluated,'official_event_ids_detected':hit,
        'official_id_event_recall':hit/evaluated if evaluated else None,
        'mean_first_detection_delay_grid_steps':float(np.mean(delay)) if delay else None,
        'per_id':outcomes}
