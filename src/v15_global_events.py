"""V1.5 global official ESA event-ID chronology and strict onset-only alarm evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from v13_protocol import Partition,Event

CHANNELS=('channel_29','channel_30','channel_31')
TARGET='channel_29'

def get_global_event_records(catalog,timestamps,eval_codes,channels=CHANNELS,target=TARGET):
    ts=pd.DatetimeIndex(pd.to_datetime(timestamps,utc=True))
    if ts.hasnans or ts.has_duplicates or not ts.is_monotonic_increasing:
        raise ValueError('Timestamp must be strictly increasing')
    codes=np.asarray(eval_codes,int)
    if len(codes)!=len(ts):raise ValueError('Label length mismatch')
    cat=catalog[catalog['Channel'].isin(channels)].copy()
    if cat.empty:raise ValueError('No official annotations for selected channels')
    cat['StartTime']=pd.to_datetime(cat.StartTime,utc=True,format='mixed')
    cat['EndTime']=pd.to_datetime(cat.EndTime,utc=True,format='mixed')
    if cat.ID.isna().any() or cat.Category.isna().any():raise ValueError('Missing official category/ID')
    records=[]
    for ident,g in cat.groupby('ID',sort=False):
        category=str(g.Category.iloc[0])
        if g.Category.nunique()!=1:raise ValueError('ID has multiple category values')
        begin=int(ts.searchsorted(g.StartTime.min(),'left'))
        finish=int(ts.searchsorted(g.EndTime.max(),'right'))
        row={'id':str(ident),'category':category,'global_start':begin,'global_stop':finish,
             'affected_channels':sorted(g.Channel.unique().tolist()),'target_start':None,
             'target_stop':None,'target_evaluable':False,'annotation_rows':int(len(g))}
        hit=g[g.Channel==target]
        if not hit.empty and category=='Anomaly':
            a=int(ts.searchsorted(hit.StartTime.min(),'left'))
            b=int(ts.searchsorted(hit.EndTime.max(),'right'))
            row['target_start']=a;row['target_stop']=b
            row['target_evaluable']=bool(a<b and np.any(codes[a:b]==1))
        records.append(row)
    return sorted(records,key=lambda x:(x['global_start'],x['global_stop'],x['id']))


def global_id_partitions(records,n,lookback=16,min_target_ids=2):
    """Use target anomaly ID counts, but bar event-ID crossing in ALL input channels.

    Cut candidates lie strictly outside the global span of EVERY official ID,
    including context-only anomalous and rare nominal events.
    """
    target=[r for r in records if r['target_evaluable']]
    if len(target)<3*min_target_ids:
        raise ValueError('BLOCKED: insufficient target official anomaly IDs')
    safe=[]
    candidates=set()
    for r in target:
        candidates.add(int(r['global_start'])-lookback)
        candidates.add(int(r['global_stop'])+lookback)
    candidates.update([n//3,2*n//3])
    spans=[(r['global_start'],r['global_stop']) for r in records]
    for pos in sorted(candidates):
        if lookback*2<=pos<n-lookback*2 and not any(a<pos<b for a,b in spans):
            safe.append(pos)
    if len(safe)<2:raise ValueError('BLOCKED: no global-ID-safe cut candidates')
    counts_before={b:sum(r['global_stop']<=b for r in target) for b in safe}
    counts_after={b:sum(r['global_start']>=b for r in target) for b in safe}
    options=[]
    n_events=len(target)
    for b1 in safe:
        a=counts_before[b1]
        if a<min_target_ids:continue
        for b2 in safe:
            if b2-b1<4*lookback:continue
            c=counts_after[b2];mid=n_events-a-c
            if min(mid,c)<min_target_ids:continue
            if sum(r['global_stop']<=b1 or r['global_start']>=b1 for r in records)!=len(records):
                raise AssertionError('Broken first event separation')
            if sum(r['global_stop']<=b2 or r['global_start']>=b2 for r in records)!=len(records):
                raise AssertionError('Broken second event separation')
            cost=(a/n_events-.6)**2+(mid/n_events-.2)**2+(c/n_events-.2)**2
            cost+=.001*((b1/n-.6)**2+(b2/n-.8)**2)
            options.append((cost,b1,b2,a,mid,c))
    if not options:raise ValueError('BLOCKED: cannot form 3 global-ID-pure partitions')
    _,b1,b2,_,_,_=min(options)
    bounds=[(0,b1),(b1,b2),(b2,n)]
    parts=[]
    id_splits={}
    for name,(a,b) in zip(('train','validation','test'),bounds):
        members=[r for r in records if r['global_start']>=a and r['global_stop']<=b]
        id_splits[name]={c:[r['id'] for r in members if r['category']==c]
                         for c in ('Anomaly','Rare Event')}
        target_ids=[r['id'] for r in members if r['target_evaluable']]
        ev=tuple(Event(r['target_start'],r['target_stop'],1) for r in members if r['target_evaluable'])
        parts.append(Partition(name,a,b,ev))
        id_splits[name]['target_evaluable_anomaly']=target_ids
    for name in id_splits:
        if len(id_splits[name]['target_evaluable_anomaly'])<min_target_ids:
            raise ValueError(f'BLOCKED: {name} too few target events')
    allids=[r['id'] for r in records]
    for ident in allids:
        covered=sum(ident in ids['Anomaly'] or ident in ids['Rare Event'] for ids in id_splits.values())
        category=next(r['category'] for r in records if r['id']==ident)
        if category in ('Anomaly','Rare Event') and covered!=1:
            raise AssertionError('ID missing or duplicated across splits: '+ident)
    return tuple(parts),{'cut_indices':[b1,b2],'global_id_splits':id_splits,
        'selected_channels':list(CHANNELS),'target':TARGET,
        'split_unit':'global ESA ID with envelope across all selected channels',
        'note':'Global ID separation does not guarantee independent physical anomaly mechanisms'}


def event_alarm_metrics(records,part,ends,codes,alerts,stride,grid_step_minutes=10):
    """Report BOTH covered-event recall and stricter in-event new alarm onset.

    A sustained false alert already active before event onset is never counted
    as a newly triggered event alarm.
    """
    ends=np.asarray(ends,int);codes=np.asarray(codes,int);on=np.asarray(alerts,bool)
    if len(ends)!=len(on):raise ValueError('Alert/window length mismatch')
    contiguous=np.r_[False,np.diff(ends)==int(stride)]
    rising=on&(~np.r_[False,on[:-1]]|~contiguous)
    events=[r for r in records if r['target_evaluable'] and
            part.start<=r['global_start'] and r['global_stop']<=part.stop]
    rows=[]
    for r in events:
        considered=(ends>=r['target_start'])&(ends<r['target_stop'])&(codes[ends]==1)
        coverage=np.flatnonzero(considered & on)
        onset=np.flatnonzero(considered & rising)
        rows.append({'id':r['id'],'evaluable_at_stride':bool(considered.any()),
                     'alarm_coverage':bool(len(coverage)),
                     'new_alarm_onset_inside_event':bool(len(onset)),
                     'delay_grid_steps':int(ends[onset[0]]-r['target_start']) if len(onset) else None})
    eligible=[r for r in rows if r['evaluable_at_stride']]
    fp_onsets=rising & (codes[ends]!=1)
    elapsed_days=(part.stop-part.start)*grid_step_minutes/(60*24)
    return {'official_ID_total_target_anomalies':len(rows),
            'official_ID_evaluable_at_stride':len(eligible),
            'coverage_ID_recall':float(np.mean([r['alarm_coverage'] for r in eligible])) if eligible else None,
            'strict_new_onset_ID_recall':float(np.mean([r['new_alarm_onset_inside_event'] for r in eligible])) if eligible else None,
            'false_alarm_new_onsets':int(fp_onsets.sum()),
            'false_alarm_new_onsets_per_30_elapsed_days':float(fp_onsets.sum()/max(elapsed_days,1e-12)*30),
            'calendar_days_in_partition':float(elapsed_days),
            'mean_strict_onset_delay_grid_steps':float(np.mean([r['delay_grid_steps'] for r in eligible if r['delay_grid_steps'] is not None]))
                if any(r['delay_grid_steps'] is not None for r in eligible) else None,
            'event_details':rows}
