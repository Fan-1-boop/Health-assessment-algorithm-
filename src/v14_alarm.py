"""Predeclared causal alarm policies; NO test-set threshold fitting."""
import numpy as np


def sustained_alert(scores, threshold, ends=None, min_run=3, expected_stride=None):
    """An alarm is on from the third consecutive high-score evaluated window.

    Reset run across skipped windows/gaps. Unlike centered smoothing, this uses
    only historical and current scores and is online-causal.
    """
    s=np.asarray(scores,float)
    if s.ndim!=1 or not np.isfinite(s).all():raise ValueError('scores must be 1D finite')
    if min_run<1:raise ValueError('min_run must be positive')
    e=np.arange(len(s)) if ends is None else np.asarray(ends,int)
    if len(e)!=len(s) or np.any(np.diff(e)<=0):raise ValueError('Bad window-end sequence')
    stride=1 if expected_stride is None else int(expected_stride)
    out=np.zeros(len(s),bool);run=0
    for i,z in enumerate(s):
        if i and e[i]-e[i-1]!=stride:run=0
        run=(run+1) if z>=threshold else 0
        out[i]=run>=min_run
    return out


def validation_nominal_threshold(scores, ends, raw_codes, false_positive_fraction=.002):
    """Only official nominal (code 0) validation windows determine cutoff.

    Rare nominal (code 2) is held out from threshold choice and separately
    audited for false alerts. This differs from ESA's official rare-as-anomaly.
    """
    s=np.asarray(scores,float);e=np.asarray(ends,int);c=np.asarray(raw_codes,int)
    x=s[c[e]==0]
    if not len(x):raise ValueError('No validation nominal windows for threshold setting')
    if not (0<false_positive_fraction<1):raise ValueError('Budget outside (0,1)')
    return float(np.quantile(x,1-false_positive_fraction))


def alert_statistics(ends, alarms, codes, expected_stride):
    e=np.asarray(ends,int);a=np.asarray(alarms,bool);c=np.asarray(codes,int)
    if len(e)!=len(a):raise ValueError('Bad alert length')
    raw_nom=c[e]==0;rare=c[e]==2;true=c[e]==1
    fp=a&(~true)
    prev=np.r_[False,fp[:-1]]
    contiguous=np.r_[False,np.diff(e)==int(expected_stride)]
    starts=fp&(~prev|~contiguous)
    return {'normal_evaluated':int(raw_nom.sum()),
            'normal_alert_windows':int((a&raw_nom).sum()),
            'normal_alert_rate':float((a&raw_nom).sum()/max(raw_nom.sum(),1)),
            'rare_nominal_evaluated':int(rare.sum()),
            'rare_nominal_alert_windows':int((a&rare).sum()),
            'rare_nominal_alert_rate':float((a&rare).sum()/max(rare.sum(),1)),
            'true_anomaly_evaluated':int(true.sum()),
            'true_anomaly_alert_windows':int((a&true).sum()),
            'false_alarm_episodes':int(starts.sum()),
            'false_alarm_episodes_per_1000_evaluated_windows':float(starts.sum()/max(len(e),1)*1000),
            'metric_denominator':'sampled evaluation windows, not hours or raw timestamps' }
