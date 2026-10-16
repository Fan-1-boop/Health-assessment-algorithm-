"""Lightweight anomaly baselines: NOT a Telemanom-ESA reproduction.

No supervised binary anomaly classifier is fitted on the test events.
All feature encodings are calculated from causal RAW time series windows.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score


def _features(w):
    w = np.asarray(w, float)
    if w.ndim != 3:
        raise ValueError('expected (n, lookback, channels) raw telemetry')
    d = np.diff(w, axis=1)
    # Transformations are INTERNAL to this raw-sequence anomaly detector.
    return np.concatenate([w.mean(1), w.std(1), w[:,-1,:],
                           w.max(1)-w.min(1), d.mean(1), d.std(1)],axis=1)


class RobustLevelDeviation:
    """Training-nominal reference of raw measurements; strong/simple baseline."""
    def fit(self, windows):
        x = np.asarray(windows)[:,-1,:]
        self.med = np.median(x,axis=0)
        self.scale = 1.4826*np.median(np.abs(x-self.med),axis=0)
        self.scale = np.maximum(self.scale, .05 * np.std(x,axis=0) + 1e-5)
        return self

    def score(self, windows):
        x = np.asarray(windows)[:,-1,:]
        return np.max(np.abs((x-self.med)/self.scale),axis=1)


class WindowedIForest:
    def __init__(self, seed=42):
        self.seed=seed
    def fit(self, windows):
        f = _features(windows)
        self.scaler=StandardScaler().fit(f)
        self.model=IsolationForest(n_estimators=160,max_samples=min(256,len(f)),
                                   random_state=self.seed,n_jobs=1).fit(self.scaler.transform(f))
        return self
    def score(self, windows):
        return -self.model.score_samples(self.scaler.transform(_features(windows)))


class ForecastRidge:
    """Causal linear autoregressive forecast; separate cheap predictive baseline."""
    def fit(self, windows):
        a=np.asarray(windows,float)
        self.lags=min(8,a.shape[1]-1)
        u=a[:,-self.lags-1:-1,:].reshape(len(a),-1)
        y=a[:,-1,:]
        self.sc=StandardScaler().fit(u)
        self.reg=Ridge(alpha=10.).fit(self.sc.transform(u),y)
        err=y-self.reg.predict(self.sc.transform(u)).reshape(len(a),-1)
        self.scale=np.maximum(np.std(err,axis=0),1e-5)
        return self
    def score(self, windows):
        a=np.asarray(windows,float)
        u=a[:,-self.lags-1:-1,:].reshape(len(a),-1)
        e=(a[:,-1,:]-self.reg.predict(self.sc.transform(u)).reshape(len(a),-1))/self.scale
        return np.sqrt(np.mean(e*e,axis=1))


def threshold_at_normal_false_positive_budget(scores, true_anomaly, budget=.02):
    scores=np.asarray(scores,float);y=np.asarray(true_anomaly,int)
    nominal=scores[y==0]
    if not len(nominal):
        raise ValueError('Validation data need nominal or rare-normal samples')
    if not (0 < budget < 1):
        raise ValueError('Budget must be in (0,1)')
    return float(np.quantile(nominal,1-budget))


def episode_metrics(ends, y, scores, th, all_events):
    """Event recall/delay; no point adjustment, no event expansion of TP."""
    pred=np.asarray(scores) >= th
    ends=np.asarray(ends,int)
    considered=[e for e in all_events if e.label==1]
    hit=0;delays=[]
    for e in considered:
        inside=(ends>=e.start)&(ends<e.stop)
        flagged=ends[inside & pred]
        if len(flagged):
            hit+=1;delays.append(int(flagged[0]-e.start))
    true=np.asarray(y,int)
    # Count consecutive predicted false alarms as episodes; units = evaluated steps.
    fps=(pred & (true==0))
    fp_events=int(np.sum(fps & ~np.r_[False,fps[:-1]]))
    return {'n_anomaly_events':len(considered), 'n_events_detected':hit,
            'event_recall':hit/len(considered) if considered else None,
            'mean_detection_delay_steps':float(np.mean(delays)) if delays else None,
            'false_alarm_episodes':fp_events,
            'false_alarm_episodes_per_1000_evaluated_points':fp_events/max(len(y),1)*1000}


def score_metrics(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float)
    return {'point_ap':float(average_precision_score(y,s)),
            'point_roc_auc':float(roc_auc_score(y,s)) if len(set(y))==2 else None,
            'anomaly_prevalence':float(y.mean())}
