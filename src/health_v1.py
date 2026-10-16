"""Health V1 algorithms. All outputs are research proxies, NOT flight fault diagnoses."""
import numpy as np
from scipy.special import softmax,expit,logit
from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

STATES=('healthy','degraded_proxy','fault_proxy')

def causal_windows(x,lookback=64):
    """At step t, use samples [t-lookback+1, ..., t] only."""
    x=np.asarray(x,float).ravel()
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(x,lookback).copy(),np.arange(lookback-1,len(x))

def features(w):
    w=np.asarray(w,float);d=np.diff(w,axis=1);n=w.shape[1]
    q=np.quantile(w,[.1,.25,.5,.75,.9],axis=1).T
    t=np.arange(n); slope=((w-w.mean(axis=1,keepdims=True)) @ (t-t.mean()))/np.sum((t-t.mean())**2)
    sig=w-w.mean(axis=1,keepdims=True)
    f=np.abs(np.fft.rfft(sig,axis=1))[:,1:]
    f=f/(f.sum(axis=1,keepdims=True)+1e-12)
    entropy=-np.sum(f*np.log(f+1e-12),axis=1)
    ac=lambda lag: np.sum(sig[:,:-lag]*sig[:,lag:],axis=1)/(np.sum(sig*sig,axis=1)+1e-12)
    tail=w[:,-min(16,n):]
    X=np.column_stack([w.mean(1),w.std(1),w.min(1),w.max(1),np.ptp(w,axis=1),
        q, w[:,-1], w[:,0],slope,np.mean(np.abs(d),axis=1),np.std(d,axis=1),
        np.max(np.abs(d),axis=1),np.sqrt(np.mean(w**2,axis=1)),
        np.mean(tail,axis=1),np.std(tail,axis=1),
        ac(1),ac(2),ac(4),entropy,np.max(f,axis=1),
        np.mean(w[:,:n//2],axis=1)-np.mean(w[:,n//2:],axis=1)])
    return np.nan_to_num(X,nan=0,posinf=0,neginf=0)

def proxy_labels(n,intervals,edge_fraction=.2):
    """EXPERIMENTAL labels only; source only says anomaly/non-anomaly.

    Beginning of annotated intervals is arbitrary transition proxy; remaining
    interval is arbitrary severe proxy. NOT verified degradation/fault.
    """
    y=np.zeros(n,dtype=int);binary=np.zeros(n,dtype=int)
    for a,b in intervals:
        a=max(0,int(a));b=min(n-1,int(b));k=max(1,int((b-a+1)*edge_fraction))
        binary[a:b+1]=1
        y[a:min(b+1,a+k)]=1;y[min(b+1,a+k):b+1]=2
    return y,binary

def split_points(ends,train_end=2600,val_end=4000):
    return (ends<train_end),((ends>=train_end)&(ends<val_end)),(ends>=val_end)

def train_base(X,y,train):
    X0=X[train];y0=y[train]
    sc=StandardScaler();f=sc.fit_transform(X0)
    b1=LogisticRegression(max_iter=500,class_weight='balanced',random_state=42)
    b1.fit(f,(y0>0).astype(int))
    b2=LogisticRegression(max_iter=500,class_weight='balanced',random_state=42)
    non=y0>0
    if len(np.unique(y0[non]))<2:raise ValueError('Proxy degraded/fault classes missing in training')
    b2.fit(f[non],(y0[non]==2).astype(int))
    return sc,b1,b2

def base_proba(X,model):
    sc,b1,b2=model;f=sc.transform(X)
    u=b1.predict_proba(f)[:,1];q=b2.predict_proba(f)[:,1]
    return np.column_stack([1-u,u*(1-q),u*q])

def markov_transitions(y,alpha=1.0):
    T=np.full((3,3),alpha)
    for a,b in zip(y[:-1],y[1:]):T[a,b]+=1
    return T/T.sum(1,keepdims=True)

def hmm_filter(discriminative_proba,T,train_prior,temperature=0.55):
    """Discriminative HMM approximation: class posterior / training prior.

    Markov prior is from proxy training labels. These are NOT true calibrated
    observation likelihoods and output is NOT genuine equipment health belief.
    """
    p=np.clip(np.asarray(discriminative_proba),1e-6,1)
    prior=np.clip(np.asarray(train_prior),1e-4,None)
    emission=(p/prior[None,:])**temperature
    out=np.empty_like(emission); state=prior/prior.sum()
    for i in range(len(p)):
        pred=state@T
        state=pred*emission[i];state=state/state.sum()
        out[i]=state
    return out

def train_anomaly(raw_window,y_binary,mask):
    clf=RandomForestClassifier(n_estimators=90,max_depth=10,min_samples_leaf=8,
        class_weight='balanced_subsample',n_jobs=-1,random_state=42)
    clf.fit(raw_window[mask],y_binary[mask]);return clf

def calibrate_fusion(base_probs,anomaly_probs,labels,mask):
    """Stacked logistic correction on held-out-to-upstream validation data."""
    x=np.column_stack([np.log(np.clip(base_probs,1e-6,1)),logit(np.clip(anomaly_probs,1e-4,1-1e-4))])
    y=labels[mask]
    if len(np.unique(y))<3:
        return None
    clf=LogisticRegression(max_iter=700,class_weight='balanced',random_state=42)
    clf.fit(x[mask],y);return clf

def fusion_proba(base_probs,anomaly_probs,clf):
    if clf is None:return base_probs.copy()
    x=np.column_stack([np.log(np.clip(base_probs,1e-6,1)),logit(np.clip(anomaly_probs,1e-4,1-1e-4))])
    p=np.zeros_like(base_probs)
    p[:,clf.classes_]=clf.predict_proba(x)
    return p

def health_score(probabilities,weights=(100,60,0)):
    return np.asarray(probabilities)@np.asarray(weights,float)

def fit_user_mapping(p,y,prior=(100,60,0),strength=8.0):
    """Constrained 3-weight calibration; numerical optimizer enforces order."""
    from scipy.optimize import minimize
    p=np.asarray(p,float);y=np.asarray(y,float);prior=np.asarray(prior,float)
    def fun(w):
        e=p@w-y
        return float(np.mean(e*e) + strength*np.mean((w-prior)**2))
    res=minimize(fun,prior,method='SLSQP',bounds=[(0,100)]*3,
        constraints=[{'type':'ineq','fun':lambda x: x[0]-x[1]},
                     {'type':'ineq','fun':lambda x: x[1]-x[2]}],
        options={'maxiter':500,'ftol':1e-11})
    if not res.success:raise RuntimeError('Mapping optimization failed: '+str(res.message))
    return res.x

def metrics(y,p):
    from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
    pred=np.argmax(p,axis=1)
    return {'n':int(len(y)),'accuracy':round(float(accuracy_score(y,pred)),4),
        'macro_f1':round(float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),4),
        'balanced_accuracy':round(float(balanced_accuracy_score(y,pred)),4),
        'multi_brier':round(float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),5),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist()}

def binary_metrics(y,score,threshold):
    from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,precision_score,recall_score
    pred=(score>=threshold).astype(int)
    return {'n':int(len(y)),'positive_n':int(sum(y)),'average_precision':round(float(average_precision_score(y,score)),4) if np.any(y) else None,
      'roc_auc':round(float(roc_auc_score(y,score)),4) if len(np.unique(y))==2 else None,
      'precision':round(float(precision_score(y,pred,zero_division=0)),4),
      'recall':round(float(recall_score(y,pred,zero_division=0)),4),
      'f1':round(float(f1_score(y,pred,zero_division=0)),4)}
