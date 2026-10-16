"""Independent satellite telemetry branches (V1.2).

State labels are interval-position PROXIES, never physical fault truth.
Anomaly detection uses published binary anomaly intervals only.
No anomaly score is supplied to the state model; no fusion is performed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp, expit, logit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, log_loss


@dataclass
class TwoStage:
    selected_healthy: np.ndarray
    selected_severity: np.ndarray
    first: object
    second: object
    prior_unhealthy: float
    prior_fault_if_unhealthy: float


def _binary_model(X, y, *, seed=42):
    y=np.asarray(y,int)
    if len(np.unique(y))!=2:raise ValueError('Each stage needs both binary labels')
    pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=800,class_weight='balanced',C=.3,random_state=seed))
    pipe.fit(X,y)
    return pipe


def train_two_stage(X,states,train_mask,s1,s2):
    states=np.asarray(states,int); t=np.asarray(train_mask,bool)
    unhealthy=(states[t]!=0).astype(int)
    sick=states[t]!=0
    return TwoStage(np.asarray(s1,int),np.asarray(s2,int),
        _binary_model(X[t][:,s1],unhealthy),
        _binary_model(X[t][sick][:,s2],(states[t][sick]==2).astype(int),seed=43),
        float(np.mean(unhealthy)),float(np.mean(states[t][sick]==2)))


def _prior_correct_balanced(p,prior):
    # A class-balanced binary logistic objective represents prior ~0.5.
    # This is an approximate prior-shift correction, not an automatic calibration guarantee.
    return expit(logit(np.clip(p,1e-6,1-1e-6))+logit(np.clip(prior,1e-5,1-1e-5)))


def two_stage_probability(X,model,prior_correction=True):
    u=model.first.predict_proba(X[:,model.selected_healthy])[:,1]
    q=model.second.predict_proba(X[:,model.selected_severity])[:,1]
    if prior_correction:
        u=_prior_correct_balanced(u,model.prior_unhealthy)
        q=_prior_correct_balanced(q,model.prior_fault_if_unhealthy)
    return np.column_stack([1-u,u*(1-q),u*q])


def transition_matrix(states,alpha=0.4):
    """Training-only transitions; no future labels in inference."""
    a=np.asarray(states,int)
    t=np.full((3,3),float(alpha))
    for i,j in zip(a[:-1],a[1:]):t[i,j]+=1
    return t/t.sum(axis=1,keepdims=True)


def filtering_from_log_evidence(log_evidence,T,prior,gamma=1.0,memory=1.0):
    """Forward-filter in log space. log_evidence is a class-conditional log-likelihood
    (up to common additive constant). memory=0 -> no temporal memory.
    """
    obs=np.asarray(log_evidence,float); T=np.asarray(T,float)
    assert obs.ndim==2 and obs.shape[1]==3
    prior=np.maximum(np.asarray(prior,float),1e-9)
    prior=prior/prior.sum()
    T_eff=memory*T+(1-memory)*np.tile(prior,(3,1))
    lT=np.log(np.maximum(T_eff,1e-14))
    last=np.log(prior);out=np.zeros_like(obs)
    for i in range(len(obs)):
        pred=logsumexp(last[:,None]+lT,axis=0)
        nxt=pred+gamma*obs[i]
        nxt=nxt-logsumexp(nxt)
        out[i]=np.exp(nxt);last=nxt
    return out


def posterior_as_likelihood(p,prior):
    """Approximate log P(x|state) from calibrated posterior / class prior.

    Strictly valid only if p is calibrated under stated state prior.
    """
    return np.log(np.clip(p,1e-8,1))-np.log(np.clip(prior,1e-8,1))


@dataclass
class GaussianEmissions:
    scaler: object
    pca: object
    means: np.ndarray
    variances: np.ndarray
    dimensions: int


def fit_gaussian_emissions(X,y,n_components=4,seed=42):
    """Train explicit diagonal-Gaussian state-conditional observation densities.

    PCA is fitted to training only; covariance shrinkage avoids tiny proxy classes.
    """
    X=np.asarray(X,float);y=np.asarray(y,int)
    if set(np.unique(y))!={0,1,2}:raise ValueError('three states needed')
    dim=min(int(n_components),X.shape[1],min(np.bincount(y)))
    scaler=StandardScaler().fit(X)
    pca=PCA(n_components=dim,random_state=seed).fit(scaler.transform(X))
    z=pca.transform(scaler.transform(X))
    var_global=np.maximum(z.var(axis=0),1e-4)
    means=np.stack([z[y==k].mean(axis=0) for k in range(3)])
    variances=np.stack([.65*z[y==k].var(axis=0)+.35*var_global+1e-3 for k in range(3)])
    return GaussianEmissions(scaler,pca,means,variances,dim)


def emission_loglik(m,X):
    z=m.pca.transform(m.scaler.transform(np.asarray(X,float)))
    d=z[:,None,:]-m.means[None,:,:]
    return -0.5*np.sum(np.log(2*np.pi*m.variances)[None,:,:]+d*d/m.variances[None,:,:],axis=2)


def state_metrics(y,p):
    from sklearn.metrics import balanced_accuracy_score
    p=np.asarray(p,float)
    if not np.all(np.isfinite(p)):raise ValueError('non-finite probability')
    if not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('probabilities must sum to one')
    pred=p.argmax(axis=1)
    return {'n':int(len(y)),
        'accuracy':round(float(np.mean(y==pred)),5),
        'macro_f1':round(float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),5),
        'balanced_accuracy':round(float(balanced_accuracy_score(y,pred)),5),
        'multi_brier':round(float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),6),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
        'prediction_counts':np.bincount(pred,minlength=3).tolist()}


def anomaly_raw_features(w):
    """All transformations are causal and computed from the input RAW temporal window.

    Include raw levels AND differences to preserve information on shifts and shape.
    """
    w=np.asarray(w,float)
    d=np.diff(w,axis=1)
    recent=w[:,-8:]
    return np.column_stack([w,d,recent.mean(1),recent.std(1),
        np.mean(w,axis=1), np.std(w,axis=1),
        np.max(w,axis=1)-np.min(w,axis=1),
        np.abs(w[:,-1]-w[:,0])])


def candidate_anomaly_models(seed=42):
    return {
      'random_forest_raw':RandomForestClassifier(n_estimators=180,min_samples_leaf=3,
                 max_depth=None,max_features='sqrt',class_weight='balanced_subsample',random_state=seed,n_jobs=-1),
      'extra_trees_raw':ExtraTreesClassifier(n_estimators=230,min_samples_leaf=2,
                 max_features=.8,class_weight='balanced_subsample',random_state=seed,n_jobs=-1),
      'logistic_raw':make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=700,class_weight='balanced',random_state=seed)),
      'rbf_svm_raw':make_pipeline(StandardScaler(),SVC(C=2,gamma='scale',class_weight='balanced',random_state=seed)),
      'hgb_raw':HistGradientBoostingClassifier(max_iter=170,max_leaf_nodes=9,learning_rate=.05,
                 l2_regularization=5,random_state=seed),
    }


def anomaly_score(model,X):
    if hasattr(model,'predict_proba'):
        return np.asarray(model.predict_proba(X)[:,1],float)
    if hasattr(model,'decision_function'):
        return np.asarray(model.decision_function(X),float)
    raise TypeError('model does not expose anomaly score')


def choose_anomaly_threshold(y,score):
    """Validation-only F1 threshold; never selects on test labels.

    precision_recall_curve gives exact thresholds rather than quantile grid.
    """
    precision,recall,thresholds=precision_recall_curve(y,score)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    if not len(thresholds):raise ValueError('threshold requires scores')
    if int(np.sum(y))==0:raise ValueError('validation must include positive anomaly')
    # Highest precision on ties, rather than selecting the all-positive threshold.
    pick=int(np.lexsort((precision[:-1],f1))[-1])
    return float(thresholds[pick]),float(f1[pick])


def anomaly_metrics(y,score,threshold):
    from sklearn.metrics import precision_score,recall_score
    yp=np.asarray(score)>=threshold;y=np.asarray(y,int)
    return {'n':int(len(y)), 'positive_n':int(y.sum()),
       'base_rate':round(float(y.mean()),5),
       'average_precision':round(float(average_precision_score(y,score)),6) if y.sum() else None,
       'roc_auc':round(float(roc_auc_score(y,score)),6) if len(set(y))==2 else None,
       'precision':round(float(precision_score(y,yp,zero_division=0)),6),
       'recall':round(float(recall_score(y,yp,zero_division=0)),6),
       'f1':round(float(f1_score(y,yp,zero_division=0)),6),
       'false_positive_count':int(np.sum(yp & (y==0))),
       'false_negative_count':int(np.sum((~yp) & (y==1)))}


def train_anomaly_candidates(Xtr,ytr,Xval,yval,seed=42):
    from sklearn.base import clone
    models=candidate_anomaly_models(seed);rank={};best=None;best_metric=(-1,-1)
    for name,model in models.items():
        model.fit(Xtr,ytr)
        score=anomaly_score(model,Xval)
        th,val_f1=choose_anomaly_threshold(yval,score)
        ap=float(average_precision_score(yval,score))
        auc=float(roc_auc_score(yval,score))
        rank[name]={'validation_ap':round(ap,6),'validation_auc':round(auc,6),
                   'validation_threshold':th,'validation_f1':round(val_f1,6)}
        # Selection is ONLY on validation AP (threshold-free), F1 tie-break.
        if (ap,val_f1)>best_metric:
            best_metric=(ap,val_f1);best=(name,model,th)
    return best,rank


def tune_hmm(evidence_val,y_val,T,prior):
    """HMM selection on validation, never test; mixed macro-F1 / Brier objective."""
    best=None;scores=[]
    for gamma in (.15,.35,.7,1.0,1.5,2.5):
        for memory in (.0,.35,.7,1.):
            p=filtering_from_log_evidence(evidence_val,T,prior,gamma,memory)
            m=state_metrics(y_val,p)
            objective=m['macro_f1']-0.08*m['multi_brier']
            scores.append((objective,gamma,memory,m['macro_f1'],m['multi_brier']))
            if best is None or objective>best[0]:best=(objective,gamma,memory)
    return {'gamma':best[1],'memory':best[2],
            'validation_objective':round(float(best[0]),6),
            'n_candidates':len(scores)}
