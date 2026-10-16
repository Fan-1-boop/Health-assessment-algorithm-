"""Independent V1.3 proxy-state research components.

Never describe interval-position proxy classes as confirmed physical faults.
Data with too few independent event groups triggers an explicit error.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logit, expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from msf_mofs import fisher_scores, sizes, _fronts, _crowding, _survivors
from independent_branches import filtering_from_log_evidence


def _as_groups(groups):
    groups=np.asarray(groups)
    if len(np.unique(groups))<5:
        raise ValueError('Grouped feature selection needs >=5 independent training groups; do not random-split windows')
    return groups


def select_grouped_msf(X, labels, event_groups, *, population=12, generations=5,
                       seed=42, tolerance=.03):
    """Strictly separated Fisher seed group and event-grouped wrapper folds.

    Fisher initialization uses an independent seed group; inner wrapper evaluation
    excludes all seed observations. No inner-validation label enters the ranking.
    Outer test is NEVER used. Search on inner validation is still model selection;
    report objective as search fitness, never as unbiased generalization.
    """
    X=np.asarray(X,float); y=np.asarray(labels,int);g=_as_groups(event_groups)
    if X.ndim!=2 or len(X)!=len(y) or len(y)!=len(g) or set(np.unique(y))!={0,1}:
        raise ValueError('Expect one binary-labeled row/group per feature row')
    ug=np.unique(g)
    # Use first event-group containing both classes, kept out of wrapper folds.
    seed_g=next((z for z in ug if set(np.unique(y[g==z]))=={0,1}),None)
    if seed_g is None:
        raise ValueError('No independent mixed-class group for Fisher seeding')
    seed_rows=(g==seed_g); search_rows=~seed_rows
    ix=np.where(search_rows)[0]; group_cv=GroupKFold(n_splits=min(3,len(np.unique(g[ix]))))
    folds=[]
    for t,v in group_cv.split(X[ix],y[ix],g[ix]):
        ti,vi=ix[t],ix[v]
        if set(np.unique(y[ti]))!={0,1} or set(np.unique(y[vi]))!={0,1}:
            raise ValueError('Insufficient independent events/classes for grouped wrapper CV')
        folds.append((ti,vi))
    rng=np.random.default_rng(seed);d=X.shape[1]
    rank=np.argsort(-fisher_scores(X[seed_rows],y[seed_rows]))
    pop=rng.random((population,d))<.2
    ks=sizes(d)
    for i,k in enumerate(ks):
        if i>=population:break
        pop[i]=False;pop[i,rank[:k]]=True
    cache={}
    def fitness(bit):
        key=tuple(np.flatnonzero(bit))
        if key not in cache:
            if not key:result=(1.0,0.0)
            else:
                perf=[]
                for t,v in folds:
                    est=make_pipeline(StandardScaler(),LogisticRegression(
                        max_iter=400,C=.5,class_weight='balanced',solver='liblinear',random_state=seed))
                    est.fit(X[t][:,key],y[t]);p=est.predict(X[v][:,key])
                    perf.append(balanced_accuracy_score(y[v],p))
                result=(1-float(np.mean(perf)),len(key)/d)
            cache[key]=result
        return cache[key]
    objs=np.asarray([fitness(p) for p in pop]);
    for _ in range(generations):
        rank_val=np.zeros(len(pop));dist={}
        for fi,fr in enumerate(_fronts(objs)):
            rank_val[fr]=fi;dist.update(_crowding(fr,objs))
        def pick():
            a,b=rng.integers(len(pop),size=2)
            return int(a if (rank_val[a],-dist[a])<=(rank_val[b],-dist[b]) else b)
        children=[]
        while len(children)<population:
            p,q=pop[pick()].copy(),pop[pick()].copy()
            if d>1 and rng.random()<.9:
                cut=rng.integers(1,d)
                p,q=np.r_[p[:cut],q[cut:]],np.r_[q[:cut],p[cut:]]
            for child in (p,q):
                child ^= rng.random(d)<1/max(d,1)
                if not child.any():child[rng.integers(d)]=True
                children.append(child)
                if len(children)>=population:break
        c=np.asarray(children)
        pop,objs=_survivors(np.vstack([pop,c]), np.vstack([objs,[fitness(z) for z in c]]),population)
    front=_fronts(objs)[0]
    top=max(1-objs[i,0] for i in front)
    candidates=[i for i in front if 1-objs[i,0]>=top-tolerance]
    win=min(candidates,key=lambda i:(objs[i,1],objs[i,0]))
    selected=np.flatnonzero(pop[win])
    return selected,{'fisher_seed_group':str(seed_g),
                     'wrapper_group_folds':len(folds),
                     'wrapper_rows_exclude_seed':int(len(ix)),
                     'selected_dim':int(len(selected)),
                     'total_dim':int(d),
                     'inner_cv_fitness_balanced_accuracy_NOT_unbiased':float(1-objs[win,0]),
                     'evaluated_subsets':len(cache)}


@dataclass
class HierarchicalCalibrated:
    features_stage1: np.ndarray
    features_stage2: np.ndarray
    model_stage1: object
    model_stage2: object
    cal_stage1: object
    cal_stage2: object
    evidence_class_prior: np.ndarray
    transition: np.ndarray


def _train_binary(X,y):
    if len(np.unique(y))!=2:
        raise ValueError('Both binary classes must be present')
    return make_pipeline(StandardScaler(),LogisticRegression(max_iter=700,C=.5,
                         class_weight='balanced',solver='liblinear',random_state=42)).fit(X,y)


def _calibrator(raw_probability,y):
    y=np.asarray(y,int)
    if len(np.unique(y))!=2:
        raise ValueError('Calibration requires both true binary labels from independent groups')
    z=logit(np.clip(raw_probability,1e-6,1-1e-6))[:,None]
    return LogisticRegression(C=1.,max_iter=1000,random_state=42).fit(z,y)


def grouped_transitions(y,groups,alpha=.5):
    y=np.asarray(y,int);g=np.asarray(groups)
    T=np.full((3,3),alpha,float)
    # No artificial transition across event or device boundaries.
    for i in range(1,len(y)):
        if g[i]==g[i-1]: T[y[i-1],y[i]]+=1
    return T/T.sum(axis=1,keepdims=True)


def fit_hierarchical(X_fit,y_fit,group_fit,X_cal,y_cal,s1,s2):
    """Fit on events A; fit Platt probabilities on disjoint calibration events B."""
    X_fit=np.asarray(X_fit,float); y_fit=np.asarray(y_fit,int)
    X_cal=np.asarray(X_cal,float);y_cal=np.asarray(y_cal,int)
    if set(np.unique(y_fit))!={0,1,2} or set(np.unique(y_cal))!={0,1,2}:
        raise ValueError('Each stage and independent calibration need all three states')
    s1=np.asarray(s1,int);s2=np.asarray(s2,int)
    m1=_train_binary(X_fit[:,s1],(y_fit!=0).astype(int))
    sick=y_fit!=0
    m2=_train_binary(X_fit[sick][:,s2],(y_fit[sick]==2).astype(int))
    q1=m1.predict_proba(X_cal[:,s1])[:,1]
    cal1=_calibrator(q1,(y_cal!=0).astype(int))
    sick2=y_cal!=0
    q2=m2.predict_proba(X_cal[sick2][:,s2])[:,1]
    cal2=_calibrator(q2,(y_cal[sick2]==2).astype(int))
    prior=np.bincount(y_cal,minlength=3).astype(float)
    prior=(prior+.5)/(prior.sum()+1.5)
    return HierarchicalCalibrated(s1,s2,m1,m2,cal1,cal2,prior,
                                  grouped_transitions(y_fit,group_fit))


def predict_hierarchical(model,X):
    X=np.asarray(X,float)
    raw1=model.model_stage1.predict_proba(X[:,model.features_stage1])[:,1]
    raw2=model.model_stage2.predict_proba(X[:,model.features_stage2])[:,1]
    u=model.cal_stage1.predict_proba(logit(np.clip(raw1,1e-6,1-1e-6))[:,None])[:,1]
    q=model.cal_stage2.predict_proba(logit(np.clip(raw2,1e-6,1-1e-6))[:,None])[:,1]
    return np.column_stack([1-u,u*(1-q),u*q])


def filter_groups(model,p,groups,gamma=.5,memory=.7):
    """Reset causal forward filter at each independent event/sequence boundary."""
    p=np.asarray(p,float);groups=np.asarray(groups)
    out=np.zeros_like(p)
    log_e=np.log(np.clip(p,1e-8,1))-np.log(model.evidence_class_prior)
    for g in np.unique(groups):
        idx=np.where(groups==g)[0]
        if len(idx) and (idx[-1]-idx[0]+1!=len(idx)):
            raise ValueError('Group timestamps must be contiguous for causal filtering')
        out[idx]=filtering_from_log_evidence(log_e[idx],model.transition,
                         model.evidence_class_prior,gamma=gamma,memory=memory)
    return out


def evaluation(y,p):
    p=np.asarray(p,float);y=np.asarray(y,int);pred=p.argmax(axis=1)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Invalid state probabilities')
    return {'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'recalls':[float(np.mean(pred[y==k]==k)) if np.any(y==k) else None for k in range(3)],
            'brier':float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),
            'nll':float(-np.mean(np.log(np.maximum(p[np.arange(len(y)),y],1e-9))))}
