"""Budgeted MSF-MOFS population initialization + standard NSGA-II subset selection.

Implements the core design in Fan et al. EIC_Paper43: one TRAIN-only Fisher
ranking, exactly three multi-scale top-K initial chromosomes, and no MSF-
specific search operators after generation zero. Wrapper: logistic regression,
not the paper's 5-NN. This is an application adaptation, not an exact paper
benchmark reproduction.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def fisher_scores(X, y):
    X = np.asarray(X, float); y = np.asarray(y, int)
    assert set(np.unique(y)) == {0, 1}, 'MSF Fisher ranking requires binary labels'
    a, b = X[y==0], X[y==1]
    return ((a.mean(axis=0)-b.mean(axis=0))**2 /
            (a.var(axis=0)+b.var(axis=0)+1e-12))


def sizes(d):
    candidates = [10, round(.05*d), round(.10*d), round(.20*d)]
    vals = sorted(set(max(1,min(d,int(x))) for x in candidates))
    # At very low d there may be fewer than three distinct valid sizes.
    if len(vals)<3:
        vals = sorted(set(vals + [1, max(1,d//2),d]))
    return vals[:3]


def _fronts(objs):
    n=len(objs); dominates=[[] for _ in range(n)]; degree=np.zeros(n, int)
    first=[]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if np.all(objs[p]<=objs[q]) and np.any(objs[p]<objs[q]): dominates[p].append(q)
            elif np.all(objs[q]<=objs[p]) and np.any(objs[q]<objs[p]): degree[p]+=1
        if degree[p] == 0: first.append(p)
    fronts=[]; current=first
    while current:
        fronts.append(current); future=[]
        for p in current:
            for q in dominates[p]:
                degree[q]-=1
                if degree[q]==0: future.append(q)
        current=future
    return fronts


def _crowding(front, objs):
    dist={i:0.0 for i in front}
    if len(front)<3:
        return {i:float('inf') for i in front}
    for j in range(objs.shape[1]):
        order=sorted(front,key=lambda i:objs[i,j]); span=objs[order[-1],j]-objs[order[0],j]
        dist[order[0]]=dist[order[-1]]=float('inf')
        if span>0:
            for k in range(1,len(order)-1):
                dist[order[k]]+=(objs[order[k+1],j]-objs[order[k-1],j])/span
    return dist


def _survivors(pop, objs, n):
    keep=[]
    for front in _fronts(objs):
        if len(keep)+len(front)<=n: keep+=front
        else:
            crowd=_crowding(front,objs)
            keep+=sorted(front,key=lambda i:crowd[i],reverse=True)[:n-len(keep)]
            break
    return np.asarray(pop)[keep],np.asarray(objs)[keep]


def _gmean(y_true, y_pred):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    return float(np.sqrt((cm[0,0]/max(cm[0].sum(),1)) * (cm[1,1]/max(cm[1].sum(),1))))


def select_features(X,y,population=12,generations=5,seed=42,tolerance=.03):
    X=np.asarray(X,float);y=np.asarray(y,int); d=X.shape[1]
    if len(np.unique(y))<2: raise ValueError('Both labels required for feature selection')
    rng=np.random.default_rng(seed);ranking=np.argsort(-fisher_scores(X,y))
    pop=rng.random((population,d))<.2
    ks=sizes(d)
    for i,k in enumerate(ks):
        pop[i]=False; pop[i,ranking[:k]]=True
    folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,y))
    cache={}
    def fit(bit):
        key=tuple(np.flatnonzero(bit))
        if key in cache: return cache[key]
        if not key: obj=(1.,0.)
        else:
            gs=[]
            for train,val in folds:
                sc=StandardScaler();xt=sc.fit_transform(X[train][:,key]);xv=sc.transform(X[val][:,key])
                lr=LogisticRegression(max_iter=250,class_weight='balanced',solver='liblinear',random_state=seed)
                lr.fit(xt,y[train]);gs.append(_gmean(y[val],lr.predict(xv)))
            obj=(1-float(np.mean(gs)),len(key)/d)
        cache[key]=obj;return obj
    objs=np.asarray([fit(bit) for bit in pop])
    for _ in range(generations):
        fs=_fronts(objs); rank=np.zeros(len(pop));dist={}
        for k,front in enumerate(fs):
            rank[front]=k;dist.update(_crowding(front,objs))
        def pick():
            a,b=rng.integers(len(pop),size=2)
            return a if (rank[a],-dist[a])<=(rank[b],-dist[b]) else b
        children=[]
        while len(children)<population:
            p,q=pop[pick()].copy(),pop[pick()].copy()
            if rng.random()<.9:
                cut=rng.integers(1,d) if d>1 else 1
                p,q=np.r_[p[:cut],q[cut:]],np.r_[q[:cut],p[cut:]]
            for child in (p,q):
                child ^= rng.random(d)<1/max(d,1)
                if not np.any(child): child[rng.integers(d)]=True
                children.append(child)
                if len(children)==population:break
        child=np.array(children); childobj=np.asarray([fit(x) for x in child]);
        pop,objs=_survivors(np.vstack([pop,child]),np.vstack([objs,childobj]),population)
    front=_fronts(objs)[0]; best_g=max(1-objs[i,0] for i in front)
    allowed=[i for i in front if 1-objs[i,0]>=best_g-tolerance]
    idx=min(allowed,key=lambda i:(objs[i,1],objs[i,0]))
    selected=np.flatnonzero(pop[idx])
    return selected, {'candidate_dim':d,'selected_dim':len(selected),
        'guided_sizes':ks,'unique_wrapper_evaluations':len(cache),
        'selection_cv_gmean':round(float(1-objs[idx,0]),5),
        'pareto_objectives':[(round(float(x[0]),5),round(float(x[1]),5)) for x in objs[front]]}
