"""Fast real-OPS-SAT MVP smoke tests; do not fabricate real satellite metrics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from run import prepare_raw,median_imputer_fit,median_imputer_apply,threshold_f1,classification_metrics,candidate

def test_features_never_use_label_or_split():
    features=['mean','std']
    df=pd.DataFrame({'segment':[1,2],'anomaly':[1,0],'train':[0,1],
                     'channel':['a','b'],'mean':[1.,2.],'std':[.3,.6]})
    out=prepare_raw(df,features)
    assert out.shape==(2,2)
    assert np.array_equal(out,np.array([[1,.3],[2,.6]]))

def test_imputation_only_learns_fit_data():
    fit=np.array([[1,np.nan],[3,5],[5,7]],float)
    median=median_imputer_fit(fit)
    assert np.allclose(median,[3,6])
    test=np.array([[np.nan,np.inf]])
    assert median_imputer_apply(test,median).tolist()==[[3,6]]

def test_validation_threshold_and_metrics():
    y=np.array([0,0,1,1])
    scores=np.array([.1,.2,.8,.9])
    th=threshold_f1(y,scores)
    assert .2<th<=.8
    m=classification_metrics(y,scores,th)
    assert m['f1']==1
    assert m['confusion_matrix_TN_FP_FN_TP']==[[2,0],[0,2]]

def test_registered_models_probabilities():
    x=np.random.default_rng(42).normal(size=(80,4))
    y=np.array([0,1]*40)
    for name in ('rf','extra_trees','rbf_svm','hist_gb'):
        m=candidate(name).fit(x,y)
        q=m.predict_proba(x[:5])[:,1]
        assert len(q)==5
        assert np.all(np.isfinite(q))
        assert np.all((q>=0)&(q<=1))
