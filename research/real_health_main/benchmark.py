#!/usr/bin/env python3
"""First REAL-TELEMETRY HMM health-index benchmark; not physical failure staging.

Only OPS-SAT-AD original train=1, genuine point measurements and published
segment binary groundtruth. Every feature window is causal INSIDE its segment.
Weak health/deviation/high-risk classes are an explicit INITIALIZATION from
nominal reference quantiles, NOT groundtruth. HMM has no true state transition
labels. Distinct fit/fusion/threshold/holdout segment-group phases.
Final display health score is an INITIAL POLICY, never expert-calibrated.
"""
import argparse,hashlib,io,json,math,sys,time
from pathlib import Path
import numpy as np,pandas as pd,requests
from scipy.special import expit, logsumexp,logit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (average_precision_score,roc_auc_score,
    f1_score,recall_score,precision_score,brier_score_loss,log_loss)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[2]
SPEC=ROOT/"research/real_health_main/PROTOCOL_FROZEN.json"
SOURCE="https://zenodo.org/api/records/12588359/files/segments.csv/content"


def split_groups(keys,groups,frac,seed):
    keys=np.asarray(keys,int);group=np.asarray(groups)
    left,right=next(GroupShuffleSplit(n_splits=1,test_size=float(frac),
             random_state=int(seed)).split(keys,groups=group[keys]))
    result=keys[left],keys[right]
    assert set(group[result[0]]).isdisjoint(set(group[result[1]]))
    return tuple(np.sort(x) for x in result)


def features_last16(w):
    z=np.asarray(w,float)
    dd=np.diff(z);zz=z-z.mean()
    t=np.arange(len(z),dtype=float);t=t-t.mean()
    slope=float(np.dot(zz,t)/(np.dot(t,t)+1e-12))
    return np.array([
       z.mean(),np.log1p(np.std(z)),
       z.min(),z.max(),np.median(z),
       np.sqrt(np.mean(z*z)),slope,np.mean(np.abs(dd))])


def build_sequences(original_train):
    # Temporal order proven within each source segment; never cross segments.
    sequences={};table=[]
    for segment,s in original_train.groupby("segment",sort=True):
        s=s.sort_values("timestamp",kind="stable")
        val=s.value.to_numpy(float)
        if len(val)<16:continue
        ends=list(range(15,len(val),8))
        if ends[-1]!=len(val)-1:ends.append(len(val)-1)
        x=np.stack([features_last16(val[i-15:i+1]) for i in ends])
        assert np.isfinite(x).all()
        sequences[int(segment)]={"x":x,
            "binary_y":int(s.anomaly.iloc[0]),
            "source_label":str(s.label.iloc[0]),
            "channel":str(s.channel.iloc[0]),"start":str(s.timestamp.iloc[0]),
            "last":str(s.timestamp.iloc[-1]),
            "n_raw_samples":len(val),"n_windows":len(ends)}
        table.append({"segment":int(segment),"channel":str(s.channel.iloc[0]),
            "binary_y":int(s.anomaly.iloc[0]),"source_label":str(s.label.iloc[0])})
    return sequences,pd.DataFrame(table).sort_values("segment").reset_index(drop=True)


def learned_nominal_reference(sequences,fitids):
    """Only historical FIT nominal segment values may define the reference."""
    all_nom={}
    for i in fitids:
        s=sequences[int(i)]
        if s["binary_y"]!=0:continue
        all_nom.setdefault(s["channel"],[]).append(s["x"][:,[0,1]])
    global_nom=np.vstack([x for a in all_nom.values() for x in a])
    def obj(mat):
        med=np.median(mat,axis=0)
        scale=np.maximum(1.4826*np.median(np.abs(mat-med),axis=0),
                        np.std(mat,axis=0)*.10+1e-8)
        return np.array(med),np.array(scale)
    ref_all=obj(global_nom)
    references={ch:obj(np.vstack(arr)) if len(arr)>=2 else ref_all
                for ch,arr in all_nom.items()}
    return references,ref_all


def risk_sequence(sequences,ids,refs,fallback):
    risks={}
    for i in ids:
        s=sequences[int(i)]
        med,scale=refs.get(s["channel"],fallback)
        z=np.clip((s["x"][:,[0,1]]-med)/scale,-30,30)
        risks[int(i)]=np.sqrt((z*z).sum(axis=1))
    return risks


def reference_thresholds(sequences,fitids,risks):
    nom=np.concatenate([risks[int(i)] for i in fitids
                       if sequences[int(i)]["binary_y"]==0])
    q=np.quantile(nom,[.50,.70,.95])
    if q[2]-q[1]<1e-3:raise ValueError("Unable to determine nominal risk quantiles")
    return {"q50":float(q[0]),"q70":float(q[1]),"q95":float(q[2])}


def risk_tier(r,q):
    return np.where(r>q["q95"],2,np.where(r>q["q70"],1,0))


def fit_hmm(sequences,fitids,risks,q):
    # Actual observed train risk evolution; state classes INIT by nominal quantiles.
    obs=[];labels=[]
    trans=np.full((3,3),.5)
    for i in fitids:
        s=sequences[int(i)];r=risks[int(i)]
        y=risk_tier(r,q);o=np.log1p(r)
        obs.extend(o);labels.extend(y)
        for a,b in zip(y[:-1],y[1:]):trans[int(a),int(b)]+=1
    values=np.asarray(obs,float);classes=np.asarray(labels,int)
    prior=(np.bincount(classes,minlength=3)+.5).astype(float);prior/=prior.sum()
    gmean=np.array([values[classes==j].mean() for j in range(3)])
    var=np.array([max(values[classes==j].var(),.06**2) for j in range(3)])
    return {"mean":gmean,"variance":var,
         "transition":trans/trans.sum(axis=1,keepdims=True),
         "prior":prior}


def static_and_causal_hmm(r,hmm,gamma=.6,memory=.5):
    x=np.log1p(r)
    mean,var,prior=hmm["mean"],hmm["variance"],hmm["prior"]
    emiss=-.5*(np.log(2*np.pi*var)[None,:]+(x[:,None]-mean[None,:])**2/var[None,:])
    # Static is a genuinely non-temporal Bayes classifier with SAME emissions.
    st=emiss+np.log(prior)[None,:]
    st=np.exp(st-logsumexp(st,axis=1,keepdims=True))
    T=hmm["transition"]
    te=memory*T+(1-memory)*np.tile(prior,(3,1))
    logT=np.log(np.maximum(te,1e-12))
    current=np.log(prior)
    states=np.zeros((len(r),3))
    for t in range(len(r)):
        prediction=logsumexp(current[:,None]+logT,axis=0)
        current=prediction+gamma*emiss[t]
        current-=logsumexp(current)
        states[t]=np.exp(current)
    # EMA is a conventional non-HMM causal smoothing baseline.
    ema=np.empty_like(st);current=st[0].copy()
    for t in range(len(r)):
        current=.4*st[t]+.6*current
        ema[t]=current
    return st,states,ema


def grouped_subset(sequences,ids,per_segment=12):
    arr=[];yt=[];weights=[]
    for i in ids:
        s=sequences[int(i)]
        x=s["x"]
        # Future-to-select schedule is for OFFLINE candidate model fitting only.
        ix=np.unique(np.linspace(0,len(x)-1,min(len(x),per_segment),dtype=int))
        arr.append(x[ix]);yt.extend([s["binary_y"]]*len(ix))
        weights.extend([1/len(ix)]*len(ix))
    return np.vstack(arr),np.asarray(yt,int),np.asarray(weights,float)


def fit_supervised_aux(sequences,ids):
    x,y,w=grouped_subset(sequences,ids)
    if len(np.unique(y))!=2:raise ValueError("Missing binary anomaly class in fit")
    model=make_pipeline(StandardScaler(),LogisticRegression(
        C=.5,class_weight="balanced",solver="liblinear",max_iter=500,random_state=42))
    model.fit(x,y,logisticregression__sample_weight=w)
    return model


def values_for_part(sequences,ids,risks,hmm,aux):
    vals=[]
    for i in ids:
        s=sequences[int(i)]
        a,b,e=static_and_causal_hmm(risks[int(i)],hmm)
        # Alarm probability at last causal window of the true real segment.
        anomaly=float(aux.predict_proba(s["x"][-1:])[:,1][0])
        vals.append({
         "segment":int(i),"channel":s["channel"],
         "official_label":s["source_label"],"binary_y":s["binary_y"],
         "num_windows":s["n_windows"],"num_samples":s["n_raw_samples"],
         "p_static_healthy":float(a[-1,0]),
         "p_hmm_healthy":float(b[-1,0]),
         "p_hmm_deviation":float(b[-1,1]),
         "p_hmm_highrisk":float(b[-1,2]),
         "p_static_risk":float(1-a[-1,0]),
         "p_ema_risk":float(1-e[-1,0]),
         "p_hmm_risk":float(1-b[-1,0]),
         "p_aux_anomaly":anomaly,
         "main_health_index_initial":float(b[-1]@np.array([100.,60.,0.])),
        })
    return pd.DataFrame(vals)


def fusion_columns(f):
    p=np.clip(f[["p_hmm_healthy","p_hmm_deviation",
                  "p_hmm_highrisk"]].to_numpy(float),1e-7,1.)
    a=np.clip(f["p_aux_anomaly"].to_numpy(float),1e-7,1-1e-7)
    return np.column_stack([np.log(p),logit(a)])


def train_aux_fusion(f):
    y=f.binary_y.to_numpy(int)
    if set(np.unique(y))!={0,1}:raise ValueError("Fusion needs both real anomaly label classes")
    m=LogisticRegression(C=.5,solver="lbfgs",max_iter=1200,random_state=42)
    m.fit(fusion_columns(f),y)
    return m


def threshold_for_f1(y,p):
    from sklearn.metrics import precision_recall_curve
    a,b,t=precision_recall_curve(y,p)
    f=2*a[:-1]*b[:-1]/np.maximum(a[:-1]+b[:-1],1e-12)
    return float(t[int(np.argmax(f))])


def report(y,probs,threshold,calibrated=False):
    y=np.asarray(y,int);probs=np.asarray(probs,float)
    pred=probs>=threshold
    d={"segments":int(len(y)),"anomalies":int(y.sum()),
       "AP":float(average_precision_score(y,probs)),
       "ROC_AUC":float(roc_auc_score(y,probs)),
       "F1":float(f1_score(y,pred)),"Recall":float(recall_score(y,pred)),
       "Precision":float(precision_score(y,pred,zero_division=0)),
       "FPR":float(np.mean(pred[y==0])),
       "FP":int(np.sum(pred&(y==0))),"FN":int(np.sum((~pred)&(y==1))),
       "threshold_from_independent_calibration":float(threshold)}
    if calibrated:
        d["binary_Brier"]=float(brier_score_loss(y,probs))
        d["binary_NLL"]=float(log_loss(y,np.column_stack([1-probs,probs]),labels=[0,1]))
    return d


def save_plot(frame,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols={"static":"p_static_risk","EMA":"p_ema_risk","HMM":"p_hmm_risk",
          "aux":"p_aux_anomaly","aux_cal":"p_aux_cal_anomaly",
          "fusion":"p_fused_anomaly"}
    fig,ax=plt.subplots(figsize=(11,5))
    for key,col in cols.items():
        # Raw scores are never assumed calibrated P(physical fault).
        a=frame[frame.binary_y==0][col];b=frame[frame.binary_y==1][col]
        ax.scatter([list(cols).index(key)+.05]*len(a),a,s=2,alpha=.15)
        ax.scatter([list(cols).index(key)+.20]*len(b),b,s=3,alpha=.22)
    ax.set_xticks(np.arange(len(cols))+.13,list(cols))
    ax.set_ylabel("End-of-real-segment anomaly-related score")
    ax.set_title("REAL OPS-SAT: nominal/anomalous segment evidence (not faults)")
    fig.tight_layout();fig.savefig(out/"real_segment_scores.png",dpi=160);plt.close(fig)


def run(path):
    out=Path(path);out.mkdir(parents=True,exist_ok=True)
    cfgraw=SPEC.read_bytes()
    cfg=json.loads(cfgraw)
    t0=time.perf_counter()
    response=requests.get(SOURCE,timeout=130);response.raise_for_status()
    source_sha=hashlib.sha256(response.content).hexdigest()
    # Explicit original-test firewall BEFORE constructing any reference/model.
    full=pd.read_csv(io.BytesIO(response.content),parse_dates=["timestamp"])
    source=full.loc[full.train==1].copy()
    assert len(source)==225178 and source.segment.nunique()==1594
    assert all(v.nunique()==1 for _,v in source.groupby("segment")["label"])
    seq,meta=build_sequences(source)
    seg_ids=meta.segment.to_numpy(int)
    group=np.empty(len(meta),dtype=object)
    for ch,sub in meta.groupby("channel"):
        inds=sub.sort_values("segment").index.to_numpy(int)
        for j,idx in enumerate(inds):group[idx]=f"{ch}_{j//16:03}"
    fitpos,remaining=split_groups(np.arange(len(meta)),group,.4,20261026)
    # Remaining 40% segment blocks: 45% fusion fit, rest 25% cal, 30% audit.
    fusedpos,rest=split_groups(remaining,group,.55,20261026+7)
    calpos,auditpos=split_groups(rest,group,.30/.55,20261026+11)
    splits={"fit":meta.segment.iloc[fitpos].to_numpy(int),
      "fusion_fit":meta.segment.iloc[fusedpos].to_numpy(int),
      "calibration":meta.segment.iloc[calpos].to_numpy(int),
      "audit":meta.segment.iloc[auditpos].to_numpy(int)}
    allids=np.concatenate(list(splits.values()))
    assert len(set(allids))==len(allids)==len(seq)
    for part,ids in splits.items():
        y=meta.set_index("segment").loc[ids,"binary_y"].to_numpy(int)
        assert len(y)>20 and 0<y.sum()<len(y),(part,len(y),y.sum())
    refs,fallback=learned_nominal_reference(seq,splits["fit"])
    risk=risk_sequence(seq,allids,refs,fallback)
    q=reference_thresholds(seq,splits["fit"],risk)
    hmm=fit_hmm(seq,splits["fit"],risk,q)
    aux=fit_supervised_aux(seq,splits["fit"])
    frames={k:values_for_part(seq,ids,risk,hmm,aux) for k,ids in splits.items()
            if k in ("fusion_fit","calibration","audit")}
    fuse=train_aux_fusion(frames["fusion_fit"])
    # Fair comparator: ordinary single-score Platt calibration trained on
    # EXACT SAME real labeled fusion_fit segments as the HMM-assisted fusion.
    fit_f=frames["fusion_fit"]
    aux_cal=LogisticRegression(C=.5,solver="lbfgs",max_iter=1200,random_state=42)
    aux_cal.fit(logit(np.clip(fit_f["p_aux_anomaly"].to_numpy(float),
                      1e-7,1-1e-7)).reshape(-1,1),
                fit_f.binary_y.to_numpy(int))
    for key,f in frames.items():
        f["p_fused_anomaly"]=fuse.predict_proba(fusion_columns(f))[:,1]
        f["p_aux_cal_anomaly"]=aux_cal.predict_proba(
            logit(np.clip(f["p_aux_anomaly"].to_numpy(float),
                          1e-7,1-1e-7)).reshape(-1,1))[:,1]
        # Initial DISPLAY POLICY after all evidence: NOT expert-rated.
        f["final_display_health_index_initial"]=np.clip(
            f["main_health_index_initial"]-15*f["p_fused_anomaly"],0,100)
        # Alarm stays EXACTLY from objective detector, not post-score override.
        f["alarm_input_p"]=f["p_aux_anomaly"]
    names={"static":"p_static_risk","EMA":"p_ema_risk","HMM":"p_hmm_risk",
           "aux":"p_aux_anomaly","aux_cal":"p_aux_cal_anomaly",
           "fusion":"p_fused_anomaly"}
    calibration=frames["calibration"]
    audit=frames["audit"]
    stats={}
    for name,col in names.items():
        th=threshold_for_f1(calibration.binary_y,calibration[col])
        stats[name]=report(audit.binary_y,audit[col],th,
                           calibrated=name in ("aux","aux_cal","fusion"))
    for item in (calibration,audit):
        assert np.array_equal(item.alarm_input_p.to_numpy(),item.p_aux_anomaly.to_numpy())
        assert (item.final_display_health_index_initial.between(0,100)).all()
        assert (item.main_health_index_initial.between(0,100)).all()
    audit.to_csv(out/"audit_segment_predictions.csv",index=False)
    save_plot(audit,out)
    counts={k:{"segments":len(v),"binary_anomalies":int(np.sum([seq[int(i)]["binary_y"]
                       for i in v]))} for k,v in splits.items()}
    result={
      "study":"REAL OPS-SAT retrospective causal-within-segment HMM weak-tier health init plus supervised objective anomaly evidence",
      "source_sha256":source_sha,
      "protocol_sha256":hashlib.sha256(cfgraw).hexdigest(),
      "original_official_test_0_excluded":True,
      "all_raw_inputs_genuine":True,
      "no_synthetic_data":True,
      "no_true_physical_H_D_R_labels":True,
      "no_expert_score_anchors":True,
      "split":counts,
      "raw_windows":int(sum(v["n_windows"] for v in seq.values())),
      "nominal_reference_quantiles":q,
      "hmm_initial_state_prior":hmm["prior"].tolist(),
      "hmm_weak_state_transition_matrix":hmm["transition"].tolist(),
      "hmm_state_semantics":"Weak rule-based nominal-reference risk tiers. NOT observed physical state transitions.",
      "classification_metrics_against_PUBLISHED_REAL_SEGMENT_ANOMALY_LABELS":stats,
      "fair_fusion_increment_vs_same_data_calibrated_aux":{
        "delta_AP":float(stats["fusion"]["AP"]-stats["aux_cal"]["AP"]),
        "delta_binary_Brier":float(stats["fusion"]["binary_Brier"]-stats["aux_cal"]["binary_Brier"]),
        "delta_binary_NLL":float(stats["fusion"]["binary_NLL"]-stats["aux_cal"]["binary_NLL"]),
        "delta_F1":float(stats["fusion"]["F1"]-stats["aux_cal"]["F1"]),
        "delta_FPR":float(stats["fusion"]["FPR"]-stats["aux_cal"]["FPR"]),
        "same_fusion_training_segments_for_both":True
      },
      "score":"Initial policy 100 H + 60 D + 0 R minus 15 times fused observed anomaly probability; not tuned to experts.",
      "elapsed_seconds":time.perf_counter()-t0,
      "critical_limitations":[
         "No physical 3-state labels; auxiliary binary class only is genuine expert-published groundtruth",
         "Entire anomaly truth refers to the original SEGMENT; prefix windows inherit it solely for supervised auxiliary FIT, not for per-point scoring",
         "Within-segment timesteps are chronological but changing risk pseudo-states cannot be verified as physical transitions",
         "Data partitions refer to the same earlier-examined source; this result exploratory, not fresh independent blind",
         "Original test 529 segments NEVER used; source entire CSV loaded then excluded before training/validation",
         "Risk/nonhealthy output is NOT a calibrated probability of fault and uses nominal-based author-defined thresholds",
         "No REAL expert score anchors; mapping is initialization only, do not claim human agreement improvement",
         "The main and auxiliary branch use correlated views of the same signal; fusion must beat auxiliary alone to support incremental value",
         "This quick risk baseline does not yet contain the frozen MSF 64-budget and two-level logistic classifiers; these are separate existing research modules",
         "Historic full-segment classification results are not comparable: prefix16 has less information than entire official segment"
      ]}
    (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    lines=["# 真实OPS-SAT：健康度主线HMM初始化和异常辅线比较",
      "","**真实遥测，无生成数据；三级运行等级仅由正常参考分布的预设分位规则初始化，非物理真值；原529条官方测试未使用。**",
      "","| 方法 | 审计AP | 审计F1 | 审计Recall | 正常误报FPR |",
      "|---|---:|---:|---:|---:|"]
    for n,m in stats.items():
        lines.append(f"| {n} | {m['AP']:.4f} | {m['F1']:.4f} | {m['Recall']:.4f} | {m['FPR']:.4f} |")
    lines+=["",f"真实训练内部阶段：{counts}；风险等级初始阈值：{q}。",
       f"来源哈希：{source_sha}；真实时序窗口总数：{result['raw_windows']}。",
      "真实监督目标仅是整段正常/异常；HMM是弱定义运行风险状态的因果滤波，不是设备真实故障状态识别。",
      "显示健康度采用**人为初始评分政策**，没有专家评分，不能汇报评分MAE。客观异常告警始终独立于主观分数。",
      "本次采用轻量风险特征基线验证主辅逻辑，完整MSF64/两级状态分类器的真实融合效果尚待后续对照。",
      "![real](real_segment_scores.png)"]
    (out/"RESULTS.md").write_text("\n".join(lines))
    print("REAL_MAIN_RESULT",json.dumps({
     "split":counts,"q":q,
     "metrics":{k:{f:m[f] for f in ("AP","F1","Recall","FPR")}
                for k,m in stats.items()},
      "seconds":result["elapsed_seconds"]}),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/real_health_main")
    a=parser.parse_args()
    run(a.out)
