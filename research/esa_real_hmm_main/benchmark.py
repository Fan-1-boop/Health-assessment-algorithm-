#!/usr/bin/env python3
"""Real ESA anomaly-labelled time series: initialized engineering risk HMM.

This is a controlled REAL DATA E1 ablation: no fabricated physical-state
labels, no score anchors, no adaptive learning and no new patent mechanism.
Uses the established V1.5 globally event-ID-separated chronological partitions
and exactly the original 3-channel real telemetry; original test data have
previously been observed in V1.5 and are EXPLORATORY only.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from v13_protocol import windows_in_partition
from v13_anomaly import RobustLevelDeviation
from v14_alarm import sustained_alert,validation_nominal_threshold,alert_statistics
from v15_global_events import CHANNELS,TARGET,get_global_event_records,global_id_partitions,event_alarm_metrics
SPEC=ROOT/"research/esa_real_hmm_main/PROTOCOL_FROZEN.json"


def nominal_three_tier_fit(risk,ids,seed_cfg):
    """All thresholds, density and pseudo transition use FIT period only."""
    x=np.asarray(risk,float);e=np.asarray(ids,int)
    if len(x)<100 or len(x)!=len(e) or np.any(x<0) or not np.isfinite(x).all():
        raise ValueError("Bad real nominal reference training examples")
    q70,q95=np.quantile(x,[.70,.95])
    if q70>=q95:raise ValueError("Nominal reference risk tiers not separable")
    labels=np.where(x>q95,2,np.where(x>q70,1,0))
    logx=np.log1p(x)
    mu=np.array([logx[labels==c].mean() for c in range(3)])
    var=np.array([max(float(logx[labels==c].var()),.0036) for c in range(3)])
    counts=np.bincount(labels,minlength=3).astype(float)
    prior=(counts+.5)/(counts.sum()+1.5)
    T=np.full((3,3),.5,dtype=float)
    correct_next=e[1:]-e[:-1]==int(seed_cfg["stride"])
    for before,after in zip(labels[:-1][correct_next],labels[1:][correct_next]):
        T[before,after]+=1.
    T/=T.sum(axis=1,keepdims=True)
    return {"quantile70":float(q70),"quantile95":float(q95),
      "weak_labels":"train-normal-reference quantile tiers, not actual physical states",
      "mean":mu,"var":var,"prior":prior,"transition":T,
      "consecutive_nominal_transition_pairs":int(correct_next.sum())}


def static_ema_hmm(risk,ends,params,gamma=.6,memory=.5,ema_alpha=.4,stride=4):
    """Log-space forward filter. Causal: outputs at t do not depend on t+1."""
    x=np.asarray(risk,float);e=np.asarray(ends,int)
    mu,var,prior=(params[k] for k in ("mean","var","prior"))
    obs=-.5*(np.log(2*np.pi*var)[None,:]+(np.log1p(x[:,None])-mu[None,:])**2/var[None,:])
    staticlog=np.log(prior)[None,:]+gamma*obs
    static=np.exp(staticlog-logsumexp(staticlog,axis=1,keepdims=True))
    ema=np.empty_like(static);hmm=np.empty_like(static)
    T_eff=memory*params["transition"]+(1-memory)*np.tile(prior,(3,1))
    LT=np.log(np.maximum(T_eff,1e-12))
    prev_log=np.log(prior)
    for t in range(len(x)):
        if t==0 or e[t]-e[t-1]!=stride:
            prev_log=np.log(prior)
            emap=static[t].copy()
        else:
            emap=ema_alpha*static[t]+(1-ema_alpha)*ema[t-1]
        predictive=logsumexp(prev_log[:,None]+LT,axis=0)
        current=predictive+gamma*obs[t]
        current-=logsumexp(current)
        hmm[t]=np.exp(current)
        ema[t]=emap
        prev_log=current
    for z in [static,ema,hmm]:
        if not np.allclose(z.sum(axis=1),1,atol=1e-6) or not np.isfinite(z).all():
            raise ValueError("Broken probability simplex")
    return static,ema,hmm


def evaluate_one(s,ends,codes,records,part,threshold,stride):
    alarm=sustained_alert(s,threshold,ends,min_run=3,expected_stride=stride)
    y=(codes[ends]==1).astype(int)
    return {
      "test_label_positive_prevalence":float(y.mean()),
      "point_AP":float(average_precision_score(y,s)),
      "point_ROC_AUC":float(roc_auc_score(y,s)),
      "alert_occupancy":alert_statistics(ends,alarm,codes,stride),
      "strict_global_official_event_ID":event_alarm_metrics(
            records,part,ends,codes,alarm,stride,grid_step_minutes=10),
      "threshold_from_nominal_validation_only":float(threshold),
      "alert_definition":"consecutive 3 40min-evaluation windows above threshold"
    }


def run(input_csv,catalog_csv,out_dir):
    start=time.perf_counter()
    config_bytes=SPEC.read_bytes();cfg=json.loads(config_bytes)
    v15=json.loads((ROOT/"configs/v15_esa_protocol_frozen.json").read_text())
    if v15["primary_target"]!=TARGET or tuple(v15["context_channels"])!=CHANNELS[1:]:
        raise ValueError("Original ESA V15 source protocol mismatch")
    source=Path(input_csv);catalog=Path(catalog_csv)
    if not source.is_file() or not catalog.is_file():
        raise FileNotFoundError("Genuine ESA data or official event catalog missing")
    raw=source.read_bytes();src_sha=hashlib.sha256(raw).hexdigest()
    expected_sha="bf42672b0a611cb855e44b93f00af535c774c1f25965e6d459c5781e8e86d58c"
    if src_sha!=expected_sha:raise ValueError("Unexpected ESA 3-channel source hash")
    df=pd.read_csv(source)
    times=pd.to_datetime(df.timestamp,utc=True,format="mixed")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Real time axis invalid")
    signals=df[list(CHANNELS)].to_numpy(float)
    codes=df.eval_label.to_numpy(int)
    if not np.isfinite(signals).all() or not np.isin(codes,[0,1,2,3,4]).all():
        raise ValueError("Signal or annotation invalid")
    records=get_global_event_records(pd.read_csv(catalog),times,codes)
    part,audit=global_id_partitions(records,len(df),int(v15["lookback"]),
                       int(v15["min_target_anomaly_ids_per_partition"]))
    if audit["cut_indices"]!=[431304,593284]:
        raise ValueError("Not original frozen event-ID split")
    windows={};end={};ys={}
    for p in part:
        w,e,y,_=windows_in_partition(signals,codes,p,lookback=16,stride=4)
        windows[p.name]=w;end[p.name]=e;ys[p.name]=y
        if y.sum()==0 or y.sum()==len(y):raise ValueError("No positive and negative official labels")

    train_w=windows["train"];train_e=end["train"]
    cs=np.r_[0,np.cumsum((codes!=0).astype(int))]
    pure_train=(cs[train_e+1]-cs[train_e+1-16])==0
    train_nominal=train_w[pure_train]
    if len(train_nominal)>20000:
        nominal_fit=train_nominal[np.linspace(0,len(train_nominal)-1,20000,dtype=int)]
    else:nominal_fit=train_nominal
    model=RobustLevelDeviation().fit(nominal_fit)
    # Use full historical nominal sequence, not 20k downsampled fit points,
    # for actual consecutive transition counts.
    nominal_risk=model.score(train_nominal)
    state=nominal_three_tier_fit(nominal_risk,train_e[pure_train],
                                     {"stride":4})
    experiments={}
    all_pred={}
    for p in part[1:]:
        name=p.name
        measured_risk=model.score(windows[name])
        st,ema,hmm=static_ema_hmm(measured_risk,end[name],state,
           gamma=float(cfg["filter"]["gamma"]),
           memory=float(cfg["filter"]["memory"]),
           ema_alpha=float(cfg["filter"]["ema_alpha"]),stride=4)
        experiments[name]={
          "robust_level_reference":measured_risk,
          "static_three_tier":1-st[:,0],
          "EMA_three_tier":1-ema[:,0],
          "HMM_three_tier":1-hmm[:,0]
        }
        if name=="test":
            all_pred={"end":end[name],"code":codes[end[name]],
               "HMM_prob_H":hmm[:,0],"HMM_prob_D":hmm[:,1],
               "HMM_prob_R":hmm[:,2],"risk_score":measured_risk}
    output={}
    for name in cfg["models"]:
        score_cal=experiments["validation"][name]
        t=validation_nominal_threshold(score_cal,end["validation"],codes,
                   false_positive_fraction=float(cfg["alarm"]["nominal_validation_false_positive_fraction"]))
        output[name]=evaluate_one(experiments["test"][name],end["test"],
                                  codes,records,part[-1],t,4)
        output[name]["validation_AP_diagnostic_no_model_selection"]=float(
            average_precision_score(ys["validation"],score_cal))
    outputdir=Path(out_dir);outputdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({
      "actual_global_end_index":all_pred["end"],
      "official_anomaly_label_at_endpoint":(all_pred["code"]==1).astype(int),
      "official_rare_nominal_at_endpoint":(all_pred["code"]==2).astype(int),
      "main_state_H_weak":all_pred["HMM_prob_H"],
      "main_state_D_weak":all_pred["HMM_prob_D"],
      "main_state_R_weak":all_pred["HMM_prob_R"],
      "display_score_policy_initial_only":100*all_pred["HMM_prob_H"]+60*all_pred["HMM_prob_D"],
      **{f"score_{k}":v for k,v in experiments["test"].items()}
    }).to_csv(outputdir/"real_timepoint_scores.csv",index=False)
    serial={
      "study":"ESA Mission1 THREE CHANNELS real weak-risk HMM, first continuous chronological E1",
      "research_only":True,"raw_telemetry_real":True,"any_generated_synthetic_labels":False,
      "weak_risk_tiers_are_author_defined_NOT_physical_fault_truth":True,
      "expert_health_score_ground_truth_count":0,
      "source":str(source),"source_sha256":src_sha,
      "catalog_sha256":hashlib.sha256(catalog.read_bytes()).hexdigest(),
      "protocol_sha256":hashlib.sha256(config_bytes).hexdigest(),
      "source_data_sampling":"10min previous observation max-staleness60min, NOT official 30second ESA preprocessing",
      "full_real_grid_points":len(df),
      "official_event_ID_splits":audit["global_id_splits"],
      "cut_indices":audit["cut_indices"],
      "target":TARGET,"channel_inputs":list(CHANNELS),
      "fit_windows_pure_nominal":int(len(nominal_fit)),
      "train_nominal_windows_for_weak_transition":int(len(train_nominal)),
      "validation_windows":int(len(end["validation"])),
      "test_windows":int(len(end["test"])),
      "initial_reference_thresholds":{"70pct":state["quantile70"],"95pct":state["quantile95"]},
      "weak_HMM_transition":state["transition"].tolist(),
      "weak_HMM_prior":state["prior"].tolist(),
      "weak_HMM_density_mu":state["mean"].tolist(),
      "weak_HMM_transition_pairs":state["consecutive_nominal_transition_pairs"],
      "outputs":output,
      "timer_seconds":float(time.perf_counter()-start),
      "caveats":[
        "Existing V1.5 test event IDs have been viewed; all outcomes are exploratory, no fresh blind test",
        "True target normal/anomaly annotation and rare nominal notes are authentic ESA but no true H/D/R physical state labels",
        "Training-derived weak tier transitions are about changes in reference deviation, not actual onboard hardware state",
        "HMM reset when missing window violates stride, filter never looks forward",
        "Equal 99.8% nominal validation cutoff and three-in-a-row alert for all arms",
        "No model selected by heldout outcome; no threshold recalibration using test",
        "Display score uses initial 100/60/0 mapping, no human expert calibration data",
        "No MSF64 integration in this exploratory continuous HMM risk baseline; historical OPS-SAT fixed budget unchanged",
        "Our resampling is 10min, evaluation stride 40min; short anomalies can be unobserved",
        "Strict new alert onset is distinct from being covered by an alarm already on",
        "Rare nominal events are negative only under this study protocol and are separately audited",
        "No true operator update or autonomous model replacement"
      ]
    }
    (outputdir/"metrics.json").write_text(json.dumps(serial,indent=2,ensure_ascii=False)+"\n")
    table=[
      "# ESA真实连续卫星遥测：初始健康风险HMM vs 静态/EMA vs 常规异常基线",
      "",
      "**仅真实ESA卫星遥测和官方异常事件标签；H/D/R是正常参考自定义风险层级，不是真实物理故障等级。原V1.5已曝光的测试事件，因此是探索性而非盲测结果。**",
      "",
      "| 方法 | 测试AP | 测试AUC | 正常告警时间占比 | Rare正常告警时间占比 | 官方事件覆盖召回 | 官方事件新起点召回 | 每30天假告警起点 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k,z in output.items():
        a=z["alert_occupancy"];e=z["strict_global_official_event_ID"]
        table.append(f"| {k} | {z['point_AP']:.5f} | {z['point_ROC_AUC']:.4f} | {a['normal_alert_rate']:.2%} | {a['rare_nominal_alert_rate']:.2%} | {e['coverage_ID_recall']:.2%} | {e['strict_new_onset_ID_recall']:.2%} | {e['false_alarm_new_onsets_per_30_elapsed_days']:.3f} |")
    table+=[
      "",f"数据点：{len(df)}，纯正常训练窗口：{len(train_nominal)}，验证{len(end['validation'])}，审计{len(end['test'])}。",
      f"弱层阈值：nominal risk q70={state['quantile70']:.4f}、q95={state['quantile95']:.4f}。",
      "所有模型告警统一使用**验证正常99.8%分位阈值、连续3窗口触发**，未按测试优化。",
      "本研究只检验因果HMM相对于静态/EMA及其母体稳健偏离分数的实际作用。若不能明显改善稀有正常误报及新告警，不能宣称健康主线完成真实有效性验收。",
      "所有健康分数是未由专家校准的初始展示值，不能由此报告人工评分MAE或真实故障检测准确率。",
      ""]
    (outputdir/"RESULTS.md").write_text("\n".join(table),encoding="utf-8")
    print("ESA_REAL_HMM",json.dumps({
      "train_nominal":len(train_nominal),"val":len(end["validation"]),"test":len(end["test"]),
      "tier_q":[state["quantile70"],state["quantile95"]],
      "arms":{k:{"AP":z["point_AP"],
         "FPR":z["alert_occupancy"]["normal_alert_rate"],
         "rare_FPR":z["alert_occupancy"]["rare_nominal_alert_rate"],
         "new_onset":z["strict_global_official_event_ID"]["strict_new_onset_ID_recall"]}
         for k,z in output.items()},
       "seconds":serial["timer_seconds"]},ensure_ascii=False),flush=True)
    return serial


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/esa_v15_multichannel/three_channels.csv.gz")
    p.add_argument("--catalog",default="data/esa_v15_multichannel/mission1_event_catalog.csv")
    p.add_argument("--out",default="results/real_esa_health_hmm")
    args=p.parse_args()
    run(args.input,args.catalog,args.out)
