#!/usr/bin/env python3
"""ESA real continuous telemetry: fixed observation E2, static vs EMA vs HMM.

All source telemetry is authentic. Engineering H/D/R risk states are weakly
INITIALIZED from training normal references; never physical failure labels.
Real official abnormal vs rare-normal targets used ONLY for external evaluation
and for validation-only alarm thresholds. Previous mission test IDs were seen.
This is exploratory; not independent blind evidence or patent core.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"research/esa_real_hmm_main"))
from v13_protocol import windows_in_partition
from v13_anomaly import RobustLevelDeviation,ForecastRidge
from v14_alarm import sustained_alert,validation_nominal_threshold,alert_statistics
from v15_global_events import CHANNELS,TARGET,get_global_event_records,global_id_partitions,event_alarm_metrics
from research.esa_real_hmm_main.benchmark import nominal_three_tier_fit,static_ema_hmm

SPEC=ROOT/"research/esa_observation_e2/PROTOCOL_FROZEN.json"
SOURCE="data/esa_v15_multichannel/three_channels.csv.gz"
CATALOG="data/esa_v15_multichannel/mission1_event_catalog.csv"
EXPOSED_TEST_CUTS=[431304,593284]

def local_change_matrix(w):
    """Last observed value minus median of EIGHT preceding real measurements."""
    a=np.asarray(w,dtype=float)
    if a.ndim!=3 or a.shape[1]!=16 or a.shape[2]!=3:
        raise ValueError("Expected 16x3 causal real windows")
    return a[:,-1,:]-np.median(a[:,-9:-1,:],axis=1)

def robust_component_reference(nominal_window):
    raw=local_change_matrix(nominal_window)
    loc=np.median(raw,axis=0)
    scale=1.4826*np.median(np.abs(raw-loc),axis=0)
    scale=np.maximum(scale,.05*np.std(raw,axis=0)+1e-5)
    return loc,scale

def local_change_score(w,reference):
    loc,scale=reference
    return np.max(np.abs((local_change_matrix(w)-loc)/scale),axis=1)

def normal_reference_rank(score,sorted_nominal_scores):
    """Rank only vs train pure-nominal. Test labels never enter reference."""
    z=np.asarray(score,float)
    if not np.isfinite(z).all():raise ValueError("Nonfinite risk observation")
    ref=np.asarray(sorted_nominal_scores,float)
    if len(ref)<100 or np.any(np.diff(ref)<0):raise ValueError("Invalid ranked nominal reference")
    # Uncertainty in empirical far right tail is acknowledged.
    return np.searchsorted(ref,z,side="right")/(len(ref)+1.)

def compute_risks(w,models,rank_refs):
    level=models["level"].score(w)
    innovation=models["innovation"].score(w)
    change=local_change_score(w,models["change_ref"])
    components={"level":level,"innovation":innovation,"change":change}
    ranks={k:normal_reference_rank(v,rank_refs[k]) for k,v in components.items()}
    combo=.35*ranks["level"]+.45*ranks["innovation"]+.20*ranks["change"]
    if np.any(combo<0) or np.any(combo>1+1e-9):
        raise AssertionError("Nominal-ranked combined evidence out of bounds")
    return components,combo,ranks

def parse_esa(source,catalog,reference_v15):
    src=Path(source)
    if hashlib.sha256(src.read_bytes()).hexdigest()!="bf42672b0a611cb855e44b93f00af535c774c1f25965e6d459c5781e8e86d58c":
        raise ValueError("Expected actual pinned three-channel ESA satellite data")
    df=pd.read_csv(src)
    if not set(["timestamp","eval_label",*CHANNELS]).issubset(df.columns):
        raise ValueError("ESA expected channel/label schema missing")
    times=pd.to_datetime(df.timestamp,utc=True,format="mixed")
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Noncausal or duplicate raw time grid")
    codes=df.eval_label.to_numpy(int)
    a=df[list(CHANNELS)].to_numpy(float)
    if not np.isfinite(a).all() or not np.isin(codes,[0,1,2,3,4]).all():
        raise ValueError("Unexpected real ESA data")
    catalog_data=pd.read_csv(catalog)
    records=get_global_event_records(catalog_data,times,codes)
    parts,audit=global_id_partitions(records,len(df),16,2)
    if audit["cut_indices"]!=EXPOSED_TEST_CUTS:
        raise ValueError("Changed official-ID-pure frozen split")
    train_nom_count=0
    windows={};ends={};y={}
    for p in parts:
        wi,e,yi,_=windows_in_partition(a,codes,p,lookback=16,stride=4)
        windows[p.name]=wi
        ends[p.name]=e
        y[p.name]=yi
        if sum(yi)==0 or sum(yi)==len(yi):
            raise ValueError("Missing observed ESA binary anomaly class")
    return codes,records,parts,audit,windows,ends,y

def measure(score,ends,codes,events,part,validation_score,validation_ends):
    t=validation_nominal_threshold(validation_score,validation_ends,codes,
                                  false_positive_fraction=.002)
    alarm=sustained_alert(score,t,ends,min_run=3,expected_stride=4)
    y=(codes[ends]==1).astype(int)
    return {
      "test_AP":float(average_precision_score(y,score)),
      "test_AUROC":float(roc_auc_score(y,score)),
      "test_anomaly_prevalence":float(y.mean()),
      "alarm_cutoff_from_validation":float(t),
      "alarm_occupancy":alert_statistics(ends,alarm,codes,4),
      "official_ID_metrics":event_alarm_metrics(events,part,ends,codes,alarm,4,grid_step_minutes=10)
    }

def run(source=SOURCE,catalog=CATALOG,out="results/esa_observation_e2"):
    t0=time.perf_counter()
    cfg_raw=SPEC.read_bytes()
    cfg=json.loads(cfg_raw)
    frozen=json.loads((ROOT/"configs/v15_esa_protocol_frozen.json").read_text())
    if frozen["primary_target"]!=TARGET or tuple(frozen["context_channels"])!=CHANNELS[1:]:
        raise ValueError("V15 source target changed")
    codes,events,part,audit,w,ends,y=parse_esa(source,catalog,frozen)
    pure=np.r_[0,np.cumsum((codes!=0).astype(int))]
    te=ends["train"]
    good=(pure[te+1]-pure[te+1-16])==0
    full_normal=w["train"][good]
    full_end=te[good]
    fit=(full_normal[np.linspace(0,len(full_normal)-1,20000,dtype=int)]
         if len(full_normal)>20000 else full_normal)
    # All components fit using EXACT SAME training-only pure nominal windows.
    models={
      "level":RobustLevelDeviation().fit(fit),
      "innovation":ForecastRidge().fit(fit),
      "change_ref":robust_component_reference(fit)
    }
    nominal_for_ecdf={
      "level":models["level"].score(fit),
      "innovation":models["innovation"].score(fit),
      "change":local_change_score(fit,models["change_ref"])
    }
    rank_refs={k:np.sort(v) for k,v in nominal_for_ecdf.items()}
    components_train,train_combo,_=compute_risks(full_normal,models,rank_refs)
    hmm_params=nominal_three_tier_fit(train_combo,full_end,{"stride":4})
    scores={};testing_details={}
    for phase in ("validation","test"):
        partscores,combined,_=compute_risks(w[phase],models,rank_refs)
        st,ema,hmm=static_ema_hmm(combined,ends[phase],hmm_params,
                 gamma=.6,memory=.5,ema_alpha=.4,stride=4)
        score={
          "original_robust_level":partscores["level"],
          "forecast_innovation":partscores["innovation"],
          "local_change":partscores["change"],
          "fixed_combined_static_risk":combined,
          "fixed_combined_static_3tier":1-st[:,0],
          "fixed_combined_EMA_3tier":1-ema[:,0],
          "fixed_combined_HMM_3tier":1-hmm[:,0]
        }
        assert list(score)==cfg["evaluated_models"]
        scores[phase]=score
        if phase=="test":
            testing_details={
              "raw_global_grid_indices":ends[phase],
              "official_anomaly":y[phase],
              "official_rare_nominal":(codes[ends[phase]]==2).astype(int),
              "weak_H":hmm[:,0],
              "weak_D":hmm[:,1],
              "weak_R":hmm[:,2],
              "initial_HEALTH_DISPLAY_SCORE_NO_EXPERT":100*hmm[:,0]+60*hmm[:,1]
            }
    rows={}
    for name in cfg["evaluated_models"]:
        rows[name]=measure(scores["test"][name],ends["test"],codes,events,part[-1],
                          scores["validation"][name],ends["validation"])
        rows[name]["validation_AP_descriptive"]=float(
            average_precision_score(y["validation"],scores["validation"][name]))
    result={
      "study":"Real ESA three-channel original-V15 split, causal nominal-referenced level+innovation+change, static/EMA/HMM observation ablation",
      "protocol_SHA256":hashlib.sha256(cfg_raw).hexdigest(),
      "real_ESA_source_SHA256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),
      "real_catalog_SHA256":hashlib.sha256(Path(catalog).read_bytes()).hexdigest(),
      "original_source_sample_count":int(len(codes)),
      "source_window_end_counts":{k:len(v) for k,v in ends.items()},
      "pure_nominal_train_windows":int(len(full_normal)),
      "nominal_reference_fit_windows":int(len(fit)),
      "official_event_ID_partition":audit["global_id_splits"],
      "true_target_anomalies_test":int(y["test"].sum()),
      "no_simulated_satellite_input_or_labels":True,
      "author_defined_tiers_NOT_physical_health_fault_truth":True,
      "no_human_health_score_anchors":True,
      "results":rows,
      "weak_state_HMM":{
        "training_normal_risk_q70":hmm_params["quantile70"],
        "training_normal_risk_q95":hmm_params["quantile95"],
        "transition":hmm_params["transition"].tolist(),
        "prior":hmm_params["prior"].tolist(),
        "consecutive_only_nominal_pairs":hmm_params["consecutive_nominal_transition_pairs"]
      },
      "elapsed_seconds":time.perf_counter()-t0,
      "critical_limitations":cfg["forbidden"],
      "interpretation":[
         "Combined weights PRE-fixed, comparison is exploratory because source ESA test IDs have been previously viewed",
         "AUC/AP are measured on real ESA target anomaly labels; weak H/D/R values remain author-defined",
         "Test alarms require 3 consecutive 40-min sampled steps, not 3 original 10-min raw samples",
         "Anomalies without amplitude deviation are not necessarily innovations; no guarantee of improved classification",
         "MSF max64 and two-level healthy/degraded classification from previous synthetic/full segment branches are NOT integrated here",
         "No operator-review updates; real expert health display score labels absent"
      ]}
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    frame=pd.DataFrame(testing_details)
    for name,values in scores["test"].items():
        frame["score_"+name]=values
    frame.to_csv(output/"real_timepoint_scores.csv",index=False)
    lines=[
      "# ESA真实遥测：名义正常参考条件下的幅值、预测创新、短期变化观测对照",
      "",
      "**全部遥测真实，真实监督外部标签为目标通道Anomaly；作者定义的H/D/R只是工程风险等级。ESA Mission1测试事件曾被观察，属于同源探索，不能称为新盲测。**",
      "",
      "| 方法 | Test AP↑ | AUROC↑ | 普通正常告警占比↓ | Rare正常告警占比↓ | 真实异常新告警事件/10 | 事件覆盖/10 |",
      "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for name,v in rows.items():
        a=v["alarm_occupancy"];e=v["official_ID_metrics"]
        lines.append(f"| {name} | {v['test_AP']:.5f} | {v['test_AUROC']:.4f} | "
              f"{a['normal_alert_rate']:.2%} | {a['rare_nominal_alert_rate']:.2%} | "
              f"{e['strict_new_onset_ID_recall']:.0%} | {e['coverage_ID_recall']:.0%} |")
    lines+=["","不同观测方法均采用验证通常正常窗口的99.8%分位阈值及连续3个评价窗口报警，没有针对测试事件改动规则。",
         "预测创新来自正常参考拟合的真实遥测因果窗口，风险组合权重固定0.35/0.45/0.20。",
         "只有当合并观测静态本身具有信息，而且HMM相比相同观测静态与EMA不损失真实事件能力时，才可主张HMM主线有实证收益。",
         "健康指数仍只是未由专家校准的初始显示策略；不能报告专家偏好或物理故障判定效果。",
         ""]
    (output/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    print("ESA_E2_REAL",json.dumps({
       "official_test_event_count":int(result["true_target_anomalies_test"]),
       "selected":{k:{"AP":v["test_AP"],
                     "normal_alert":v["alarm_occupancy"]["normal_alert_rate"],
                     "rare_alert":v["alarm_occupancy"]["rare_nominal_alert_rate"],
                     "new_onset":v["official_ID_metrics"]["strict_new_onset_ID_recall"]}
                     for k,v in rows.items()},
        "seconds":result["elapsed_seconds"]},ensure_ascii=False),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default=SOURCE)
    p.add_argument("--catalog",default=CATALOG)
    p.add_argument("--out",default="results/esa_observation_e2")
    a=p.parse_args()
    run(a.csv,a.catalog,a.out)
