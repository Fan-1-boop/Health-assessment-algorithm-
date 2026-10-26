#!/usr/bin/env python3
"""Raw-label semantics audit only. REAL OPS-SAT, no model fitting.

Use only official train=1 for investigating whether fine-grained time/labels
support HMM state-transition experiments. All original official test=0
observations are excluded from analysis. Keep risk/physical-fault semantics separate.
"""
import hashlib,io,json
from pathlib import Path
import numpy as np,pandas as pd,requests
OUT=Path("results/real_health_main"); OUT.mkdir(parents=True,exist_ok=True)
URL="https://zenodo.org/api/records/12588359/files/segments.csv/content"
r=requests.get(URL,timeout=130);r.raise_for_status()
raw_bytes=r.content
df=pd.read_csv(io.BytesIO(raw_bytes),parse_dates=["timestamp"])
df=df.loc[df["train"]==1].copy()
g=df.groupby("segment",sort=False)
def hist(col):
 x=df[col].astype(str).value_counts(dropna=False)
 return {str(k):int(v) for k,v in x.items()}
byseg=g.agg(channel=("channel","first"),n=("value","size"),
  raw_labels=("label","nunique"),anomalies=("anomaly","nunique"),
  first=("timestamp","min"),last=("timestamp","max"))
all_ids=g.agg(one_raw_label=("label","first"),anomaly=("anomaly","first"))
first_label=pd.crosstab(all_ids["one_raw_label"].astype(str),all_ids["anomaly"].astype(str))
within=byseg["raw_labels"].value_counts().to_dict()
time_good=0; time_bad=0; repeated_times=0
for _,v in g:
 t=v["timestamp"]
 if t.is_monotonic_increasing:time_good+=1
 else:time_bad+=1
 repeated_times+=int(t.duplicated().any())
report={
 "source":"OPS-SAT-AD official original raw segments.csv, train==1 only",
 "source_url":URL,
 "raw_source_sha256":hashlib.sha256(raw_bytes).hexdigest(),
 "train_only_rows":int(len(df)),
 "train_only_segments":int(len(byseg)),
 "unique_channels":int(df.channel.nunique()),
 "raw_point_label_hist":hist("label"),
 "official_binary_segment_label_hist":hist("anomaly"),
 "raw_label_nunique_per_segment":{str(k):int(v) for k,v in within.items()},
 "point_annotation_changed_within_segment":int((byseg.raw_labels>1).sum()),
 "label_by_official_segment_anomaly":first_label.to_dict(),
 "segment_timestamp_monotonic":int(time_good),
 "segment_timestamp_nonmonotonic":int(time_bad),
 "segment_timestamp_duplicate_count":int(repeated_times),
 "segments_length_quantiles":{str(k):float(v) for k,v in byseg["n"].quantile([0,.1,.25,.5,.75,.9,1]).items()},
 "physical_health_degradation_fault_triple_label_present":False,
 "human_health_score_groundtruth_present":False,
 "strong_warning":"Within-segment constant groundtruth does NOT validate HMM transition between physical states or early event onset. Never sequence arbitrary segment IDs for HMM."
}
(OUT/"raw_label_feasibility.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(report,ensure_ascii=False,indent=2))
