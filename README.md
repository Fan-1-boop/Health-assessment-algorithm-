# 卫星健康度算法 V1：双分支异常证据融合与概率健康映射

> **研究性质**：基于 NASA SMAP 真实卫星遥测的 **异常检测** 概念验证。数据集没有真实设备 `健康/退化/故障` 三状态、维修人员确认记录和专家 0–100 分，因此后三者均明确标记为 **模拟/代理标签**；**不得把这些指标宣传为真实故障识别性能**。

## 架构

```
SMAP P-1 telemetry (single target channel)
├── engineered causal-window features -> MSF-MOFS -> two logistic classifiers
│                                                   -> discriminative HMM filter -> base 3-state probabilities
└── raw telemetry causal windows -> supervised RF anomaly detector -> anomaly evidence
(base probabilities + anomaly evidence) -> validation-trained stacked logistic fusion
-> final 3-state PROXY probabilities -> default/user-calibrated health score

Simulated user scoring: actual window -> synthetic 0..100 label -> fit ordered mapping
Simulated repair feedback: record manual-confirmation field only; no claimed real repair confirmation
```

## 数据出处及限制

- Telemanom original [NASA SMAP/MSL dataset](https://github.com/khundman/telemanom) and [KDD 2018 paper](https://arxiv.org/abs/1802.04431).
- This repo includes SMAP `P-1` channel CSV **reformatted/converted by [ICC2023](https://github.com/aminmalekisadr/ICC2023)**: `P-1-train.csv`, `P-1-test.csv` and Telemanom's anomaly interval metadata `labeled_anomalies.csv`; original author/license [Telemanom LICENSE](https://github.com/khundman/telemanom/blob/master/LICENSE.txt). Please preserve original Caltech copyright acknowledgement and notice.
- CSV includes **only one anonymized telemetry target value** per timestep, not the original 24 command inputs and not spacecraft operational condition metadata. Physical drift versus mechanical degradation cannot be validated with these data.
- Annotated anomaly intervals are **not** physical failure labels. Artificial `degraded_proxy` is the **first 20%** of a real anomaly interval and `fault_proxy` is the **remaining 80%**. These names are only to exercise three-state software interfaces.
- `P-1` original separate `train.csv` is mainly treated as healthy-training data. `P-1-test.csv`, whose anomalies have public labels, is temporally partitioned into **[0,2600) supervised train**, **[2600,4000) validation**, and **[4000,end) untouched test**. This is *not* the original unsupervised Telemanom train/test protocol.
- Satellite anomaly intervals are only measured on 1 channel, 1 holdout interval: statistical conclusions do not generalize to the SMAP satellite or equipment fault diagnosis.

## 运行

```bash
python -m pip install -r requirements.txt
pytest -q
python scripts/run_v1.py --data-dir data/raw --output-dir results
```

Outputs include `results/metrics.json`, `results/RESULTS.md`, `results/heldout_predictions.csv` and `results/simulated_feedback.jsonl`. For automatic execution see `.github/workflows/experiment.yml`.

## 模型说明

- **MSF-MOFS**: implements single Fisher ranking; three absolute/relative top-K Fisher chromosomes; unmodified standard NSGA-II operators thereafter; binary G-mean and feature sparsity objectives. Wrapper adapted from paper 5-NN to logistic regression, and evolution budget reduced for CPU V1. `src/msf_mofs.py` labels this adaptation clearly.
- **Base status**: hierarchical logistic classifiers estimate `P(unhealthy)` and `P(fault_proxy | unhealthy)`; combined into three probabilities. A Markov transition model learned from train proxy states and a discriminative likelihood correction filter (posterior/train-prior) generate temporally filtered probabilities; it is a *discriminative HMM approximation*, not a fully identified generative HMM.
- **Anomaly**: random forest classifier on 64-sample raw lag vector, trained with early labeled test samples plus original healthy train samples. No online updates in V1. Threshold selected from **validation** only.
- **Fusion**: learned logistic stacking on **validation segment**, using base log-probabilities and anomaly logit, evaluated once on later test segment. Because inputs share the same telemetry no independence claim is made.
- **Health score**: initial mapping `[100,60,0]`; simulated user scoring labels from `[96,74,12]` with random noise; ordered bounded regression fits a new mapping from a small validation sample. Both true physical scores and actual human judgments are **unavailable**.
- **Maintenance**: only simulated record shape, no real technician-confirmed failures. A clinician/engineer can attach real confirmation later.

## Fair interpretation

Only the test **binary anomaly detection** numbers are evaluated against actual Telemanom ground-truth labels. All other reported state classification and score MAE are about synthetic labels. Report false positives/negatives, confusion matrix, Brier and MAE without paper-style inflated point-adjusted F1. No real normal-drift/fault separation experiment is possible with this dataset.

## Next research steps

1. Add multiple SMAP channels and grouped/channel-held-out evaluation with original command inputs; compare Telemanom, M2N2, AnDri and feature-based detectors.
2. Acquire independent operating-condition, maintenance and physically verified fault labels (or an experimentally injected/validated equivalent).
3. Study controlled incremental anomaly adaptation and old/new model evidence calibration; prohibit unconfirmed data from contaminating healthy baseline.
4. Replace proxy states by real health/degradation/fault labels, and calibrate health mapping using human-entered scores.

## Research references

- Fan et al., *MSF-MOFS: Budget-Efficient Multi-Scale Fisher Seeding for High-Dimensional Multi-Objective Feature Selection*, supplied manuscript.
- Hundman et al., [*Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*](https://arxiv.org/abs/1802.04431), KDD 2018.
- Kim et al., [*When Model Meets New Normals*](https://ojs.aaai.org/index.php/AAAI/article/view/29210), AAAI 2024.
- [AnDri code](https://github.com/mac-dsl/AnDri) (candidate for V2, **not** reproduced in V1).