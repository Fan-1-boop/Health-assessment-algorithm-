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

## V1.1：冻结代理标签与高维多尺度特征（新增）

标签规则在实现与测试前独立提交：[configs/proxy_label_rules_v1.json](configs/proxy_label_rules_v1.json)（冻结提交 `861c46bf12a6acd1002c5b88cf8119297d44c3ea`）。沿用 V1 代理标签定义：未标注异常区间为“健康代理”；异常区间的前 20% 为“退化代理”；剩余 80% 为“故障代理”。**这些标签不是维修确认的物理故障状态**。规则独立于观测幅值与模型结果；任何性能数字都是这套冻结规则的代理标签预测结果。

- `src/highdim_features.py`：在单通道原始 64 点因果窗口上，构造 **510 个候选特征**（多个长度和滞后位置的统计量/变化率/高阶矩/自相关，以及 FFT 分量等），并导出对应特征名称。**不是**卫星多通道特征。
- `src/msf_mofs.py`：训练数据上执行单次 Fisher 排序 + 3 个多尺度初始化染色体，使用基于逻辑回归分类性能和特征稀疏度的预算受限 NSGA-II 搜索。本项目版本是对论文方法的工程适配，**并非 5-NN 原文基准的严格复现**。
- `scripts/run_v11.py`：对比原 25 维特征、新 510 维全部特征、MSF-MOFS 选择特征各自的静态分层逻辑回归与时序滤波，并测试最终异常证据融合。异常分支暂时沿用 V1 原始时序随机森林，避免一次改变多个实验因素。
- `configs/proxy_label_rules_v1.json`：冻结规则与禁止将模拟标签解释为真实物理状态的限制；在 `results/v11/metrics.json` 中记录其 SHA-256。
- `results/v11/`：自动生成特征名称、选择索引、时间保留测试结果、三状态代理标签消融、真实异常区间指标与模拟评分反馈。

运行（需先按上文获取 data/raw/ 原始 CSV）：

```bash
pytest -q
python scripts/run_v11.py --generations 5 --population 12
```

查看 V1.1 结果：[results/v11/RESULTS.md](results/v11/RESULTS.md)，[results/v11/metrics.json](results/v11/metrics.json)。V1 与 V1.1 保留相同的 P-1 数据和时间段边界，方便对照；单通道、单测试异常事件下的结论仅为探索性结果。

完整逐时间点代理标签 CSV：[data/derived/P-1_proxy_labels_v1.csv](data/derived/P-1_proxy_labels_v1.csv)；可重复生成工具 `python scripts/export_proxy_labels.py`；详见 [docs/FROZEN_PROXY_LABELS.md](docs/FROZEN_PROXY_LABELS.md)。每一行明确标注 `physical_fault_confirmed=False`。
