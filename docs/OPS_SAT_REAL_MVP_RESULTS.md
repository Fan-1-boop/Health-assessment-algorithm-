# OPS-SAT-AD 真实卫星遥测异常检测 MVP：已执行的实验结果

**日期：2026-10-16 | 状态：完整成功运行，真实二分类异常检测展示可交付。**

该实验是针对**真实卫星遥测片段的正常/异常分类**，没有虚构退化或故障的物理三级真值。冻结的**完整模拟健康评估 MVP**位于 \`release/mvp-v1.0-synthetic-frozen\`，与本次实验相互独立。

## 数据来源

- 官方论文与数据项目：https://github.com/kplabs-pl/OPS-SAT-AD
- 采用原始 Zenodo 发布记录：https://doi.org/10.5281/zenodo.12588359
- 原始文件：\`dataset.csv\`（官方预计算统计特征，不是自行模拟或修改的标签）
- 数据 SHA-256：\`da20bc4bd3f30a3302ce06fe6a073bea6f036b297a92e83311d3f83b9c5e619b\`
- 数据规模：2123 个卫星遥测片段；18 个官方统计特征。
- 官方 \`train=1\`：1594（正常 1273 / 异常 321）；官方 \`train=0\`：529（正常 416 / 异常 113）。
- 样本划分：在官方训练片段中二次分出拟合片段 1195、独立验证片段 399；原始官方测试 529 个片段完全保留。阈值和最终模型均由验证片段确定，官方测试只作最终评价。
- 18 个特征来自官方已经完成的片段统计，**不能用于声称片段开始时的因果早期预警**。

## 实际采用的训练方案

在官方训练拟合片段执行一次现有仓库的 MSF-MOFS（Fisher 引导 + NSGA-II 的 logistic-wrapper 工程版本），并与全部 18 个特征比较。MSF 选择 3 个：

\`n_peaks\`、\`var_div_len\`、\`diff2_var\`。

对 2 个特征模式（MSF / 全特征）× 4 个固定分类器（RF / ExtraTrees / RBF-SVM / HistGradientBoosting）进行验证集 F1 选择，异常阈值也由验证集选择。不能利用官方测试数据选模型或反复调阈值。

### 验证集：所有模型按既定协议选模

| 特征 | 分类器 | 验证 F1 | 验证 AP |
|---|---|---:|---:|
| MSF（3 特征） | RF | 0.7843 | 0.7029 |
| MSF（3 特征） | ExtraTrees | 0.7755 | 0.6870 |
| MSF（3 特征） | RBF-SVM | 0.7383 | 0.7549 |
| MSF（3 特征） | HistGradientBoosting | 0.7871 | 0.8541 |
| 全部 18 特征 | RF | 0.9007 | 0.9564 |
| 全部 18 特征 | ExtraTrees | 0.9221 | 0.9600 |
| 全部 18 特征 | RBF-SVM | 0.8727 | 0.9182 |
| **全部 18 特征** | **HistGradientBoosting（选中）** | **0.9299** | **0.9635** |

**结论：当前 MSF-MOFS 对这份低维特征数据压缩过度，损失了分类性能。** 因而交付版本按验证集模型选择协议保留 18 个特征。MSF 仍作为真实数据上的消融结果，而不是声称它显著提升效果。

## 官方测试成绩（一次性最终评价）

官方测试片段 **529**；其中正常 **416**、异常 **113**。

| 指标 | 值 |
|---|---:|
| Accuracy | **0.9622** |
| Precision（异常类） | **0.9429** |
| Recall（异常类） | **0.8761** |
| F1（异常类） | **0.9083** |
| Average Precision (AP) | **0.9648** |
| ROC-AUC | **0.9850** |
| MCC | **0.8854** |

混淆矩阵（真实 × 预测，先正常后异常）：

\`\`\`text
                    预测正常    预测异常
真实正常                410           6
真实异常                 14          99
\`\`\`

选择的阈值 **0.4674859460**，来自官方训练段中的单独验证部分。

## 复现与交付

- 成功运行：https://github.com/Fan-1-boop/Health-assessment-algorithm-/actions/runs/38909254719
- 源码：\`real_mvp/run.py\` 和 \`real_mvp/predict.py\`
- 固定配置：\`real_mvp/PROTOCOL_FROZEN.json\`
- 真实卫星数据与出处：\`results/real_ops_sat/dataset.csv\`、\`source_manifest.json\`
- 全部实际指标：\`results/real_ops_sat/metrics.json\`
- 测试集逐片段预测：\`results/real_ops_sat/test_predictions.csv\`
- 保存模型：\`results/real_ops_sat/model_bundle.joblib\`
- 模型重载输出：\`results/real_ops_sat/reload_predictions.csv\`；GitHub Actions 已验证全部 529 个官方测试片段预测与首次执行一致。
- 自动生成的中文简报：\`results/real_ops_sat/RESULTS.md\`

运行：

\`\`\`bash
pip install -r requirements.txt
pip install requests joblib
python real_mvp/run.py
python real_mvp/predict.py --official-test-only
\`\`\`

## 严格的解释边界

1. 真实卫星数据，**真实的是正常/异常二分类标签**；没有确认的健康、退化、故障三状态、真实健康 0–100 分、维修记录，不允许宣传为真实故障状态 Macro-F1。
2. OPS-SAT 是预分割、带完整统计特征的遥测片段分类，不是在线逐秒因果异常事件检测，也没有证明真实告警延迟。
3. HMM / 动态贝叶斯时序状态推断和双分支融合只在已有**完全模拟数据 MVP**展示，不跨独立 OPS-SAT 片段伪造真实状态转移。
4. MSF-MOFS 包装交叉验证适用于实验优化，但不是独立泛化指标；实际优选了全特征模型。
5. 从此只允许人工上传、校验标签后进行**候选模型训练**；不得将无标签流式数据自动归类为新正常状态；人工批准通过后才可发布新模型，旧版本必须保留。

**交付建议：在一个真实卫星数据集上，我们已实现 F1≈0.91 的专家标注异常二分类实验，可以与模拟完整算法 MVP 组成双份明确标记来源的交付材料。**
