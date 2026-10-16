# 真实卫星异常检测 MVP：OPS-SAT-AD

这是**真实 ESA OPS-SAT 卫星遥测数据**上的最小可复现异常检测实验，针对 OPS-SAT-AD 官方 \`anomaly = 0/1\` 二分类标签。原模拟数据完整双分支 MVP 已单独冻结在 \`release/mvp-v1.0-synthetic-frozen\`，不受这里的实验影响。

## 数据与监督标签

官方来源：https://doi.org/10.5281/zenodo.12588359
官方代码：https://github.com/kplabs-pl/OPS-SAT-AD

使用官方 \`dataset.csv\`，2123 个真实遥测**片段**、18 个已生成的时序统计特征。官方文件中 \`train=1\` 为训练，\`train=0\` 为原始测试，共 1594/529 个片段。标签是由专家参与标注的正常/异常，**不是**确认的三级健康—退化—故障状态，也不包含真实 0–100 设备健康分。

- 18 个特征：mean, var, std, len, duration, len_weighted, gaps_squared, n_peaks, smooth10_n_peaks, smooth20_n_peaks, var_div_duration, var_div_len, diff_peaks, diff2_peaks, diff_var, diff2_var, kurtosis, skew。
- 不将 segment ID、channel ID、\`anomaly\` 或 \`train\` 当作预测输入。
- 使用原始人工/专家注释，没有注入人工故障或更改标签。真实数值标注以官方源文件为准。

## 一键运行

在仓库根目录执行：

\`\`\`bash
python -m pip install -r requirements.txt
python -m pip install joblib requests
python -m pytest -q real_mvp/test_real_mvp.py
python real_mvp/run.py --out results/real_ops_sat

# 以后只做预测，无需再次训练
python real_mvp/predict.py \
  --input results/real_ops_sat/dataset.csv \
  --model results/real_ops_sat/model_bundle.joblib \
  --official-test-only \
  --out results/real_ops_sat/reload_predictions.csv
\`\`\`

\`run.py\` 自动从官方 Zenodo 下载特征 CSV。固定训练集内部按分层随机种子 42 拆出 25% 验证段；MSF-MOFS 的搜索仅访问训练拟合段。固定比较 \`MSF\` 特征与全部 18 特征，四种分类器（RandomForest/ExtraTrees/RBF-SVM/HistGradientBoosting）。从内部验证 F1 选模型与阈值，在官方原始测试集上计算一次最终指标，并导出混淆矩阵、预测 CSV、模型和中文结果说明。

当前采用的是**真实卫星片段级回顾性分类**。18 个特征提取时可能使用一个片段的完整长度，因此不能据此宣称“异常刚开始就能实时检测”。样本是片段而非连续健康状态序列；不应在任意片段顺序上运行 HMM 并解释为真实设备的状态转移。

## 输出

\`results/real_ops_sat\` 中应生成：

- \`dataset.csv\`：未经改标签的官方真实卫星特征数据（来源校验见下）。
- \`source_manifest.json\`：官方数据来源、SHA-256、标签计数。
- \`metrics.json\`：固定候选的验证指标、MSF 所选特征、唯一验证选中模型的官方测试指标及实验限制。
- \`test_predictions.csv\`：官方测试片段的真实标签、异常概率、二分类结果、纯演示用 normality index。
- \`model_bundle.joblib\`：模型、预处理和阈值；**仅从可信来源加载 joblib 文件**。
- \`reload_predictions.csv\`：独立重载模型得到的预测，用于与初始测试结果一致性验证。
- \`RESULTS.md\`：中文实验总结。

## 更新原则：仅人工监督、受控更新

模型在使用期间**不进行自动增量学习或伪标签自训练**。后续新增训练集必须由人工提供可信异常标签、记录来源，经过质量检查和验证后，人工批准模型版本替换；数据上传本身不会触发正式模型的自动更新。

## 范围边界

此实验可以展示真实卫星遥测的**异常二分类**与 MSF-MOFS 特征选择，但**不能**展示真实三状态物理健康诊断。双分支 HMM—异常证据融合、模拟用户打分和 0–100 健康分在冻结的模拟 MVP 中展示，不把真实异常概率冒充物理健康分。
