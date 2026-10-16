# V1.3 ESA Mission1 单通道真实遥测探索实验

> 此实验使用公开 ESA-ADB v2 原始真实卫星遥测与专家异常事件标签；不是 ESA 官方完整基准复现，更不是物理故障诊断性能实验。两条分支独立研究，不进行融合。

## 数据来源及固定方案

- 数据：ESA Anomaly Dataset v2，DOI https://doi.org/10.5281/zenodo.15237121；原始 ESA-Mission1.zip；许可 CC BY 3.0 IGO。
- 数据通道：channel_41，在训练前由官方事件元数据确认有 51 个不同的真实 Anomaly 事件 ID，时间重叠会使连续事件段数变少。
- 获取方式：remotezip HTTP Range 仅读取两张标签表及 channel_41.zip（约 83 MB）；不依赖使用不受支持 ZIP 压缩方式 9 的 channels.csv。
- 固定时间粒度 10 分钟，使用以前最后观测到的遥测值；超过 60 分钟没有实测数据标为 gap (3)，不允许作为正常序列。
- Anomaly=1, Rare Event=2, Gap=3, Invalid=4, Nominal=0。仅真正异常为正例；rare nominal 事件作为负类并单独报告误报。
- 10 分钟研究采样可能遗漏短时异常，不等于 ESA 官方的 30 秒预处理，不能直接与官方排行榜比较。
- 此第一版通道不包含 telecommands，尚不能验证物理工况切换/故障的可靠区分。

## 固定独立评估

现有 V1.3 代码根据连续事件片段、而非官方事件 ID，按时间先划 train/validation/test，要求每段至少两个真实异常事件片段。窗口不得跨分区，含 gap 的窗口排除。正式论文需改成按官方 ID 去重、跨通道聚类的独立事件验证。

模型：robust level deviation、windowed Isolation Forest、Ridge forecasting residual。三者在训练区间纯正常窗口拟合，仅用验证 AP 选模、以验证集正常误报预算 2% 定阈值。测试只作评价，不参与决策。

指标：Point AP, ROC-AUC, true event recall, false alarm episodes per 1000 windows, rare nominal false positives, detection delay. No point adjustment.

这不属于 Telemanom-ESA 复现；官方 Docker 基线尚未实现。

## 执行命令

    pip install numpy pandas scipy scikit-learn remotezip
    python scripts/fetch_esa_remote_subset.py --channel channel_41 --resample 10min --out data/esa_lightweight
    python scripts/run_v13_anomaly.py --esa-preprocessed-csv data/esa_lightweight/channel_41.csv --channel channel_41 --lookback 16 --stride 4 --out results/v13/esa_mission1

GitHub Actions 文件 .github/workflows/esa_lightweight_research.yml 在联网 Ubuntu Runner 上运行，保存指标并尝试提交派生 CSV。失败不作性能声明。历史 V1.2/Proxy-V1 不修改。
