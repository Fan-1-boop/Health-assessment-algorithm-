# V1.1 冻结规则与可重复的代理标签数据

冻结记录：[configs/proxy_label_rules_v1.json](../configs/proxy_label_rules_v1.json)；冻结提交 `861c46bf12a6acd1002c5b88cf8119297d44c3ea`。

生成完整的 8505 行时间点级标注 CSV：

```bash
python scripts/export_proxy_labels.py
```

输出为 `data/derived/P-1_proxy_labels_v1.csv`，每行包含：

- `timestep`：P-1 原始公开标注序列中的零基下标；
- `nasa_binary_anomaly_annotation`：唯一来自 NASA/Telemanom 的真实异常区间标注；
- `proxy_state_code` / `proxy_state_name`：依据提前冻结规则人工设计的三级**代理状态**；
- `published_anomaly_event_index`：公开异常事件编号（不在事件内为 -1）；
- `offset_within_event`：对应异常区间内部位置；
- `label_source`：固定 `deterministic_anomaly_interval_position`；
- `physical_fault_confirmed`：所有行都是 `False`；
- `frozen_rule_id` / `frozen_rule_sha256`：标签生成器使用的规则身份和字节哈希。

规则：正常区间代理类 0；公开异常事件前 `max(1,floor(interval_length*0.20))` 样本代理类 1，其余样本代理类 2。起止下标均为**包含端点**，来源为官方公开异常区间，**绝不使用遥测值、诊断预测分数和测试评价指标重新定义标签**。

标签表示的是人为预注册的“异常开始段与持续段”分割，不能视为退化和故障已由维修人员确认。评价三级代理标签不能等价声称故障诊断准确率；真实工况切换/设备老化问题仍无法使用这批数据验证。
