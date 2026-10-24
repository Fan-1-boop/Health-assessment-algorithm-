# Feature-budget decision (engineering fixed configuration)

**Scope:** OPS-SAT-AD real-satellite retrospective binary anomaly detection,
using 2,578 engineered features (2,560 from genuine segment telemetry and 18
official statistical features). This is NOT a universal rule for all feature
selection tasks or an established patent novelty claim.

## Agreed decision

- **Freeze the MSF-MOFS experiment's feature budget at MAXIMUM 64.**
- The previously executed exploratory study at budget 64 **selected 37 actual descriptors**, not exactly 64.
- Treat this as **“实验分析中满足当前研究要求的较好计算成本—分类性能折中配置”**, NOT “全局最优” / “经过统计验证的最优” / “最佳泛化结果”.
- Prior exploratory internal-audit results: 37/2578 descriptors, F1=0.8926, AP=0.9531, supervised classifier fit median about 4.9 ms, MSF search about 399 ms, in the same action run. Direct full-2578 reference: F1=0.8667, AP=0.9484, fit ~521 ms.
- **Explicit caveat:** the calibration-only rule favored the official 18-feature configuration; budget64's calibration F1 was 0.8167 versus all 2578's 0.8800. A budget chosen after reviewing an exploratory audit must not be represented as unbiased independent optimality confirmation.
- **Future work on this branch should not perform further budget tuning unless expressly requested.** Keep existing results untouched.

## Transfer to other modules

The 64 maximum was established for the **2,578-dimensional OPS-SAT anomaly-feature study only**. The frozen synthetic HMM+fusion MVP uses its own 400-dimensional feature system and pre-trained selectors (50 and 10 features; union 59). Do **not** retrofit the OPS-SAT budget and recompute the legacy synthetic metrics, or assert the 64 budget has been evaluated for synthetic state classification.

The next module research tracks effectiveness and computational cost of the two-stage calibrated state classifier, simple causal temporal averaging, causal HMM forward filtering, anomaly-evidence fusion and simulated health scores. This phase holds the trained synthetic artifact fixed to avoid test-driven model changes and makes the physical-label limitations explicit.

Frozen delivery release: \`release/delivery-mvp-v1.0-frozen\` — DO NOT MODIFY.
