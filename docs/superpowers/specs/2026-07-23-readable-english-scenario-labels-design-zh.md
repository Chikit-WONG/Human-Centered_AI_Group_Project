# 英文实验场景自然语言命名设计

## 目标

让匹配公平性实验的英文报告与中文报告保持一致，直接展示读者能够理解的实验场景，
不再要求读者解释 `dropq5_dropg10_dropp0_dupg20` 或 `dupq10` 等内部标识符。

本规格仅覆盖英文展示，并取代
`2026-07-23-readable-chinese-scenario-labels-design-zh.md` 中“英文报告保持不变”的旧范围约束。

## 范围

- 修改英文结果报告中标准场景表和真实重复 EEG 表的 `Scenario` 列。
- 标准套件覆盖既有 27 个组合：删除 EEG 数量 `{0, 5, 10}`、删除图片数量
  `{0, 5, 10}`、重复图片数量 `{0, 10, 20}`。
- 真实重复 EEG 套件覆盖既有 3 个场景：基础 200×200、加入 10 条真实重复 EEG
  的 210×200、加入 20 条真实重复 EEG 的 220×200。
- 中文报告、实验矩阵、原始 CSV、scenario manifest、`summary.json` 和 per-query
  ledger 保持不变。

## 命名规则

标准套件按实际操作拼接自然语言：

- 三个扰动数均为 0：
  `Baseline one-to-one matching (200 EEG queries × 200 images)`。
- 删除 EEG 非零：加入 `Remove N EEG queries`。
- 删除图片非零：加入 `remove N images`。
- 重复图片非零：加入 `duplicate N images`。
- 多个操作同时存在时用英文逗号连接。
- 名称末尾写入实际矩阵尺寸，例如
  `(195 EEG queries × 210 images)`。

真实重复 EEG 套件使用：

- `Real duplicate-EEG baseline (200 EEG-A queries × 200 images)`。
- `Add 10 real duplicate EEG-B queries (210 EEG queries × 200 images)`。
- `Add 20 real duplicate EEG-B queries (220 EEG queries × 200 images)`。

## 实现方式

- 在报告生成模块中增加只负责英文展示的确定性场景标签函数。
- 英文标准表和英文真实重复 EEG 表调用英文标签函数；中文渲染继续调用现有中文标签函数。
- 不建立新的通用国际化框架，也不修改场景数据模型。
- 使用现有 450 条聚合记录重新渲染英文结果；不重新计算实验指标，也不重新训练。
- 原始 CSV 中继续保留 `scenario_index` 和 `scenario` 字段，保证机器可读性与可复现性。

## 错误处理

- 标准场景的矩阵尺寸继续由 `ScenarioSpec` 参数确定，避免手写尺寸与实验配置不一致。
- 真实重复 EEG 标签仅接受场景索引 `27`、`28` 和 `29`；其他索引继续抛出
  `ValueError`，避免静默生成错误标签。

## 验证

- 先增加失败测试，证明当前英文报告仍含内部 slug。
- 单元测试覆盖 27 个标准场景和 3 个真实重复 EEG 场景。
- 测试英文名称中的操作数量及矩阵尺寸与场景参数一致。
- 测试英文报告不再出现 `dropq`、`dropg`、`dropp`、`dupg` 或 `dupq`。
- 测试中文报告的既有自然语言标签保持不变。
- 重渲染前后比较 SHA-256，确认 `aggregate_metrics.csv` 和 `RESULTS_ZH.md` 均未改变。
- 运行完整 matching-fairness 测试套件和 Ruff。

## 交付与版本控制

- 只重渲染 `matching_fairness_v3/aggregate/RESULTS.md`。
- 在当前 `ckw` 分支提交并推送，不创建新分支。

## 非目标

- 不更改任何实验结果、checkpoint、匹配算法或评价指标。
- 不修改原始产物中的内部标识符。
- 不新增场景、不补跑实验，也不开发通用国际化框架。
