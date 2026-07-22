# NICE、ATM-S 与 Our project 匹配公平性实验设计（sub-08 / seed-42）

**日期：** 2026-07-22  
**状态：** 待用户审阅  
**仓库：** Human-Centered_AI_Group_Project  
**工作分支：** ckw（不创建额外实验分支）  
**正式交付单元：** sub-08 / seed-42

## 1. 目标与研究问题

本实验在 THINGS-EEG2 精确图片检索任务上比较 NICE、ATM-S 与 Our project，并回答：

1. 在标准 200 个 EEG 查询对 200 张图片的设置下，三种模型各自的标准独立 Top-1 和 Top-5 是多少？
2. Greedy、Hungarian、Stable Matching 与 Sinkhorn 是否只是在利用测试集的一对一结构？
3. 当 EEG 与图片不再严格一一对应时，这些匹配方法相对 Independent retrieval 是提升、无效还是退化？
4. 这种变化是否在三个 baseline 上一致，从而避免只对 Our project 使用有利后处理的不公平比较？
5. 当同一图片拥有两个来自真实、互不重叠 EEG trial 的查询时，严格一对一匹配是否出现结构性上限？

标准论文指标仍以 Independent retrieval 为主；其他算法作为传导式、批量级后处理与鲁棒性分析单独报告。

## 2. 固定范围与非目标

### 2.1 正式交付范围

- 模型：NICE、ATM-S、Our project。
- 被试：仅 sub-08。
- 训练与扰动随机种子：仅 seed-42。
- 标准扰动套件：27 个场景。
- 真实重复 EEG 套件：3 个场景。
- 匹配方法：Independent、Greedy、Hungarian、Stable Matching、Sinkhorn。
- 总规模：3 个模型 x 30 个场景 = 90 个模型-场景组合；每个组合输出 5 种匹配结果，共 450 组匹配输出。

### 2.2 明确不做

- 不执行 10 subjects x 5 seeds 全量实验。
- 不训练或评估 UBP、Hierarchical visual embeddings、EEGiT、Shallow Alignment、HCF 或 SAMGA。
- 不从单一 subject-seed 推断跨被试总体显著性或随机种子稳定性。
- 不根据正式测试集表现继续调整模型、checkpoint、Sinkhorn 参数或扰动样本。
- 不把 best-test checkpoint 指标作为正式主结果。
- 不复制相似度矩阵中的 EEG 行来伪造重复 EEG。

## 3. 论文原生配置与模型来源

### 3.1 NICE 与 ATM-S

NICE 与 ATM-S 必须来自 ATM 论文的[官方代码库](https://github.com/dongyangli-del/EEG_Image_decode)。实施时先审计官方 develop 分支，然后锁定明确的 commit hash；实验完成前不得跟随移动分支。以下内容以该锁定版本为准并写入运行清单：

- 模型结构与初始化；
- EEG 通道选择、时间采样率、裁剪窗口与归一化；
- 图片编码器、图片预处理与特征归一化；
- 训练损失、优化器、学习率、batch size、epoch 上限与 scheduler；
- 官方训练/验证划分方式；
- 官方精确图片检索定义。

如果本地 EEG_Recon-RL 的实现或默认值与锁定的官方实现不一致，不得静默用本地版本替代。必须以官方实现为准，或中止并列出无法复现的差异。

### 3.2 Our project

Our project 使用现有 sub-08 / seed-42 正式 checkpoint 与已审计的 BrainRW 评估配置。不得为了本次结果重新选择训练轮次或调参。运行清单必须记录 checkpoint 路径、SHA-256、CLIP 权重、EEG 通道、时间窗口和评估脚本版本。

### 3.3 “论文原生”的边界

“论文原生配置”表示保留官方模型、预处理、训练目标和核心超参数，不表示复制不严谨的测试集选模方式。唯一预先批准的协议修正是：正式主结果使用验证集选择 checkpoint，测试集只在配置冻结后评估。

## 4. Checkpoint 选择与测试集封存

### 4.1 主结果：validation-selected checkpoint

NICE 与 ATM-S 使用同一套官方验证划分及相同 checkpoint 规则：

1. 使用 seed-42 固定数据划分、初始化和数据顺序。
2. 训练期间保存满足官方评估间隔的 checkpoint。
3. 仅按最低 validation contrastive loss 选择 checkpoint；若 loss 完全相同，选择较早 epoch。验证集 Top-1/Top-5 只记录，不参与选模。
4. 选择完成后冻结 checkpoint、配置和哈希。
5. 每个模型只对正式测试集生成一次主结果相似度矩阵。

锁定的官方 develop 实现使用 `val_ratio=0.1`、validation contrastive loss 和
`early_stopping_patience=10`，实施原样使用并记录。若锁定 commit 的行为与上述规则
不一致，必须先提出书面差异，不能查看测试结果后临时改变选模规则。

### 4.2 核验结果：paper-style best-test

为解释与论文报告值的差异，可在一次训练结束并保存候选 checkpoint 后进行 best-test 核验。该输出必须：

- 标记为 best_test_audit_only；
- 与 val_selected_formal 分开存储和制表；
- 不进入 30 个匹配公平性场景；
- 不用于超参数、epoch、扰动或匹配算法选择；
- 不得在 README、报告或 presentation 中称为无偏正式测试结果。

best-test 核验可复用同一次训练保存的 checkpoint，不需要第二次训练。

### 4.3 原生一致性门槛

每个模型进入公平性实验前，必须通过标准 80-trial 平均、200 x 200 图库的 Independent parity gate：

- 统一矩阵重算的 Top-1/Top-5 计数与原生 evaluator 完全一致；
- 200 个 query ID、gallery ID 与 target canonical ID 顺序一致；
- 所有 ID 唯一且哈希被记录；
- 相似度为有限二维数值；
- target 通过 ID 映射，而不是假设 target 位于相同行列号。

任何一项失败，该模型不得进入汇总表，必须标记为 parity_failed。

## 5. 分数矩阵统一契约

每个正式 artifact 至少包含：

- similarity：形状 Q x G 的浮点相似度矩阵；
- query_ids：长度 Q 的唯一 EEG 查询 ID；
- gallery_entry_ids：长度 G 的唯一图库条目 ID；
- gallery_canonical_ids：图库条目对应的原始图片 ID，允许重复；
- target_canonical_ids：每个 EEG 查询的正确原始图片 ID；
- 模型、subject、seed、checkpoint role、checkpoint hash、配置 hash；
- split、trial split、图片顺序和代码 commit 来源。

模型只负责生成矩阵。同一模型和场景下，五种 decoder 必须读取完全相同的矩阵、ID 与 target relation。匹配算法绝不能读取 ground-truth label；label 只在匹配完成后用于计算指标。

## 6. 标准 27 场景套件

每个模型从原生 80-trial 平均的 200 x 200 正式矩阵出发，使用：

    drop_query in {0, 5, 10}
    drop_gallery in {0, 5, 10}
    drop_pair = 0
    duplicate_gallery in {0, 10, 20}

因此每个模型恰好有 3 x 3 x 1 x 3 = 27 个场景。

- drop_query：只移除 EEG，保留对应图片；这些图片成为没有配对 EEG 的真实干扰图片。
- drop_gallery：只移除图片，保留对应 EEG；这些 EEG 成为图库中没有正确答案的 unanswerable queries。
- drop_pair=0：不增加成对删除控制，避免扩大已确认范围。
- duplicate_gallery：复制真实图库条目和分数列，赋予新 entry ID，同时保留 canonical image ID；原图或复制图均算 canonical-correct。

所有删除和复制选择由 seed-42 的独立确定性 RNG stream 在一份 canonical master
manifest 上一次性生成，并将所选 canonical ID 写入场景 manifest。三个模型按 ID 应用
同一 manifest，不能依赖各自矩阵的行号另行抽样；三模型的基础 canonical query/gallery
集合不一致时直接中止。

## 7. 真实重复 EEG 的 3 场景套件

### 7.1 Session 内 10/10 分层拆分

测试集中每张图片有 80 个真实 trial，来自 4 个 session，每个 session 20 个 trial。不得简单使用前 40/后 40，因为这会造成 session 混杂。

对每个图片和每个 session：

1. 对每个 trial 计算
   SHA256("AIAA3800-DUPLICATE-EEG-v1\n" + "42\n" + canonical image ID +
   "\n" + session ID + "\n" + trial ID)，按摘要及 trial ID 稳定排序。
2. 前 10 个分配给 half A，后 10 个分配给 half B。
3. 四个 session 的 half A 共 40 个 trial，平均得到 EEG-A。
4. 四个 session 的 half B 共 40 个 trial，平均得到 EEG-B。

两条 EEG 来自真实、互不重叠且 session-balanced 的重复测量。trial ID、session ID、A/B 分配及 SHA-256 写入 split manifest。拆分在各模型原生通道和时间变换之前以同一原始 trial identity 完成，之后分别进入三种模型的原生预处理。

### 7.2 三个矩形场景

- dupq0：200 张图片各使用 EEG-A，得到 200 x 200。
- dupq10：在 200 个基础查询后追加 10 个所选图片的 EEG-B，得到 210 x 200。
- dupq20：追加 20 个所选图片的 EEG-B，得到 220 x 200。

对 200 个 canonical image ID 计算
SHA256("AIAA3800-DUPLICATE-QUERY-v1\n42\n" + canonical image ID)，按摘要及 ID
稳定排序并取前 20 个；dupq10 取其中前 10 个，因此严格是 dupq20 的子集。相同 ID 与
顺序用于三个模型。

该套件使用 40-trial 平均，绝对指标不得与标准 80-trial 论文分数直接比较；只能在同一模型、同一场景内比较匹配方法。

### 7.3 一对一算法的结构性上限

每个重复图片对应两条正确 EEG，但图库只有一个该图片条目。严格一对一 decoder 无法让两条查询同时匹配同一 gallery entry。按全部查询计算的理论最高 Top-1 为：

    dupq10: 200 / 210 = 95.24%
    dupq20: 200 / 220 = 90.91%

这是要检验的结构性质，不是模型失败。报告必须同时显示 unmatched 数量。

## 8. 五种匹配方法

### 8.1 Independent retrieval

每个查询独立选择最高相似度图片，允许不同查询选择同一条目。只有该方法报告标准 Top-5：按每行稳定降序排序，正确 canonical ID 在前 5 个条目中即为正确。

### 8.2 Greedy one-to-one

沿用现有实现：先按每个查询的独立 Top-1 分数从高到低处理；若首选已占用，选择该查询尚未占用的最高分图库。矩形场景允许 unmatched。

### 8.3 Hungarian

使用 linear_sum_assignment 的 maximize 模式最大化整批相似度总和。矩形矩阵只产生 min(Q,G) 个匹配，其他查询记为 unmatched。固定 seed-42 的行列置换用于稳定处理完全相等分数，并记录置换哈希。

### 8.4 Stable Matching

使用 query-proposing Gale-Shapley：查询按自身相似度降序提出，图库按对应相似度选择。相等分数按固定 query index 打破平局；矩形场景允许 unmatched。

### 8.5 Sinkhorn

沿用当前受测实现：

    temperature = 0.05
    max_iterations = 500
    tolerance = 1e-8

Sinkhorn 对矩形矩阵计算均匀行列边缘质量的 transport plan，再对每个查询执行 plan row argmax。它不是严格一对一算法，必须报告 strict_one_to_one=false。若未达到 tolerance，保留结果但标记 sinkhorn_converged=false，并在主汇总表中警告。

除 Independent 外，其他方法输出单一 assignment，不制造“assignment Top-5”；它们只报告 assignment Top-1。

## 9. 指标与报告口径

### 9.1 共同指标

每个模型、场景和 decoder 至少报告：

- overall Top-1 count / percent；
- answerable Top-1 count / percent；
- answerable 与 unanswerable query 数；
- assigned 与 unmatched query 数；
- unique gallery entry 与 unique canonical image 数；
- 相对 Independent 的 Top-1 差值；
- Independent Top-5 count / percent；
- Independent 到该 decoder 的四格转移：correct-to-correct、correct-to-wrong、wrong-to-correct、wrong-to-wrong。

总体 Top-1 将 unanswerable 和 unmatched 计为错误；answerable Top-1 用于区分图库缺失与 decoder 排序表现。

### 9.2 重复 EEG 附加指标

dupq0/10/20 额外报告：

- base EEG-A Top-1；
- appended EEG-B Top-1；
- 每个重复 canonical image 至少一条 EEG 正确的 coverage；
- 两条 EEG 均正确的 duplicated-pair both-correct；
- 严格一对一理论上限与实际值的差距；
- 重复 canonical image 导致的冲突与 unmatched 明细。

### 9.3 统计限制

正式交付只有一个 subject-seed 单元，因此：

- 报告精确计数、百分比和逐查询配对变化；
- 可以提供描述性图表；
- 不宣称跨被试、跨 seed 显著性；
- 不报告 10 subjects x 5 seeds 均值、标准差或双向 cluster bootstrap；
- 结论必须写“在 sub-08 / seed-42 上观察到”，不得写“普遍提高”或“显著优于”。

## 10. 产物与目录

实现应在仓库建立独立命名空间：

    experiments/matching_fairness/
      configs/
      scripts/
      tests/
      README.md
      README_ZH.md

大型 checkpoint、缓存、矩阵和逐查询结果不提交 Git；它们写入现有 test/brain-rw/results 下的新版本目录，并由 manifest 连接到仓库代码：

    results/matching_fairness_v3/
      manifests/
      matrices/{nice,atm_s,our_project}/subj08/seed42/
      runs/{nice,atm_s,our_project}/subj08/seed42/{standard,duplicate_eeg}/
      aggregate/
      logs/

最终交付至少包含 JSON summary、完整 CSV、trial split manifest、checkpoint/source/config/matrix SHA-256 清单、中英文报告、report/presentation 图表，以及区分 val_selected_formal 与 best_test_audit_only 的复现审计。

所有 .out、.err 和运行日志必须位于 logs/，不得散落在仓库根目录。

## 11. 自动验证与失败策略

自动测试至少覆盖：

1. 范围锁定，拒绝意外的全被试/多 seed 提交。
2. 三个模型 manifest 条目完整、来源可追溯。
3. 原生 evaluator 与统一 artifact 的 Independent Top-1/Top-5 parity。
4. query/gallery/target ID 数量、唯一性、顺序与 canonical 映射。
5. 27 个标准场景恰好生成且三模型共用相同扰动 ID。
6. 每个 session 的 A/B 均为 10/10、互不重叠、并集为 20 个 trial。
7. 每张图片的 EEG-A 与 EEG-B 均含 40 个真实 trial。
8. dupq10 重复 ID 是 dupq20 子集，矩阵形状为 200 x 200、210 x 200、220 x 200。
9. duplicate EEG 分数由真实 EEG-B 前向计算，禁止复制 EEG-A 分数行。
10. 五个 decoder 在方阵、矩形矩阵上的确定性、unmatched 与 tie-breaking。
11. duplicate gallery canonical correctness 与 duplicate EEG many-query-to-one correctness。
12. Sinkhorn finite、deterministic、marginal error 与 convergence 标记。
13. ground truth 不进入 decoder 调用路径。
14. JSON/CSV schema 稳定，相同输入重复运行产生相同内容哈希。

以下情况必须 fail closed：

- 官方 commit、模型权重或数据版本无法锁定；
- 官方原生预处理与本地数据无法对齐；
- 80 个 trial 的 session identity 无法可靠恢复；
- validation-selected checkpoint 无法与 best-test audit 隔离；
- 原生 evaluator parity 失败；
- 三模型没有复用同一 trial/扰动 manifest；
- 出现 NaN/Inf 或 ID/shape 不一致。

## 12. 执行顺序与时间预算

规格和实施计划获批后按以下顺序执行：

1. 审计并锁定官方 NICE/ATM-S commit、环境与原生配置。
2. 实现统一 artifact、checkpoint role 与 provenance contract。
3. 为三个模型生成标准 80-trial 200 x 200 矩阵并通过 parity gate。
4. 实现并测试 session-balanced 10/10 duplicate EEG exporter。
5. 运行 dupq0/10/20 正确性检查。
6. 运行每模型 27 个标准场景与 3 个 duplicate EEG 场景。
7. 汇总 450 组输出，生成中英文报告并检查表述限制。

总墙钟时间预计 6--12 小时；若官方环境适配、SLURM 排队或训练重试不顺，保守为 1--2 天。主要 GPU 成本是 NICE 与 ATM-S 各训练一次并导出 80-trial、EEG-A 和 EEG-B 表征；30 场景匹配后处理主要是 CPU 计算。

## 13. 验收标准

只有同时满足以下条件才可标记完成：

- 三模型均通过论文原生配置审计与标准 Independent parity gate；
- 主表只使用 validation-selected checkpoint；
- best-test 只出现在明确标注的 audit 部分；
- 27 个标准场景和 3 个真实重复 EEG 场景全部完成；
- 每场景包含相同五种 decoder，且跨模型共用同一扰动/trial manifest；
- 结果同时报告准确率、answerability、unmatched 与结构性上限；
- 中英文报告、CSV、JSON、manifest、哈希与日志齐全；
- 所有自动测试通过；
- 结论明确限定为 sub-08 / seed-42；
- 未启动或暗示 10 subjects x 5 seeds 全量交付。
