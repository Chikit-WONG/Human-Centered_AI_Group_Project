# 基于脑–视觉联合对齐的 EEG 图像检索

[English](README.md) | 简体中文

**AIAA3800 — 以人为中心的人工智能（Human-Centered Artificial Intelligence）**课程项目。

本仓库研究能否将非侵入式 EEG 记录映射到视觉语义嵌入空间，并用它检索受试者观看过的图像。原始正式协议覆盖 THINGS-EEG2 的全部十名受试者（`sub-01` 至 `sub-10`）和五个随机种子（`42` 至 `46`），并为每个“受试者–随机种子”组合分别训练脑编码器与经过 LoRA 适配的 CLIP 视觉编码器。一项受控 SAMGA 实验在其他条件完全一致的冻结 CLIP SAMGA 对照上，单独检验同一套视觉 LoRA/TTUR 改进的效果。第三条独立实验线使用 SAMGA 发布的训练代码和固定的推断 InternViT V2.5 特征，执行经审计的尽力复现。全局一对一匈牙利解码器仅保留为随机种子 `42`、`sub-08` 的传导式消融实验。

> **范围说明。** 原始主结果覆盖完整的“10 名受试者 × 5 个随机种子”网格：随机种子为 `42`、`43`、`44`、`45`、`46`，共 50 个独立训练的“受试者–随机种子”模型。受控 SAMGA 扩展在同一网格上另外训练 50 个匹配的 Frozen 模型和 50 个匹配的 LoRA 模型。公共代码复现另行评估发布启动器唯一给出的 seed `2025`，并在项目自定的 `42`–`46` 上运行固定 60 轮稳定性网格；后者不声称是论文未公开的五个 seed。由于论文没有公开精确视觉 checkpoint、提取器、五个 seed 值或检查点选择规则，这些结果保持分开。匈牙利结果仅涵盖随机种子 `42` / `sub-08`，不计入任何标准汇总。

> **复现指南。** 固定资产、可执行命令、完整协议细节和声明边界见
> [SAMGA 公共代码复现中文指南](experiments/samga_reproduction/README_ZH.md)。

## 项目亮点

- 将试次平均后的后部脑区 EEG 信号映射到 512 维 CLIP 图像空间。
- 联合训练脑 MLP 与 CLIP ViT-B/32 上秩为 32 的 LoRA 适配器。
- 对脑分支和视觉分支采用不同的学习率（TTUR 风格优化）。
- 为每个“受试者–随机种子”运行报告固定的最终检查点结果，而不是选择测试集表现最好的 epoch。
- 对全部 50 个“受试者–随机种子”模型分别训练和评估，再报告逐随机种子十人汇总及五随机种子标准 Top-1/Top-5 汇总。
- 运行预注册的 50 对 SAMGA Frozen-versus-LoRA 归因实验，逐对匹配任务模型初始化，并只用密封的概念不重叠 pilot 验证集选择配置。
- 审计 SAMGA 发布源码、固定一个推断的 InternViT V2.5 revision，并以不同标签报告固定/最终轮与测试集选模指标。
- 为每个标准运行提供单元测试、独立检查点重载验证和逐查询预测，并为随机种子 `42` / `sub-08` 的匈牙利消融保存相似度矩阵来源记录。

## 方法

![EEG 图像检索模型结构：训练阶段进行双侧 CLIP 对齐，推理阶段进行 Top-1/Top-5 图像检索](asserts/Architecture.png)

对于 EEG 查询嵌入 $b_i$ 和图库图像嵌入 $v_j$，两者均经过 L2 归一化，检索分数为

```math
S_{ij} = b_i^\top v_j.
```

标准检索对每一行独立排序：

```math
\hat{j}_i = \underset{j}{\arg\,\max}\; S_{ij}.
```

可选的匈牙利解码器则求解一个全局双射：

```math
\hat{\pi} = \underset{\pi \in \mathrm{Perm}(N)}{\arg\,\max}
\sum_{i=1}^{N} S_{i,\pi(i)}.
```

第二种协议优化的是完整的一对一匹配，因此可能为某个查询分配一个并非该查询行内最大值的图像。

## 已验证结果

### 五随机种子、十名受试者的标准独立检索

正式实验覆盖随机种子 `42`、`43`、`44`、`45`、`46`。全部 50 个“受试者–随机种子”组合均独立训练模型，使用第 25 个 epoch 后保存的固定检查点，并在 200 个留出 EEG 查询和 200 张唯一图库图像上评估。每次运行均通过严格产物验证及独立检查点重载重复检查，指标与逐查询预测完全一致。

主结果为：

- Top-1：**86.66% ± 0.69 个百分点**；
- Top-5：**98.38% ± 0.14 个百分点**；
- 合并计数：Top-1 **8666/10000**、Top-5 **9838/10000**。

± 项是五个“单 seed 十名受试者宏平均准确率”之间的样本标准差（`ddof=1`）。由于每个“受试者–随机种子”运行均包含 200 个查询，五随机种子均值等于合并准确率。10,000 次查询评估会在五个随机种子间重复每名受试者的同一组留出刺激，因此并非 10,000 个独立测试样本。包含 200 张候选图像时，随机基线为 Top-1 0.5%、Top-5 2.5%。

#### 各随机种子的十名受试者平均值

| 随机种子 | Top-1 | Top-5 | Top-1 正确数 | Top-5 正确数 |
|---:|---:|---:|---:|---:|
| 42 | 87.35% | 98.30% | 1747/2000 | 1966/2000 |
| 43 | 85.60% | 98.35% | 1712/2000 | 1967/2000 |
| 44 | 87.10% | 98.20% | 1742/2000 | 1964/2000 |
| 45 | 86.40% | 98.55% | 1728/2000 | 1971/2000 |
| 46 | 86.85% | 98.50% | 1737/2000 | 1970/2000 |

#### 各受试者的五随机种子结果

对每名受试者，± 项是该受试者五个随机种子分数之间的样本标准差（`ddof=1`），单位为百分点。

| 受试者 | Top-1 五随机种子均值 ± 样本标准差 | 范围 | Top-5 五随机种子均值 ± 样本标准差 | 范围 |
|---|---:|---:|---:|---:|
| sub-01 | 84.0% ± 1.84 | 81.0–86.0% | 96.6% ± 0.42 | 96.0–97.0% |
| sub-02 | 89.7% ± 0.67 | 89.0–90.5% | 99.7% ± 0.27 | 99.5–100.0% |
| sub-03 | 85.2% ± 1.25 | 83.5–87.0% | 97.4% ± 0.42 | 97.0–98.0% |
| sub-04 | 83.4% ± 1.43 | 81.5–85.5% | 96.5% ± 0.35 | 96.0–97.0% |
| sub-05 | 84.0% ± 1.27 | 82.0–85.5% | 98.3% ± 1.10 | 96.5–99.0% |
| sub-06 | 92.8% ± 1.15 | 91.5–94.0% | 99.8% ± 0.27 | 99.5–100.0% |
| sub-07 | 84.8% ± 2.25 | 82.0–87.5% | 98.0% ± 0.35 | 97.5–98.5% |
| sub-08 | 90.9% ± 1.24 | 89.0–92.0% | 99.9% ± 0.22 | 99.5–100.0% |
| sub-09 | 80.3% ± 2.02 | 77.0–82.5% | 97.7% ± 0.67 | 96.5–98.0% |
| sub-10 | 91.5% ± 1.17 | 90.5–93.5% | 99.9% ± 0.22 | 99.5–100.0% |

元数据说明：复用的旧版随机种子 `42` / `sub-08` 指标没有记录 Conda 环境名和 SciPy 版本；其中已记录的 Python、PyTorch、Transformers、Datasets、PEFT、CUDA 设备和 dtype 均与其余运行一致。独立的数据来源限制见下文。

### 受控 SAMGA + 视觉 LoRA/TTUR 扩展实验

为了回答“我们的方法放到 SAMGA 上是否仍然有效”，我们另外运行了一项受控归因实验。两个实验臂使用相同的 CLIP ViT-B/32 视觉骨干、SAMGA EEGProject 编码器、被试感知五层路由器、投影器、两阶段 MMD/对比学习目标、17 个后部通道、试次平均、训练样本、批次顺序和随机种子。唯一有意改变的可训练因素是“冻结视觉特征”与“在 q/k/v/out 及 MLP 投影上使用秩 32 视觉 LoRA”。每一对 Frozen/LoRA 运行都记录了完全相同的任务模型初始化哈希。Frozen 实验臂使用共享 float16 特征缓存；缓存/在线特征一致性检验的最小向量余弦为 0.999939，而且零 LoRA 时的检索指标完全一致。

Pilot 只使用受试者 01、05、08 在随机种子 42、43 下的概念不重叠验证划分，并锁定视觉/任务学习率比例 `0.20` 和统一停止 epoch `25`。之后在全部训练概念上重新训练十名受试者，最后只评估一次测试图库。训练遵循 SAMGA 发布代码的归一化约定：对比损失中只对图像特征做 L2 归一化、不对 EEG 特征归一化；评估仍使用余弦相似度。

| 受控实验臂 | Top-1 | Top-5 |
|---|---:|---:|
| 冻结 CLIP 的 SAMGA | 76.17% ± 0.30 | 95.79% ± 0.32 |
| SAMGA + 视觉 LoRA/TTUR | **82.68% ± 0.36** | **97.67% ± 0.17** |
| 配对变化 | **+6.51 个百分点**，95% CI **[+5.22, +7.69]** | **+1.88 个百分点**，95% CI **[+1.30, +2.50]** |

± 项仍是五个“单 seed 十人宏平均”之间的样本标准差。置信区间使用预注册的受试者/随机种子双向 cluster bootstrap，共 10,000 次重采样。Top-1 在 50 对中有 48 对提升，Top-5 有 47 对提升，而且十名受试者的 Top-1 平均变化全部为正。预注册标准——Top-1 至少提升 0.5 个百分点、Top-1 区间下界大于零、Top-5 平均下降不超过 0.2 个百分点——**通过**。该实验未使用匈牙利解码。

这说明在受控 CLIP 骨干下，视觉 LoRA/TTUR 改进可以迁移到 SAMGA 风格的训练框架。但它**不是**对 SAMGA 论文所报告 91.3%/98.8% 的精确复现：论文结果使用未披露精确来源的预计算 InternViT 特征，而这里为了识别改进本身的作用，刻意固定 CLIP ViT-B/32。该受控扩展的 Top-1 82.68% 也不会取代本仓库原始的 86.66% 主结果。

### 经审计的 SAMGA 公共代码复现尝试

我们将 SAMGA 官方源码树保持在干净提交 `1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1`，并固定推断的 `OpenGVLab/InternViT-6B-448px-V2_5` revision `9d1a4344077479c93d42584b6941c64d795d508d`。明确假设的视觉表示取实际第 20/24/28/32/36 个 block 输出、排除 CLS 后对 patch token 求均值，并且不额外执行逐向量归一化。该配置先在 Subject 08 筛选，再在 Subject 01/05 检查，因此属于经审计的近似复现，不是前瞻锁定或论文精确复现。

| Seed-2025 公共代码诊断 | Top-1 | Top-5 | Top-1 差距 | Top-5 差距 |
|---|---:|---:|---:|---:|
| 禁用早停、固定第 60 轮 | **89.55%** | **98.65%** | -1.75 个百分点 | -0.15 个百分点 |
| 逐轮查看测试集后选择 | **91.95%** | **98.95%** | +0.65 个百分点 | +0.15 个百分点 |

与发布启动器兼容的 seed-2025 对照使用 batch 512 和公共代码默认的 patience 10：

| Seed-2025 patience-10 诊断 | Top-1 | Top-5 | Top-1 差距 | Top-5 差距 |
|---|---:|---:|---:|---:|
| 实际停止/最终轮 | **88.95%** | **98.90%** | -2.35 个百分点 | +0.10 个百分点 |
| 逐轮查看测试集后选择 | **91.50%** | **98.75%** | +0.20 个百分点 | -0.05 个百分点 |

十名受试者的实际停止轮范围为 30–60，平均为 37.4 ± 9.55。由于停止规则本身监控正式测试集 Top-1，实际停止/最终轮这一行也受测试集条件影响。

项目自定随机种子 `42`–`46` 提供另一组固定 60 轮稳定性检查：

| 项目自定五随机种子协议 | Top-1 | Top-5 | Top-1 差距 | Top-5 差距 |
|---|---:|---:|---:|---:|
| 禁用早停、固定第 60 轮 | **89.02% ± 0.36** | **98.87% ± 0.06** | -2.28 个百分点 | +0.07 个百分点 |
| 逐轮查看测试集后选择 | **91.82% ± 0.20** | **98.87% ± 0.16** | +0.52 个百分点 | +0.07 个百分点 |

五随机种子行先在每个 seed 内对十名受试者取宏平均，± 是五个 seed-level 均值之间的样本 SD。选模行每轮直接查看正式测试集；Top-5 是按 Top-1 选中轮次的伴随值。公共代码的 patience-10 早停同样监控正式测试集 Top-1。这些是存在测试泄漏的诊断值，不是无泄漏项目指标。所有行均使用标准的独立 200-way 余弦检索，从未使用匈牙利分配。

[SAMGA 公共代码复现指南](experiments/samga_reproduction/README_ZH.md)记录源码/模型/数据哈希、公开材料冲突、特征提取假设、命令和局限。详细 CSV/JSON 报告保留在本地 `results/samga_reproduction`；该目录通过 Git 有意忽略。

### 与既有 EEG 图像检索工作的比较

下表采用文献中能够确认的最接近协议：在 **THINGS-EEG2 上进行被试内、200-way、零样本检索**，并对全部十名受试者取平均。每个 EEG 查询都独立地在 200 张留出刺激图像中排序，因此不纳入匈牙利结果。数值取各论文主要比较表中的主结果；每个指标列的最高值用粗体标出，本项目结果另外使用蓝色突出显示。

| 方法 | 发表状态 | Top-1 | Top-5 | 协议说明 |
|---|---|---:|---:|---|
| [NICE（Li 等人的精确图像复现）](https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf) | NeurIPS 2024，Table 8 | 21.52% | 51.57% | 依据验证集选择检查点；使用真实刺激图图库 |
| [ATM-S](https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf) | NeurIPS 2024，Table 8 | 26.13% | 55.32% | 正式会议版本结果；使用 63 个 EEG 通道 |
| [UBP](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Bridging_the_Vision-Brain_Gap_with_an_Uncertainty-Aware_Blur_Prior_CVPR_2025_paper.html) | CVPR 2025 | 50.90% | 79.70% | 17 通道；试次平均；图库使用模糊后的图像表征 |
| [分层视觉嵌入](https://openreview.net/forum?id=IEq71qS8B7) | ICLR 2026 | 75.70% | 94.60% | 17 通道；融合 RN50、CLIP-B/32 与 VAE |
| [EEGiT](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_EEGiT_Teaching_Vision_Transformers_to_Understand_the_EEG_signal_CVPR_2026_paper.html) | CVPR 2026 | 70.40% | 95.10% | 将预训练 ViT 迁移为 EEG 编码器 |
| [Shallow Alignment](https://arxiv.org/abs/2601.21948) | arXiv 2026 预印本 | 82.60% | 97.70% | 五个随机种子的平均；选择最佳中间层视觉特征 |
| [HCF](https://arxiv.org/abs/2603.07077) | arXiv 2026 预印本 | 84.60% | 98.20% | 分层融合中间层视觉特征 |
| [SAMGA](https://arxiv.org/abs/2604.17782) | arXiv 2026 预印本 | **91.30%** | **98.80%** | 五个随机种子的平均；训练 60 个 epoch 并早停 |
| 我们的项目（标准检索） | 课程项目，固定协议 | $\color{blue}{\mathbf{86.66\%}}$ | $\color{blue}{\mathbf{98.38\%}}$ | 五随机种子均值；17 通道；固定第 25 个 epoch 检查点 |

虽然我们的项目没有在这组比较中取得第一名，但 **Top-1 和 Top-5 均排名第二仍然是一项很强的结果**：Top-1 $\color{blue}{\mathbf{86.66\%}}$、Top-5 $\color{blue}{\mathbf{98.38\%}}$。我们的项目超过表中所有经过同行评审的文献结果；唯一更高的是目前尚未经过同行评审的 SAMGA 预印本。这**不是**不加限定的 SOTA 声明：我们的五随机种子均值现在与 Shallow Alignment 和 SAMGA 报告所用的随机种子数量更接近，但各论文在视觉目标、预训练编码器、训练日程和检查点选择规则上仍有差异。HCF 和 Shallow Alignment 同样是预印本。分层视觉嵌入论文的主表报告 Top-5 94.60%，但其列出的十个逐被试 Top-5 数值的算术平均约为 94.91%。

上文复现不另加为排名行，因为它是对 SAMGA 本身的再次评估，并非一个新方法。

以下两项处理用于避免混淆不同协议：

- 原始 NICE-GA 论文的 EEG Top-1 15.6% 和 Top-5 42.8% **未纳入**。该结果使用每个概念的其他图像构建类别模板，评估 200-way 类别模板识别，而非找回精确的刺激图像。上表的 NICE 行是 Li 等人在 ATM 检索协议下完成的后续精确图像复现。
- NeurIPS 正式版 ATM 论文在正式 Table 8 中报告 26.13%/55.32%。网上常见的 28.64%/58.47% 来自采用不同统计规则的 arXiv/消融结果，因此未将其混入上表。

该测试划分中每个概念恰好只有一张刺激图像，所以评分时概念身份与图像身份一一对应；但图库表征方式仍会影响任务。本项目直接对真实测试图的嵌入进行排序，不使用类别模板。以上文献数值均为论文报告值，并未在本仓库中逐一重跑。

### 随机种子 `42`、`sub-08` 匈牙利一对一消融实验

![EEG 图像检索中的匈牙利一对一分配实现流程](asserts/Hungarian_Algorithm.png)

| 评估协议（仅随机种子 `42`、`sub-08`） | Top-1 / 分配准确率 | Top-5 |
|---|---:|---:|
| 标准逐查询独立检索 | **182/200 (91.0%)** | **199/200 (99.5%)** |
| 全局匈牙利一对一分配 | **200/200 (100.0%)** | 不适用 |

### 如何理解随机种子 `42` / `sub-08` 的匈牙利算法结果

匈牙利算法结果是一项**传导式闭集消融实验**，不能替代标准 Top-1：

- 它会联合观察完整的测试查询批次；
- 它假设 200 个查询与 200 张图库图像构成一个已知的双射；
- 每张图库图像都必须且只能使用一次；
- 一次全局分配只为每个查询返回一张图像，因此不存在可直接比较的 Top-5。

在随机种子 `42` / `sub-08` 的运行中，独立 Top-1 预测只覆盖了 183 张不同的图库图像。匈牙利解码改变了 18 个分配，将全部 18 个标准 Top-1 错误转换为正确匹配，同时没有把任何原本正确的匹配改错。预先声明的九种行/列排序产生了相同的映射分配，从而排除了依靠对齐顺序打破平局而得到 100% 结果的解释。

因此，推荐采用以下报告方式：

- **主要结果：**五随机种子标准 Top-1 **86.66% ± 0.69 个百分点**、Top-5 **98.38% ± 0.14 个百分点**，并同时报告上文的逐随机种子表、逐被试表和合并计数。
- **次要消融结果：**随机种子 `42` / `sub-08` 的全局一对一分配准确率 100.0%，并与该次运行的标准 Top-1 91.0%、Top-5 99.5% 对照。

任何十名受试者、逐随机种子或五随机种子汇总分数均未使用匈牙利分配。

## 原始项目实验配置

| 组件 | 设置 |
|---|---|
| 数据集 | THINGS-EEG2 |
| 已验证受试者 / 随机种子 | `sub-01`–`sub-10` / `42, 43, 44, 45, 46` |
| 独立训练运行数 | 10 名受试者 × 5 个随机种子 = 50 个“受试者–随机种子”模型 |
| 每名受试者加载后的训练 EEG 张量 | `(16540, 4, 63, 250)` |
| 每名受试者加载后的测试 EEG 张量 | `(200, 80, 63, 250)` |
| 试次处理 | 分别对 4 个训练试次和 80 个测试试次取平均 |
| EEG 通道 | `P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2` |
| 时间窗口 | `[0, 250)` 个采样点 |
| 脑编码器 | 带残差投影块的 MLP |
| 视觉编码器 | CLIP ViT-B/32 |
| 视觉适配 | LoRA 秩 32，全部线性层 |
| 嵌入维度 | 512 |
| 脑分支 / 视觉分支学习率 | `5e-4` / `5e-5` |
| 调度器 / 权重衰减 | 余弦 / `0.05` |
| 训练 / 评估批次大小 | 512 / 100 |
| 训练 | 25 个 epoch、bf16、梯度检查点 |
| 评估范围 | 每次运行 200 个查询 × 200 张图库图像（50 次运行；共 10,000 次重复查询评估） |
| 主汇总方式 | 五个“单 seed 十人宏平均准确率”的均值 ± 样本标准差（`ddof=1`） |
| 正式实验硬件 | 每个“受试者–随机种子”任务使用一张 NVIDIA A40 |

## 仓库结构

```text
.
├── main/
│   ├── data.py                     # THINGS-EEG/图像加载和 ID 匹配
│   ├── models_brain.py             # EEG 编码器骨干网络
│   ├── models_clip.py              # 脑–CLIP 对齐模型
│   └── models_diffusion.py         # 实验性重建组件
├── scripts/
│   ├── evaluate_retrieval.py       # 标准评估与匈牙利评估
│   ├── aggregate_subject_metrics.py # 验证并汇总十名受试者指标
│   ├── aggregate_multiseed_metrics.py # 严格执行五随机种子汇总
│   ├── finalize_results.py         # 标准结果验证/报告
│   ├── finalize_hungarian_results.py
│   ├── run_subject_reproduction.sh # 通用单受试者复现脚本
│   ├── run_sub08_reproduction.sh   # 旧版 Subject 08 专用脚本
│   ├── run_hungarian_evaluation.sh # 特定站点的匈牙利评估封装脚本
│   ├── submit_subject_array.slurm  # 十名受试者 SLURM 数组任务
│   ├── submit_multiseed_array.slurm # 缺失随机种子的“受试者 × seed”数组任务
│   └── submit_*.slurm              # 其他 HKUST(GZ) SLURM 启动脚本
├── tests/
│   ├── test_hungarian_assignment.py
│   ├── test_multiseed_aggregation.py
│   ├── test_subject_metric_validation.py
│   └── test_submit_multiseed_array.py
├── experiments/samga_lora/        # 受控 SAMGA Frozen/LoRA 实验
│   ├── samga_lora/                 # 数据、模型、损失与指标工具
│   ├── scripts/                    # 预检、pilot、正式实验与聚合门控
│   ├── exploratory_internvit/      # 明确标注的推断模型探索扩展
│   └── tests/                      # 独立扩展测试
├── experiments/samga_reproduction/ # 经审计的发布代码复现
│   ├── README.md                   # 协议、结果与声明边界
│   ├── V2_5_FEATURE_PIPELINE.md    # 固定的特征提取与验证
│   ├── run_official_cell.sh        # 失败关闭的单被试启动器
│   ├── aggregate_official_results.py
│   └── tests/                      # 下载、特征、启动器和聚合测试
├── previous_work/clip_lora_baseline/ # 加入 SAMGA 前的 README 快照
├── docs/                            # 内部技术说明
├── train_clip_lora.py               # 主要训练入口
├── vanilla.py                       # 实验性重建路径
├── enhance.py                       # 实验性检索优化
└── graph.py                         # 实验性图方法优化
```

生成的检查点、缓存、日志、计划和结果产物均通过 `.gitignore` 有意排除；实验实现、协议文档、测试和启动脚本仍可纳入版本控制。

## 环境配置

请从仓库根目录运行本节中的命令。每个正式“受试者–随机种子”任务使用 Linux、一张 NVIDIA A40，以及以下经过完整测试的软件栈：

| 软件包 | 已测试版本 |
|---|---:|
| Python | 3.10.20 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| TorchVision | 0.26.0 + CUDA 12.8 |
| Transformers | 5.12.1 |
| Datasets | 5.0.0 |
| Accelerate | 1.14.0 |
| PEFT | 0.19.1 |
| Diffusers | 0.38.0 |
| Safetensors | 0.8.0 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Pillow | 12.2.0 |
| tqdm | 4.68.3 |
| einops | 0.8.2 |

`diffusers` 属于核心环境的一部分，因为即使仅运行检索，`main/models_clip.py` 也会导入其中的一个模型类。评估入口需要 SciPy，并使用它提供匈牙利算法求解器。

### 方案 A：复用已验证的集群环境

在项目集群上，`eeg_recon` 是生成所报告指标时使用的环境。如果当前 shell 中可以使用 Conda，可直接激活：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg_recon

python --version
which python
```

标准检索或匈牙利检索均无需额外安装依赖。复现原始 86.66%/98.38% 结果时应使用 `eeg_recon`；下文单独验证的 `test` 环境支持受控 SAMGA 扩展和经审计的公共代码复现，不能写成原始实验所用的软件栈。

若希望保持 `eeg_recon` 不变并创建一个独立工作副本：

```bash
conda create --name eeg-retrieval --clone eeg_recon -y
conda activate eeg-retrieval
```

### SAMGA 扩展所用环境

Frozen/LoRA 对比和发布代码复现均使用现有的 `test` 环境运行：Python 3.10.18、PyTorch 2.10.0+cu126、TorchVision 0.25.0+cu126、Transformers 4.57.6、PEFT 0.18.1、Accelerate 1.13.0、NumPy 1.26.4、SciPy 1.15.3、scikit-learn 1.7.2 和 Timm 1.0.26。请单独激活：

```bash
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
python -c "import torch, transformers, peft, scipy, timm"
```

受控扩展的可移植依赖清单位于 `experiments/samga_lora/requirements.txt`；发布代码流程和精确固定资产另见 `experiments/samga_reproduction/README_ZH.md`。PyTorch 仍需选择与目标 CUDA 驱动兼容的构建版本。集群现有的 `test` 环境没有安装可选的 `pytest` 包；仓库测试可在任意提供 pytest 的兼容环境中运行（现有 `eeg_recon` 环境即为一种选择）：

```bash
conda activate eeg_recon
PYTHONPATH=experiments/samga_lora python -m pytest -q experiments/samga_lora/tests
```

### 方案 B：从零创建已测试环境

创建干净的 Conda 环境，先安装相匹配的 CUDA 12.8 PyTorch wheel，再安装其余固定版本的依赖：

```bash
conda create --name eeg-retrieval python=3.10.20 pip -y
conda activate eeg-retrieval
python -m pip install --upgrade pip

python -m pip install \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  transformers==5.12.1 \
  datasets==5.0.0 \
  accelerate==1.14.0 \
  peft==0.19.1 \
  diffusers==0.38.0 \
  safetensors==0.8.0 \
  numpy==2.2.6 \
  scipy==1.15.3 \
  Pillow==12.2.0 \
  tqdm==4.68.3 \
  einops==0.8.2
```

CUDA wheel 必须与目标机器的 NVIDIA 驱动兼容。如果 CUDA 12.8 不适用，请从[官方安装指南](https://pytorch.org/get-started/locally/)选择兼容的 PyTorch 构建，并保持其余软件包版本固定。不要混用各自独立选择的 PyTorch 和 TorchVision 构建。

### 验证安装

提交训练任务前，请运行以下导入检查：

```bash
python - <<'PY'
import sys

import accelerate
import datasets
import diffusers
import peft
import scipy
import torch
import torchvision
import transformers
from scipy.optimize import linear_sum_assignment

from main.models_clip import BrainCLIPModel

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("Accelerate:", accelerate.__version__)
print("PEFT:", peft.__version__)
print("Diffusers:", diffusers.__version__)
print("SciPy:", scipy.__version__)
print("Compiled CUDA:", torch.version.cuda)
print("CUDA visible on this node:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Core retrieval imports: OK")
PY

python -m unittest discover -s tests -v
```

如果登录节点没有分配 GPU，出现 `CUDA visible on this node: False` 属正常现象。训练前应在 SLURM GPU 分配中再次运行该检查；正式运行时应显示所分配的 A40。下方单 GPU 复现命令只使用一个 `torchrun` 进程，因此使用仓库中双进程 `accelerate_config.yaml` 执行 `accelerate launch` 并不等价。

### 可选的重建依赖

标准检索与匈牙利检索结果不依赖实验性重建工具。若要使用 `vanilla.py`、`enhance.py`、`graph.py` 或重建指标，还需安装：

```bash
python -m pip install \
  scikit-image==0.25.2 \
  clip-anytorch==2.6.0
```

这些路径还需要另外下载 SDXL/IP-Adapter 权重。Weights & Biases、SwanLab 或 TensorBoard 等实验跟踪器是可选项，仅在通过 `--report_to` 选择时才需要。

## 数据与预训练模型

本仓库不分发数据集和模型权重。

从 [THINGS initiative](https://things-initiative.org/) 或其 [OSF 仓库](https://osf.io/3jk45/)下载 THINGS-EEG2，然后准备加载器所需的 250 Hz 白化文件：

```text
things_eeg_data/
├── Preprocessed_data_250Hz_whiten/
│   ├── sub-01/
│   │   ├── train.pt
│   │   └── test.pt
│   ├── ...
│   └── sub-10/
│       ├── train.pt
│       └── test.pt
├── training_images/
│   └── **/*.jpg
└── test_images/
    └── **/*.jpg
```

CLIP 模型必须存放在本地兼容 Hugging Face 的目录中，该目录应包含配置、图像处理器与权重，例如：

```text
CLIP-ViT-B-32-laion2B-s34B-b79K/
├── config.json
├── preprocessor_config.json
└── model.safetensors
```

运行前设置可移植路径：

```bash
export PROJECT_ROOT="$(pwd)"
export THINGS_ROOT="/path/to/things_eeg_data"
export BRAIN_DIR="$THINGS_ROOT/Preprocessed_data_250Hz_whiten"
export CLIP_PATH="/path/to/CLIP-ViT-B-32-laion2B-s34B-b79K"
export SEED=42
export SUBJECT_ID=1
printf -v SUBJECT_PADDED '%02d' "$SUBJECT_ID"
export OUTPUT_DIR="$PROJECT_ROOT/runs/all_subjects/seed${SEED}/subj${SUBJECT_PADDED}"
export RESULTS_DIR="$PROJECT_ROOT/results/all_subjects/seed${SEED}/subj${SUBJECT_PADDED}"
export CHANNELS="P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2"

mkdir -p "$OUTPUT_DIR/cache" "$RESULTS_DIR"
```

每当 `SUBJECT_ID` 或 `SEED` 改变时，都要重新计算 `SUBJECT_PADDED`、`OUTPUT_DIR` 和 `RESULTS_DIR`。

如需完全离线运行：

```bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 训练

以下命令无需依赖特定站点封装脚本中的路径，即可复现一个“受试者–随机种子”运行。正式实验网格需要将 `SUBJECT_ID` 依次设为 1 至 10、将 `SEED` 依次设为 42 至 46，并分别运行全部 50 个组合；不要把不同受试者或随机种子合并到一个模型中。

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  train_clip_lora.py \
  --dataset_name things \
  --brain_directory "$BRAIN_DIR" \
  --image_directory "$THINGS_ROOT" \
  --cache_dir "$OUTPUT_DIR/cache" \
  --subject_ids "$SUBJECT_ID" \
  --eval_subject_ids "$SUBJECT_ID" \
  --brain_column eeg \
  --brain_backbone brain_mlp \
  --dropout 0.1 \
  --pretrained_model_name_or_path "$CLIP_PATH" \
  --lora_rank 32 \
  --lora_layers all-linear \
  --gradient_checkpointing \
  --time_slice 0,250 \
  --avg_trials \
  --selected_channels "$CHANNELS" \
  --learning_rate 5e-4 \
  --vision_learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --weight_decay 0.05 \
  --seed "$SEED" \
  --dataloader_num_workers 8 \
  --mixed_precision bf16 \
  --output_dir "$OUTPUT_DIR" \
  --metrics_jsonl "$OUTPUT_DIR/validation_metrics.jsonl" \
  --save_total_limit 1 \
  --checkpointing_steps epoch \
  --validation_steps epoch \
  --num_train_epochs 25 \
  --per_device_train_batch_size 512 \
  --per_device_eval_batch_size 100
```

通用封装脚本可以执行 smoke test，也可以执行 25 epochs 的正式训练并随后进行两次全新的 checkpoint 重载评估。默认情况下它会拒绝覆盖已有正式运行：

```bash
bash scripts/run_subject_reproduction.sh smoke --subject-id 1 --seed "$SEED"
bash scripts/run_subject_reproduction.sh formal --subject-id 1 --seed "$SEED"
```

如果使用新硬件或新准备的数据集，请在正式任务前先运行 smoke test。

### 受控 SAMGA 扩展

独立的 [SAMGA + 视觉 LoRA 指南](experiments/samga_lora/README.md) 说明了数据清单、冻结特征缓存、一致性/smoke、概念不重叠 pilot、配置锁定和正式实验阶段。只有 pilot 门控通过且锁定源码哈希仍一致时，正式任务才允许评估测试集。本集群上的 100 个单元使用互不重叠、符合调度限制的区间提交；可复用的十单元启动方式为：

```bash
bash experiments/samga_lora/scripts/submit_formal_chunk.sh 0
# 完成后依次使用 10、20、……、90。

python experiments/samga_lora/scripts/aggregate_formal.py \
  --formal-root artifacts/samga_lora/formal \
  --locked-config artifacts/samga_lora/pilot_selection.json \
  --output-dir results/samga_lora
```

确认性协议只使用标准独立检索；不要把匈牙利分配加入 SAMGA 汇总。

### 经审计的 SAMGA 公共代码复现

独立的[公共代码复现指南](experiments/samga_reproduction/README_ZH.md)记录固定模型 revision 与哈希、安全下载、V2.5 特征提取、固定 60 轮与发布启动器 patience-10 协议、项目自定五随机种子稳定性网格、严格聚合和测试集选模泄漏。SAMGA 官方源码树保持未修改。

## 评估

在 Bash 中定义通用评估参数：

```bash
EVAL_ARGS=(
  --brain-model-path "$OUTPUT_DIR/brain_model"
  --vision-adapter-path "$OUTPUT_DIR/vision_model"
  --pretrained-model-name-or-path "$CLIP_PATH"
  --brain-directory "$BRAIN_DIR"
  --image-directory "$THINGS_ROOT"
  --dataset-name things
  --subject-id "$SUBJECT_ID"
  --selected-channels "$CHANNELS"
  --time-slice 0,250
  --batch-size 100
  --num-workers 0
  --device cuda
  --dtype bf16
  --cache-dir "$OUTPUT_DIR/cache"
  --seed "$SEED"
  --expected-num-samples 200
  --local-files-only
)
```

### 标准独立检索

```bash
python scripts/evaluate_retrieval.py \
  "${EVAL_ARGS[@]}" \
  --metrics-output "$RESULTS_DIR/sub${SUBJECT_PADDED}_seed${SEED}_formal_metrics.json" \
  --predictions-output "$RESULTS_DIR/sub${SUBJECT_PADDED}_seed${SEED}_formal_predictions.csv"
```

### 匈牙利一对一消融实验

该消融仅在随机种子 `42` / `sub-08` 上完成验证。请先设置 `SEED=42` 和 `SUBJECT_ID=8`、重新计算 `SUBJECT_PADDED=08`，将 `OUTPUT_DIR` 和 `RESULTS_DIR` 指向该次运行，然后重新执行上方完整的 `EVAL_ARGS=(...)` 定义，使 Bash 数组捕获更新后的值。评估器仍会写出标准逐查询指标，同时增加独立的受约束分配命名空间和 CSV：

```bash
python scripts/evaluate_retrieval.py \
  "${EVAL_ARGS[@]}" \
  --enable-hungarian \
  --metrics-output "$RESULTS_DIR/sub08_hungarian_metrics.json" \
  --predictions-output "$RESULTS_DIR/sub08_hungarian_standard_predictions.csv" \
  --hungarian-output "$RESULTS_DIR/sub08_hungarian_assignment.csv" \
  --similarity-output "$RESULTS_DIR/sub08_cosine_similarity.npz"
```

不要把 `assignment_accuracy` 标记为标准 Top-1，也不要为单次全局分配虚构匈牙利 Top-5。

## 测试

```bash
python -m unittest discover -s tests -v
```

这些测试涵盖匈牙利求解器最优性、冲突消解、无效矩阵、非对角 ID 映射，以及唯一最优解下的行/列置换不变性；同时验证随机种子列表解析、`ddof=1` 样本标准差、逐随机种子汇总顺序、逐被试跨 seed 汇总、预测字段语义，以及默认和自定义随机种子列表下的 SLURM 数组范围与任务映射。

SAMGA 复现工具使用独立测试套件：

```bash
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python \
  -m pytest -q experiments/samga_reproduction/tests
```

## SLURM 封装脚本

仓库包含在 HKUST(GZ) 集群上使用的启动脚本：

```bash
# 一个指定随机种子的全部十名受试者；数组最多同时运行两个任务。
export SEED=42
sbatch scripts/submit_subject_array.slurm smoke --seed "$SEED"
sbatch scripts/submit_subject_array.slurm formal --seed "$SEED"

# 本次复现使用以下默认设置，仅运行缺失的随机种子 43--46。
# 随机种子 42 的十次运行已经验证，因此脚本有意将其排除。
sbatch scripts/submit_multiseed_array.slurm formal

# 从零运行完整五随机种子网格时，覆盖数组范围并列出全部五个随机种子。
sbatch --array=0-49%2 scripts/submit_multiseed_array.slurm \
  formal 42 43 44 45 46

# 随机种子 42、Subject 08 专用旧版/匈牙利启动脚本。
sbatch scripts/submit_sub08.slurm formal
sbatch scripts/submit_hungarian_eval.slurm
```

汇总分为两个严格层级。首先在每个随机种子内验证并汇总十名受试者：

```bash
for SEED in 42 43 44 45 46; do
  python scripts/aggregate_subject_metrics.py \
    --results-root "$PROJECT_ROOT/results/all_subjects/seed${SEED}" \
    --subjects 1-10 \
    --seed "$SEED" \
    --expected-epochs 25
done
```

逐随机种子聚合器会检查 query/gallery 数量、25 条验证记录、指标与正确数是否一致、检索协议、已保存模型配置、CLIP 基座路径以及关键环境版本。它还会根据预测中的图像 ID、排名和有序 Top-5 列表重新推导 Top-1/Top-5 正确性，验证重复重载预测完全一致，并记录模型配置、模型权重、指标、预测和训练历史的哈希。它会在每个 `results/all_subjects/seed${SEED}/` 目录下生成 `summary.json`、`per_subject_metrics.csv`、`RESULTS_EN.md` 和 `RESULTS_ZH.md`。

然后重新验证完整的 50 次运行网格并计算五随机种子结果：

```bash
python scripts/aggregate_multiseed_metrics.py \
  --results-root "$PROJECT_ROOT/results/all_subjects" \
  --seeds 42,43,44,45,46 \
  --subjects 1-10 \
  --expected-epochs 25
```

跨随机种子聚合器会重新打开、从语义上重新验证每个源运行及其哈希，拒绝缺失或重复的“受试者–随机种子”单元格以及不兼容的模型/环境元数据，并检查逐随机种子、逐被试和合并均值是否一致。它会在 `results/all_subjects/seeds42-46/` 下生成 `summary.json`、`per_run_metrics.csv`、`per_seed_metrics.csv`、`per_subject_metrics.csv`、`RESULTS_EN.md` 和 `RESULTS_ZH.md`。主不确定性是五个“单 seed 十人宏平均准确率”之间的样本标准差（`ddof=1`），而不是 50 个单独单元格之间的离散程度。

这些 shell 与 SLURM 文件目前包含特定站点的绝对路径。在其他克隆或集群中使用前，请更新：

- `PROJECT_ROOT`、`THINGS_ROOT`、`BRAIN_DIR` 和 `CLIP_PATH`；
- 当随机种子数量变化时，更新 `#SBATCH --array`、`--chdir`、`--output` 和 `--error`；
- Conda 激活路径与环境名称；
- 分区、GPU、CPU、内存和时间请求。

前文给出的直接训练和评估命令是可移植的参考命令。

## 原始项目可复现性政策

- 正式指标使用第 25 个 epoch 后的固定最终检查点进行评估。
- 测试集峰值 epoch 仅用于诊断，不用于选择检查点。
- 完整实验网格使用随机种子 `42`、`43`、`44`、`45`、`46`；全部 50 个“受试者–随机种子”组合均训练为独立模型，训练时不进行跨被试或跨随机种子合并。
- 查询与图库身份通过唯一图像 ID 匹配，而不是假定目标位于对角线上。
- 每个标准评估均在独立重新加载模型后重复执行，并且重复评估必须复现相同的指标与逐查询预测。
- 主均值由五个“单 seed 十人宏平均准确率”计算；其 ± 项是这五个值的样本标准差（`ddof=1`），而不是 50 个“受试者–随机种子”单元格的标准差。
- 10,000 次查询评估是十名受试者各自同一组 200 个留出查询在五个随机种子下的重复评估，并非 10,000 个独立测试样本。
- 所有标准汇总只包含逐查询独立 Top-1/Top-5；匈牙利分配绝不混入逐随机种子或五随机种子结果。
- 随机种子 `42` / `sub-08` 的匈牙利评估会保存完整相似度矩阵、ID 顺序、哈希、转换记录和分配输出。
- 真实标签仅在求解分配后使用，不属于匈牙利目标函数的一部分。
- 审计多种预先声明的行/列顺序，确保输入的对齐顺序无法在完全平局时静默决定结果。

## 局限性与负责任使用

- 五个随机种子可以估计跨随机种子波动，但 `n=5` 仍然较小，所报告的样本标准差本身也存在不确定性。
- 每名受试者使用独立模型；本实验不评估跨被试泛化。
- 设置随机种子后，训练过程仍未对所有 GPU 运算强制启用 PyTorch 确定性算法，因此重新训练同一“受试者–随机种子”组合不保证逐比特一致。独立检查点重载评估验证的是已保存产物，而不是一次全新训练的逐比特可复现性。
- 10,000 次查询评估会在五个随机种子间复用每名受试者的同一组 200 个留出刺激，因此不能将其视为 10,000 个统计独立样本。
- 原始项目训练循环在 `backward()` 之前调用梯度裁剪，因此配置的最大梯度范数不会影响那 50 个“受试者–随机种子”运行。修正调用顺序会形成不同协议，并需要完整重跑。
- 试次平均使用了重复呈现的数据，并不等价于单试次解码。
- 匈牙利消融仅在随机种子 `42` / `sub-08` 上验证，不能外推到其他受试者或随机种子。它还需要完整的查询批次和已知的一对一图库先验，因此不是在线单查询检索协议。
- 复用的旧版随机种子 `42` / `sub-08` 指标记录了一个现已不可用的早期数据集路径，无法回溯验证它与当前 `EEG_Recon-RL` 数据集根目录的历史字节一致性。已保存的模型/结果产物、协议和重复重载检查仍通过验证，但该次运行的数据来源声明受到限制。
- 数据集、预处理和模型权重的版本可能显著影响结果。
- 重建路径仍处于实验阶段；本 README 不声称任何正式的重建指标。
- EEG 属于敏感的人类受试者数据。请遵守数据集关于知情同意、隐私、许可和再分发的要求，且不要将本研究系统解释为临床或诊断工具。

## 参考文献

- Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). [A large and rich EEG dataset for modeling human visual object recognition](https://doi.org/10.1016/j.neuroimage.2022.119754). *NeuroImage, 264*, 119754.
- Song, Y., Liu, B., Li, X., Shi, N., Wang, Y., & Gao, X. (2024). [Decoding Natural Images from EEG for Object Recognition](https://openreview.net/forum?id=dhLIno8FmH). *ICLR 2024*. 代码：[NICE-EEG](https://github.com/eeyhsong/NICE-EEG)。
- Li, D., Wei, C., Li, S., Zou, J., & Liu, Q. (2024). [Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ba5f1233efa77787ff9ec015877dbd1f-Abstract-Conference.html). *NeurIPS 2024*. 代码：[EEG Image Decode](https://github.com/dongyangli-del/EEG_Image_decode)。
- Wu, H., Li, Q., Zhang, C., He, Z., & Ying, X. (2025). [Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Bridging_the_Vision-Brain_Gap_with_an_Uncertainty-Aware_Blur_Prior_CVPR_2025_paper.html). *CVPR 2025*。
- Zheng, J., Jia, H., Li, M., Zheng, Y., Zeng, Y., Gao, Y., & Liang, C. (2026). [Learning Brain Representation with Hierarchical Visual Embeddings](https://openreview.net/forum?id=IEq71qS8B7). *ICLR 2026*。
- Zhou, J., Xu, C., Wang, W., Yang, E., & Deng, C. (2026). [EEGiT: Teaching Vision Transformers to Understand the EEG Signal](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_EEGiT_Teaching_Vision_Transformers_to_Understand_the_EEG_signal_CVPR_2026_paper.html). *CVPR 2026*。
- Du, Y., Dai, S., Song, Y., Thompson, P. M., Tang, H., & Zhan, L. (2026). [Deep Models, Shallow Alignment: Uncovering the Granularity Mismatch in Neural Decoding](https://arxiv.org/abs/2601.21948). *arXiv 预印本*。
- Tang, J., Jiang, S., Su, F., & Zhao, Z. (2026). [Aligning What EEG Can See: Structural Representations for Brain-Vision Matching](https://arxiv.org/abs/2603.07077). *arXiv 预印本*。
- Jiang, L., She, Q., Xu, J., Xu, H., Wu, D., & Kuang, Z. (2026). [Subject-Aware Multi-Granularity Alignment for Zero-Shot EEG-to-Image Retrieval](https://arxiv.org/abs/2604.17782). *arXiv 预印本*。

## 许可证

本课程项目仓库尚未声明开源许可证。重新分发代码或接受外部贡献前，请添加明确的许可证。
