# brain-rw 当前工作进度说明

## 1. 当前任务与阶段目标

当前仓库的主任务是 **brain-to-image retrieval**：输入脑信号，输出与之对应的图像表征，并在图库中检索出目标图像。

从任务设定上，单个样本记为 $(b_i, x_i, s_i)$，其中：

- $b_i$ 表示第 $i$ 个脑信号样本；
- $x_i$ 表示与该脑信号配对的目标图像；
- $s_i$ 表示 subject id。

当前阶段我们更关心的是：

1. 能否把脑信号稳定映射到一个可检索的视觉语义空间；
2. 在这个基础上，是否能自然接出第二条有说服力的工作线。

现阶段的结论比较明确：

- **LoRA 双端联合适配**在 retrieval 上已经形成了当前最强、最清晰的一条主线；
- 我们暂时还没有找到一个能与这条主线并列、可单独支撑故事的第二个 idea；
- 目前只沿着 **RAG + reconstruction** 方向做了若干尝试，它更像是一个自然延伸，但还不是同等级的亮点。

## 2. 变量定义与基本链路

我们把脑信号编码器记为 $\mathcal{E}_{\text{brain}}$，图像编码器记为 $\mathcal{E}_{\text{img}}$。对第 $i$ 个样本，有

$$
z_i^{(b)} = \mathcal{E}_{\text{brain}}(b_i, s_i),
\qquad
z_i^{(v)} = \mathcal{E}_{\text{img}}(x_i).
$$

为了做对比学习与检索，训练时使用归一化后的表征

$$
\bar z_i^{(b)} = \frac{z_i^{(b)}}{\|z_i^{(b)}\|_2},
\qquad
\bar z_i^{(v)} = \frac{z_i^{(v)}}{\|z_i^{(v)}\|_2}.
$$

相似度矩阵记为 $S \in \mathbb{R}^{N \times N}$，其中

$$
S_{ij} = \exp(\alpha)\, (\bar z_i^{(b)})^\top \bar z_j^{(v)}.
$$

这里 $\alpha$ 是可学习的 `logit_scale` 参数。对于一个 batch，大意上我们希望：

- 正样本对 $(b_i, x_i)$ 的相似度尽可能高；
- 负样本对 $(b_i, x_j)$，其中 $j \neq i$，的相似度尽可能低。

当前仓库中的默认实验设定大致是：

- 数据模态以 EEG 为主；
- 默认使用后部 17 个通道；
- 默认时间窗为 $[0, 250)$；
- 按 subject 分开训练；
- 当前主用视觉骨干是 `CLIP-ViT-B-32`。

因此，当前主链路可以概括为

$$
b_i
\xrightarrow{\mathcal{E}_{\text{brain}}}
z_i^{(b)}
\xrightarrow{\text{align with } \mathcal{E}_{\text{img}}}
z_i^{(v)}
\xrightarrow{\text{retrieval}}
\text{top-}K \text{ image candidates}.
$$

## 3. 主线一：LoRA 双端联合适配的 Retrieval

### 3.1 思路概述

当前最有效的思路，是把脑信号编码器和视觉编码器放在同一个对齐训练框架下联合优化：

- 脑信号侧直接训练 $\mathcal{E}_{\text{brain}}$；
- 图像侧不做完整重训，而是在 CLIP vision encoder 上加入 LoRA；
- 优化时采用 **TTUR**，即脑信号侧和视觉侧使用不同学习率；
- 两端共同被拉到同一个 retrieval 空间中。

这里说的“LoRA 双向微调”，在当前文档里统一指 **脑编码器训练 + 视觉编码器 LoRA 适配的双端联合对齐**，而不是单纯指某种双向检索损失。

### 3.2 训练目标

当前实现里使用的是 CLIP 风格的对比损失。写成公式，可以记为

$$
\mathcal{L}_{\text{ret}}
=
\frac{1}{N}\sum_{i=1}^N
-\log
\frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ij})}.
$$

但在参数更新时，当前实现并不是把所有参数用同一个学习率一起推，而是采用 **Two Time-Scale Update Rule (TTUR)**。记脑编码器参数为 $\theta_b$，视觉侧 LoRA 参数为 $\theta_v$，则更新可以写成

$$
\theta_b \leftarrow \theta_b - \eta_b \nabla_{\theta_b}\mathcal{L}_{\text{ret}},
\qquad
\theta_v \leftarrow \theta_v - \eta_v \nabla_{\theta_v}\mathcal{L}_{\text{ret}},
\qquad
\eta_b > \eta_v.
$$

它的含义是：

- 脑信号编码器承担从 EEG 到视觉语义空间的主要迁移，因此允许更快更新；
- 视觉侧只做 LoRA 级别的轻量适配，因此用更小的学习率，避免过度破坏原始 CLIP 表征；
- 两个模块在同一个损失下联合训练，但收敛速度是有意分开的。

就当前脚本而言，这一点直接体现在优化器参数分组上：脑侧使用 `learning_rate`，视觉 LoRA 侧使用 `vision_learning_rate`。默认训练脚本里也明确采用了

$$
\eta_b = 5 \times 10^{-4},
\qquad
\eta_v = 5 \times 10^{-5}.
$$

它的核心含义很直接：

- 对每个脑信号查询 $b_i$，正确图像 $x_i$ 应该在 batch 内排到最高；
- 其余图像都作为负样本参与竞争。

在验证阶段，仓库里实际使用的是 retrieval 指标，例如 `top1_acc` 和 `top5_acc`。因此，这条线的判断标准不是生成图像好不好看，而是：

- 脑信号能不能稳定找回正确图像；
- 正确图像能不能稳定排在前列。

### 3.3 为什么这条线当前最强

这条线目前之所以最有说服力，原因主要有三点。

第一，它的任务目标是清楚的。我们直接优化脑信号与图像语义表征之间的对应关系，目标函数和评估指标是一致的。

第二，它的改动方式是克制的。视觉端不是完全重训，而是通过 LoRA 做轻量适配，这使得：

- 原始 CLIP 空间的通用视觉语义仍然保留；
- 模型又能向脑信号分布做一定偏移；
- 训练成本和稳定性也更容易控制。

第三，它的优化策略也是有针对性的。TTUR 让脑侧用更快的时间尺度去适应 EEG 分布，同时让视觉 LoRA 侧以更慢的时间尺度做配合式修正，这比“所有模块同学习率一起推”更符合当前任务结构。

第四，它在当前项目里确实形成了阶段性结果。虽然这份文档不补具体数值，但我们现在可以相对有把握地说：

- **LoRA 双端联合适配在 retrieval 上表现很好**；
- 这是目前仓库里最成熟、最像一个完整工作单元的部分。

## 4. 第二条尝试：RAG + Reconstruction

### 4.1 动机

当 retrieval 主线跑通之后，一个很自然的问题是：

> 如果脑信号已经能检索到相近的图像记忆，那么能不能进一步把这些记忆组织起来，作为生成模型的条件，去做图像重建？

这就是当前 `RAG + reconstruction` 思路的出发点。

它并不是在原始像素空间里直接从脑信号生成图像，而是先检索、再聚合、再重建。

### 4.2 变量定义

这里可以把图库中的两类图像表征区分开来：

- $g_j$：用于 retrieval 打分的图像表征；
- $f_j$：用于后续生成条件构造的 frozen 图像表征。

对一个脑信号查询 $b_i$，先得到脑表征 $\bar z_i^{(b)}$，然后在图库上计算相似度

$$
r_{ij} = (\bar z_i^{(b)})^\top \bar g_j.
$$

接着取 top-$K$ 邻居

$$
\mathcal{N}_K(i) = \operatorname{TopK}_j(r_{ij}),
$$

并利用训练阶段学到的缩放规则对它们做加权：

$$
w_{ij}
=
\operatorname{softmax}_{j \in \mathcal{N}_K(i)}
\left(r_{ij}\exp(\alpha)\right).
$$

然后把检索得到的 frozen visual memories 融合成一个条件向量

$$
\tilde f_i
=
\sum_{j \in \mathcal{N}_K(i)} w_{ij} f_j.
$$

最后把 $\tilde f_i$ 送入 IP-Adapter，再接到 SDXL 上做重建：

$$
h_i = \operatorname{IPAdapter}(\tilde f_i),
\qquad
\hat x_i = \operatorname{SDXL}(h_i).
$$

### 4.3 这条线在做什么

这条线的本质可以理解为：

1. 先用 EEG 查询一个视觉记忆库；
2. 再把检索到的邻域压缩成一个生成条件；
3. 最后借助预训练的图像生成器做 reconstruction。

换句话说，这不是“从零生成”，而是“基于检索结果的条件重建”。

从项目叙事上看，它像是把 retrieval 主线自然延伸到了生成侧，因此是当前最顺手、也最合理的第二条探索方向。

### 4.4 当前状态判断

这条线已经完成了技术链路上的基本打通：

- 能从脑信号编码得到 query embedding；
- 能在图像 embedding bank 中做 top-$K$ retrieval；
- 能把检索结果转换成 IP-Adapter 条件；
- 能接入 SDXL 完成重建并计算 `pixcorr`、`SSIM`、`CLIP` 等指标。

同时，仓库里也已经包含了若干 retrieval refinement 的不同实现版本。但就当前阶段而言，我们对外更适合把它们统一归纳为 **RAG + reconstruction 的尝试**，而不是展开成多个已经独立成立的方法点。

更关键的是，这条线目前仍然是探索性的。它说明了 retrieval 结果确实可以进一步被拿来驱动生成，但还没有强到足以成为一个和主线并列的第二亮点。

## 5. 当前工作进度的总体判断

如果把现阶段工作压缩成一句话，可以概括为：

> 我们已经有了一条效果明确、逻辑清晰的 retrieval 主线，但尚未找到与其配套的第二个强 idea，目前只在 RAG + reconstruction 上做了尝试。

更具体地说，当前进展可以分成三层。

第一层，是已经站稳的部分：

- 脑信号到视觉语义空间的对齐是可行的；
- 基于 LoRA 双端联合适配的 retrieval 是当前最强结果；
- 这部分已经可以作为对外讲解时的主干内容。

第二层，是已经打通但还不够强的部分：

- 基于 retrieval 的 reconstruction 流程已经成立；
- 生成端可以利用检索到的视觉记忆作为条件；
- 但它目前更像验证“这条路可以走”，还不是验证“这条路已经走得很好”。

第三层，是当前最真实的项目状态：

- 现在的瓶颈不在于没有工作可讲；
- 而在于除了 retrieval 之外，还没有第二个同等清晰、同等有力的 idea；
- 所以整个项目更像是“一个强主线 + 一个仍在摸索的补充方向”。

## 6. 当前卡点与下一步方向

当前卡点主要有两个。

第一个卡点是，retrieval 这条线已经比较清楚，但它的优势主要集中在“对齐与检索”本身，故事结构还偏单点。

第二个卡点是，reconstruction 虽然是最自然的延伸，但目前还没有显示出足够强的独立价值。也就是说，它还没有把项目从“检索做得不错”真正推到“检索之外还有第二个明确亮点”。

因此，下一步的方向也比较明确：

1. 要么继续把 `RAG + reconstruction` 做强，使它真正成为 retrieval 的有效补强；
2. 要么寻找一个与 retrieval 更直接互补、但又不只是附属演示的新 idea。

在这两者中，当前仓库已经实际落地的，仍然主要是前者。

## 7. 代码落点

如果需要把这份进展和仓库代码对应起来，当前最关键的入口有三类：

- retrieval 训练入口：`train_clip_lora.py`
- 表征模型定义：`main/models_clip.py`、`main/models_brain.py`
- reconstruction 尝试：`vanilla.py` 及相关生成链路

因此，这份文档对应的不是一个抽象想法，而是当前仓库里已经存在的一套明确工作结构：

$$
\text{brain signals}
\rightarrow
\text{retrieval alignment}
\rightarrow
\text{top-}K \text{ visual memory}
\rightarrow
\text{reconstruction attempt}.
$$
