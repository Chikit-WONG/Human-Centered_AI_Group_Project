# 10 分钟中文讲稿

> 主幻灯片为英文。下面的时间用于排练，不显示在 PDF 中。正常语速下主讲约
> 9 分 35 秒，另外预留约 25 秒给换页、停顿和现场波动。

## Title — Let Both Sides Adapt（约 25 秒）

大家好，我们的项目研究的是 EEG-to-image retrieval：能不能根据一个人看到图片时产生的非侵入式脑电信号，从图库中找回他真正看过的那张图片。

我们的核心结论可以概括为一句话：不要强迫 EEG 单方面进入一个完全固定的视觉空间。应该让脑信号和视觉模型共同适配，但让视觉侧移动得更慢、更少。

**过渡：** 首先，我会把任务本身说清楚。

## Slide 2 — Can 17 EEG channels recover one image from 200?（约 55 秒）

我们使用 THINGS-EEG2 数据。输入是受试者观看图片时记录的 EEG，只保留 17 个后部通道和 250 个时间采样点，并对重复试次求平均。

测试时，每条 EEG query 面前都有 200 张从未参与训练的候选图片。模型不是判断图片类别，而是必须在目标图片和另外 199 张干扰图片之间，把受试者真正看过的那一张排到前面。

因此这是严格的 200-way exact-image retrieval。随机 Top-1 只有 0.5%，而我们的五随机种子结果是 86.66%。这个数字很高，但它仍然是一个封闭图库中的检索任务，不是开放式读心。

**过渡：** 那么，为什么不能直接把 EEG 映射到冻结的 CLIP 空间？

## Slide 3 — A rigid visual space makes EEG carry the whole mismatch（约 55 秒）

一个直观方案是完全冻结视觉编码器。这样能够保留 CLIP 的通用视觉语义，但脑信号噪声很大，而且存在明显的被试差异。视觉空间完全不动，就意味着所有脑—视觉错配都必须由 EEG encoder 独自承担。

另一个极端是完整微调 CLIP。视觉空间虽然能够自由移动，但 EEG 数据规模远小于视觉预训练数据，完整微调更容易过拟合，也可能破坏原来的视觉先验。

所以我们的设计目标是让两个模态在中间相遇，但是视觉侧必须受到约束。

**过渡：** 这就得到我们最重要的方法页。

## Slide 4 — EEG learns fast; vision adapts slowly（约 1 分 20 秒）

我们把方法概括为 asymmetric co-adaptation。

左边是脑信号分支。它使用 residual MLP，把 17 乘 250 的 EEG 输入映射到 512 维表征。脑侧参数全部训练，学习率是 5e-4，它承担主要的跨模态迁移。

右边是视觉分支。基础模型是 CLIP ViT-B/32，但原始 CLIP 权重保持冻结，只在视觉线性层中加入 rank-32、alpha-32 的 LoRA adapter。视觉学习率是 5e-5。

因此脑侧更新速度是视觉侧的十倍。视觉空间会向 EEG 做小幅配合，但不会像完整微调那样大幅漂移。

这里需要强调两点。第一，不是两个分支都使用 LoRA：脑侧是全量训练，只有视觉侧使用 LoRA。第二，我们不是发明 LoRA 或 TTUR，而是把参数高效视觉适配与非对称优化组合成适合小样本、噪声 EEG 的共同对齐策略。

**过渡：** 接下来看看两个分支如何被同一个目标拉到一起。

## Slide 5 — The training objective matches independent retrieval（约 1 分 05 秒）

对每个 EEG trial，脑编码器产生一个 512 维 embedding；对应图片经过带 LoRA 的视觉编码器，也产生一个 512 维 embedding。两边都先做 L2 normalization，再计算带可学习温度的余弦相似度。

当前代码优化的是 brain-to-image 的 batch contrastive loss。正确图片的相似度需要高于 batch 内的其他图片。这里不应该把它描述成双向或者对称的 CLIP loss，因为代码实际使用的是 `contrastive_loss(logits_per_brain)`。

推理与训练关系保持一致：每条 EEG query 独立计算与 200 张图片的相似度，最后报告 Top-1 和 Top-5。

正式结果覆盖 10 名受试者、5 个随机种子，并统一使用 epoch 25。标准结果没有使用 Hungarian，也没有利用“图库中每张图片必须被使用一次”的全局先验。

**过渡：** 在这个严格协议下，完整系统的表现如何？

## Slide 6 — The complete system retrieves the viewed image reliably（约 55 秒）

完整系统的 Top-1 是 86.66%，五个 seed-level 十受试者宏平均之间的标准差是 0.69 个百分点；Top-5 是 98.38%，标准差是 0.14 个百分点。

作为参照，200-way 随机基线只有 0.5% Top-1 和 2.5% Top-5。五个种子的 Top-1 均值位于 85.60% 到 87.35% 之间，说明结果并不是由某一个随机种子偶然拉高。

这里的正负号不是 50 个 subject-seed cell 的标准差，也不是一万条独立测试样本的置信区间。它是五个 seed-level 宏平均的 sample standard deviation。同一名受试者的 200 个测试刺激会在不同种子下重复评估，因此不能把一万次判断说成一万个独立样本。

**过渡：** 除了最终均值，我们还检查了每个训练过程是否稳定。

## Slide 7 — Fifty independent runs converge to the same operating point（约 55 秒）

这张图汇总了 10 名受试者乘 5 个随机种子的 50 条训练轨迹。左侧可以看到 held-out 200-way contrastive loss 持续下降；右侧 Top-1 和 Top-5 逐步上升，并且在训练后期集中到相近范围。

虚线表示我们预先统一使用的 epoch 25。必须说明，这些逐 epoch 曲线是对正式测试 split 的事后诊断，不是验证集选模。所有正式结果都固定使用最终 epoch 25，没有根据这张曲线挑选最好 epoch。

因此这张图支持的是“训练过程在 50 个独立模型中表现一致”，而不是“我们通过查看测试集选择了最优停止点”。

**过渡：** 不过，一个很高的完整系统结果仍然不能单独证明提升来自我们的视觉适配，所以我们又做了 matched control。

## Slide 8 — A matched control isolates a +6.51-point Top-1 gain（约 1 分 25 秒）

这是一条与主系统分开的受控实验线。我们把同样的视觉 LoRA/TTUR 干预放入 SAMGA-style EEG framework，并建立 Frozen CLIP 与视觉 LoRA/TTUR 两个严格匹配的实验臂。

两边使用相同的数据、EEG backbone、任务初始化、训练样本、batch 顺序、随机种子、目标函数和 CLIP ViT-B/32。唯一有意改变的因素，是视觉特征保持冻结，还是允许 rank-32 LoRA 以较小学习率更新。

Frozen CLIP 的 Top-1 是 76.17%，加入 LoRA/TTUR 后是 82.68%，提高 6.51 个百分点。subject 和 seed 双向 cluster bootstrap 的 95% 区间是正的 5.22 到 7.69。

50 对实验中有 48 对 Top-1 提升，而且 10 名受试者的平均变化全部为正。因此改进并不是由少数 subject 或 seed 驱动。

这里不能把 82.68% 和前面的 86.66% 当成同一个系统横向比较。86.66%回答“完整项目模型有多强”，而这组 matched control 回答“我们的视觉适配干预是否真的产生稳定贡献”。

**过渡：** 最后，作为 Human-Centered AI 项目，我们还必须明确这个结果的边界。

## Slide 9 — High accuracy does not turn retrieval into mind reading（约 1 分钟）

目前的证据支持的是被试内、重复试次平均、200 张已知图片组成的封闭图库检索，并且在五个随机种子上表现稳定。

它不支持跨被试泛化，不等价于 single-trial mind reading，不是开放世界图像生成，也不是临床或诊断工具。

EEG 是敏感的人类受试者数据。数据许可、知情同意、隐私保护、避免夸大声明，以及不使用测试集选 checkpoint，都应该被视为系统质量的一部分，而不是准确率之外的附加说明。

所以我们的目标不是把高准确率包装成“读心”，而是在定义清楚的任务和伦理边界内，证明一种脑—视觉对齐策略是有效的。

**过渡：** 最后把整个项目压缩成一句可以带走的设计原则。

## Slide 10 — Let the two modalities meet halfway（约 40 秒）

我们的关键结论不是 EEG encoder 必须越来越复杂，而是视觉空间不应该完全僵硬。

让 EEG 快速学习，让视觉侧通过 LoRA 缓慢适配。这个方案是非对称的、参数高效的，并且同时得到完整系统结果与 matched control 的支持。

下一步最重要的问题，是这条原则能否继续适用于跨受试者和单试次 EEG。

谢谢大家。

## 备份页与常见问题

### 如果被问“完整结构是什么？”

打开 `Backup: Full training and retrieval pipeline`。先区分训练阶段与推理阶段，再强调标准推理是每条 query 独立排序。

### 如果被问“LoRA 的具体配置怎么选的？”

打开 `Backup: Verified final LoRA configuration`。说明 rank 32、alpha 32、dropout 0、cosine scheduler、weight decay 0.05 和 25 epochs 都是最终核验配置；仓库没有保存本地 LoRA 超参数 sweep，因此不能声称这是 sweep 得到的最优值。

### 如果被问“是不是 SOTA？”

打开 literature backup。回答：我们的结果在选定文献中很强，但不同论文的视觉特征、通道数、checkpoint 规则和评估实现不同；SAMGA 预印本结果更高，而且这张图有意没有画 SAMGA。因此不做无条件 SOTA 声明。

### 如果被问“为什么 Hungarian 能到 100%？”

打开 local diagnostic backup。说明它同时查看完整 200×200 相似度矩阵，并假设 query 与 gallery 是已知一一对应关系。它是 transductive global assignment，不是标准 independent retrieval，所以只作为 Subject-08、seed-42 的诊断。

### 如果被问“reconstruction 为什么没有讲？”

打开 result-tracks backup。回答：RAG + SDXL 链路已经技术性打通，但目前没有足够强、足够稳定的正式重建结果，因此只作为探索性下游方向，不与 retrieval 主结果并列。
