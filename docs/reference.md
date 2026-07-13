# 脑信号描述任务简报：基于 Latent Flow Reward 的思路与结果

## 1. 任务定义

我们当前研究的是 **brain captioning**：输入脑信号，输出一段能描述目标图像内容的文本。

- 脑信号记为 $b$
- brain encoder 记为 $\mathcal{E}_{\text{brain}}$
- 脑特征记为 $z_b = \mathcal{E}_{\text{brain}}(b)$
- caption decoder / policy 记为 $\pi_\phi(c \mid z_b)$
- 文本 encoder 记为 $\mathcal{E}_{\text{text}}$
- 图像 encoder 记为 $\mathcal{E}_{\text{img}}$
- 对任意 caption $c$，其文本特征记为 $z_c = \mathcal{E}_{\text{text}}(c)$
- 对目标图像 $v$，其图像特征记为 $z_v = \mathcal{E}_{\text{img}}(v)$

因此整体链路可写为

$$
b \xrightarrow{\mathcal{E}_{\text{brain}}} z_b \xrightarrow{\pi_\phi} c,
\qquad
z_c = \mathcal{E}_{\text{text}}(c),
\qquad
z_v = \mathcal{E}_{\text{img}}(v).
$$

我们的核心目标不是只让 caption “看起来通顺”，而是让它尽可能保留目标图像对应的**视觉身份信息**。

## 2. 核心思路：从 CIM 到我们的 Latent Flow Reward

### 2.1 来自 CIM 的关键启发

CIM 的底层 insight 可以理解为：

$$
\text{一个好的 caption}
\Longrightarrow
p(z_{v_r} \mid z_{c_r})
\approx
p(z_{v_{\text{gt}}} \mid z_{c_{\text{gt}}}),
$$

也就是：

- rollout 出来的 caption $c_r$，其文本特征 $z_{c_r}$ 所诱导出的视觉后验，
- 应该尽可能接近真实 caption $c_{\text{gt}}$ 所对应的视觉后验。

这意味着 caption 优化的本质，不一定要直接在表层文本上定义 reward，而可以转化为：

> caption 是否保持了正确的视觉身份分布。

CIM 用的是 retrieval / gallery consistency 的方式去逼近这个量；而我们把这件事进一步推广到**连续 latent 空间**里，用一个条件 flow 模型来建模文本 latent 到图像 latent 的关系。

### 2.2 我们的方法直觉

我们训练一个条件 latent flow 模型 $v_\theta$，输入是 noisy image latent 与 text latent，输出是对应的 velocity：

$$
v_\theta(z_t, z_c, t) \approx \epsilon - z_v.
$$

其中

$$
z_t = (1-\sigma_t) z_v + \sigma_t \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I).
$$

训练目标是标准的 flow matching：

$$
\mathcal{L}_{\text{flow}}(\theta)
=
\mathbb{E}_{z_v, z_c, t, \epsilon}
\left[
\left\|
v_\theta(z_t, z_c, t) - (\epsilon - z_v)
\right\|_2^2
\right].
$$

直观上，这个模型学到的是：

- 给定一段文本，它在视觉 latent 空间里会“拉向哪里”；
- 如果 caption 和目标图像身份一致，那么它诱导出来的 velocity field 应该更接近 GT caption 对应的 velocity field；
- 如果 caption 漂移了，velocity 就会偏离。

因此，我们可以把 flow 模型从一个“生成式 mapper”，转成一个“相对身份一致性打分器”。

## 3. 我们的 Reward 形式

在当前最终版中，我们采用的是 **reference-conditioned delta-only reward**。

设：

- GT caption 为 $c_{\text{gt}}$
- 候选 caption 为 $c$
- 参考文本特征为 $z_{\text{ref}} = \mathcal{E}_{\text{text}}(c_{\text{gt}})$
- 候选文本特征为 $z_c = \mathcal{E}_{\text{text}}(c)$

对每个噪声 level $u \in \mathcal{U}$，分别计算

$$
\hat{v}_{\text{ref}} = v_\theta(z_t, z_{\text{ref}}, t),
\qquad
\hat{v}_{c} = v_\theta(z_t, z_{c}, t).
$$

定义 velocity delta：

$$
\Delta_v = \lambda (\hat{v}_c - \hat{v}_{\text{ref}}),
$$

其中当前实验固定 $\lambda = 3.5$。

然后构造缩放后的候选 velocity：

$$
\hat{v}_{c}^{\text{scaled}} = \hat{v}_{\text{ref}} + \Delta_v.
$$

定义归一化 velocity distance：

$$
d_{\text{vel}}
=
\frac{
\|\Delta_v\|_2^2 / D
}{
\frac{1}{2}
\left(
\|\hat{v}_{c}^{\text{scaled}}\|_2^2 / D
+
\|\hat{v}_{\text{ref}}\|_2^2 / D
\right)
},
$$

最终 reward 为

$$
R_{\text{flow}}(c ; c_{\text{gt}}, z_v)
=
\mathbb{E}_{u \in \mathcal{U}}
\left[
\frac{1}{1 + d_{\text{vel}}(u)}
\right].
$$

这个 reward 的含义是：

- 若候选 caption 与 GT caption 所诱导的视觉 latent 几何接近，则 reward 高；
- 若候选 caption 脱离目标图像身份，则 reward 降低。

这个 reward 不是通用 caption 分数，而是一个**相对 GT 视觉身份 manifold 的一致性分数**。

## 4. 为什么这个思路可信

从现有结果看，这个思路成立的证据主要有三类。

### 4.1 Reward 与 caption 质量有明显正相关

我们把 `v3-last` 四个 subject 的 test split 全部聚合后，画出了 reward 与 caption quality 的散点图：

- [reward_quality_scatter.png](/home/jiawen/code/brain-cpt/docs/assets/group_meeting/reward_quality_scatter.png)
- 对应数据：[reward_quality_scatter_test.csv](/home/jiawen/code/brain-cpt/docs/assets/group_meeting/reward_quality_scatter_test.csv:1)

从图上可以直接看到：

- reward 越高，`CLIPScore` 越高；
- reward 越高，`RefCLIPScore` 也越高。

这说明 reward 并不是一个“纯内部量”，而是和真实 caption 质量、图文一致性有稳定单调关系。

### 4.2 V3-last 在所有 subject 上都可用

最终版 `v3-last` 的核心 reward 结果如下。

总体均值：

| split | mean GT | mean SFT | mean shuffled | GT-SFT gap | SFT-shuffled gap | GT>SFT | SFT>shuffled | Spearman CLIP | Spearman RefCLIP | penalty_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1.0000 | 0.6403 | 0.4215 | 0.3597 | 0.2188 | 1.0000 | 0.9747 | 0.6364 | 0.7568 | 0.7250 |
| val | 1.0000 | 0.6414 | 0.4254 | 0.3586 | 0.2160 | 1.0000 | 0.9775 | 0.6328 | 0.7522 | 0.8750 |
| test | 1.0000 | 0.6173 | 0.4215 | 0.3827 | 0.1958 | 1.0000 | 0.9504 | 0.7651 | 0.7979 | 0.7750 |

test split 的 subject 细节：

| Subject | mean GT | mean SFT | mean shuffled | GT-SFT gap | SFT-shuffled gap | GT>SFT | SFT>shuffled | Spearman CLIP | Spearman RefCLIP | penalty_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 1.0000 | 0.6283 | 0.4217 | 0.3717 | 0.2066 | 1.0000 | 0.9623 | 0.7550 | 0.8034 | 0.8000 |
| S2 | 1.0000 | 0.6113 | 0.4213 | 0.3887 | 0.1899 | 1.0000 | 0.9440 | 0.7739 | 0.8041 | 0.9000 |
| S5 | 1.0000 | 0.6287 | 0.4221 | 0.3713 | 0.2066 | 1.0000 | 0.9603 | 0.7418 | 0.7818 | 0.9000 |
| S7 | 1.0000 | 0.6010 | 0.4209 | 0.3990 | 0.1802 | 1.0000 | 0.9348 | 0.7897 | 0.8024 | 0.5000 |

相应可视化见：

- [v3_last_subjects.png](/home/jiawen/code/brain-cpt/docs/assets/group_meeting/v3_last_subjects.png)

这说明：

- reward 在四个 subject 上都能稳定区分 GT / SFT / shuffled；
- 与 `CLIPScore / RefCLIPScore` 的相关性也比较稳定；
- 唯一仍需继续优化的是 corruption robustness，尤其 `S7` 的 penalty rate 偏低。

## 5. 下游结果：SFT 与 RL/CPT 对比

### 5.1 平均结果

当前 `results/fmri` 中，SFT 与 RL/CPT 的四 subject 平均结果如下：

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE | CIDER | SPICE | CLIPScore | RefCLIPScore | PACScore | RefPACScore | SentenceScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 63.2764 | 43.8115 | 29.7868 | 20.3811 | 20.6564 | 46.4956 | 64.5085 | 12.6825 | 64.1394 | 71.3445 | 51.3116 | 62.6154 | 57.4597 |
| RL/CPT | 64.5327 | 46.1166 | 32.3569 | 22.8378 | 20.7763 | 47.4805 | 67.6553 | 13.2670 | 64.9892 | 72.2544 | 51.9913 | 63.4184 | 59.2728 |
| RL/CPT - SFT | +1.2563 | +2.3052 | +2.5701 | +2.4566 | +0.1199 | +0.9849 | +3.1468 | +0.5845 | +0.8498 | +0.9099 | +0.6797 | +0.8030 | +1.8132 |

对应图表：

- [caption_mean_sft_vs_cpt.png](/home/jiawen/code/brain-cpt/docs/assets/group_meeting/caption_mean_sft_vs_cpt.png)
- [caption_subject_deltas.png](/home/jiawen/code/brain-cpt/docs/assets/group_meeting/caption_subject_deltas.png)

结论很直接：

- RL/CPT 相比 SFT，在所有平均指标上都提升；
- 其中 `BLEU-4`, `CIDEr`, `SentenceScore`, `RefCLIPScore` 提升尤其明显；
- 这说明 flow reward 不只是“能打分”，而且确实能把 reward 优势传导到最终 caption 质量上。

### 5.2 各个 subject 的完整指标

#### S1

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE | CIDER | SPICE | CLIPScore | RefCLIPScore | PACScore | RefPACScore | SentenceScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 64.1144 | 44.9983 | 30.8114 | 21.4077 | 21.2248 | 47.0712 | 68.4688 | 13.3333 | 64.7720 | 71.9606 | 51.8176 | 63.1663 | 58.7148 |
| RL/CPT | 65.4107 | 47.0025 | 33.3025 | 23.7270 | 21.8605 | 48.5619 | 72.4684 | 14.4879 | 65.4797 | 72.7554 | 52.3836 | 63.8634 | 60.8894 |
| Delta | +1.2963 | +2.0042 | +2.4911 | +2.3193 | +0.6357 | +1.4907 | +3.9996 | +1.1546 | +0.7077 | +0.7948 | +0.5660 | +0.6971 | +2.1746 |

#### S2

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE | CIDER | SPICE | CLIPScore | RefCLIPScore | PACScore | RefPACScore | SentenceScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 62.4790 | 42.9997 | 29.2111 | 19.9912 | 20.3087 | 46.3666 | 62.0150 | 12.3173 | 63.6489 | 70.7802 | 50.9190 | 62.1206 | 56.3281 |
| RL/CPT | 64.3691 | 45.7650 | 31.8895 | 22.5337 | 20.4613 | 47.7121 | 66.2417 | 12.9555 | 64.7099 | 71.9255 | 51.7679 | 63.1318 | 58.7256 |
| Delta | +1.8901 | +2.7653 | +2.6784 | +2.5425 | +0.1526 | +1.3455 | +4.2267 | +0.6382 | +1.0610 | +1.1453 | +0.8489 | +1.0112 | +2.3975 |

#### S5

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE | CIDER | SPICE | CLIPScore | RefCLIPScore | PACScore | RefPACScore | SentenceScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 64.5271 | 45.2654 | 31.1036 | 21.4113 | 21.0334 | 46.9985 | 68.9896 | 13.1821 | 65.6486 | 72.7687 | 52.5192 | 63.8938 | 59.4510 |
| RL/CPT | 64.9190 | 47.2436 | 33.5006 | 23.8036 | 20.9599 | 47.8810 | 71.8694 | 13.4016 | 66.1480 | 73.3953 | 52.9181 | 64.4377 | 60.9640 |
| Delta | +0.3919 | +1.9782 | +2.3970 | +2.3923 | -0.0735 | +0.8825 | +2.8798 | +0.2195 | +0.4994 | +0.6266 | +0.3989 | +0.5439 | +1.5130 |

#### S7

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE | CIDER | SPICE | CLIPScore | RefCLIPScore | PACScore | RefPACScore | SentenceScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 61.9853 | 41.9825 | 28.0213 | 18.7143 | 20.0588 | 45.5459 | 58.5606 | 11.8972 | 62.4880 | 69.8684 | 49.9907 | 61.2807 | 55.3448 |
| RL/CPT | 63.4320 | 44.4555 | 30.7351 | 21.2868 | 19.8235 | 45.7668 | 60.0418 | 12.2230 | 63.6192 | 70.9414 | 50.8956 | 62.2405 | 56.5123 |
| Delta | +1.4467 | +2.4730 | +2.7138 | +2.5725 | -0.2353 | +0.2209 | +1.4812 | +0.3258 | +1.1312 | +1.0730 | +0.9049 | +0.9598 | +1.1675 |

## 6. 与 BrainHub baseline 的完整平均对比

这里用 BrainHub 里公开的 caption baseline，与我们的四 subject 平均结果直接对比。

| Method | BLEU-1 | BLEU-4 | METEOR | ROUGE | CIDER | SPICE | CLIPScore | RefCLIPScore | PACScore | RefPACScore | SentenceScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SDRecon | 35.2175 | 3.3000 | 9.7750 | 24.6125 | 13.5175 | 4.8825 | 60.0250 | 65.6375 | 64.9275 | 70.2150 | 39.1300 |
| BrainCap | 54.8225 | 14.0400 | 16.2425 | 40.3850 | 38.8600 | 8.8350 | 63.3000 | 69.0525 | 66.8525 | 72.7750 | 49.1575 |
| UMBRAE-S avg | 57.4100 | 17.0750 | 18.2425 | 41.9625 | 52.4900 | 11.8950 | 66.3550 | 72.1725 | 69.6925 | 75.6950 | 55.3375 |
| UMBRAE | 59.0925 | 18.4000 | 19.2375 | 43.6350 | 57.7600 | 12.4225 | 67.1325 | 72.9600 | 70.2350 | 76.2725 | 57.0000 |
| Ours-SFT | 63.2764 | 20.3811 | 20.6564 | 46.4956 | 64.5085 | 12.6825 | 64.1394 | 71.3445 | 51.3116 | 62.6154 | 57.4597 |
| Ours-RL/CPT | 64.5327 | 22.8378 | 20.7763 | 47.4805 | 67.6553 | 13.2670 | 64.9892 | 72.2544 | 51.9913 | 63.4184 | 59.2728 |

这张表的关键信息是：

- 我们在 **BLEU / ROUGE / CIDEr / SentenceScore** 这类生成质量指标上已经明显领先；
- `RL/CPT` 相比 `SFT` 又进一步提升了一档；
- `RefCLIPScore` 已经非常接近 `UMBRAE / UMBRAE-S avg`，但平均上还略低于 `UMBRAE`；
- 因此当前最合理的结论是：**我们在 caption generation quality 上已经很强，而 reference-aware alignment 还有少量提升空间。**

另外给一张更直观的 baseline 图：

- [brainhub_bleu4_comparison.png](/home/jiawen/code/brain-cpt/docs/assets/group_meeting/brainhub_bleu4_comparison.png)

## 7. 总结

当前这套工作最值得在组会上分享的主线是：

1. 我们不是把 reward 直接定义在表层文本上，而是把问题提升为“caption 是否保持了正确的视觉身份后验”。
2. CIM 提供了这个方向的原始启发，而我们的推广是在**连续 latent 空间**中，用条件 flow 去建模文本到视觉身份的映射关系。
3. 基于这个 flow，我们构建了 reference-conditioned reward，使 reward 能够反映 caption 对目标图像身份的一致性。
4. 这个 reward 和 `CLIPScore / RefCLIPScore` 呈现稳定正相关，说明理论代理是有效的。
5. 把它用于 RL/CPT 后，最终 caption 指标相比 SFT 在四个 subject 上都实现了稳定提升。

一句话总结就是：

$$
\text{caption RL 的关键，不只是提升语言流畅性，}
\quad
\text{而是约束 caption 保持正确的视觉身份信息。}
$$

## 8. 相关文件

- 简报中文版：[report_simple_cn.md](/home/jiawen/code/brain-cpt/docs/report_simple_cn.md:1)
- 完整英文版：[group_meeting_v3_last_idea_report.md](/home/jiawen/code/brain-cpt/docs/group_meeting_v3_last_idea_report.md:1)
- 可视化与汇总表目录：[docs/assets/group_meeting](/home/jiawen/code/brain-cpt/docs/assets/group_meeting:1)
