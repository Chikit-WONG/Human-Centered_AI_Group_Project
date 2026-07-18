# SAMGA x LoRA/TTUR 实验计划

[English](SAMGA_LORA_EXPERIMENT_PLAN_EN.md) | 简体中文

## 主要问题

在完全相同的 CLIP ViT-B/32 视觉骨干下，给 SAMGA 加入秩为 32、采用较小
视觉学习率的 LoRA，能否优于其他设置完全相同的冻结视觉对照？实验仅考察
THINGS-EEG2 的 intra-subject 图像检索。

## 锁定协议

- 标准独立 200-way Top-1/Top-5，不使用匈牙利解码。
- 17 个后枕叶通道并对重复试次取平均。
- 使用 CLIP 第 4、6、8、10、12 个 block；取 CLS token 后通过冻结的 CLIP
  post-layer normalization。
- 保留 SAMGA 的 EEGProject、subject-aware router、线性投影、共享编码器、
  第一阶段 MMD 加双向对比损失，以及第二阶段纯对比损失。
- 按发布的 SAMGA 启动命令，仅在训练对比目标中对图像特征做 L2 归一化，
  不归一化 EEG 特征；评估仍使用余弦相似度。
- 对 q/k/v/out projection 和两层 MLP 线性层使用 rank/alpha 32 LoRA。
- 随机种子为 42--46。测试集在 concept-disjoint 验证集锁定视觉学习率比例
  和统一停止 epoch 以前保持封存。

## 执行门控

1. 验证 manifest、冻结特征缓存一致性、梯度、checkpoint 重载和 debug smoke。
2. 对 sub-01、sub-05、sub-08 和 seeds 42、43 进行 pilot；比较视觉学习率比例
   0.05、0.10、0.20，并共享同一个冻结视觉对照。
3. 只有当最佳验证配置的平均 Top-1 提升至少 0.5 个百分点、六个配对单元中
   至少四个为正、且任一受试者下降不超过 2 个百分点时，才扩展正式实验。
4. 使用全部训练概念重训十名受试者、五个种子，只评估一次测试集，并聚合
   paired difference。
5. 若 CLIP 主实验成功，再用推断的 InternViT-6B-448px-V1-5 特征进行明确标注
   的探索性冻结视觉复现。

## 成功标准

正式实验的平均配对 Top-1 增益至少为 0.5 个百分点，并且 subject/seed 双向
cluster bootstrap 的 95% 置信区间下界大于零；Top-5 下降不得超过 0.2 个百分点。

## 执行记录（2026-07-16）

- 预检、缓存/在线一致性、梯度、重载与 smoke 门控全部通过。
- 六对 pilot 通过，并锁定视觉学习率比例 0.20、epoch 25。
- 完整“十名受试者 x 五个随机种子”网格的 100 个正式单元全部成功完成，
  stderr 为空：50 个 Frozen、50 个 LoRA。
- Frozen：Top-1 76.17% +/- 0.30，Top-5 95.79% +/- 0.32。
- LoRA/TTUR：Top-1 82.68% +/- 0.36，Top-5 97.67% +/- 0.17。
- 配对变化：Top-1 +6.51 个百分点，双向 bootstrap 95% CI
  [+5.22, +7.69]；Top-5 +1.88 个百分点，95% CI [+1.30, +2.50]。
- 预注册确认性成功标准通过，因此已单独开启推断 InternViT 冻结视觉的探索性
  后续实验。
- 探索性后续已完成十名受试者、seed 2025、固定 epoch 60：Top-1 83.05%
  （1661/2000），Top-5 98.00%（1960/2000）。所有 checkpoint 重载和 200 行
  预测审计均通过。由于论文没有披露精确 checkpoint、提取器和层编号语义，
  而且论文报告五个 seed，该数值仍是推断模型的单 seed 诊断结果，不是论文
  91.3%/98.8% 的精确复现。
