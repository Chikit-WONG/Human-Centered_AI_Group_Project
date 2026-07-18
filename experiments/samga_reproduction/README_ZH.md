# 经审计的 SAMGA 公共代码复现尝试

[English](README.md) | 简体中文

本实验是对 SAMGA 的 THINGS-EEG2 被试内检索结果所做的尽力数值复现。
实验运行发布的训练代码，且不编辑 SAMGA 源代码树，但这并非论文级精确
复现：论文和代码仓库都没有指明视觉检查点、发布特征提取器、定义层/词元
语义，也没有公开五个随机种子的取值。因此，复现结果仍然是经过审计且
明确依赖假设的结果。

本实验独立于
[`experiments/samga_lora`](../samga_lora/README.md)；后者是一项使用
CLIP ViT-B/32、控制了泄漏的 Frozen 与 LoRA 归因研究。本实验也不会取代
课程项目原有的 CLIP/LoRA 主要结果。

## 范围与公开材料中的不确定性

本地论文为 arXiv v1，官方源码固定在干净提交
`1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1`。截至 2026-07-18，该代码仓库
没有 release、检查点、预计算特征下载、特征提取脚本或已公布的五随机种子
列表。尚未解决的
[特征版本问题](https://github.com/LinJiang8/SAMGA/issues/1) 没有得到作者答复。

论文中的若干表述与公开启动器不同：

| 项目 | 论文 | 发布的 `intra.sh` / 启动器所用的代码默认值 |
|---|---|---|
| 批量大小 | 1024 | 512 |
| 随机种子 | 对五个未公开种子取均值 | 仅 seed 2025 |
| 检索温度 | 可学习 | 固定，除非添加 `--t_learnable` |
| 层丢弃 | 已应用；概率未公开 | `0.0` |
| 训练相似度 | 余弦 | 训练时仅对图像侧执行 L2 归一化 |
| 早停 | 已应用；规则未公开 | patience 10，由正式测试集 Top-1 驱动 |

PDF 本身从未提及 InternViT。代码仓库的目录约定暗示使用第
20/24/28/32/36 层的五个 3,200 维层特征，但无法确定模型版本、检查点、
隐藏状态偏移、词元池化、LayerNorm 前/后的表示或图像处理器。正因存在
这些缺口，以下协议被标记为推断所得，而不是作者确认。

## 固定的源码、模型、数据与特征协议

下载的模型资产仅存放在用户指定的模型根目录中：

```text
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models
```

所选模型为：

```text
OpenGVLab/InternViT-6B-448px-V2_5
revision 9d1a4344077479c93d42584b6941c64d795d508d
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/InternViT-6B-448px-V2_5/9d1a4344077479c93d42584b6941c64d795d508d
```

`model_provenance.json` 的 SHA-256 为
`56f4218ae3f636e32521a719808767313c093a3d85a78053bfb1404d1463342c`；
三个官方权重分片的 SHA-256 分别为
`9818659d13d932da8bc0c3b8ee15f5b5d68d8c94d66eb525be566066630111da`、
`4f0c10e72d6f6513f421baa6ec843d5508657435059c1d18b6b5fd7789f9d5b7` 和
`d21c4fe0bc4af1425cfae1a59a8f5fbb00fde9d8e2888325a60913ac61b0494d`。
失效即关闭的下载设计见 [DOWNLOADER_SAFETY.md](DOWNLOADER_SAFETY.md)，
提取和验证命令见
[V2_5_FEATURE_PIPELINE.md](V2_5_FEATURE_PIPELINE.md)。

所选特征假设如下：

- 使用固定的处理器，将图像缩放并中心裁剪至 448 像素；
- ImageNet 均值 `[0.485, 0.456, 0.406]` 和标准差
  `[0.229, 0.224, 0.225]`；
- 实际的第 20、24、28、32 和 36 个块输出；
- 对 patch tokens 取均值，排除 CLS；
- 输入 SAMGA 前不执行额外的逐向量 LayerNorm 或 L2 归一化；
- 磁盘缓存为 float16，由训练代码加载为 float32。

所得特征溯源信息的 SHA-256 为
`d12c29387738cdd76fedd547221e33ada2db2fe12c123be7ef904e3f58732fb1`。
特征选择先在 Subject 08 上筛选，再在 Subjects 01 和 05 上检查，因此即使
是固定 epoch 的结果也属于探索性结果，而不是前瞻性锁定的复现。

本次运行没有下载数据集，而是复用以下路径中已存在的、经过 MVNN 白化的
THINGS-EEG2 文件：

```text
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/EEG_Recon-RL/datasets/things_eeg_data/Preprocessed_data_250Hz_whiten
```

这些文件在不重新归一化的情况下被转换为 SAMGA 的布局。源 `.pt` 训练/测试
张量的形状为 `[16540,4,63,250]` 和 `[200,80,63,250]`。转换后，SAMGA
`.npy` 张量的形状为 `[1654,10,4,63,250]` 和 `[200,1,80,63,250]`。
源数据和转换后数据的存储格式均为 float16；发布的加载器会将数组转换为
float32。生成的数据溯源信息 SHA-256 为
`b76b3db21010d4a55da7855e2e6cd3a00c4439c8855b645d09d02bfbd5e46463`。

训练使用现有的 `test` 环境：Python 3.10.18、PyTorch 2.10.0+cu126、
TorchVision 0.25.0+cu126、Transformers 4.57.6、Timm 1.0.26、
NumPy 1.26.4、SciPy 1.15.3 和 scikit-learn 1.7.2。

## 指标定义与检查点选择边界

所有结果均采用标准的独立 200 路余弦检索，不使用匈牙利匹配。

- **实际最终/停止 epoch：** 发布的 CSV 中的 `top1 acc` 和 `top5 acc`。
  只有在全部 60 个 epoch 均已完成时，它们才是固定 epoch-60 指标。
- **逐 epoch 测试集选择的诊断指标：** 发布的代码在每个 epoch 都评估正式
  测试集，并选择测试 Top-1 最高的 epoch；若测试 Top-1 完全相同，则以较低
  测试损失打破平局。其 Top-5 是该 Top-1 所选 epoch 的配套值，而不是独立
  最大化所得的值。
- **Patience-10 终点：** 发布的早停规则同样由正式测试集 Top-1 驱动。
  因此，停止 epoch 的指标也存在测试泄漏；它不是由验证集选择的。

论文的 91.30% Top-1 和 98.80% Top-5 是参考值，并非这一公开协议未完整
指定情况下的真实标准。

## 已验证结果

对于 seed 2025、batch 512、60 epochs 且禁用早停的协议，全部十名被试均
已完成，并通过 CSV 与日志以及配置之间的审计：

| Seed-2025 协议 | Top-1 | Top-5 | Top-1 差距 | Top-5 差距 |
|---|---:|---:|---:|---:|
| Epoch 60，不早停 | 89.55% | 98.65% | -1.75 points | -0.15 points |
| 测试集选择的 epoch 诊断指标 | 91.95% | 98.95% | +0.65 points | +0.15 points |

第一行的跨被试样本标准差分别为 3.90 和 1.36 points；第二行则为 4.04 和
1.40 points。所选 epoch 为 40.0 ± 11.96（均值 ± 跨被试样本标准差）。
绝不能将第二行呈现为无泄漏的最终项目估计。

### 与发布启动器兼容的 seed-2025 验证

batch-512、patience-10 的验证完成了全部十名被试：

| 指标解释 | Top-1 均值 ± 跨被试 SD | Top-5 均值 ± 跨被试 SD | Top-1 差距 | Top-5 差距 |
|---|---:|---:|---:|---:|
| 实际停止/最终 epoch | 88.95% ± 4.78 | 98.90% ± 1.26 | -2.35 points | +0.10 points |
| 测试集选择的诊断指标 | 91.50% ± 4.00 | 98.75% ± 1.46 | +0.20 points | -0.05 points |

Subjects 01–10 的实际停止 epoch 分别为 34、31、40、30、33、47、60、30、
32 和 37。其均值为 37.4 ± 9.55，范围为 30–60。停止规则监控正式测试集
Top-1，因此终点行以测试集为条件；所选行则在同一测试集上进一步进行了
直接的最佳 epoch 选择。

### 项目自定五随机种子稳定性网格

Seeds 42–46 由本项目自行定义，因为论文没有公开其五个随机种子的取值。
全部 50 个单元都在禁用早停的情况下完成了 60 个 epoch：

| Seed | Epoch-60 Top-1 | Epoch-60 Top-5 | 测试集选择的 Top-1 | 配套 Top-5 |
|---:|---:|---:|---:|---:|
| 42 | 88.75% | 98.80% | 91.85% | 98.70% |
| 43 | 89.10% | 98.95% | 91.70% | 98.75% |
| 44 | 89.40% | 98.85% | 91.65% | 99.10% |
| 45 | 88.55% | 98.90% | 91.75% | 98.95% |
| 46 | 89.30% | 98.85% | 92.15% | 98.85% |
| 均值 ± seed-level SD | **89.02% ± 0.36** | **98.87% ± 0.06** | **91.82% ± 0.20** | **98.87% ± 0.16** |

先对每个 seed 的十名被试取宏平均；表中显示的 SD 是这五个 seed-level
均值之间的样本标准差，并非将 50 个被试–种子单元汇总后的标准差。相对于
论文主要结果，epoch-60 的差距是 -2.28/+0.07 points，测试集选择结果的
Top-1/Top-5 差距是 +0.52/+0.07 points。

详细的本地报告生成于：

```text
results/samga_reproduction
```

Git 有意忽略该目录；因此，为了让远程 GitHub 读者可以查看，这里内嵌了
核心结果和边界说明。

## 复现流程

从项目根目录运行命令。首先按照
[DOWNLOADER_SAFETY.md](DOWNLOADER_SAFETY.md) 操作，然后使用
[V2_5_FEATURE_PIPELINE.md](V2_5_FEATURE_PIPELINE.md) 提取并验证固定的
特征。准备现有 EEG 资产，并将所选训练/测试特征缓存转换为发布的 SAMGA
加载器所需的五层文件名：

```bash
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
python experiments/samga_reproduction/prepare_official_assets.py eeg \
  --source-root /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/EEG_Recon-RL/datasets/things_eeg_data/Preprocessed_data_250Hz_whiten \
  --output-root "$PWD/artifacts/samga_reproduction/data/preprocessed_eeg" \
  --subjects 1 2 3 4 5 6 7 8 9 10

python experiments/samga_reproduction/prepare_official_assets.py features \
  --train-cache "$PWD/artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/variants/train_idx0_patch_mean/features.npy" \
  --test-cache "$PWD/artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/variants/test_idx0_patch_mean/features.npy" \
  --output-dir "$PWD/artifacts/samga_reproduction/features/v2_5_idx0_patch_mean_raw" \
  --normalization none
```

运行一个固定 60 epoch 的被试：

```bash
SAMGA_VARIANT=v2_5_idx0_patch_mean_raw \
SAMGA_FEATURE_ROOT="$PWD/artifacts/samga_reproduction/features/v2_5_idx0_patch_mean_raw" \
SAMGA_SEED=2025 \
SAMGA_BATCH_SIZE=512 \
SAMGA_EARLY_STOP_PATIENCE=0 \
SAMGA_NUM_EPOCHS=60 \
bash experiments/samga_reproduction/run_official_cell.sh 8
```

使用 `official_array.slurm` 运行 Subjects 01–10。请在执行 `sbatch` 前创建
日志目录，因为 SLURM 会在脚本启动前打开输出文件。显式导出主协议的每个
值：

```bash
mkdir -p logs/samga_reproduction
sbatch --array=0-9%2 \
  --export=ALL,SAMGA_VARIANT=v2_5_idx0_patch_mean_raw,SAMGA_FEATURE_ROOT="$PWD/artifacts/samga_reproduction/features/v2_5_idx0_patch_mean_raw",SAMGA_SEED=2025,SAMGA_BATCH_SIZE=512,SAMGA_EARLY_STOP_PATIENCE=0,SAMGA_NUM_EPOCHS=60 \
  experiments/samga_reproduction/official_array.slurm
```

对于与发布启动器兼容的早停验证，使用单独的 variant
`v2_5_idx0_patch_mean_raw_launcher_p10`，将其指向相同的特征根目录，并将
`SAMGA_EARLY_STOP_PATIENCE=10`。训练前，运行器会拒绝格式错误的被试、
随机种子、variant、开关值和缺失的资产。

在聚合前创建隔离的结果快照。聚合器会拒绝重复项、符号链接、格式错误的
CSV、意外单元以及不完整的预期矩阵：

```bash
python experiments/samga_reproduction/aggregate_official_results.py \
  --input-root /absolute/path/to/clean/source_cells \
  --output-dir /absolute/path/to/clean/aggregate \
  --expected-subjects 1-10 \
  --expected-seeds 2025
```

发布的 SAMGA 源代码树必须在提交 `1a63745...` 上保持干净。生成的 EEG
布局、特征、检查点和运行输出位于 `artifacts/samga_reproduction`；整理后的
本地报告位于 `results/samga_reproduction`；下载的权重仅存放在
`EEG_Project/models`。

## 仅用于敏感性分析的开关

以下三个开关默认关闭。报告的主协议还显式设置其 variant、特征根目录、
batch 512、patience、seed 和 60-epoch 上限；运行器的通用 batch 默认值
并非报告所用协议。这些可选择启用的开关会开放发布的 `train.py` 已支持的
消融：

| 环境变量 | 默认值 | 接受的值 | 启用时附加的 CLI |
|---|---:|---|---|
| `SAMGA_EEG_L2NORM` | `0` | `0` 或 `1` | `--eeg_l2norm` |
| `SAMGA_T_LEARNABLE` | `0` | `0` 或 `1` | `--t_learnable` |
| `SAMGA_ROUTER_LAYER_DROPOUT` | `0` | `[0, 1)` 内的有限数值 | 非零时为 `--router_layer_dropout VALUE` |

每个协议都应使用不同的 `SAMGA_VARIANT`。例如：

```bash
SAMGA_VARIANT=v2_5-patch-eegl2-tlearn-rdrop025 \
SAMGA_FEATURE_ROOT=/absolute/path/to/v2_5-patch-features \
SAMGA_EEG_L2NORM=1 \
SAMGA_T_LEARNABLE=1 \
SAMGA_ROUTER_LAYER_DROPOUT=0.25 \
bash experiments/samga_reproduction/run_official_cell.sh 8
```

## 局限性

- 模型版本和特征语义是推断所得，因为作者没有将其公布。InternViT V2.5
  与 patch-mean 语义均为推断，并非作者确认。数值接近不能证明配置相同。
- V2.5 patch-mean 语义是在有限的被试筛选后选定的。
- Seed 2025 是发布启动器仅有的 seed。任何 42–46 网格均由项目自行定义，
  因为论文未公开其五个 seed 值。
- 逐 epoch 的最佳选择和公开的 patience-10 早停规则都会检查正式测试集。
  测试集选择的指标在每个 epoch 检查正式测试集；patience-10 终点也以
  测试集为条件，因为早停监控正式测试集 Top-1。
- 论文与启动器在若干重要超参数上存在分歧。
- GPU 重训练已设置随机种子，但不保证逐位确定性。
- 试次平均解码不是单试次解码。
