# NICE / ATM-S / Our project 匹配公平性实验

[English](README.md) | 中文

本实验在 THINGS-EEG2 上固定比较 **NICE**、**ATM-S** 与
**Our project（BrainRW）**。正式交付单元严格限定为 `sub-08 / seed-42`，
不是全被试或多随机种子 benchmark。

本文档不提前填写任何正式分数。密封实验完成后，自动生成的中英文报告分别位于
下文结果根目录中的 `aggregate/RESULTS_ZH.md` 与 `aggregate/RESULTS.md`。

## 研究问题与报告边界

标准论文指标会让每个 EEG 查询独立地在图库中排序。其余四种 decoder 会同时查看
完整的“查询 × 图库”分数矩阵，因此属于**传导式、整批级分析**，不能替代标准检索。
本实验检验：当测试集已知的一对一结构被“缺失 EEG、缺失图片、重复图库条目、同一
图片对应多个真实 EEG 测量”削弱后，这些方法看似获得的提升是否仍然存在。

在同一个“模型–场景”中，五种 decoder 必须读取完全相同的相似度矩阵和身份映射。
真实 target 只在解码结束后用于评分，不允许进入匹配算法。正式主指标仍是
Independent Top-1/Top-5；受约束匹配结果单独报告。

由于密封范围只有一个 subject-seed 单元，结论必须写成“**在 sub-08 / seed-42 上
观察到**”。可以报告精确计数、百分比和逐查询转移，但不能声称存在跨被试或跨 seed
的统计显著性。

## 模型、源码锁定与 checkpoint 封存

NICE 与 ATM-S 来自官方
[`dongyangli-del/EEG_Image_decode`](https://github.com/dongyangli-del/EEG_Image_decode)
仓库的 `develop` 分支。预检会抓取当时的分支头，将工作树切换到 detached HEAD，
要求源码树保持干净，并把远程 URL、分支、精确 commit 与已跟踪源码树 SHA-256 写入：

```text
test/brain-rw/results/matching_fairness_v3/manifests/upstream_lock.json
```

后续所有原生训练和导出阶段都会验证该锁。流程不会静默地用本地 NICE/ATM-S 实现
替换已经锁定的官方源码。

NICE 与 ATM-S 保留论文原生训练配置，只采用一项预先声明的协议修正：正式
checkpoint 依据最低 **validation contrastive loss** 选择（`val_ratio=0.1`、
patience `10`；loss 完全相同时选择更早的 epoch）。在 checkpoint、配置与哈希冻结
之前，正式测试集保持封存；测试集 Top-1/Top-5 不参与正式选模。

原生导出器还会生成 `best_test_audit.json`。它是明确经过测试集选模的**复现诊断**，
与 30 个公平性场景隔离，绝不能称为无偏正式结果。Our project 使用已有的固定正式
BrainRW checkpoint 与经过审计的评估配置，本实验不会重新为它选轮次。

进入匹配阶段前，三个模型都必须在标准 `200 x 200` 矩阵上通过“原生 evaluator 与
统一 evaluator 的 Independent Top-1/Top-5 完全一致”门禁。源码、哈希、身份顺序、
shape、有限值或 parity 任一不一致都会 fail closed。

## 场景套件

### 标准套件：27 个场景

标准 artifact 对每张图片的全部 80 个 trial 求平均。流程用 seed `42` 生成一份
canonical manifest，并由三个模型共同复用：

```text
drop_query        in {0, 5, 10}
drop_gallery      in {0, 5, 10}
drop_pair         = 0
duplicate_gallery in {0, 10, 20}
```

因此每个模型恰好有 `3 x 3 x 1 x 3 = 27` 个场景。`drop_query` 会留下没有配对 EEG
的真实图片作为干扰项；`drop_gallery` 会产生“正确图片不在图库中”的 EEG 查询；
`duplicate_gallery` 会增加一个新的图库 entry，但它与原图共享 canonical image ID。

### 真实重复 EEG 套件：3 个场景

重复查询套件不会复制 EEG 行。每张图片在四个 session 中各有 20 条真实 trial。
确定性的 SHA-256 排序会执行严格的**每个 session 内 10/10 拆分**：10 条 trial 分到
half A，另外 10 条分到 half B。因此 EEG-A 与 EEG-B 都是由 40 条真实 trial 得到的平均值，二者互不重叠，
而且 session-balanced。

- `dupq0`：200 条 EEG-A 查询 × 200 张图片；
- `dupq10`：追加 10 条对应的 EEG-B 查询，得到 `210 x 200`；
- `dupq20`：追加 20 条对应的 EEG-B 查询，得到 `220 x 200`。

10 个重复身份严格是 20 个重复身份的子集。这三个 40-trial 场景属于鲁棒性套件，
其绝对分数不能与 80-trial 标准套件直接比较。严格一对一分配在 `dupq10` 上的结构性
Top-1 上限为 `200/210 = 95.24%`，在 `dupq20` 上为 `200/220 = 90.91%`，所以必须
同时报告 unmatched 数量。

三个模型的完整实验共有 `3 x (27 + 3) = 90` 个“模型–场景”单元。

## 五种 decoder 的语义

| Decoder | 语义 | Top-5 |
|---|---|---|
| Independent | 对每一行稳定排序；允许多个查询复用同一图库条目 | 有定义并报告 |
| Greedy | 按 independent Top-1 置信度处理查询，再选择尚未占用的最高分条目 | 无定义 |
| Hungarian | 用 `linear_sum_assignment` 最大化全局总分；矩形矩阵只匹配 `min(Q,G)` 对 | 无定义 |
| Stable Matching | query-proposing Gale-Shapley，并采用确定性的偏好排序与 tie-breaking | 无定义 |
| Sinkhorn | temperature `0.05`、最多 `500` 次迭代、tolerance `1e-8` 的 transport plan，再逐行 argmax；不是严格一对一 | 无定义 |

Greedy、Hungarian 与 Stable Matching 只为已匹配查询产生一个 assignment；Sinkhorn
从 transport plan 中为每行产生一个 argmax。它们都没有输出包含五个候选项的排序表，
所以“assignment Top-5”属于人为制造的指标，本实验不会报告。只有 Independent 报告
标准 Top-5。Sinkhorn 若未收敛，结果仍保留，但必须明确标记，不能隐藏。

## 环境配置

从仓库根目录运行：

```bash
cd /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
```

环境声明使用显式官方 URL `https://conda.anaconda.org/conda-forge` 并加入
`nodefaults`，因此用户级 channel alias 无法把依赖求解重定向到镜像，而且本环境
不会用 Anaconda Defaults 求解或下载依赖。Conda 26 在读取文件前仍可能为全局配置
频道查询 notices/ToS 元数据；命令不会启用或执行 ToS 接受。Conda 只创建隔离的
Python 3.12/pip 骨架；构建工具链另固定 setuptools 75.8.0，因为锁定的 CLIP
setup 仍会 import `pkg_resources`。同一条
环境创建命令随后会安装精确版本的 pip 依赖。PyTorch 三件套从官方 CUDA 12.4
wheel 索引严格固定为 `2.5.0+cu124` / `0.20.0+cu124` / `2.5.0+cu124`，OpenAI
CLIP 则固定到 commit `a9b1bf5920416aaeaec965c25dd9e8f98c864f16`。命令关闭 pip
build isolation，让 CLIP 使用该锁定工具链，并仅在 pip 调用系统 Git 客户端时清除
继承的 `LD_LIBRARY_PATH`，避免 Conda 的 libffi 被注入 Git；它不会修改任何已有环境。

首次创建独立原生环境：

```bash
PIP_NO_BUILD_ISOLATION=1 env -u LD_LIBRARY_PATH conda env create -n atm_native \
  -f experiments/matching_fairness/configs/atm_native_environment.yml
```

如果 `atm_native` 已经存在，则按声明更新，而不是再次创建：

```bash
PIP_NO_BUILD_ISOLATION=1 env -u LD_LIBRARY_PATH conda env update -n atm_native --prune \
  -f experiments/matching_fairness/configs/atm_native_environment.yml
```

仅当命令在 Git 获取锁定 CLIP commit 时，因相同的 GitHub TLS/传输错误反复失败，
才使用下面已预取的精确 checkout。先确认 HEAD 等于预期 commit，且工作区干净：

```bash
git -C /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/openai_clip_a9b1bf5920416aaeaec965c25dd9e8f98c864f16_shallow \
  rev-parse HEAD
git -C /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/openai_clip_a9b1bf5920416aaeaec965c25dd9e8f98c864f16_shallow \
  status --short
```

`rev-parse` 必须输出 `a9b1bf5920416aaeaec965c25dd9e8f98c864f16`，而
`status --short` 必须没有输出。然后在不访问包索引的情况下，仅安装这个缺失的
VCS 包：

```bash
PIP_NO_INDEX=1 PIP_NO_BUILD_ISOLATION=1 env -u LD_LIBRARY_PATH \
  conda run -n atm_native python -m pip install \
  --no-build-isolation --no-deps \
  "git+file:///hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/openai_clip_a9b1bf5920416aaeaec965c25dd9e8f98c864f16_shallow@a9b1bf5920416aaeaec965c25dd9e8f98c864f16"
```

本次 `atm_native` 在 Conda 骨架及声明的科学计算 wheels 成功安装后，通过该
fallback 完成；本次远程 VCS 步骤本身没有稳定地完整成功。

然后运行精确版本与 import 门禁：

```bash
conda run -n atm_native python \
  experiments/matching_fairness/scripts/preflight.py --environment-only
```

不要把这些依赖安装进已有的 `test` 或 `eeg_recon` 环境，也不要更新这两个环境。
NICE/ATM-S 原生作业激活 `atm_native`；BrainRW 导出与 CPU 匹配/汇总则由各自的
SLURM 脚本使用已经存在的 `eeg_recon`。

## 预检、dry-run 与正式提交

wrapper 会在本地运行正式预检，所以使用前先激活原生环境：

```bash
conda activate atm_native
```

先执行源码、数据、环境与来源预检，不提交作业：

```bash
bash experiments/matching_fairness/run_matching_fairness.sh --phase preflight
```

在不创建运行目录、不提交作业的前提下渲染完整固定 DAG：

```bash
bash experiments/matching_fairness/run_matching_fairness.sh \
  --phase all --dry-run
```

dry-run 必须显示恰好 **2 个训练单元、3 个主模型导出、2 个原生 audit 单元、90 个
场景和 450 条 decoder 记录**。正式依赖图为：

```text
preflight -> 原生训练 array -> 原生 main export array
preflight ------------------> BrainRW main export
原生 main exports + BrainRW export -> 原生 audit array
原生 audits -> matching -> aggregation
```

核对 dry-run 后，只提交一次完整 DAG：

```bash
bash experiments/matching_fairness/run_matching_fairness.sh \
  --phase all --submit
```

提交 ledger 会在每次 `sbatch` 调用前持久写入 job ID 与依赖关系。重复执行
`all --submit` 会被有意拒绝，它不是恢复命令。

## 分阶段执行与恢复语义

如果要有意识地分阶段执行，应先完成 preflight，再提交原生训练：

```bash
bash experiments/matching_fairness/run_matching_fairness.sh \
  --phase train --submit
```

等两个 validation-selected checkpoint 都完整且与当前输入一致后，再提交 3 个 main
export 和 2 个原生 audit 单元：

```bash
bash experiments/matching_fairness/run_matching_fairness.sh \
  --phase export --submit
```

这些作业成功后，可以在本地运行或恢复带哈希绑定的 CPU 阶段：

```bash
bash experiments/matching_fairness/run_matching_fairness.sh --phase match
bash experiments/matching_fairness/run_matching_fairness.sh --phase aggregate
```

已经完成且输入哈希仍一致的 preflight、matching 与 aggregation 会通过 manifest 验证
后自动跳过。partial、orphaned、哈希不一致、failed 或 unknown 状态都会 fail closed，
恢复前必须检查 ledger 与日志。`--overwrite` 只用于经过审阅后有意识地替换派生产物；
它不是通用重试/恢复开关，也不会清除 submission ledger 历史。

### 经过审计的一次性失败 DAG 恢复

下面的命令只用于已经审阅的 spool-entrypoint 故障，是固定的事故恢复路径；它**不是**
overwrite、resume 或通用重试。流程会在同一个提交锁内验证不可变的原始 ledger，并用
权威 `sacct -X` 记录确认五个精确 root job 都处于失败终态；同时要求 checkpoint、
matrix、run 与 aggregate 根目录不存在或为空。通过后，它会在第一次新 `sbatch` 前先
保留 `manifests/submission_recovery.json`，而原始 `submission.json` 必须逐字节不变。

```bash
bash experiments/matching_fairness/run_matching_fairness.sh \
  --phase all \
  --submit \
  --recover-failed-all \
  --original-request-id 3ae8dc60c2df4166b7d4021f48146487 \
  --original-ledger-sha256 2125615c73c156bea4137c1c764aba6b7893e94cb64d819b6856b8a93b4042be \
  --recovery-reason spool-entrypoint-bug
```

只要 recovery 路径已经存在——即使它为空、格式错误、处于 failed/completed 状态或是
symlink——都永久禁止第二次恢复。调度器记录不匹配、旧作业仍未终止或已经成功、派生
输出根目录不安全、原始 ledger 被修改，都会 fail closed；不得通过删除或编辑任一
ledger 来强行重试。

匹配树完整后，也可以直接执行汇总器：

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python \
  experiments/matching_fairness/scripts/aggregate_results.py \
  --results-root /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3
```

## 结果与日志目录

大型 checkpoint、分数矩阵、逐场景 ledger 与生成的报告均不进入 Git，而是位于：

```text
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/
  manifests/                         # 源码、资产、预检、trial、阶段和提交锁
  checkpoints/{nice,atm_s}/          # validation-selected 原生 checkpoint
  matrices/{nice,atm_s,our_project}/ # standard、eeg_a 与 eeg_b 分数 artifact
  runs/<model>/subj08/seed42/
    standard/                         # 27 对场景 JSON/CSV
    duplicate_eeg/                    # 3 对场景 JSON/CSV
  aggregate/
    RESULTS.md
    RESULTS_ZH.md
    aggregate_metrics.csv
    aggregate_summary.json
    presentation_standard.md
    presentation_duplicate_eeg.md
```

所有调度器 stdout/stderr 文件统一位于：

```text
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/logs/matching_fairness_v3/
```

## 正式提交前验证

集群当前的 `test` 环境没有安装 `pytest`，因此仓库内完整测试与 fixture smoke 使用
已有且兼容的 `eeg_recon` 测试运行器，不对该环境作任何修改。运行：

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest \
  experiments/matching_fairness/tests \
  tests/test_things_trial_selection.py \
  tests/test_train_clip_lora_grad_clip.py -v
```

在声明的原生环境中运行 native-import 子集：

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/atm_native/bin/python -m pytest \
  experiments/matching_fairness/tests/test_native_training.py \
  experiments/matching_fairness/tests/test_native_export.py -v
```

最后确认两个彼此独立的 fixture 运行会生成字节完全一致的 aggregate CSV、summary JSON
与两份报告：

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest \
  experiments/matching_fairness/tests/test_orchestration.py::test_fixture_pipeline_is_byte_stable -v
```
