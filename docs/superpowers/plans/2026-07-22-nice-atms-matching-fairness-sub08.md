# NICE、ATM-S 与 Our project（sub-08 / seed-42）匹配公平性实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用论文官方原生配置为 NICE、ATM-S 和 Our project 生成 `sub-08 / seed-42` 可审计分数矩阵，完成 27 个标准非一一对应场景和 3 个真实重复 EEG 场景下的五种匹配算法公平比较。

**Architecture:** 在仓库内新增独立的 `experiments/matching_fairness` Python 包，第三方 ATM 代码保持只读并在外部 detached checkout 中按运行时 commit 锁定。训练、正式测试导出和 paper-style best-test audit 分离；所有模型先转换成带显式 canonical ID 的统一 ScoreArtifact，再由同一场景清单和 decoder 实现完成后处理与汇总。

**Tech Stack:** Python 3.12、PyTorch 2.5.0、torchvision 0.20.0、braindecode 0.8.1、NumPy 1.26.4、SciPy 1.15.3、pandas 2.3.3、pytest、SLURM、Bash、Hugging Face CLI。

## Global Constraints

- 工作分支固定为 `ckw`；不创建新的 Git 分支。
- 正式范围固定为 NICE、ATM-S、Our project，且只允许 `sub-08 / seed-42`。
- 不提交或启动 `10 subjects x 5 seeds`。
- NICE 使用官方 `ENCODER=NICE`；ATM-S 使用官方 `ENCODER=ATMS`。
- 官方源为 `https://github.com/dongyangli-del/EEG_Image_decode.git` 的 `develop` 分支；每次正式运行先解析 commit，再 detached checkout 并记录 SHA-1。
- 主 checkpoint 以最低 validation contrastive loss 选择；完全相同时选较早 epoch。
- 固定 `val_ratio=0.1`、`early_stopping_patience=10`、`epochs=500`、`batch_size=1024`、`lr=3e-4`、`ema_decay=0.999`、`logit_scale_type=exp`、`avg_trials=true`、`n_chans=63`、`n_times=250`。
- 训练入口不得接收测试 EEG 或测试图片特征路径；正式测试只在 checkpoint 与配置冻结后运行。
- paper-style best-test 只写入 `best_test_audit_only`，不得进入匹配公平性场景。
- 标准套件为 `drop_query={0,5,10}` x `drop_gallery={0,5,10}` x `drop_pair={0}` x `duplicate_gallery={0,10,20}`，恰好 27 个场景。
- duplicate EEG 套件固定为 `dupq0=200x200`、`dupq10=210x200`、`dupq20=220x200`。
- duplicate EEG 必须来自每个 session 的真实 10/10 不重叠 trial 平均；禁止复制 EEG-A 分数行。
- decoder 固定为 Independent、Greedy、Hungarian、Stable Matching、Sinkhorn。
- Sinkhorn 固定 `temperature=0.05`、`max_iterations=500`、`tolerance=1e-8`。
- 只有 Independent 报告标准 Top-5；其余 decoder 只报告 assignment Top-1。
- 所有模型复用同一 canonical 场景清单和 trial split manifest；ground truth 不得传入 decoder。
- 大型数据、checkpoint、矩阵和逐查询结果写入 `test/brain-rw/results/matching_fairness_v3`，不提交 Git。
- 所有 `.out`、`.err` 和运行日志写入 `test/brain-rw/logs/matching_fairness_v3`。
- 结论只允许写“在 sub-08 / seed-42 上观察到”，不得宣称跨被试或跨随机种子显著性。
- 当前 `test` 与 `eeg_recon` 环境均缺少 `braindecode==0.8.1`；不得原地修改它们，使用隔离环境 `atm_native`。
- 相关确认规格为 `docs/superpowers/specs/2026-07-22-nice-atms-matching-fairness-sub08-design-zh.md`。

---

## 文件结构

实施后新增或修改以下文件；第三方 checkout 和运行产物不纳入 Git：

```text
experiments/matching_fairness/
├── README.md                         # 英文复现和结果说明
├── README_ZH.md                      # 中文复现和结果说明
├── RESULTS.md                        # 完成正式运行后生成的小型英文摘要
├── RESULTS_ZH.md                     # 完成正式运行后生成的小型中文摘要
├── run_matching_fairness.sh          # 固定 sub-08/seed-42 一键入口
├── configs/
│   ├── protocol_sub08_seed42.json    # 唯一正式协议
│   └── atm_native_environment.yml    # 最小论文原生检索环境
├── matching_fairness/
│   ├── __init__.py
│   ├── config.py                     # 协议解析和范围锁
│   ├── provenance.py                 # Git、文件和内容哈希
│   ├── artifacts.py                  # 统一 ScoreArtifact 契约
│   ├── trial_splits.py               # session 内 10/10 拆分
│   ├── native_training.py            # 无测试泄漏的 NICE/ATM-S 训练循环
│   ├── native_export.py              # NICE/ATM-S standard/A/B 导出和 audit
│   ├── scenarios.py                  # 27+3 场景与共享 manifest
│   ├── decoders.py                   # 五种匹配算法
│   ├── evaluation.py                 # 共同和 duplicate EEG 指标
│   └── reporting.py                  # CSV/JSON/中英文报告
├── scripts/
│   ├── fetch_upstream.py
│   ├── fetch_assets.py
│   ├── preflight.py
│   ├── train_native.py
│   ├── export_native_scores.py
│   ├── export_brainrw_scores.py
│   ├── run_scenarios.py
│   ├── aggregate_results.py
│   └── submit_pipeline.py
├── slurm/
│   ├── train_native_array.slurm
│   ├── export_native_array.slurm
│   ├── export_brainrw.slurm
│   └── fairness_cpu.slurm
└── tests/
    ├── test_config.py
    ├── test_provenance.py
    ├── test_artifacts.py
    ├── test_trial_splits.py
    ├── test_native_training.py
    ├── test_native_export.py
    ├── test_scenarios.py
    ├── test_decoders.py
    ├── test_evaluation.py
    ├── test_reporting.py
    └── test_orchestration.py

main/data.py                             # 增加显式 trial-index 平均
scripts/evaluate_retrieval.py            # 增加 standard/A/B 查询导出参数
tests/test_things_trial_selection.py     # BrainRW trial 选择回归测试
.gitignore                               # 仅补充新实验的本地 source/runtime 目录
```

运行时外部路径：

```text
EEG_Project/reference_code/codes_for_papers/EEG_Image_decode/
EEG_Project/models/EEG_Image_decode_assets/
test/brain-rw/results/matching_fairness_v3/
test/brain-rw/logs/matching_fairness_v3/
```

---

### Task 1: 固定协议、环境与范围锁

**Files:**
- Create: `experiments/matching_fairness/configs/protocol_sub08_seed42.json`
- Create: `experiments/matching_fairness/configs/atm_native_environment.yml`
- Create: `experiments/matching_fairness/matching_fairness/__init__.py`
- Create: `experiments/matching_fairness/matching_fairness/config.py`
- Create: `experiments/matching_fairness/tests/test_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `Protocol.load(path: Path) -> Protocol`
- Produces: `Protocol.assert_formal_scope() -> None`
- Produces: immutable `ModelSpec(slug, encoder_type, checkpoint_role)`
- Consumes: no earlier task.

- [ ] **Step 1: Write the failing scope tests**

```python
from pathlib import Path
import json
import pytest
from matching_fairness.config import Protocol

CONFIG = Path("experiments/matching_fairness/configs/protocol_sub08_seed42.json")

def test_formal_protocol_is_exactly_sub08_seed42() -> None:
    protocol = Protocol.load(CONFIG)
    assert protocol.subject == "sub-08"
    assert protocol.seed == 42
    assert tuple(model.slug for model in protocol.models) == (
        "nice", "atm_s", "our_project"
    )
    assert protocol.standard_scenario_count == 27
    assert protocol.duplicate_query_counts == (0, 10, 20)
    protocol.assert_formal_scope()

def test_scope_guard_rejects_multisubject(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["subject"] = "all"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sub-08 / seed-42"):
        Protocol.load(path).assert_formal_scope()
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
PYTHONPATH=experiments/matching_fairness /hpc2hdd/home/ckwong627/miniconda3/envs/test/bin/python -m pytest experiments/matching_fairness/tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'matching_fairness'`.

- [ ] **Step 3: Add the exact protocol JSON**

The JSON must encode the Global Constraints without optional subject/seed lists. The core fields are:

```json
{
  "schema_version": 1,
  "subject": "sub-08",
  "seed": 42,
  "models": [
    {"slug": "nice", "encoder_type": "NICE", "checkpoint_role": "val_selected_formal"},
    {"slug": "atm_s", "encoder_type": "ATMS", "checkpoint_role": "val_selected_formal"},
    {"slug": "our_project", "encoder_type": "BrainRW", "checkpoint_role": "fixed_formal"}
  ],
  "native_training": {
    "mode": "intra",
    "epochs": 500,
    "batch_size": 1024,
    "lr": 0.0003,
    "val_ratio": 0.1,
    "early_stopping_patience": 10,
    "ema_decay": 0.999,
    "logit_scale_type": "exp",
    "avg_trials": true,
    "n_chans": 63,
    "n_times": 250,
    "checkpoint_metric": "validation_contrastive_loss",
    "checkpoint_direction": "min"
  },
  "standard_grid": {
    "drop_query": [0, 5, 10],
    "drop_gallery": [0, 5, 10],
    "drop_pair": [0],
    "duplicate_gallery": [0, 10, 20]
  },
  "duplicate_query_counts": [0, 10, 20],
  "sinkhorn": {"temperature": 0.05, "max_iterations": 500, "tolerance": 1e-8}
}
```

- [ ] **Step 4: Implement the immutable parser and guard**

```python
@dataclass(frozen=True)
class ModelSpec:
    slug: str
    encoder_type: str
    checkpoint_role: str

@dataclass(frozen=True)
class Protocol:
    subject: str
    seed: int
    models: tuple[ModelSpec, ...]
    standard_grid: Mapping[str, tuple[int, ...]]
    duplicate_query_counts: tuple[int, ...]
    native_training: Mapping[str, object]
    sinkhorn: Mapping[str, object]

    @classmethod
    def load(cls, path: Path) -> "Protocol":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            subject=str(payload["subject"]),
            seed=int(payload["seed"]),
            models=tuple(ModelSpec(**item) for item in payload["models"]),
            standard_grid={
                key: tuple(int(value) for value in values)
                for key, values in payload["standard_grid"].items()
            },
            duplicate_query_counts=tuple(payload["duplicate_query_counts"]),
            native_training=dict(payload["native_training"]),
            sinkhorn=dict(payload["sinkhorn"]),
        )

    @property
    def standard_scenario_count(self) -> int:
        return math.prod(len(values) for values in self.standard_grid.values())

    def assert_formal_scope(self) -> None:
        actual = (self.subject, self.seed, tuple(m.slug for m in self.models))
        expected = ("sub-08", 42, ("nice", "atm_s", "our_project"))
        if actual != expected:
            raise ValueError(f"formal scope must be sub-08 / seed-42: {actual}")
        if self.standard_scenario_count != 27:
            raise ValueError("formal standard grid must contain exactly 27 scenarios")
        if self.duplicate_query_counts != (0, 10, 20):
            raise ValueError("duplicate-query counts must be exactly (0, 10, 20)")
```

- [ ] **Step 5: Add the isolated conda environment**

Write the complete `atm_native_environment.yml` as:

```yaml
name: atm_native
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.12
  - pytorch=2.5.0
  - torchvision=0.20.0
  - torchaudio=2.5.0
  - pytorch-cuda=12.4
  - numpy=1.26.4
  - pandas=2.3.3
  - scipy=1.15.3
  - scikit-learn=1.6.1
  - mne=1.9.0
  - einops=0.8.1
  - pytest
  - pip
  - pip:
      - braindecode==0.8.1
      - wandb==0.19.10
      - open-clip-torch==2.26.1
      - git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16
```

Do not include diffusion or image-generation packages because no retrieval source imports them.

- [ ] **Step 6: Run the tests and commit**

Run the Step 2 command. Expected: `2 passed`.

```bash
git add .gitignore experiments/matching_fairness/configs   experiments/matching_fairness/matching_fairness   experiments/matching_fairness/tests/test_config.py
git commit -m "feat(fairness): lock sub08 seed42 protocol"
```

---

### Task 2: 锁定官方源、下载最小资产并执行 preflight

**Files:**
- Create: `experiments/matching_fairness/matching_fairness/provenance.py`
- Create: `experiments/matching_fairness/scripts/fetch_upstream.py`
- Create: `experiments/matching_fairness/scripts/fetch_assets.py`
- Create: `experiments/matching_fairness/scripts/preflight.py`
- Create: `experiments/matching_fairness/tests/test_provenance.py`

**Interfaces:**
- Consumes: `Protocol` from Task 1.
- Produces: `SourceLock(url, branch, commit, checkout_sha256)`.
- Produces: `resolve_detached_checkout(path, url, branch) -> SourceLock`.
- Produces: `sha256_file(path: Path) -> str`.
- Produces runtime: `manifests/upstream_lock.json` and `manifests/preflight.json`.

- [ ] **Step 1: Write failing provenance tests**

```python
def test_sha256_file_is_content_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )

def test_source_lock_rejects_non_detached_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"], check=True
    )
    required = (
        "Retrieval/train_unified.py",
        "Retrieval/retrieval_engine.py",
        "Retrieval/eeg_encoders.py",
        "eegdatasets.py",
        "encoder_utils.py",
        "models/atms.py",
    )
    for relative in required:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-m", "fixture"], check=True
    )
    with pytest.raises(ValueError, match="detached"):
        inspect_checkout(checkout)
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
PYTHONPATH=experiments/matching_fairness /hpc2hdd/home/ckwong627/miniconda3/envs/test/bin/python -m pytest experiments/matching_fairness/tests/test_provenance.py -v
```

Expected: FAIL because `provenance.py` does not exist.

- [ ] **Step 3: Implement detached source locking**

`fetch_upstream.py` must execute only argument-list subprocesses:

```python
def resolve_detached_checkout(path: Path, url: str, branch: str) -> SourceLock:
    if not (path / ".git").exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--branch", branch, url, str(path)],
            check=True,
        )
    subprocess.run(["git", "-C", str(path), "fetch", "origin", branch], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "FETCH_HEAD"], text=True
    ).strip()
    subprocess.run(
        ["git", "-C", str(path), "checkout", "--detach", commit], check=True
    )
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise RuntimeError(f"checkout mismatch: expected {commit}, found {actual}")
    return inspect_checkout(path, expected_url=url)
```

`inspect_checkout` must require detached HEAD, clean worktree, the exact remote URL, and the files `Retrieval/train_unified.py`, `Retrieval/retrieval_engine.py`, `Retrieval/eeg_encoders.py`, `eegdatasets.py`, `encoder_utils.py`, and `models/atms.py`.

- [ ] **Step 4: Implement the minimal asset fetcher**

Use Hugging Face dataset repo `LidongYang/EEG_Image_decode` and write under:

```text
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/EEG_Image_decode_assets/
```

The downloader invokes:

```bash
hf download LidongYang/EEG_Image_decode --repo-type dataset   --include 'Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy'   --include 'Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy'   --include 'ViT-H-14_features_train.pt'   --include 'ViT-H-14_features_test.pt'   --local-dir /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/EEG_Image_decode_assets
```

After download, record byte size and SHA-256 for all four files. Reject symbolic links that resolve outside the asset root.

- [ ] **Step 5: Implement preflight gates**

`preflight.py` must verify:

- environment name is `atm_native`;
- Python is 3.12.x;
- exact critical versions from Task 1 import successfully;
- source checkout is detached and clean;
- official NPY train/test shapes end in `(63, 250)`, test shape is `(200, 80, 63, 250)`;
- official feature files contain `img_features` and `text_features`, with 16,540 training and 200 test rows;
- existing BrainRW `test.pt` contains `eeg,label,img,text,session,ch_names,times`;
- BrainRW test shape is `(200, 80, 63, 250)`;
- each image has session counts `{0:20,1:20,2:20,3:20}`;
- sorted official test images match `Path(img).stem` from the BrainRW file.

- [ ] **Step 6: Run tests, dry-run preflight, and commit**

Run unit tests first; expected all pass. Do not download assets in the unit test. Mock the HF subprocess and use tiny NPY/PT fixtures.

```bash
git add experiments/matching_fairness/matching_fairness/provenance.py   experiments/matching_fairness/scripts/fetch_upstream.py   experiments/matching_fairness/scripts/fetch_assets.py   experiments/matching_fairness/scripts/preflight.py   experiments/matching_fairness/tests/test_provenance.py
git commit -m "feat(fairness): pin native ATM source and assets"
```

---

### Task 3: 建立统一 ScoreArtifact 契约

**Files:**
- Create: `experiments/matching_fairness/matching_fairness/artifacts.py`
- Create: `experiments/matching_fairness/tests/test_artifacts.py`

**Interfaces:**
- Consumes: hashing helpers from Task 2.
- Produces: `ScoreArtifact`.
- Produces: `write_score_artifact(directory, artifact) -> None`.
- Produces: `read_score_artifact(directory) -> ScoreArtifact`.
- Produces: `independent_ranks(artifact) -> np.ndarray`.

- [ ] **Step 1: Write failing artifact tests**

```python
def test_round_trip_preserves_ids_and_matrix(tmp_path: Path) -> None:
    artifact = ScoreArtifact(
        similarity=np.eye(3),
        query_ids=("q0", "q1", "q2"),
        gallery_entry_ids=("e0", "e1", "e2"),
        gallery_canonical_ids=("i0", "i1", "i2"),
        target_canonical_ids=("i0", "i1", "i2"),
        metadata={"model_slug": "fixture"},
    )
    write_score_artifact(tmp_path / "score", artifact)
    loaded = read_score_artifact(tmp_path / "score")
    np.testing.assert_array_equal(loaded.similarity, artifact.similarity)
    assert loaded.query_ids == artifact.query_ids
    assert loaded.target_canonical_ids == artifact.target_canonical_ids

def test_targets_are_resolved_by_canonical_id_not_diagonal() -> None:
    artifact = ScoreArtifact(
        similarity=np.array([[0.1, 0.9], [0.8, 0.2]]),
        query_ids=("q-a", "q-b"),
        gallery_entry_ids=("entry-b", "entry-a"),
        gallery_canonical_ids=("image-b", "image-a"),
        target_canonical_ids=("image-a", "image-b"),
        metadata={"model_slug": "fixture"},
    )
    assert independent_ranks(artifact).tolist() == [1, 1]
```

- [ ] **Step 2: Verify failure**

Run `pytest experiments/matching_fairness/tests/test_artifacts.py -v` with the Task 1 `PYTHONPATH`. Expected: import failure.

- [ ] **Step 3: Implement validation and atomic persistence**

```python
@dataclass(frozen=True)
class ScoreArtifact:
    similarity: np.ndarray
    query_ids: tuple[str, ...]
    gallery_entry_ids: tuple[str, ...]
    gallery_canonical_ids: tuple[str, ...]
    target_canonical_ids: tuple[str, ...]
    metadata: Mapping[str, object]

    def validate(self) -> None:
        rows, cols = self.similarity.shape
        if self.similarity.ndim != 2 or rows < 1 or cols < 1:
            raise ValueError("similarity must be a non-empty 2-D matrix")
        if not np.isfinite(self.similarity).all():
            raise ValueError("similarity contains NaN or Inf")
        if len(self.query_ids) != rows or len(self.target_canonical_ids) != rows:
            raise ValueError("query metadata does not match rows")
        if len(self.gallery_entry_ids) != cols or len(self.gallery_canonical_ids) != cols:
            raise ValueError("gallery metadata does not match columns")
        if len(set(self.query_ids)) != rows:
            raise ValueError("query IDs must be unique")
        if len(set(self.gallery_entry_ids)) != cols:
            raise ValueError("gallery entry IDs must be unique")
```

Persist `similarity.npy` and canonical, sorted `metadata.json`. Metadata contains matrix hash, ordered-ID hashes, subject, seed, model, checkpoint role/hash, source commit, data hashes, query mode and score semantics.

- [ ] **Step 4: Add failure tests**

Add tests rejecting non-finite values, duplicate entry IDs, row/ID mismatch, tampered matrix hashes, and a target absent from a nominally answerable gallery.

- [ ] **Step 5: Run tests and commit**

Expected: artifact tests pass.

```bash
git add experiments/matching_fairness/matching_fairness/artifacts.py   experiments/matching_fairness/tests/test_artifacts.py
git commit -m "feat(fairness): add auditable score artifacts"
```

---

### Task 4: 生成共享 standard 场景和真实 session-balanced trial manifest

**Files:**
- Create: `experiments/matching_fairness/matching_fairness/trial_splits.py`
- Create: `experiments/matching_fairness/matching_fairness/scenarios.py`
- Create: `experiments/matching_fairness/tests/test_trial_splits.py`
- Create: `experiments/matching_fairness/tests/test_scenarios.py`

**Interfaces:**
- Consumes: `Protocol`, `ScoreArtifact`.
- Produces: `build_trial_manifest(image_ids, sessions, seed=42) -> dict`.
- Produces: `average_trial_half(eeg, image_ids, manifest, half) -> np.ndarray`.
- Produces: `select_duplicate_image_ids(image_ids, seed=42) -> tuple[str, ...]`.
- Produces: `standard_scenarios() -> tuple[ScenarioSpec, ...]`.
- Produces: `apply_standard_scenario(artifact, manifest, scenario) -> ScoreArtifact`.
- Produces: `build_duplicate_query_artifact(a, b, repeated_ids, count) -> ScoreArtifact`.

- [ ] **Step 1: Write failing trial split tests**

```python
def test_every_session_is_split_ten_ten_without_overlap() -> None:
    image_ids = ("img-0", "img-1")
    sessions = np.tile(np.repeat(np.arange(4), 20), (2, 1))
    manifest = build_trial_manifest(image_ids, sessions, seed=42)
    for image_id in image_ids:
        for session in range(4):
            a = set(manifest["images"][image_id][str(session)]["a"])
            b = set(manifest["images"][image_id][str(session)]["b"])
            assert len(a) == len(b) == 10
            assert not (a & b)
            assert a | b == set(np.flatnonzero(sessions[0] == session))

def test_half_averages_use_different_real_trials() -> None:
    eeg = np.arange(2 * 80 * 3 * 4).reshape(2, 80, 3, 4)
    sessions = np.tile(np.repeat(np.arange(4), 20), (2, 1))
    manifest = build_trial_manifest(("a", "b"), sessions, seed=42)
    a = average_trial_half(eeg, ("a", "b"), manifest, "a")
    b = average_trial_half(eeg, ("a", "b"), manifest, "b")
    assert a.shape == b.shape == (2, 3, 4)
    assert not np.array_equal(a, b)
```

- [ ] **Step 2: Implement the exact SHA-256 split**

For each trial sort by:

```python
def trial_key(image_id: str, session_id: int, trial_index: int) -> tuple[str, int]:
    payload = (
        "AIAA3800-DUPLICATE-EEG-v1\n42\n"
        f"{image_id}\n{session_id}\n{trial_index}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), trial_index
```

For repeated images sort by:

```python
def duplicate_key(image_id: str) -> tuple[str, str]:
    payload = f"AIAA3800-DUPLICATE-QUERY-v1\n42\n{image_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), image_id
```

Use the first 20 sorted IDs; `dupq10` uses the first 10.

- [ ] **Step 3: Write failing 27-scenario tests**

```python
def test_standard_grid_has_exactly_27_unique_scenarios() -> None:
    scenarios = standard_scenarios()
    assert len(scenarios) == len(set(scenarios)) == 27
    assert {s.drop_pair for s in scenarios} == {0}

def test_all_models_apply_the_same_canonical_manifest() -> None:
    canonical_ids = tuple(f"image-{index:03d}" for index in range(200))
    manifest = build_standard_manifest(canonical_ids, seed=42)
    scenarios = standard_scenarios()
    common = dict(
        query_ids=tuple(f"q-{index:03d}" for index in range(200)),
        gallery_entry_ids=canonical_ids,
        gallery_canonical_ids=canonical_ids,
        target_canonical_ids=canonical_ids,
    )
    first = apply_standard_scenario(
        ScoreArtifact(similarity=np.eye(200), metadata={"model_slug": "a"}, **common),
        manifest,
        scenarios[7],
    )
    second = apply_standard_scenario(
        ScoreArtifact(similarity=2 * np.eye(200), metadata={"model_slug": "b"}, **common),
        manifest,
        scenarios[7],
    )
    assert first.metadata["selected_canonical_ids"] == second.metadata["selected_canonical_ids"]
```

- [ ] **Step 4: Implement shared-ID standard perturbations**

Generate selections once from the canonical master order with NumPy `SeedSequence([42, stream])`. Apply selections by canonical ID, never by model-local row number. Duplicated gallery entries get unique `__duplicate_entry_NNNN` entry IDs but retain their source canonical IDs and score columns.

- [ ] **Step 5: Implement duplicate-query composition**

```python
def build_duplicate_query_artifact(a, b, repeated_ids, count):
    selected = tuple(repeated_ids[:count])
    row_by_target = {target: i for i, target in enumerate(b.target_canonical_ids)}
    rows = [row_by_target[target] for target in selected]
    similarity = np.concatenate([a.similarity, b.similarity[rows]], axis=0)
    query_ids = a.query_ids + tuple(f"{target}__eeg_b" for target in selected)
    targets = a.target_canonical_ids + selected
    return ScoreArtifact(
        similarity=similarity,
        query_ids=query_ids,
        gallery_entry_ids=a.gallery_entry_ids,
        gallery_canonical_ids=a.gallery_canonical_ids,
        target_canonical_ids=targets,
        metadata={**a.metadata, "query_mode": f"dupq{count}"},
    )
```

Reject any A/B matrix hash mismatch in gallery IDs, any missing repeated target, and any byte-identical A/B source row set presented as “real repeat”.

- [ ] **Step 6: Run tests and commit**

Expected: all Task 4 tests pass.

```bash
git add experiments/matching_fairness/matching_fairness/trial_splits.py   experiments/matching_fairness/matching_fairness/scenarios.py   experiments/matching_fairness/tests/test_trial_splits.py   experiments/matching_fairness/tests/test_scenarios.py
git commit -m "feat(fairness): add shared perturbation and trial manifests"
```

---

### Task 5: 实现不访问测试集的官方 NICE/ATM-S 训练封装

**Files:**
- Create: `experiments/matching_fairness/matching_fairness/native_training.py`
- Create: `experiments/matching_fairness/scripts/train_native.py`
- Create: `experiments/matching_fairness/tests/test_native_training.py`

**Interfaces:**
- Consumes: protocol, detached official checkout, official training NPY, official training ViT-H-14 feature file.
- Produces: `train_native(config: NativeTrainConfig) -> TrainingResult`.
- Produces runtime: `checkpoints/{nice,atm_s}/epoch_NNNN.pth`, `best_val.pth`, `history.csv`, `checkpoint_manifest.json`.

- [ ] **Step 1: Write failing selector and sealing tests**

```python
def test_validation_loss_selects_lowest_then_earliest() -> None:
    records = [
        EpochRecord(epoch=1, val_loss=0.4, checkpoint=Path("1.pth")),
        EpochRecord(epoch=2, val_loss=0.2, checkpoint=Path("2.pth")),
        EpochRecord(epoch=3, val_loss=0.2, checkpoint=Path("3.pth")),
    ]
    assert select_validation_checkpoint(records).epoch == 2

def test_training_config_has_no_test_inputs() -> None:
    fields = {field.name for field in dataclasses.fields(NativeTrainConfig)}
    assert fields.isdisjoint({"test_eeg", "test_features", "test_images"})
```

- [ ] **Step 2: Run tests and verify failure**

Expected: missing `native_training` module.

- [ ] **Step 3: Implement official-module loading**

Load official modules from the detached checkout after validating the source lock. Map:

```python
ENCODERS = {
    "nice": {"encoder_type": "NICE", "use_subject_id": False, "normalize_feats": False},
    "atm_s": {"encoder_type": "ATMS", "use_subject_id": True, "normalize_feats": True},
}
```

Use official `EEGDataset`, `build_encoder`, `train_epoch`, `compute_val_loss`, `EMA`, and `stratified_condition_split`. Do not call official `train_loop`, because it evaluates test data every epoch.

- [ ] **Step 4: Implement the sealed training loop**

The loop must:

1. seed Python, NumPy, CPU and CUDA with 42;
2. load only `preprocessed_eeg_training.npy` and `ViT-H-14_features_train.pt`;
3. instantiate official training dataset with `avg_trials=True`;
4. make official stratified 90/10 condition split;
5. use AdamW at `3e-4`, batch 1024, `drop_last=True`;
6. call official `train_epoch`;
7. apply EMA before `compute_val_loss`;
8. save the EMA state for every epoch;
9. restore raw weights before the next epoch;
10. update `best_val.pth` only when `val_loss < best_val_loss`;
11. stop after 10 consecutive non-improvements;
12. write history and all hashes atomically.

The selector is exactly:

```python
def select_validation_checkpoint(records: Sequence[EpochRecord]) -> EpochRecord:
    if not records:
        raise ValueError("no epoch records")
    return min(records, key=lambda record: (record.val_loss, record.epoch))
```

- [ ] **Step 5: Add fake-upstream integration tests**

Use a two-epoch fake upstream module. Assert:

- test loaders are never constructed;
- only validation loss changes the selected epoch;
- EMA weights, not raw weights, are saved;
- patience stops on the expected epoch;
- NICE calls `model(eeg)`;
- ATM-S calls `model(eeg, subject_ids)`;
- restart with the same seed produces identical history hashes.

- [ ] **Step 6: Run tests and commit**

```bash
git add experiments/matching_fairness/matching_fairness/native_training.py   experiments/matching_fairness/scripts/train_native.py   experiments/matching_fairness/tests/test_native_training.py
git commit -m "feat(fairness): add sealed native baseline training"
```

---

### Task 6: 导出 NICE/ATM-S 与 Our project 的 standard、EEG-A、EEG-B 分数

**Files:**
- Create: `experiments/matching_fairness/matching_fairness/native_export.py`
- Create: `experiments/matching_fairness/scripts/export_native_scores.py`
- Create: `experiments/matching_fairness/scripts/export_brainrw_scores.py`
- Create: `experiments/matching_fairness/tests/test_native_export.py`
- Modify: `main/data.py`
- Modify: `scripts/evaluate_retrieval.py`
- Create: `tests/test_things_trial_selection.py`

**Interfaces:**
- Consumes: `ScoreArtifact`, trial manifest, frozen checkpoints.
- Produces: three artifacts per model: `standard`, `eeg_a`, `eeg_b`.
- Produces: `native_scores(model, eeg, image_features) -> np.ndarray`.
- Produces: `best_test_audit.json` for NICE and ATM-S.
- Extends: `load_things_brain_dataset(*, data_directory, split, subject_ids, brain_column, avg_trials, selected_channels, trial_indices_by_image=None)`.

- [ ] **Step 1: Write failing BrainRW trial-selection tests**

```python
def test_explicit_trial_indices_are_averaged_per_image(tmp_path: Path) -> None:
    eeg = torch.arange(2 * 8 * 63 * 250, dtype=torch.float32).reshape(
        2, 8, 63, 250
    )
    subject_dir = tmp_path / "sub-08"
    subject_dir.mkdir()
    torch.save(
        {
            "eeg": eeg,
            "label": torch.arange(2),
            "img": np.array(
                [["image-0.jpg"] * 8, ["image-1.jpg"] * 8], dtype=object
            ),
        },
        subject_dir / "test.pt",
    )
    selection = {"image-0": [0, 2, 4, 6], "image-1": [1, 3, 5, 7]}
    dataset = load_things_brain_dataset(
        data_directory=str(tmp_path),
        split="test",
        subject_ids=8,
        avg_trials=True,
        trial_indices_by_image=selection,
    )
    assert len(dataset) == 2
    np.testing.assert_allclose(
        dataset[0]["eeg"], eeg[0, [0, 2, 4, 6]].mean(dim=0).numpy()
    )
```

Also test missing IDs, duplicate indices, out-of-range indices, non-test use, and a selection not containing exactly 40 trials in formal mode.

- [ ] **Step 2: Implement the loader extension**

Resolve image IDs before averaging. When `trial_indices_by_image` is present, require the original tensor to be 4-D and compute:

```python
selected_rows = []
for image_index, image_id in enumerate(image_ids):
    indices = np.asarray(trial_indices_by_image[image_id], dtype=np.int64)
    selected_rows.append(x[image_index, indices].mean(dim=0))
x = torch.stack(selected_rows)
```

Existing calls without the new argument must remain byte-for-byte equivalent in outputs.

- [ ] **Step 3: Extend the BrainRW evaluator CLI**

Add:

```text
--trial-split-manifest PATH
--trial-half {standard,a,b}
```

`standard` rejects a manifest and preserves current behavior. `a` or `b` requires a manifest, loads its canonical image-to-index mapping, and passes it to `load_things_brain_dataset`. Export A and B as separate 200 x 200 ScoreArtifacts; do not append rows inside the model evaluator.

- [ ] **Step 4: Write failing native score tests**

```python
def test_atms_uses_normalized_native_scores() -> None:
    eeg = np.array([[3.0, 4.0], [0.0, 2.0]])
    image = np.array([[4.0, 3.0], [2.0, 0.0]])
    scores = native_scores("atm_s", eeg, image, logit_scale=2.0)
    eeg_unit = eeg / np.linalg.norm(eeg, axis=1, keepdims=True)
    image_unit = image / np.linalg.norm(image, axis=1, keepdims=True)
    expected = 2.0 * eeg_unit @ image_unit.T
    np.testing.assert_allclose(scores, expected)

def test_nice_uses_official_raw_logit_scores() -> None:
    eeg = np.array([[3.0, 4.0], [0.0, 2.0]])
    image = np.array([[4.0, 3.0], [2.0, 0.0]])
    scores = native_scores("nice", eeg, image, logit_scale=2.0)
    np.testing.assert_allclose(scores, 2.0 * eeg @ image.T)
```

- [ ] **Step 5: Implement native score export**

For `standard`, average all 80 trials before the model. For A/B, average the 40 manifest-selected official NPY trials before the model. Build query IDs from the sorted official test image paths and gallery IDs from the same 200 images. Load only `best_val.pth` for `val_selected_formal`.

Before paper-style audit, finish and hash all three main artifacts. Then evaluate every saved epoch checkpoint, record Independent Top-1/Top-5, and write only the highest-test row and full audit table under `best_test_audit_only`. Do not create fairness inputs from those checkpoints.

- [ ] **Step 6: Add parity tests**

For fixtures, assert native evaluator ranking equals ScoreArtifact Independent ranking. For the real preflight, require:

- native standard matrix is 200 x 200;
- recomputed Top-1/Top-5 equals the exporter’s native metric counts;
- A/B matrices use identical gallery IDs;
- A and B query embeddings are not byte-identical;
- Our project standard artifact matches the already recorded formal Top-1/Top-5 counts.

- [ ] **Step 7: Run tests and commit**

```bash
git add main/data.py scripts/evaluate_retrieval.py   tests/test_things_trial_selection.py   experiments/matching_fairness/matching_fairness/native_export.py   experiments/matching_fairness/scripts/export_native_scores.py   experiments/matching_fairness/scripts/export_brainrw_scores.py   experiments/matching_fairness/tests/test_native_export.py
git commit -m "feat(fairness): export native and repeated EEG scores"
```

---

### Task 7: 迁移并验证五种 decoder 与 30 个场景

**Files:**
- Create: `experiments/matching_fairness/matching_fairness/decoders.py`
- Create: `experiments/matching_fairness/matching_fairness/evaluation.py`
- Create: `experiments/matching_fairness/scripts/run_scenarios.py`
- Create: `experiments/matching_fairness/tests/test_decoders.py`
- Create: `experiments/matching_fairness/tests/test_evaluation.py`

**Interfaces:**
- Consumes: ScoreArtifact and scenario outputs.
- Produces: `decode_independent`, `decode_greedy`, `decode_hungarian`, `decode_stable`, `decode_sinkhorn`.
- Produces: `evaluate_artifact(artifact, decoder_config) -> EvaluationResult`.
- Produces runtime: exactly 450 decoder records.

- [ ] **Step 1: Write failing decoder tests**

Cover:

- stable rowwise Independent argmax;
- deterministic greedy de-duplication;
- Hungarian rectangular unmatched behavior;
- query-proposing stable matching without blocking pairs;
- Sinkhorn finite deterministic plan and convergence metadata;
- tie behavior independent of ground-truth order.

Use the current tested semantics from `test/brain-rw/scripts/compare_matching_decoders.py`.

- [ ] **Step 2: Implement decoders without labels**

Required signatures:

```python
@dataclass(frozen=True)
class Assignment:
    gallery_indices: np.ndarray
    unmatched_mask: np.ndarray
    strict_one_to_one: bool
    metadata: Mapping[str, object]

def _validated_matrix(similarity: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(similarity, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("similarity must be a non-empty 2-D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("similarity contains NaN or Inf")
    return matrix

def _assignment(
    gallery_indices: np.ndarray,
    *,
    strict_one_to_one: bool,
    metadata: Mapping[str, object] | None = None,
) -> Assignment:
    indices = np.asarray(gallery_indices, dtype=np.int64)
    return Assignment(
        gallery_indices=indices,
        unmatched_mask=indices < 0,
        strict_one_to_one=strict_one_to_one,
        metadata=dict(metadata or {}),
    )

def decode_independent(similarity: np.ndarray) -> Assignment:
    matrix = _validated_matrix(similarity)
    return _assignment(np.argmax(matrix, axis=1), strict_one_to_one=False)

def decode_greedy(similarity: np.ndarray) -> Assignment:
    matrix = _validated_matrix(similarity)
    rows, columns = matrix.shape
    ranking = np.argsort(-matrix, axis=1, kind="stable")
    top1 = ranking[:, 0]
    top_scores = matrix[np.arange(rows), top1]
    row_order = np.lexsort((np.arange(rows), -top_scores))
    indices = -np.ones(rows, dtype=np.int64)
    used = np.zeros(columns, dtype=bool)
    for row in row_order:
        gallery = int(top1[row])
        if not used[gallery]:
            indices[int(row)] = gallery
            used[gallery] = True
    for row in np.flatnonzero(indices < 0):
        available = ranking[row][~used[ranking[row]]]
        if available.size:
            gallery = int(available[0])
            indices[int(row)] = gallery
            used[gallery] = True
    return _assignment(indices, strict_one_to_one=True)

def decode_hungarian(similarity: np.ndarray, seed: int) -> Assignment:
    matrix = _validated_matrix(similarity)
    rows, columns = matrix.shape
    generator = np.random.default_rng(seed)
    row_permutation = generator.permutation(rows)
    column_permutation = generator.permutation(columns)
    selected_rows, selected_columns = linear_sum_assignment(
        matrix[row_permutation][:, column_permutation], maximize=True
    )
    indices = -np.ones(rows, dtype=np.int64)
    indices[row_permutation[selected_rows]] = column_permutation[selected_columns]
    matched = indices >= 0
    return _assignment(
        indices,
        strict_one_to_one=True,
        metadata={
            "seed": int(seed),
            "matched_count": int(matched.sum()),
            "unmatched_count": int((~matched).sum()),
            "assigned_sum_similarity": float(
                matrix[np.flatnonzero(matched), indices[matched]].sum()
            ),
        },
    )

def decode_stable(similarity: np.ndarray) -> Assignment:
    matrix = _validated_matrix(similarity)
    rows, columns = matrix.shape
    preferences = np.argsort(-matrix, axis=1, kind="stable")
    next_choice = np.zeros(rows, dtype=np.int64)
    gallery_partner = -np.ones(columns, dtype=np.int64)
    indices = -np.ones(rows, dtype=np.int64)
    free_queries: deque[int] = deque(range(rows))
    while free_queries:
        query = free_queries.popleft()
        if next_choice[query] >= columns:
            continue
        gallery = int(preferences[query, next_choice[query]])
        next_choice[query] += 1
        current = int(gallery_partner[gallery])
        if current < 0:
            gallery_partner[gallery] = query
            indices[query] = gallery
            continue
        challenger_wins = matrix[query, gallery] > matrix[current, gallery] or (
            matrix[query, gallery] == matrix[current, gallery] and query < current
        )
        if challenger_wins:
            indices[current] = -1
            if next_choice[current] < columns:
                free_queries.append(current)
            gallery_partner[gallery] = query
            indices[query] = gallery
        elif next_choice[query] < columns:
            free_queries.append(query)
    return _assignment(indices, strict_one_to_one=True)

def decode_sinkhorn(
    similarity: np.ndarray,
    temperature: float,
    max_iterations: int,
    tolerance: float,
) -> Assignment:
    matrix = _validated_matrix(similarity)
    if temperature <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid Sinkhorn parameters")
    rows, columns = matrix.shape
    log_kernel = matrix / float(temperature)
    log_u = np.zeros(rows, dtype=np.float64)
    log_v = np.zeros(columns, dtype=np.float64)
    marginal_error = float("inf")
    converged = False
    for iteration in range(1, max_iterations + 1):
        log_u = -np.log(float(rows)) - logsumexp(
            log_kernel + log_v[None, :], axis=1
        )
        log_v = -np.log(float(columns)) - logsumexp(
            log_kernel + log_u[:, None], axis=0
        )
        if iteration == 1 or iteration % 10 == 0 or iteration == max_iterations:
            plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
            row_error = np.max(np.abs(plan.sum(axis=1) - 1.0 / rows))
            column_error = np.max(np.abs(plan.sum(axis=0) - 1.0 / columns))
            marginal_error = float(max(row_error, column_error))
            if marginal_error <= tolerance:
                converged = True
                break
    plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
    return _assignment(
        np.argmax(plan, axis=1),
        strict_one_to_one=False,
        metadata={
            "temperature": float(temperature),
            "max_iterations": int(max_iterations),
            "iterations": int(iteration),
            "tolerance": float(tolerance),
            "marginal_error": marginal_error,
            "converged": converged,
        },
    )
```

Import `deque` from `collections`, `linear_sum_assignment` from `scipy.optimize`, and `logsumexp` from `scipy.special`. No decoder accepts target IDs.

- [ ] **Step 3: Write failing metric tests**

Assert:

- unanswerable and unmatched count wrong in overall Top-1;
- answerable Top-1 excludes unanswerable queries;
- duplicate gallery entries with the same canonical ID count correct;
- only Independent has Top-5;
- `dupq10` strict one-to-one ceiling is 200/210;
- `dupq20` strict one-to-one ceiling is 200/220;
- duplicated-pair both-correct is computed by canonical ID, not query position.

- [ ] **Step 4: Implement evaluation**

Compute target ranks from canonical IDs after decoding. For duplicate EEG additionally emit base-A, appended-B, at-least-one coverage, both-correct, theoretical ceiling, unmatched repeated queries and distance from ceiling.

- [ ] **Step 5: Implement the run matrix**

`run_scenarios.py` must:

1. load the fixed protocol and assert formal scope;
2. require all nine source artifacts: 3 models x standard/A/B;
3. compare cross-model canonical gallery order before running;
4. generate 27 standard plus 3 duplicate-query artifacts per model;
5. run five decoders per artifact;
6. write one JSON and one per-query CSV per model/scenario;
7. fail unless it observes exactly `3 x 30 x 5 = 450` decoder records.

- [ ] **Step 6: Run tests and commit**

```bash
git add experiments/matching_fairness/matching_fairness/decoders.py   experiments/matching_fairness/matching_fairness/evaluation.py   experiments/matching_fairness/scripts/run_scenarios.py   experiments/matching_fairness/tests/test_decoders.py   experiments/matching_fairness/tests/test_evaluation.py
git commit -m "feat(fairness): evaluate five decoders across all scenarios"
```

---

### Task 8: 汇总、双语报告与防止过度结论

**Files:**
- Create: `experiments/matching_fairness/matching_fairness/reporting.py`
- Create: `experiments/matching_fairness/scripts/aggregate_results.py`
- Create: `experiments/matching_fairness/tests/test_reporting.py`

**Interfaces:**
- Consumes: 450 decoder result records plus parity and audit manifests.
- Produces: `aggregate_metrics.csv`, `aggregate_summary.json`, `RESULTS_EN.md`, `RESULTS_ZH.md`.
- Produces: presentation-ready standard and duplicate-EEG tables.

- [ ] **Step 1: Write failing report tests**

```python
def valid_records() -> list[dict[str, object]]:
    records = []
    for model in ("nice", "atm_s", "our_project"):
        for scenario_index in range(30):
            suite = "standard" if scenario_index < 27 else "duplicate_eeg"
            for decoder in (
                "independent", "greedy", "hungarian", "stable_matching", "sinkhorn"
            ):
                records.append(
                    {
                        "model": model,
                        "suite": suite,
                        "scenario_index": scenario_index,
                        "decoder": decoder,
                        "correct": 1,
                        "total": 1,
                        "top1": 100.0,
                        "top5": 100.0 if decoder == "independent" else None,
                    }
                )
    return records

def test_report_rejects_incomplete_grid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="450"):
        aggregate_records(valid_records()[:-1])

def test_report_contains_single_cell_limitation() -> None:
    report = render_english_report(aggregate_records(valid_records()))
    assert "sub-08 / seed-42" in report
    assert "does not establish cross-subject significance" in report

def test_assignment_top5_is_never_rendered() -> None:
    report = render_english_report(aggregate_records(valid_records()))
    assert "Hungarian Top-5" not in report
    assert "Stable Matching Top-5" not in report
```

- [ ] **Step 2: Implement strict aggregation**

Use model order `nice, atm_s, our_project`, suite order `standard, duplicate_eeg`, decoder order `independent, greedy, hungarian, stable_matching, sinkhorn`, and numeric scenario sort. Include exact counts before percentages.

Highlight the best number only within comparable columns. Never compare 40-trial duplicate EEG absolute values with 80-trial standard values.

- [ ] **Step 3: Add audit separation**

Create a separate “Reproduction audit” section containing validation-selected formal and best-test-audit-only values. Include the checkpoint epoch, validation loss and explicit test-selection warning. No audit artifact may appear in the 450-record input manifest.

- [ ] **Step 4: Run tests and commit**

```bash
git add experiments/matching_fairness/matching_fairness/reporting.py   experiments/matching_fairness/scripts/aggregate_results.py   experiments/matching_fairness/tests/test_reporting.py
git commit -m "feat(fairness): add strict bilingual result aggregation"
```

---

### Task 9: 一键 Bash、SLURM 依赖链和日志收拢

**Files:**
- Create: `experiments/matching_fairness/run_matching_fairness.sh`
- Create: `experiments/matching_fairness/scripts/submit_pipeline.py`
- Create: `experiments/matching_fairness/slurm/train_native_array.slurm`
- Create: `experiments/matching_fairness/slurm/export_native_array.slurm`
- Create: `experiments/matching_fairness/slurm/export_brainrw.slurm`
- Create: `experiments/matching_fairness/slurm/fairness_cpu.slurm`
- Create: `experiments/matching_fairness/tests/test_orchestration.py`

**Interfaces:**
- Consumes: every CLI from Tasks 2, 5, 6, 7 and 8.
- Produces: `run_matching_fairness.sh --phase {preflight,train,export,match,aggregate,all}`.
- Produces: dependency chain `train -> native export + BrainRW export -> fairness/aggregate`.

- [ ] **Step 1: Write failing dry-run tests**

Assert that:

- dry-run prints subject 08 and seed 42 only;
- two native train cells are NICE and ATMS;
- no array range can generate other subjects/seeds;
- all `#SBATCH --output/--error` paths are under `test/brain-rw/logs/matching_fairness_v3`;
- training uses two A40 cells with `--array=0-1%2`;
- CPU matching depends on successful completion of all exports.

- [ ] **Step 2: Implement the Bash phase interface**

Supported commands:

```bash
bash experiments/matching_fairness/run_matching_fairness.sh --phase preflight
bash experiments/matching_fairness/run_matching_fairness.sh --phase train --submit
bash experiments/matching_fairness/run_matching_fairness.sh --phase export --submit
bash experiments/matching_fairness/run_matching_fairness.sh --phase match
bash experiments/matching_fairness/run_matching_fairness.sh --phase aggregate
bash experiments/matching_fairness/run_matching_fairness.sh --phase all --submit
bash experiments/matching_fairness/run_matching_fairness.sh --phase all --dry-run
```

Use `set -Eeuo pipefail`, absolute roots resolved from the script, offline model flags on compute nodes, and no user-provided subject/seed options.

- [ ] **Step 3: Implement SLURM resources**

Use:

- native training: `i64m1tga40u`, A40 x1 per task, 8 CPU, 64 GB, 8 hours, array `0-1%2`;
- native export/audit: `i64m1tga40u`, A40 x1, 4 CPU, 48 GB, 3 hours, array `0-1%2`;
- BrainRW A/B export: `debug`, A40 x1, 4 CPU, 32 GB, 30 minutes;
- matching and aggregation: `i64m512u`, 4 CPU, 16 GB, 2 hours.

`submit_pipeline.py` captures numeric job IDs and submits the final CPU job with `afterok:<native-export>:<brainrw-export>`; native export itself depends on the training array.

- [ ] **Step 4: Enforce safe resume semantics**

Every phase checks input hashes and skips an existing output only if its manifest matches. Mismatched output paths fail unless `--overwrite` is explicitly supplied. `--overwrite` only replaces the selected phase’s derived outputs, never checkpoints or downloaded assets.

- [ ] **Step 5: Run tests and commit**

```bash
git add experiments/matching_fairness/run_matching_fairness.sh   experiments/matching_fairness/scripts/submit_pipeline.py   experiments/matching_fairness/slurm   experiments/matching_fairness/tests/test_orchestration.py
git commit -m "feat(fairness): add fixed-scope SLURM pipeline"
```

---

### Task 10: 文档、全套静态验证和 smoke test

**Files:**
- Create: `experiments/matching_fairness/README.md`
- Create: `experiments/matching_fairness/README_ZH.md`
- Modify: `experiments/matching_fairness/configs/atm_native_environment.yml` only if the environment gate exposes a declared-version import incompatibility.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: documented exact commands and bilingual cross-links.
- Produces: a green unit/integration test suite before GPU submission.

- [ ] **Step 1: Write bilingual READMEs**

Both documents must explain:

- research question and why matching is a separate transductive analysis;
- NICE/ATM-S official source and commit-lock mechanism;
- validation-loss checkpoint rule and test sealing;
- best-test audit warning;
- 27 standard and 3 duplicate EEG scenarios;
- 10/10 per-session trial split;
- five decoder semantics and why assignment Top-5 is undefined;
- exact environment, preflight, dry-run, submit, resume and aggregation commands;
- result and log directories;
- single-cell statistical limitation.

Add `[中文](README_ZH.md)` to English and `[English](README.md)` to Chinese.

- [ ] **Step 2: Create the environment and run its gate**

```bash
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda env create -n atm_native   -f experiments/matching_fairness/configs/atm_native_environment.yml
conda run -n atm_native python experiments/matching_fairness/scripts/preflight.py   --environment-only
```

Expected: exact critical versions and all imports pass. If the environment already exists, run:

```bash
conda env update -n atm_native --prune \
  -f experiments/matching_fairness/configs/atm_native_environment.yml
```

Never modify `test` or `eeg_recon`.

- [ ] **Step 3: Run the complete tracked test suite**

```bash
PYTHONPATH=experiments/matching_fairness /hpc2hdd/home/ckwong627/miniconda3/envs/test/bin/python -m pytest experiments/matching_fairness/tests tests/test_things_trial_selection.py tests/test_train_clip_lora_grad_clip.py -v
```

Then run the native-import subset:

```bash
PYTHONPATH=experiments/matching_fairness /hpc2hdd/home/ckwong627/miniconda3/envs/atm_native/bin/python -m pytest experiments/matching_fairness/tests/test_native_training.py experiments/matching_fairness/tests/test_native_export.py -v
```

Expected: all tests pass, no warnings about test-set checkpoint selection.

- [ ] **Step 4: Run dry-run and fixture end-to-end smoke**

```bash
bash experiments/matching_fairness/run_matching_fairness.sh   --phase all --dry-run
```

Expected counts: 2 training cells, 3 model exports, 90 scenarios and 450 decoder records. Then run the fixture pipeline and verify byte-stable repeated output hashes.

Run the exact fixture smoke test twice inside one pytest case:

```bash
PYTHONPATH=experiments/matching_fairness /hpc2hdd/home/ckwong627/miniconda3/envs/test/bin/python -m pytest \
  experiments/matching_fairness/tests/test_orchestration.py::test_fixture_pipeline_is_byte_stable -v
```

The test writes two independent temporary output roots, executes the complete fixture pipeline in each, and asserts identical SHA-256 hashes for the aggregate CSV, summary JSON and both rendered reports.

- [ ] **Step 5: Commit documentation**

```bash
git add experiments/matching_fairness/README.md   experiments/matching_fairness/README_ZH.md   experiments/matching_fairness/configs/atm_native_environment.yml
git commit -m "docs(fairness): document native sub08 workflow"
```

---

### Task 11: 执行 sub-08 / seed-42 正式实验并验收结果

**Files:**
- Runtime only: `test/brain-rw/results/matching_fairness_v3/**`
- Runtime only: `test/brain-rw/logs/matching_fairness_v3/**`
- Create after successful run: `experiments/matching_fairness/RESULTS.md`
- Create after successful run: `experiments/matching_fairness/RESULTS_ZH.md`

**Interfaces:**
- Consumes: green tests, official assets, frozen code and protocol.
- Produces: final formal and audit outputs for the course report/presentation.

- [ ] **Step 1: Fetch and lock source/assets**

```bash
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate atm_native
python experiments/matching_fairness/scripts/fetch_upstream.py
python experiments/matching_fairness/scripts/fetch_assets.py
bash experiments/matching_fairness/run_matching_fairness.sh --phase preflight
```

Expected: detached clean upstream lock, four asset hashes, valid session map and no formal test metric.

- [ ] **Step 2: Inspect the queue and submit the fixed pipeline**

Check partitions using the AGENTS.md queue command. If A40 shared is not clearly more congested, submit:

```bash
bash experiments/matching_fairness/run_matching_fairness.sh   --phase all --submit
```

Record returned job IDs in `manifests/submission.json`. Do not submit duplicate jobs while valid jobs are pending or running.

- [ ] **Step 3: Verify training completion before accepting exports**

For NICE and ATM-S require:

- non-empty history;
- best validation loss is finite;
- selected epoch equals the minimum `(val_loss, epoch)`;
- checkpoint SHA-256 matches manifest;
- early stopping state is recorded;
- training manifest contains no test paths;
- source/config/data hashes match preflight.

- [ ] **Step 4: Verify parity and the real-repeat provenance**

For all three models require standard Independent parity. For A/B require:

- 200 x 200 matrices;
- 40 unique real trial indices per image per half;
- 10 A and 10 B trials per session;
- no A/B overlap;
- all 80 trials accounted for;
- A/B query hashes differ;
- common canonical gallery order across models.

Any failure stops the downstream aggregate.

- [ ] **Step 5: Verify the 450-record fairness aggregate**

```bash
/hpc2hdd/home/ckwong627/miniconda3/envs/test/bin/python   experiments/matching_fairness/scripts/aggregate_results.py   --results-root   /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3
```

Require 405 standard decoder records plus 45 duplicate-EEG decoder records. For Greedy, Hungarian and Stable Matching, confirm unmatched counts are at least 10 for `dupq10` and 20 for `dupq20`; Independent and Sinkhorn must remain labelled non-strict.

- [ ] **Step 6: Generate and review tracked result summaries**

Generate `RESULTS.md` and `RESULTS_ZH.md` from the verified aggregate. They contain small tables, exact artifact hashes, the official source commit, validation-selected scores, best-test audit warning, and single-cell limitation. They do not include raw matrices or checkpoint paths that only exist on this HPC.

- [ ] **Step 7: Run final verification and commit summaries**

```bash
git diff --check
PYTHONPATH=experiments/matching_fairness /hpc2hdd/home/ckwong627/miniconda3/envs/test/bin/python -m pytest experiments/matching_fairness/tests tests/test_things_trial_selection.py -q
git add experiments/matching_fairness/RESULTS.md   experiments/matching_fairness/RESULTS_ZH.md
git commit -m "docs(fairness): report sub08 seed42 results"
```

Expected: tests pass; Git contains only source, configuration, documentation and small summaries; runtime artifacts remain ignored.

- [ ] **Step 8: Push the existing ckw branch without creating another branch**

```bash
git status --short --branch
git push origin ckw
```

Before pushing, require a clean worktree and confirm the branch name is exactly `ckw`.
