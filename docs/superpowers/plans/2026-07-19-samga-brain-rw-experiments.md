# SAMGA + brain-rw Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run the validation-sealed Stage 0–5 SAMGA + brain-rw experiment ladder, select at most one candidate per stage, and determine whether a locked candidate can exceed the local 89.02% Top-1 SAMGA benchmark without using the formal test set for development decisions.

**Architecture:** A new self-contained `experiments/samga_brain_rw` package owns immutable protocol/config parsing, deterministic split manifests, data-access seals, score artifacts, model variants, paired gates, candidate registry, and SLURM entry points. It reuses validated data/model primitives from `experiments/samga_lora` and the pinned V2.5 feature contract from `experiments/samga_reproduction`, but never edits the upstream SAMGA checkout. Development proceeds through `train`, `val-dev`, and one-time `val-confirm`; formal-test code requires a separately generated final-run seal and is not invoked by this plan until one candidate is locked.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy, scikit-learn, Transformers, PEFT, pytest, Bash, JSON/CSV/Markdown, and SLURM on A40 GPUs.

## Global Constraints

- Treat `docs/superpowers/specs/2026-07-19-samga-brain-rw-combination-design.md` as the normative specification.
- Keep `/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/reference_code/codes_for_papers/SAMGA` read-only at commit `1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1`.
- Pin InternViT to `OpenGVLab/InternViT-6B-448px-V2_5` revision `9d1a4344077479c93d42584b6941c64d795d508d`.
- Use the locked `idx0`, patch-token mean, no-normalization cache semantics at logical layers `20, 24, 28, 32, 36`.
- Abort unless the canonical training manifest contains exactly 1,654 concepts and 16,540 rows shared by all subjects.
- Keep every formal-test manifest, feature cache, checkpoint score, prediction, and error analysis inaccessible during Stages 0–5.
- Use standard independent cosine retrieval only; never use Hungarian assignment in candidate selection or primary reporting.
- Use pilot evaluation subjects `01, 05, 08` and seeds `42, 43`. The sole exception is Stage 4: its pre-registered shared adapter may consume `train`-scope rows from subjects `01`–`10`, but pilot gating still evaluates only subjects `01, 05, 08`; this exception never authorizes `val-confirm` or formal-test access. Use subjects `01`–`10` and seeds `42`–`46` for evaluation only at `val-confirm` and the final paired evaluation.
- Write generated artifacts to `artifacts/samga_brain_rw`, logs to `logs/samga_brain_rw`, and compact outputs to `results/samga_brain_rw`.
- Put every SLURM `.out` and `.err` file below `logs/samga_brain_rw`.
- Default to `HF_DATASETS_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Use `eeg_recon` only for CPU pytest because `test` lacks pytest; record that its dependency versions differ from the runtime environment.
- Run every new-package Python or pytest entry point with `PYTHONPATH=experiments/samga_brain_rw`; every documented command and every SLURM launcher must set it explicitly and must never depend on inherited shell state.
- Use `test` for every GPU/runtime smoke and experiment: Python 3.10.18, PyTorch 2.10.0+cu126, Transformers 4.57.6, PEFT 0.18.1, NumPy 1.26.4, and SciPy 1.15.3.
- Do not install or enable FlashAttention during parity; the locked cache was generated with naive attention.
- Treat all historical brain-rw/SAMGA checkpoints and formal-test score matrices as engineering diagnostics only because they expose the new validation concepts or formal test.
- Never copy the 11 GB InternViT base model into run directories; checkpoints store task weights, LoRA adapters, optimizer/RNG/sampler state, and base-model hashes only.
- Use `debug` first for one-batch GPU smoke tests; submit longer pilots to `i64m1tga40u` only after queue inspection.
- Require idempotent run keys and refuse silent overwrite of manifests, checkpoints, score matrices, decisions, and final seals.
- Make no README performance claim until a validation gate and, where applicable, the single formal evaluation have actually completed.

---

### Task 1: Scaffold the experiment package and immutable protocol

**Files:**
- Create: `experiments/samga_brain_rw/__init__.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/__init__.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/config.py`
- Create: `experiments/samga_brain_rw/configs/protocol_v1.json`
- Create: `experiments/samga_brain_rw/configs/internvit_baseline_v1.json`
- Create: `experiments/samga_brain_rw/configs/brainrw_clip_lora_v1.json`
- Create: `experiments/samga_brain_rw/configs/stage1_fusion_v1.json`
- Create: `experiments/samga_brain_rw/configs/stage2_candidates_v1.json`
- Create: `experiments/samga_brain_rw/tests/conftest.py`
- Create: `experiments/samga_brain_rw/tests/test_config.py`

**Interfaces:**
- `ProtocolConfig.from_path(path: Path) -> ProtocolConfig`
- `ProtocolConfig.canonical_payload() -> dict[str, object]`
- `ProtocolConfig.sha256 -> str`
- `resolve_run_config(protocol: ProtocolConfig, candidate: Mapping[str, object], input_hashes: Mapping[str, str]) -> ResolvedRunConfig`
- `make_run_key(stage: str, config_id: str, subject: int, seed: int, semantic_config_sha256: str, input_bundle_sha256: str) -> str`

- [ ] **Step 1: Write failing configuration tests**

Assert exact stage grids, subjects, seeds, hashes, thresholds, paths, and rejection of unknown keys. Assert that JSON key order does not change `ProtocolConfig.sha256` and that changing any semantic value does.

- [ ] **Step 2: Run RED**

Run:

```bash
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate eeg_recon
export PYTHONPATH=experiments/samga_brain_rw
pytest -q experiments/samga_brain_rw/tests/test_config.py
```

Expected: collection fails because `samga_brain_rw.config` does not exist.

- [ ] **Step 3: Implement strict immutable parsing**

The JSON must encode:

```json
{
  "schema_version": 1,
  "split_salt": "AIAA3800-SAMGA-SPLIT-v1\n",
  "stimulus_salt": "AIAA3800-SAMGA-STIM-v1\n",
  "expected_non_test_concepts": 1654,
  "split_sizes": {"train": 1254, "val-dev": 200, "val-confirm": 200},
  "pilot_subjects": [1, 5, 8],
  "pilot_seeds": [42, 43],
  "confirmation_subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "confirmation_seeds": [42, 43, 44, 45, 46],
  "historical_top1": 0.8902,
  "historical_top5": 0.9887,
  "paper_top1": 0.913,
  "paper_top5": 0.988,
  "pilot_gate": {
    "stage1_min_top1_delta": 0.003,
    "other_min_top1_delta": 0.005,
    "minimum_positive_cells": 4,
    "minimum_top5_delta": -0.002,
    "minimum_subject_mean_top1_delta": -0.02
  },
  "confirmation_gate": {
    "minimum_top1_delta": 0.005,
    "ci95_lower_must_exceed": 0.0,
    "minimum_top5_delta": -0.002,
    "minimum_positive_subjects": 8,
    "minimum_subject_mean_top1_delta": -0.02
  },
  "bootstrap": {
    "samples": 10000,
    "seed": 20260719,
    "resampling": "independent_subject_and_seed_indices_with_replacement_cartesian_mean",
    "quantile_method": "linear"
  }
}
```

The four semantic configs are also strict and versioned:

- `internvit_baseline_v1.json` locks the upstream commit, cache/model hashes, layers 20/24/28/32/36, image dimension 3200, prior center 28, `router_eval_mode="global"`/`force_global=true`, ordered channels `P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2`, four-trial averaging, smoothing probability 0.3, batch 512, epochs 60, stage-1 epochs 20, task LRs `1e-4`/`5e-5`, MMD `0.9`→`0.5`, image L2 normalization on, and EEG L2 normalization off.
- `brainrw_clip_lora_v1.json` locks the full local LAION CLIP ViT-B/32 path/hash, BrainMLP dropout 0.1, resolved targets q/k/v/out/fc1/fc2/visual_projection, rank/alpha 32, LoRA dropout 0.0, brain LR `5e-4`, visual LR `5e-5`, AdamW weight decay 0.05, cosine schedule, 25 fixed epochs, BF16, batch 512, four-trial averaging, and the same ordered 17 channels.
- `stage1_fusion_v1.json` contains exactly the 47 pre-registered fusion candidates and their formulas.
- `stage2_candidates_v1.json` contains only the one-factor transforms, controls, adapter grid, and averaging windows defined below.

The fully resolved semantic config hash includes protocol, candidate semantics, model/cache/checkpoint/manifest hashes, subject, seed, and runtime-affecting settings. Output/log paths are excluded. No run is keyed only by the protocol hash.

- [ ] **Step 4: Run GREEN and commit**

Run the focused test and `git diff --check`.

Commit:

```bash
git add experiments/samga_brain_rw
git commit -m "feat: scaffold sealed SAMGA brain-rw protocol"
```

### Task 2: Build the exact train/val-dev/val-confirm manifests

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/hashing.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/splits.py`
- Create: `experiments/samga_brain_rw/scripts/build_protocol_manifests.py`
- Create: `experiments/samga_brain_rw/tests/test_splits.py`
- Modify: `experiments/samga_brain_rw/tests/conftest.py`

**Interfaces:**
- `concept_digest(concept_id: str) -> str`
- `stimulus_digest(split: Literal["val-dev", "val-confirm"], concept_id: str, stimulus_id: str) -> str`
- `partition_concepts(records: Sequence[Mapping[str, object]]) -> SplitAssignment`
- `build_subject_protocol_manifest(source_manifest: Path, assignment: SplitAssignment) -> dict[str, object]`

- [ ] **Step 1: Write failing golden-vector tests**

Use literal UTF-8 IDs and assert:

```python
assert concept_digest("00001_aardvark") == hashlib.sha256(
    "AIAA3800-SAMGA-SPLIT-v1\n00001_aardvark".encode("utf-8")
).hexdigest()
```

Test exact rank boundaries 200/201/400/401, ten stimuli per concept, unique selected validation stimulus, and byte-identical split assignment across all ten subject manifests.

Serialize every ordered ID list exactly as UTF-8 `"\n".join(ordered_ids)` with no trailing newline, then apply SHA-256.

Lock these read-only simulation digests as golden outputs:

- train concept-list SHA-256 `ae5aeda4101f8740ebcb63464ca9cf5e126c81b2f124f5caa8f7b57b7a9fad24`;
- val-dev concept-list SHA-256 `c8c00ff2b15d98cdcb74d533037d52435bc12e09797151e46b52b86aedba1d15`;
- val-dev query-list SHA-256 `512c222859a31b753ee31c5d6a1ddd1c81bb06e2dd5784d325f4480967162314`;
- val-confirm concept-list SHA-256 `27cfd5b3d0b46f3e8303953ede106e0716f410aee2b5756dd6fb5ad0324908bb`;
- val-confirm query-list SHA-256 `7a77db6d8d214e4a8192472dc7a760b58763d49c8c9f88fcb55c97bb124ec9fd`.

Preserve the canonical record payload/hash `f59500f36e273f66fce5c2019670b076d75d538feccf296c7d7ed75f19ae3fac`; store new roles and selected stimuli in a separate split registry so existing cache alignment remains valid.

- [ ] **Step 2: Add failure tests**

Reject:

- 1,653 or 1,655 concepts;
- duplicate `(concept_id, stimulus_id)` pairs;
- a concept with anything other than ten training stimuli;
- non-contiguous row indices;
- subject manifests with different record hashes/order;
- a validation concept appearing in `train`;
- any input manifest whose `split` is `test`.

- [ ] **Step 3: Run RED**

Run:

```bash
env PYTHONPATH=experiments/samga_brain_rw conda run -n eeg_recon pytest -q experiments/samga_brain_rw/tests/test_splits.py
```

Expected: import failure.

- [ ] **Step 4: Implement deterministic assignment and atomic output**

Each output manifest stores:

- concept ID, split role, split rank, split digest;
- selected stimulus ID and stimulus digest for validation concepts;
- explicit row indices for all ten stimuli;
- ordered query/gallery IDs and their SHA-256 summary;
- source manifest path/hash and shared record hash;
- protocol config hash.

The CLI opens exactly `sub-01_train.json` through `sub-10_train.json` by constructed filename. It must never glob/scan the mixed directory, open `sub-XX_test.json`, or read the test-containing historical summary.

It writes `split_assignment.json`, `sub-XX_protocol.json`, and `manifest_summary.json` only after all ten train inputs validate.

- [ ] **Step 5: Run GREEN and generate Stage 0 manifests**

Run:

```bash
env PYTHONPATH=experiments/samga_brain_rw conda run -n test python experiments/samga_brain_rw/scripts/build_protocol_manifests.py \
  --protocol experiments/samga_brain_rw/configs/protocol_v1.json \
  --source-manifest-dir artifacts/samga_lora/manifests \
  --output-dir artifacts/samga_brain_rw/protocol/manifests
```

Expected:

- exactly 1,254/200/200 concepts;
- 12,540 training rows per subject;
- exactly 200 ordered queries/gallery rows in each validation split;
- all ten subjects share the same split and stimulus hashes.

- [ ] **Step 6: Commit code and the compact split assignment**

Commit the compact, immutable `split_assignment.json` and summary unconditionally before training; keep only subject-expanded manifests under ignored artifacts.

### Task 3: Enforce formal-test resealing in every development entry point

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/access.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/artifacts.py`
- Create: `experiments/samga_brain_rw/tests/test_access_seal.py`

**Interfaces:**
- `AccessScope = Literal["train", "val-dev", "val-confirm", "formal-refit", "formal-input", "formal-test"]`
- `require_typed_artifacts(scope: AccessScope, artifacts: Sequence[TypedArtifact]) -> None`
- `RefitArtifactLedger.create(cells: Sequence[RefitCell], output_path: Path) -> RefitArtifactLedger`
- `ConfirmationSeal.create(survivor_config_sha256: Sequence[str], registry_sha256: str, job_map_sha256: str, output_path: Path) -> ConfirmationSeal`
- `ConfirmationCellLedger.claim(seal_sha256: str, stage: int, role: str, subject: int, seed: int) -> CellClaim`
- `FormalPreparationSeal.create(final_selection_sha256: str, confirmation_registry_sha256: str, refit_plan_sha256: str, refit_artifact_ledger_sha256: str, formal_input_request_sha256: str, expected_formal_cell_keys_sha256: str, git_sha: str, upstream_sha: str, output_path: Path) -> FormalPreparationSeal`
- `FormalPreparationAudit.create(preparation_seal_sha256: str, expected_payload_sha256: str, output_path: Path) -> FormalPreparationAudit`
- `FormalInputLedger.claim(preparation_seal_sha256: str, preparation_audit_sha256: str, recipe_id: str) -> FormalInputClaim`
- `FinalRunSeal.create(final_selection_sha256: str, candidate_config_sha256: str, control_config_sha256: str, confirmation_registry_sha256: str, refit_plan_sha256: str, refit_artifact_ledger_sha256: str, formal_input_ledger_sha256: str, formal_job_map_sha256: str, git_sha: str, upstream_sha: str, output_path: Path) -> FinalRunSeal`
- `FinalRunSeal.verify(path: Path, expected_payload_sha256: str) -> FinalRunSeal`
- `FinalRunAudit.create(final_run_seal_sha256: str, expected_payload_sha256: str, output_path: Path) -> FinalRunAudit`

- [ ] **Step 1: Write failing guard tests**

Prove that development CLIs reject:

- `sub-XX_test.json`;
- any path containing the canonical test record hash `02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a`;
- feature metadata with `"split": "test"`;
- a checkpoint containing test metrics;
- `formal-test` scope without an immutable final-run seal.

- [ ] **Step 2: Run RED**

Run:

```bash
env PYTHONPATH=experiments/samga_brain_rw conda run -n eeg_recon pytest -q experiments/samga_brain_rw/tests/test_access_seal.py
```

- [ ] **Step 3: Implement content-based guards**

Every entry point accepts only typed manifests/artifacts whose mandatory sidecar declares schema, scope, source record hash, ordered ID hash, payload hash, and provenance hash.
Reject unknown schema versions, missing sidecars, absent scope, unrecognized payload types, renamed raw `.npy`/`.pt` inputs, and any metadata field that is not cryptographically bound.
The allowlist for development contains only `train`, `val-dev`, and the separately authorized one-time `val-confirm`; anything else fails closed.
Known formal-test record hashes and any `test_images` provenance remain explicit deny rules in addition to the allowlist.
A formal-preparation seal must be created before the first formal-test read. It binds the locked selection, confirmation registry, train-only refit plan/ledger, a deterministic formal-input request, the expected 100 formal cell keys, and git/upstream SHAs. It authorizes only claimed deterministic input materialization and cannot authorize EEG loading, similarities, predictions, or metrics. A separately created preparation-audit artifact must bind the exact seal hash; no experiment command creates that audit automatically.

After sealed input materialization, a final-run seal must bind:

- the final-selection record;
- selected candidate/config hash;
- strict control/config hash;
- confirmation registry hash;
- the component-level schedule/config hashes in the immutable refit-plan DAG, including both 60-epoch InternViT-SAMGA and 25-epoch brain-rw/CLIP-LoRA policies when Stage 1 wins;
- all 1,654-concept refit manifest hashes;
- the immutable refit-plan and refit-artifact-ledger hashes, whose cells bind subject set, seed, role, component schedule, checkpoint hash, adapter hash when present, train-cache hash when present, and frozen base-model hash;
- the formal-input-ledger hash, including every authorized derived formal cache hash when present;
- the exact immutable 100-row formal job-map hash;
- git and upstream SHAs;
- the explicit subject/seed grid through the job map.

Creating confirmation, formal-preparation, audit, and final seals is a separate command and must use exclusive creation (`O_CREAT | O_EXCL`).
Each `val-confirm` cell must claim an immutable ledger slot with `O_CREAT | O_EXCL` before evaluation and atomically attach its output hashes on completion; failed claims require an audited stale-claim recovery record.
The generic evaluator must reject `val-confirm` unless the matching confirmation seal, immutable job-map hash, and unconsumed cell claim all validate.
Formal-input materialization must reject a missing/mismatched preparation audit and must never load formal EEG or compute retrieval outputs. Formal evaluation must reject a missing/mismatched final-run audit, job-map hash, refit-ledger cell, input-ledger dependency, or unconsumed claim before loading formal EEG/images.

- [ ] **Step 4: Run GREEN and commit**

Run all three focused test modules before committing.

### Task 4: Lock model, cache, source, environment, and LoRA target provenance

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/provenance.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/internvit.py`
- Create: `experiments/samga_brain_rw/scripts/preflight.py`
- Create: `experiments/samga_brain_rw/scripts/check_stage0_cache_parity.py`
- Create: `experiments/samga_brain_rw/scripts/resolve_lora_targets.py`
- Create: `experiments/samga_brain_rw/tests/test_provenance.py`
- Create: `experiments/samga_brain_rw/tests/test_stage0_cache_parity.py`
- Create: `experiments/samga_brain_rw/tests/test_internvit_targets.py`

**Interfaces:**
- `build_provenance_manifest(inputs: ProvenanceInputs) -> dict[str, object]`
- `build_stage0_cache_parity(manifest_dir: Path, canonical_cache: Path, scopes: Sequence[str]) -> dict[str, object]`
- `resolve_internvit_components(model: nn.Module) -> tuple[nn.Module, Sequence[nn.Module]]`
- `resolve_lora_targets(model: nn.Module, first_block: int = 28, last_block: int = 36) -> Sequence[LoraTarget]`

- [ ] **Step 1: Write failing provenance tests**

Assert the exact upstream commit, model revision, three model-weight hashes, preprocessing hash, train-cache hash, cache shape `[16540, 5, 3200]`, logical layer route, pooling semantics, and normalization.

Pin these exact provenance oracles:

- canonical train manifest SHA-256 `42fd7316314eb02d69ee2234d4d8430afcfcc2a5f6834e9c7be64f38eccdbc85`;
- InternViT config SHA-256 `4fc4a1187b20575c0da8d27df2ad17f5ad6e8ac1c8b2af707bc8b263bd40c0a2`;
- preprocessor SHA-256 `0658115064c561026539aeeead9ed3b1a8e0cc90967df8c142849199f955d2b4`;
- InternViT weight SHA-256 values `9818659d13d932da8bc0c3b8ee15f5b5d68d8c94d66eb525be566066630111da`, `4f0c10e72d6f6513f421baa6ec843d5508657435059c1d18b6b5fd7789f9d5b7`, and `d21c4fe0bc4af1425cfae1a59a8f5fbb00fde9d8e2888325a60913ac61b0494d`;
- cache-generator git revision `a97b97a110c0fea7d4adafd5abce477c6cce525c`;
- canonical selected-cache SHA-256 `539c7b62ae41c8112e22b3ddc3a6566d997465a10c36d16c8f2378855ba94c71`;
- CLIP train-cache SHA-256 `a31c1871082e1f052da3d055702455b464ea2345890eee33e447e09328c45ebb`.
Compute and record a SHA-256 for every subject's source `train.pt`, not only its path and byte count.
Resolve the effective data root to `/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/EEG_Recon-RL/datasets/things_eeg_data` and the CLIP model to `/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/CLIP-ViT-B-32-laion2B-s34B-b79K`.
Count concepts from the non-empty canonical manifest, not filesystem directories, because `training_images/00010_alligator` is an empty extra directory.

- [ ] **Step 2: Write semantic target-resolution tests**

Using a small fake InternViT hierarchy, require the concrete V2.5 mapping for one-based blocks 28–36, which are zero-based Python indices 27–35:

- `encoder.layers.<index>.attn.qkv`, carrying the query/key/value semantic roles;
- `encoder.layers.<index>.attn.proj`, carrying the attention-output role;
- `encoder.layers.<index>.mlp.fc1`;
- `encoder.layers.<index>.mlp.fc2`.

Require exactly 4 modules per block and 36 modules total. Reject missing, duplicate, aliased, non-linear, out-of-range, or CLIP-style `q_proj`/`k_proj`/`v_proj` assumptions. Store both numbering conventions explicitly to prevent the 28/29 ambiguity.

- [ ] **Step 3: Run RED**

Run all three focused modules.

- [ ] **Step 4: Implement preflight and resolver**

`resolve_lora_targets.py` loads the pinned model offline and writes `artifacts/samga_brain_rw/protocol/lora_targets_v1.json` with exact concrete module paths, shapes, semantic roles, block numbering, and a payload hash.

`check_stage0_cache_parity.py` is CPU-only and exhaustive. It verifies exactly 12,540 `train` rows and 200 `val-dev` rows, ordered row/ID hashes, `[N, 5, 3200]` shape, float16 dtype, and bit-identical features against direct indexing of the pinned canonical cache. It checks split/view and ID-to-row semantics only and never loads InternViT online. It rejects `val-confirm`, formal-test scope/metadata, duplicate rows, and every cache/hash mismatch.

- [ ] **Step 5: Run CPU preflight**

Run:

```bash
env PYTHONPATH=experiments/samga_brain_rw conda run -n test python experiments/samga_brain_rw/scripts/preflight.py \
  --protocol experiments/samga_brain_rw/configs/protocol_v1.json \
  --manifest-dir artifacts/samga_brain_rw/protocol/manifests \
  --feature-directory artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/merged/train \
  --variant-directory artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/variants/train_idx0_patch_mean \
  --model-path /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/InternViT-6B-448px-V2_5/9d1a4344077479c93d42584b6941c64d795d508d \
  --upstream-root /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/reference_code/codes_for_papers/SAMGA \
  --output artifacts/samga_brain_rw/protocol/preflight.json
```

Expected: `"passed": true`, no test-cache path in the output, and environment versions recorded.

Then run:

```bash
env PYTHONPATH=experiments/samga_brain_rw conda run -n test python experiments/samga_brain_rw/scripts/check_stage0_cache_parity.py \
  --manifest-dir artifacts/samga_brain_rw/protocol/manifests \
  --canonical-cache artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/variants/train_idx0_patch_mean/features.npy \
  --scopes train val-dev \
  --output artifacts/samga_brain_rw/protocol/stage0_train_valdev_cache_parity.json
```

Expected: the exhaustive train/`val-dev` row and feature parity report passes without opening `val-confirm` or formal-test artifacts.

- [ ] **Step 6: Defer the GPU model-load smoke until Task 17's safe launcher exists**

Task 4 completes CPU/fake-model checks only. Task 17 later submits the dedicated no-array preflight launcher; never reuse `extract_v2_5_debug.slurm` with its default array because its final task reads test data.

### Task 5: Add explicit score-matrix artifacts and independent retrieval metrics

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/scores.py`
- Create: `experiments/samga_brain_rw/scripts/emit_scores.py`
- Create: `experiments/samga_brain_rw/tests/test_scores.py`

**Interfaces:**
- `ScoreArtifact.save(directory: Path, similarity: np.ndarray, query_ids: Sequence[str], gallery_ids: Sequence[str], metadata: Mapping[str, object]) -> None`
- `ScoreArtifact.load(directory: Path, allowed_scopes: Collection[str]) -> ScoreArtifact`
- `independent_retrieval_metrics(scores: np.ndarray, query_ids: Sequence[str], gallery_ids: Sequence[str]) -> RetrievalMetrics`

- [ ] **Step 1: Write failing artifact tests**

Assert stable tie-breaking, exact target lookup by ID rather than row position, Top-1/Top-5 counts, duplicate/missing ID rejection, score hash validation, and rejection of a formal-test artifact in development scope.

- [ ] **Step 2: Run RED**

Run:

```bash
env PYTHONPATH=experiments/samga_brain_rw conda run -n eeg_recon pytest -q experiments/samga_brain_rw/tests/test_scores.py
```

- [ ] **Step 3: Implement the artifact contract**

Write:

```text
<directory>/
  similarity.npy
  metadata.json
  predictions.csv
```

Metadata binds ordered query/gallery IDs and hashes, subject, seed, stage, config hash, split role, checkpoint hash, git SHA, and protocol hash.

- [ ] **Step 4: Expose the reusable score-emitter API; integrate it into the safe trainers/evaluator in Task 12**

Use the new evaluator on `val-dev` only. Keep the old formal-test evaluator unchanged.

- [ ] **Step 5: Run GREEN and commit**

### Task 6: Implement Stage 1 score-fusion grids

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/fusion.py`
- Create: `experiments/samga_brain_rw/scripts/run_score_fusion.py`
- Create: `experiments/samga_brain_rw/tests/test_fusion.py`

**Interfaces:**
- `assert_aligned(left: ScoreArtifact, right: ScoreArtifact) -> None`
- `querywise_zscore(scores: np.ndarray) -> np.ndarray`
- `convex_fusion(internvit: np.ndarray, clip: np.ndarray, alpha: float) -> np.ndarray`
- `temperature_fusion(internvit: np.ndarray, clip: np.ndarray, alpha: float, internvit_temperature: float, clip_temperature: float) -> np.ndarray`
- `reciprocal_rank_fusion(internvit: np.ndarray, clip: np.ndarray, k: int, internvit_weight: float) -> np.ndarray`
- `enumerate_stage1_configs() -> Sequence[FusionConfig]`

- [ ] **Step 1: Write failing exact-grid tests**

Require:

- 11 z-normalized convex configs for alpha `0.0` through `1.0`;
- 27 temperature configs for `3 × 3 × 3`;
- 9 RRF configs for `3 × 3`;
- 47 unique deterministic config IDs total.

- [ ] **Step 2: Add numeric and provenance tests**

Verify formulas on hand-computed matrices, no NaN for constant query rows, exact ID/hash equality, validation-only calibration, and tie-breaking by lower compute then config ID.

Lock query-wise z-score to float64 population variance (`ddof=0`) over each gallery row. Reject non-finite inputs; if the variance is exactly zero, return an all-zero row with no epsilon. Convex fusion is `alpha * z(S_I) + (1 - alpha) * z(S_C)`.

Lock temperature fusion to raw aligned cosine scores divided by positive scalar temperatures before convex combination: `alpha * S_I / T_I + (1 - alpha) * S_C / T_C`. Do not softmax either branch.

Lock RRF to one-based ordinal ranks from descending raw branch scores. Break branch-score ties by UTF-8 bytewise gallery-ID order, then compute `w / (k + rank_I) + (1 - w) / (k + rank_C)` in float64. Use no z-score, softmax, dense ranks, or average-tie ranks; break final fused-score ties by the same gallery-ID rule.

- [ ] **Step 3: Run RED, implement, and run GREEN**

- [ ] **Step 4: Lock the global stronger-single-branch comparator**

Select one comparator globally over the six `val-dev` cells by higher macro mean Top-1, then Top-5, lower measured inference cost, and lexicographic branch ID. Use that same locked branch in every paired cell; never select the stronger branch separately per cell.

- [ ] **Step 5: Defer data-dependent fusion execution to Task 18**

Task 18 trains both branches through the safe Task 12 entry points, emits aligned `val-dev` scores, evaluates all 47 configs, applies the Stage 1 gate, and locks at most one survivor. Historical formal-test scores are forbidden.

### Task 7: Implement paired gates, bootstrap, and candidate registry

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/statistics.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/registry.py`
- Create: `experiments/samga_brain_rw/scripts/aggregate_stage.py`
- Create: `experiments/samga_brain_rw/scripts/lock_survivor.py`
- Create: `experiments/samga_brain_rw/tests/test_statistics.py`
- Create: `experiments/samga_brain_rw/tests/test_registry.py`

**Interfaces:**
- `pilot_gate(candidate: CellMatrix, control: CellMatrix, stage: int) -> GateDecision`
- `confirmation_gate(candidate: CellMatrix, control: CellMatrix, bootstrap: BootstrapConfig) -> GateDecision`
- `two_way_cluster_bootstrap(delta: np.ndarray, config: BootstrapConfig) -> tuple[float, float]`
- `CandidateRegistry.append(decision: CandidateDecision) -> None`
- `CandidateRegistry.lock_stage_survivor(stage: int) -> CandidateDecision`

- [ ] **Step 1: Write failing gate-boundary tests**

Test exact inclusive/exclusive semantics:

- `>= 0.003` for Stage 1 pilot;
- `>= 0.005` for other pilot/confirmation deltas;
- CI lower bound strictly `> 0`;
- Top-5 `>= -0.002`;
- pilot positive cells `>= 4`;
- confirmation positive subjects `>= 8`;
- subject floor `>= -0.02`.

Use NumPy `default_rng(20260719)` for exactly 10,000 draws. For every draw, sample the declared number of subject indices with replacement and seed indices with replacement independently, take their Cartesian submatrix, and record its mean. Compute 0.025/0.975 quantiles with `method="linear"`.

- [ ] **Step 2: Test matrix completeness**

Reject missing/duplicate subject-seed cells, unequal effective batches/steps, mixed split roles, and mismatched data-order hashes.

When architectures match, require full task-initialization hashes to match. When candidate-specific modules differ, require the hash of the named shared-parameter intersection to match and record separate candidate/control-specific initialization hashes; never compare whole state dictionaries of different architectures.

- [ ] **Step 3: Implement deterministic registry semantics**

The registry is append-only JSONL plus a hash-chained compact JSON state. It allows at most one survivor per stage and refuses hyperparameter changes after `val-confirm`.

- [ ] **Step 4: Run GREEN and commit**

### Task 8: Refactor reusable SAMGA data/model primitives without changing historical code

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/data.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/model.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/upstream_samga.py`
- Create: `experiments/samga_brain_rw/tests/test_data.py`
- Create: `experiments/samga_brain_rw/tests/test_model_parity.py`

**Interfaces:**
- `ProtocolSubjectDataset(manifest_path: Path, scope: str, seed: int, selected_channels: Sequence[str], feature_cache: Path | None, smooth_probability: float)`
- `load_locked_upstream_components(upstream_root: Path, expected_commit: str) -> UpstreamComponents`
- `SAMGABaseConfig`
- `SAMGATaskModel`

- [ ] **Step 1: Write failing subset tests**

Assert:

- `train` exposes all ten stimuli for 1,254 concepts;
- each validation split exposes exactly one query/gallery stimulus per 200 concepts;
- no validation concept enters a training batch;
- subject EEG and shared image-cache row mappings remain exact;
- smoothing is deterministic from seed and source row.

- [ ] **Step 2: Write frozen-model parity tests**

Import the read-only upstream SAMGA component classes after verifying the upstream commit, and prove state-dict key, output-shape, normalization, router, loss, and optimizer-group parity on fixed synthetic inputs. Configure the InternViT task path explicitly with `layer_ids=(20, 24, 28, 32, 36)`, `image_dim=3200`, and `prior_center=28`.

- [ ] **Step 3: Implement by extraction, not by importing CLI modules**

Never call the upstream `train.py` main because it constructs and evaluates the formal test loader every epoch. Import verified classes/functions only, and keep both the upstream checkout and historical `experiments/samga_lora` code unchanged.

- [ ] **Step 4: Run GREEN and commit**

### Task 9: Implement Stage 2 preprocessing candidates

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/feature_transforms.py`
- Create: `experiments/samga_brain_rw/tests/test_feature_transforms.py`

**Interfaces:**
- `LayerNormTransform(enabled: bool, eps: float = 1e-6, affine: bool = False, compute_dtype: torch.dtype = torch.float32)`
- `TrainWhitening.fit(canonical_features: np.ndarray, canonical_train_rows: Sequence[int], eps: float = 1e-5) -> TrainWhitening`
- `TrainWhitening.transform(features: Tensor) -> Tensor`
- `SharedImagePreProjector`
- `SeparateImagePreProjector`

- [ ] **Step 1: Write failing tests**

Verify layer normalization independently per sample/layer, whitening fit only on train rows, serialized whitening statistics/hash, no validation leakage, shared/separate parameter counts, and unchanged output contracts.

Lock LayerNorm to non-affine float32 computation over the final 3,200-dimensional axis independently for each sample/layer, with `eps=1e-6`, before casting back to the task dtype.

Lock whitening to independent per-layer ZCA fitted on exactly the sorted, unique canonical cache row indices assigned to `train`: float64 `mu`, `X_centered = X - mu`, `C = X_centered.T @ X_centered / (n - 1)`, `W = U @ diag((maximum(eigenvalue, 0) + 1e-5) ** -0.5) @ U.T`, and float32 serialized mean/matrix with row-list and payload hashes.

- [ ] **Step 2: Implement one-factor-only candidate configs**

Create exact config IDs:

- `s2-layernorm-off` and `s2-layernorm-on`;
- `s2-whitening-off` and `s2-whitening-on`;
- `s2-preproj-shared` and `s2-preproj-separate`.

The shared candidate uses one `Linear(3200, 1024)` for all five layers; the separate candidate uses five independent `Linear(3200, 1024)` modules. Both retain five `Linear(1024, 512)` layer projectors and the locked router/loss.

The runner must reject configs that enable more than one Stage 2 factor.

- [ ] **Step 3: Run GREEN and commit**

### Task 10: Implement the residual cached-feature adapter and matched controls

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/adapters.py`
- Create: `experiments/samga_brain_rw/tests/test_adapters.py`

**Interfaces:**
- `ResidualFeatureAdapter(hidden_size: int, rank: int, layers: int)`
- `DenseBottleneckControl(hidden_size: int, target_parameters: int, layers: int)`
- `MatchedPerLayerProjectorControl(input_dim: int, output_dim: int, layers: int, target_parameters: int)`
- `match_dense_width(hidden_size: int, layers: int, target_parameters: int, tolerance: float = 0.01) -> int`
- `match_per_layer_widths(adapter_rank: int, layer_ids: Sequence[int] = (20, 24, 28, 32, 36), tolerance: float = 0.01) -> tuple[int, ...]`

- [ ] **Step 1: Write identity and gradient RED tests**

For:

```text
h'_l = h_l + gamma_l * B_l GELU(A_l LayerNorm(h_l))
```

assert `B_l` is zero, `gamma_l == 1`, output is bit-identical to input at initialization, and gradients reach `B_l` on the first step and `A_l` after the adapter departs identity.

- [ ] **Step 2: Write parameter-control tests**

Require dense bottleneck and separate-projector controls to fall within 1% of the adapter parameter count or abort before training.

Define the dense control as one global residual bottleneck over flattened `[B, L * D]`: non-affine LayerNorm, bias-free `Linear(L*D, r)`, GELU, zero-initialized bias-free `Linear(r, L*D)`, and scalar gamma initialized to one. This permits cross-layer mixing and matches the per-layer adapter's dominant parameter count exactly while remaining a distinct architecture.

Lock every adapter/control LayerNorm to non-affine float32 computation with `eps=1e-6`; make `A`, `B`, `R`, and `Q` bias-free and use one trainable scalar `gamma_l=1` per layer.

Define the matched separate-projector control as a residual output-space branch per layer: `q'_l = q_l + gamma_l * Q_l GELU(R_l LayerNorm(h_l))`, with `R_l: 3200 -> m_l` and zero-initialized `Q_l: m_l -> 512`.

For adapter rank `r`, `D=3200`, `O=512`, and `L=5`, lock `P_adapter = L * (2 * D * r + 1) = 32000 * r + 5` and `P_projector = (D + O) * sum(m_l) + L = 3712 * M + 5`. Choose `M = round(32000 * r / 3712)` with half ties downward, distribute `M` evenly, and assign the remainder to the earliest locked layer order `[20, 24, 28, 32, 36]`.

Lock the resulting vectors and counts:

- rank 8: widths `[14, 14, 14, 14, 13]`, adapter 256,005 parameters, control 256,133, error 0.050%;
- rank 16: widths `[28, 28, 28, 27, 27]`, adapter 512,005 parameters, control 512,261, error 0.050%;
- rank 32: widths `[56, 55, 55, 55, 55]`, adapter 1,024,005 parameters, control 1,024,517, error 0.050%.

Persist the widths, counts, absolute error, and relative error; reject any relative error above 1%.

- [ ] **Step 3: Implement exact grid generation**

Generate ranks `{8, 16, 32}` crossed with LR ratios `{0.05, 0.10}` and no other candidates.

Lock the Stage 2 control map: LayerNorm-on versus LayerNorm-off; whitening-on versus whitening-off; separate full preprojectors versus the locked shared-preprojector baseline; and every averaging/SWA artifact versus the raw epoch-60 checkpoint from the identical trajectory.

Each adapter configuration is compared with adapter-off identity, a same-rank/same-LR-ratio global dense control, and the layer-specific matched projector above under the same cell, schedule, shared-parameter initialization hash, and data-order hash. It is eligible only if it passes the complete Stage 2 gate against all three controls. Before confirmation, lock the strongest of those controls globally over the six pilot cells by macro Top-1, then Top-5, lower inference cost, and config ID. Among eligible Stage 2 candidates select macro Top-1, then Top-5, fewer added parameters, then lexicographic config ID.

Treat the off/shared/raw-epoch configurations as aliases of one baseline artifact when their resolved hashes are identical, not as independent candidates.

- [ ] **Step 4: Run GREEN and commit**

### Task 11: Implement checkpoint averaging and SWA candidates

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/checkpoints.py`
- Create: `experiments/samga_brain_rw/scripts/build_averaged_checkpoint.py`
- Create: `experiments/samga_brain_rw/tests/test_checkpoints.py`

**Interfaces:**
- `average_state_dicts(paths: Sequence[Path]) -> dict[str, Tensor]`
- `swa_state_dicts(paths: Sequence[Path]) -> dict[str, Tensor]`

- [ ] **Step 1: Write failing tests**

Reject different configs, key sets, tensor shapes, seeds, subjects, schedules, incomplete windows, and optimizer-stage mismatches. Assert arithmetic results on synthetic checkpoints.

Define arithmetic averaging as post-hoc equal-weight model-state averaging and SWA as `torch.optim.swa_utils.AveragedModel` updated once at the end of each epoch in the same last-5/last-10 window. Average floating model parameters only; require non-floating buffers to be identical and copy them unchanged. Never average optimizer state.

Because the locked late-stage LR is constant and there are no running-statistic layers to update, compare resulting state hashes/tensors. If arithmetic and SWA are equivalent, record the SWA config as an alias of the arithmetic artifact and reuse one evaluation; never present the alias as independent evidence.

- [ ] **Step 2: Implement last-5/last-10 only**

Candidate IDs are exactly:

- `s2-avg-last5`;
- `s2-avg-last10`;
- `s2-swa-last5`;
- `s2-swa-last10`.

The trainer must retain consecutive epoch-51 through epoch-60 model snapshots until all averaging artifacts validate. Last-5 means epochs 56–60 and last-10 means epochs 51–60; cleanup before successful averaging is forbidden.

Every averaging/SWA candidate uses the raw epoch-60 checkpoint from the identical training trajectory as its strict paired control. If arithmetic and SWA hashes are identical, their registry entries alias one evaluation.

- [ ] **Step 3: Run GREEN and commit**

### Task 12: Build a unified development-only trainer and Stage 2 runner

**Files:**
- Create: `experiments/samga_brain_rw/train.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/brainrw.py`
- Create: `experiments/samga_brain_rw/train_brainrw.py`
- Create: `experiments/samga_brain_rw/scripts/check_baseline_parity.py`
- Create: `experiments/samga_brain_rw/scripts/emit_brainrw_scores.py`
- Create: `experiments/samga_brain_rw/evaluate.py`
- Create: `experiments/samga_brain_rw/scripts/run_stage2_cell.sh`
- Create: `experiments/samga_brain_rw/tests/test_train_cli.py`
- Create: `experiments/samga_brain_rw/tests/test_run_stage2_cell.py`
- Create: `experiments/samga_brain_rw/tests/test_brainrw_training_guard.py`
- Create: `experiments/samga_brain_rw/tests/test_brainrw_scores.py`
- Create: `experiments/samga_brain_rw/tests/test_baseline_parity.py`

**Interfaces:**
- `train.py --scope train --validation-scope val-dev --stage 0|2 --subject <1..10> --seed <int> --resume <none|checkpoint.pt> --config <config.json> --manifest <sub-XX_protocol.json> --feature-cache <features.npy> --output-dir <run-directory>`
- `evaluate.py --scope val-dev --subject <1..10> --seed <int> --config <config.json> --manifest <sub-XX_protocol.json> --feature-cache <features.npy> --checkpoint <checkpoint.pt> --output-dir <score-directory>`
- `evaluate.py --scope val-confirm --subject <1..10> --seed <int> --confirmation-seal <seal.json> --job-map <map.json> --cell-claim <claim.json> --config <config.json> --manifest <sub-XX_protocol.json> --feature-cache <features.npy> --checkpoint <checkpoint.pt> --output-dir <score-directory>`
- `BrainRWCLIPLoRAModel` builds the audited BrainMLP and exact CLIP LoRA target set.
- `BrainRWDevelopmentDataset` and `BrainRWCollator` load only split-registry train/validation rows and their raw image paths/pixels.
- `train_brainrw.py --scope train --validation-scope val-dev --subject <1..10> --seed <int> --resume <none|checkpoint.pt> --config experiments/samga_brain_rw/configs/brainrw_clip_lora_v1.json --manifest <sub-XX_protocol.json> --clip-path /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/CLIP-ViT-B-32-laion2B-s34B-b79K --output-dir <run-directory>`
- `emit_brainrw_scores.py --scope val-dev --subject <1..10> --seed <int> --checkpoint <checkpoint.pt> --manifest <sub-XX_protocol.json> --output-dir <score-directory>`
- `check_baseline_parity.py --run-dir <fresh-baseline-run> --scope val-dev --output <parity.json>`

- [ ] **Step 1: Write CLI RED tests**

Require all paths via CLI/config, forbid test inputs, forbid overwrite, record the fully resolved run/config/input hashes, and enforce the paired initialization/data-order rules from Task 7.

Open only explicitly enumerated `sub-01_train.json` through `sub-10_train.json`; never glob the mixed source-manifest directory or read its test-containing summary. Lock `[4,63,250] -> mean over four trials -> [63,250] -> the 17 ordered posterior channels -> [17,250]`, training smoothing probability 0.3, and `force_global=True` for every SAMGA validation evaluation.

The development-only brain-rw trainer has no test dataset argument or loader. Its dataset exposes only split-registry rows, its model factory resolves and hashes the exact BrainMLP/CLIP-LoRA modules, and its score emitter is covered by a test proving that renamed or metadata-free test artifacts fail before image/EEG loading.

Require explicit subject and seed on every train/evaluate command and verify both against the manifest, checkpoint metadata, job-map row, and output run key. Training accepts only explicit `--resume none` or a checkpoint path. Resume restores model, optimizer, scheduler, epoch/step, RNG, sampler, and DataLoader-generator state and rejects every config/input/subject/seed/data-order mismatch. Evaluators use `--checkpoint`, never `--resume`, and reject a checkpoint whose recorded subject or seed differs.

- [ ] **Step 2: Implement resumable atomic checkpoints**

Save task state, candidate state, optimizer state, epoch, RNG states, sampler/DataLoader-generator state, data-order hash, effective batch, steps, environment, git/upstream/model/cache hashes, and validation metrics.

Retain task/candidate snapshots for every epoch 51–60 regardless of `save_total_limit`; averaging completion metadata must exist before any snapshot cleanup.

For online InternViT runs, save only LoRA adapter tensors plus their target-manifest/base-model hashes; never serialize frozen 11 GB base weights.

- [ ] **Step 3: Run CPU max-step smoke**

Use synthetic fixtures and `--max-train-steps 1`.

- [ ] **Step 4: Define the one-batch A40 smoke contract; defer submission to Task 18**

Expected: candidate and strict control complete, score artifacts validate, and no formal-test path appears in logs/manifests.

For the frozen baseline, compare the in-loop validation scores with a separate evaluation of the saved checkpoint, repeat score emission, and reload once more. Require identical ordered IDs, Top-1/Top-5 predictions and metrics, and score tensors with maximum absolute difference at most `1e-6`; record all hashes in `baseline_parity.json`.

Run a separate one-step development-only brain-rw/CLIP-LoRA smoke and prove its manifest/score sidecars contain only `train` and `val-dev`.

- [ ] **Step 5: Prepare the fresh matched-split Stage 0 baseline; Task 18 launches it after Task 17**

Prepare immutable job rows that train the six frozen InternViT-SAMGA pilot cells from scratch on the new `train` split, emit `val-dev` scores, and complete reload/repeat parity. Task 18 launches them, and launches Stage 1/2 candidates only after these baseline cells and the safe brain-rw trainer validate.

### Task 13: Implement true online InternViT and zero-LoRA parity

**Files:**
- Modify: `experiments/samga_brain_rw/samga_brain_rw/internvit.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/lora.py`
- Create: `experiments/samga_brain_rw/scripts/check_online_cache_parity.py`
- Create: `experiments/samga_brain_rw/tests/test_lora.py`
- Create: `experiments/samga_brain_rw/tests/test_online_parity.py`

**Interfaces:**
- `OnlineInternViTProvider(model_path: Path, target_manifest_path: Path, layer_ids: Sequence[int], pooling: str, normalization: str)`
- `inject_locked_lora(model: nn.Module, target_manifest: Path, rank: int, alpha: int, dropout: float)`
- `online_hidden_states(pixel_values) -> Tensor[B, 5, 3200]`

- [ ] **Step 1: Write fake-model RED tests**

Assert all 36 exact target paths, rank/alpha/dropout, frozen base parameters, intended trainable parameters only, zero-LoRA equality, gradient isolation, and deterministic save/reload. Require 1,843,200 trainable LoRA parameters for rank 4 and 3,686,400 for rank 8.

- [ ] **Step 2: Implement explicit module injection**

Do not use suffix-only PEFT targeting. Resolve each full path from `lora_targets_v1.json` and abort if the loaded model hierarchy or base tensor hash differs.

- [ ] **Step 3: Implement the memory-safe partial forward**

Run embeddings and blocks 1–27 frozen, capture patch means at block outputs 20 and 24, detach the block-27 output, then checkpoint blocks 28–36 with `use_reentrant=False`, capturing patch means at outputs 28, 32, and 36. Stop after block 36. Emit `[B, 5, 3200]`, exclude CLS, and apply no additional normalization.

- [ ] **Step 4: Add BF16/checkpointing controls**

Enable gradient checkpointing and record microbatch, accumulation, effective batch, peak memory, and autocast dtype.

- [ ] **Step 5: Define the 32-sample online/cache parity job; defer submission to Task 18**

This Task 13 GPU check is distinct from Stage 0 parity: it compares 32 freshly decoded images through frozen online InternViT against cached vectors and validates extraction semantics; it does not replace the exhaustive Stage 0 split/row parity report.

Require:

- `rtol=1e-3`;
- `atol=1e-4`;
- mean cosine `>= 0.9999`;
- identical Top-1/Top-5 predictions and aggregate metrics between cached and online-frozen outputs;
- reported mean top-20 set overlap.

If any requirement fails, stop Stage 3 and reconcile extraction semantics before training.

### Task 14: Implement the Stage 3 epoch-46 online transition

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/schedule.py`
- Create: `experiments/samga_brain_rw/scripts/run_stage3_cell.sh`
- Create: `experiments/samga_brain_rw/tests/test_schedule.py`
- Create: `experiments/samga_brain_rw/tests/test_stage3_transition.py`

**Interfaces:**
- `VisionSchedule.provider_for_epoch(epoch: int) -> Literal["cache", "online-frozen", "online-lora"]`
- epochs 1–45: cache;
- epochs 46–60: online-frozen or online-LoRA.

- [ ] **Step 1: Write transition RED tests**

Assert the exact boundary, task optimizer continuity, online optimizer group creation, LR ratios, paired batch-order hashes, and no duplicate/omitted optimization step at transition.

Train one epoch-45 warm-up checkpoint per subject-seed and fork all four LoRA configs plus the online-frozen control from that exact checkpoint hash. Preserve task-parameter Adam state by adding the LoRA parameter group at epoch 46; do not rebuild the task optimizer.

- [ ] **Step 2: Implement four exact Stage 3 configs**

- rank 4, LR ratio 0.01, alpha 8;
- rank 4, LR ratio 0.05, alpha 8;
- rank 8, LR ratio 0.01, alpha 16;
- rank 8, LR ratio 0.05, alpha 16;
- dropout 0.05 for all.

- [ ] **Step 3: Implement save/reload and one-step GPU-smoke checks; defer execution to Task 18**

- [ ] **Step 4: Prepare the paired 3 × 2 pilot job map; defer launch to Task 18**

Every candidate cell has an online-frozen control with identical warm-up, online steps, microbatch, accumulation, task initialization, and data-order hash.

- [ ] **Step 5: Aggregate and lock at most one Stage 3 survivor**

Choose mean `val-dev` Top-1, then Top-5, lower rank, lower LR ratio. Stop the family if the +0.50-point gate fails.

### Task 15: Implement Stage 4 shared subject-aware LoRA and cache rebuilding

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/multisubject.py`
- Create: `experiments/samga_brain_rw/scripts/train_shared_lora.py`
- Create: `experiments/samga_brain_rw/scripts/extract_shared_lora_cache.py`
- Create: `experiments/samga_brain_rw/scripts/run_stage4_cell.sh`
- Create: `experiments/samga_brain_rw/tests/test_multisubject.py`
- Create: `experiments/samga_brain_rw/tests/test_shared_cache.py`

**Interfaces:**
- one shared adapter per seed;
- pooled training subjects exactly `01`–`10` with explicit subject IDs for every seed-specific shared adapter;
- exact Stage 3 winner, with no new hyperparameters.

- [ ] **Step 1: Write shared-sampler RED tests**

Assert deterministic, balanced subject sampling; train-concept-only rows; no validation/test rows; independent adapter initialization per seed; and matched shared-frozen control ordering. Use image-major sampling so every batch contains unique image IDs and never treats another subject's EEG for the same image as a contrastive negative.

The pilot shared adapter trains on train-concept EEG from all ten subjects but is gated only on `val-dev` subjects 01, 05, and 08. Confirmation repeats the identical all-ten-subject training method for each seed and evaluates all ten subjects; it introduces no subject-pool choice.

Subjects `02, 03, 04, 06, 07, 09, 10` contribute `train`-scope gradients only during the Stage 4 pilot; their `val-dev`, `val-confirm`, and formal-test rows remain unopened.

- [ ] **Step 2: Implement deterministic train/validation cache extraction**

Cache metadata binds adapter checkpoint hash and ordered IDs. Build paired candidate and shared-frozen train/validation caches under the same job-map hash. The development extractor always refuses formal-test manifests. A separate formal-input materializer may create Stage 4 formal-test caches only after the final selection, a valid `FormalPreparationSeal`, its separately recorded audit, and an unconsumed input-recipe claim all validate.

After cache creation, train downstream SAMGA from scratch separately for every evaluated subject/seed on candidate and shared-frozen caches, with identical task initialization, data order, schedule, and `force_global=True` validation. Do not reuse the Stage 3 downstream checkpoint.

- [ ] **Step 3: Prepare the one-seed smoke and 3 × 2 pilot; defer execution to Task 18**

- [ ] **Step 4: Apply the Stage 4 gate**

If shared LoRA fails, retain only the already locked Stage 3 survivor; introduce no fallback hyperparameters.

### Task 16: Implement conditional Stage 5 dual-branch training

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/dual_branch.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/dual_data.py`
- Create: `experiments/samga_brain_rw/scripts/run_stage5_cell.sh`
- Create: `experiments/samga_brain_rw/tests/test_dual_branch.py`
- Create: `experiments/samga_brain_rw/tests/test_dual_data.py`

**Interfaces:**
- `DualBranchBatch(eeg: Tensor, subject_ids: Tensor, image_ids: Sequence[str], internvit_layer_features: Tensor, clip_pixel_values: Tensor)`
- `DualBranchFusion` with scalar parameters `a`, `t_i`, and `t_c`;
- explicit InternViT and CLIP branches;
- no modification to SAMGA's subject-aware router.

- [ ] **Step 1: Write formula RED tests**

First prove the dual dataset/collator emits the same ordered IDs with cached InternViT features and decoded CLIP pixels simultaneously; reject either branch's row/hash mismatch.

Define `ce` below as SAMGA's symmetric contrastive cross-entropy, `0.5 * (CE(S, y) + CE(S.T, y))`, and assert:

```python
lam = torch.sigmoid(a)
t_i = torch.exp(raw_t_i)
t_c = torch.exp(raw_t_c)
fused = lam * s_i / t_i + (1.0 - lam) * s_c / t_c
loss = ce(fused, y) + 0.25 * ce(s_i, y) + 0.25 * ce(s_c, y)
```

and initial `a = raw_t_i = raw_t_c = 0`.

- [ ] **Step 2: Write arm/schedule tests**

Allow exactly:

- InternViT only;
- CLIP only;
- dual frozen;
- dual CLIP-LoRA.

Epochs 1–45 train locked task branches with both visual backbones frozen. During epochs 46–60, continue task branches, optimize fusion scalars, and activate CLIP LoRA only in the CLIP-LoRA arm. It uses rank 32, alpha 32, dropout 0.05, targets q/k/v/out/fc1/fc2/visual_projection, and visual/task LR ratio 0.10. Apply the shared `visual_projection` to every selected intermediate-layer CLS feature in both the frozen and LoRA arms, configure the branch projector for the resulting 512-dimensional inputs, require a non-zero `visual_projection` gradient in the LoRA arm, and keep `force_global=True` for validation.

- [ ] **Step 3: Implement only if the Stage 1 registry gate passed**

The launcher must read the locked Stage 1 decision and abort otherwise.

- [ ] **Step 4: Prepare pilot aggregation/locking; Task 18 launches and applies it after Task 17**

Compare with the better of the strongest single branch and locked Stage 1 fusion.

### Task 17: Add SLURM launchers, queue checks, and log discipline

No GPU command defined in Tasks 12–16 may be submitted before this task's launcher tests and immutable job-map tests pass.

**Files:**
- Create: `experiments/samga_brain_rw/slurm/preflight_debug.slurm`
- Create: `experiments/samga_brain_rw/slurm/online_parity_debug.slurm`
- Create: `experiments/samga_brain_rw/slurm/pilot_array.slurm`
- Create: `experiments/samga_brain_rw/slurm/confirmation_array.slurm`
- Create: `experiments/samga_brain_rw/scripts/build_job_map.py`
- Create: `experiments/samga_brain_rw/scripts/submit_pilot.py`
- Create: `experiments/samga_brain_rw/tests/test_slurm.py`

- [ ] **Step 1: Write static launcher tests**

Require:

- lowercase `--gres=gpu:a40:1`;
- `.out/.err` below `logs/samga_brain_rw`;
- offline environment variables;
- validated array bounds;
- explicit subject/seed/config lookup;
- explicit `source` of the conda profile followed by `conda activate test`;
- `PYTHONPATH=experiments/samga_brain_rw`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`;
- no test path in preflight/pilot/confirmation launchers;
- `debug` time `<= 00:30:00`;
- no job submission when an idempotent completion artifact already validates;
- rejection when the submitted job-map hash, SLURM array bounds, and selected row do not agree.

- [ ] **Step 2: Implement immutable job maps and queue-aware submission**

Write one atomic job map per stage and homogeneous resource class. Every sorted row binds `array_index`, stage, role, config ID/hash, input-bundle hash, subject, seed, argv list, partition/GPU/CPU/memory/time request, stdout/stderr paths, completion path, and expected completion schema. Hash the full map; launchers require that hash and refuse missing, duplicate, reordered, or out-of-range rows.

Retries reuse the same row/run key and may run only when no valid completion exists. Any stale partial claim is recovered through an explicit audited recovery record, never by deleting or overwriting the original claim.

Then inspect the queue before submission:

Before longer pilots, inspect:

```bash
squeue -h -p debug,i64m1tga40u,i64m1tga40ue,emergency_gpua40 \
  -o "%.10i %.14P %.10u %.2t %.10M %.6D %R"
```

Use `debug` for short smoke and `i64m1tga40u` for longer A40 pilots unless the lower tier is clearly more congested.

- [ ] **Step 3: Run static tests and submit only the current-stage jobs**

Do not submit Stage 3–5 arrays before their prerequisite gates pass.

### Task 18: Execute Stage 0 baseline parity and the prioritized pilot ladder

**Files:**
- Create generated: `artifacts/samga_brain_rw/protocol/preflight.json`
- Create generated: `artifacts/samga_brain_rw/registry/decisions.jsonl`
- Create generated: `results/samga_brain_rw/stage-*/pilot_summary.json`
- Create generated: `results/samga_brain_rw/stage-*/pilot_cells.csv`

- [ ] **Step 1: Verify all local tests**

Run:

```bash
env PYTHONPATH=experiments/samga_brain_rw conda run -n eeg_recon pytest -q experiments/samga_brain_rw/tests
env PYTHONPATH=experiments/samga_brain_rw conda run -n test python -m compileall -q experiments/samga_brain_rw
git diff --check
```

- [ ] **Step 2: Run Stage 0**

Generate and hash:

- protocol manifests;
- provenance manifest;
- exact LoRA targets;
- environment report;
- exhaustive CPU train/`val-dev` cache-view parity report at `artifacts/samga_brain_rw/protocol/stage0_train_valdev_cache_parity.json`;
- frozen baseline `val-dev` scores.

- [ ] **Step 3: Run Stage 1**

Train both fresh branches, emit aligned scores, evaluate the exact fusion grid, apply the gate, and lock or stop Stage 1 before starting Stage 2.

- [ ] **Step 4: Run Stage 2**

Run the frozen-feature one-factor candidates and controls, apply the gate, and lock or stop Stage 2 before starting Stage 3.

- [ ] **Step 5: Run Stage 3 only after online parity passes**

- [ ] **Step 6: Run Stage 4 only if Stage 3 passes**

- [ ] **Step 7: Run Stage 5 only if Stage 1 passed and all higher-priority applicable stages are complete**

- [ ] **Step 8: Record negative results and stop failed families**

Do not expand a failed family or alter its grid without a reviewed specification revision.

### Task 19: One-time val-confirm ranking and final refit preparation

**Files:**
- Create: `experiments/samga_brain_rw/scripts/create_confirmation_seal.py`
- Create: `experiments/samga_brain_rw/scripts/build_confirmation_job_map.py`
- Create: `experiments/samga_brain_rw/scripts/prepare_confirmation.py`
- Create: `experiments/samga_brain_rw/scripts/run_confirmation.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/refit.py`
- Create: `experiments/samga_brain_rw/samga_brain_rw/formal_inputs.py`
- Create: `experiments/samga_brain_rw/scripts/build_refit_plan.py`
- Create: `experiments/samga_brain_rw/scripts/run_formal_refit.py`
- Create: `experiments/samga_brain_rw/scripts/select_final_candidate.py`
- Create: `experiments/samga_brain_rw/scripts/create_formal_preparation_seal.py`
- Create: `experiments/samga_brain_rw/scripts/record_formal_preparation_audit.py`
- Create: `experiments/samga_brain_rw/scripts/materialize_formal_inputs.py`
- Create: `experiments/samga_brain_rw/scripts/build_formal_job_map.py`
- Create: `experiments/samga_brain_rw/scripts/create_final_run_seal.py`
- Create: `experiments/samga_brain_rw/tests/test_final_selection.py`
- Create: `experiments/samga_brain_rw/tests/test_confirmation_ledger.py`
- Create: `experiments/samga_brain_rw/tests/test_formal_refit.py`
- Create: `experiments/samga_brain_rw/tests/test_formal_input_seal.py`

**Interfaces:**
- `RefitJob(job_id: str, kind: Literal["component_train", "pooled_adapter_train", "train_cache", "downstream_train", "composition"], family: Literal["stage1", "stage2", "stage3", "stage4", "stage5"], role: str, subjects: tuple[int, ...], seed: int, config_sha256: str, schedule_sha256: str, manifest_sha256s: tuple[str, ...], dependency_ids: tuple[str, ...])`
- `RefitPlan.create(final_selection_sha256: str, jobs: Sequence[RefitJob], output_path: Path) -> RefitPlan`
- `run_formal_refit.py --scope formal-refit --validation-scope none --refit-plan <refit-plan.json> --job-id <immutable-job-id> --resume <none|checkpoint.pt> --output-dir <run-directory>`
- `run_confirmation.py --scope val-confirm --confirmation-seal <seal.json> --job-map <map.json> --cell-claim <claim.json> --method-family <stage1|stage2|stage3|stage4|stage5> --role <candidate|control> --subject <1..10> --seed <42..46> --component-ledger <ledger.json> --output-dir <score-directory>`

- [ ] **Step 1: Write survivor-selection RED tests**

Require:

- only pilot-gated, locked stage survivors;
- one `val-confirm` evaluation per survivor/control on the complete 10 × 5 grid;
- confirmation gate;
- ranking by absolute Top-1;
- within 0.001 Top-1: fewer encoders, lower FLOPs, fewer parameters, higher Top-5, lexicographic config ID.

- [ ] **Step 2: Implement method-aware confirmation with no tuning path**

After pilot survivors are frozen, expand every survivor/control to the complete 10 × 5 grid using `train` scope only and the already locked configs, schedules, seeds, and epochs. This preparation may not open `val-confirm`.

The immutable confirmation plan handles every eligible family:

- Stage 1 trains separate InternViT-SAMGA and brain-rw/CLIP-LoRA component jobs with their locked 60-epoch and 25-epoch policies, composes the locked fusion, and aliases the globally selected stronger component as control;
- Stage 2 prepares the exact candidate and its registered strict control, including epoch-60 dependencies for averaging/SWA;
- Stage 3 prepares the locked online-LoRA and online-frozen trajectories;
- Stage 4 trains one candidate shared adapter and one shared-frozen control per seed from the explicit ten-manifest subject set `01`–`10`, builds train caches, and trains 50 downstream candidate plus 50 control cells from scratch;
- Stage 5 prepares both locked input branches and the exact dual-branch/control composition.

Then create one exclusive confirmation seal and an immutable job map containing any family-specific `val-confirm` asset rows plus every survivor/control 10 × 5 evaluation cell. Stage 4 validation-cache asset rows use exclusive claims and may not compute metrics. Each evaluation claims exactly one ledger slot; the method-aware evaluator verifies its seal, map row, typed component dependencies, subject, seed, and unconsumed claim before opening `val-confirm`.

After a confirmation result is recorded, the registry refuses every config, component, epoch, or schedule change. Repeated or early confirmation calls fail before loading EEG or image features.

- [ ] **Step 3: Select and refit the chosen pair without formal-test access**

Create an immutable `FinalSelectionRecord`, then build a hash-sealed `RefitPlan` DAG. Every component policy has its own schedule hash; no scalar epoch/schedule may stand in for a heterogeneous method.

Run every refit job on all 1,654 non-test concepts. `formal-refit` rejects validation and formal-test artifacts, early stopping, checkpoint selection, and runtime hyperparameter overrides. Exact resume is allowed only when the job/config/input/subject-set/seed and saved optimizer/RNG/data-order state all match.

Stage 1 refits both branch components under their separate locked schedules and records the composition dependencies. Stage 4 refits, for each seed, one candidate shared LoRA and one shared-frozen control from all ten subject manifests, materializes only all-1,654-concept train caches, and then refits the 50 downstream candidate plus 50 matched-control cells. It must not read or create a formal-test cache in this step.

Finalize a train-only `RefitArtifactLedger` that binds every job, subject set, seed, role, component schedule/config, manifest set, checkpoint, adapter, train cache, base model, and dependency hash.

- [ ] **Step 4: Create the formal-preparation seal and stop for the first explicit audit**

Without opening formal-test paths, build a `FormalInputRequest` that contains deterministic family-specific cache recipes, expected formal record hashes, and the expected 100 evaluation cell keys. Create an exclusive `FormalPreparationSeal` binding the final selection, confirmation registry, refit plan/ledger, request, cell-key hash, git SHA, and upstream SHA.

Stop before the first formal-test read. A separate `FormalPreparationAudit` must be created only after an explicit audit of the exact preparation-seal payload/hash; no training, selection, materialization, or submission command may create it automatically.

- [ ] **Step 5: After the first audit, materialize sealed inputs and create the final-run seal**

Require the exact preparation audit, then claim each formal-input recipe with `O_CREAT | O_EXCL`. For a Stage 4 winner, generate candidate/control formal-test caches from the locked shared adapters/base model and record ordered IDs, manifest, preprocessing, adapter/base-model, and payload hashes in `FormalInputLedger`. This phase must not load formal EEG, compute similarities, predictions, or metrics. Other families record seal-bound direct-input descriptors and their expected manifest/record hashes when no derived cache is needed; the ledger is never empty or implicit.

Build the immutable 100-row formal job map (candidate/control × 10 subjects × 5 seeds) from the refit and formal-input ledgers. Create `FinalRunSeal` with `O_CREAT | O_EXCL`, binding the exact map hash and every upstream ledger/hash.

Stop again. A separate `FinalRunAudit` must bind the exact final-seal payload/hash before any formal cell claim or metric evaluation is allowed.

### Task 20: Formal paired evaluation and final reporting

**Files:**
- Create: `experiments/samga_brain_rw/samga_brain_rw/formal_ledger.py`
- Create: `experiments/samga_brain_rw/formal_evaluate.py`
- Create: `experiments/samga_brain_rw/scripts/record_final_run_audit.py`
- Create: `experiments/samga_brain_rw/scripts/run_formal_cell.sh`
- Create: `experiments/samga_brain_rw/scripts/aggregate_formal.py`
- Create: `experiments/samga_brain_rw/slurm/formal_array.slurm`
- Create: `experiments/samga_brain_rw/tests/test_formal_ledger.py`
- Create: `experiments/samga_brain_rw/tests/test_formal_aggregation.py`
- Create generated: `results/samga_brain_rw/final/summary.json`
- Create generated: `results/samga_brain_rw/final/paired_metrics.csv`
- Create generated: `results/samga_brain_rw/final/per_subject_metrics.csv`
- Create generated: `results/samga_brain_rw/final/RESULTS_EN.md`
- Create generated: `results/samga_brain_rw/final/RESULTS_ZH.md`

- [ ] **Step 1: Write formal aggregation RED tests**

Reject missing/duplicate 10 × 5 cells, mismatched pair provenance, invalid predictions, non-standard retrieval, wrong component policy, wrong preparation/final seal or audit, wrong job-map row, and repeated evaluation.

Test the immutable 100-row job map (candidate/control × 10 subjects × 5 seeds), `O_CREAT | O_EXCL` claim creation, atomic completion, stale-claim audit recovery, checkpoint/adapter/cache hash verification against the refit and formal-input ledgers, and refusal before any formal EEG/image payload is loaded.

- [ ] **Step 2: Implement seal-aware evaluation, cell consumption, and reporting**

`formal_evaluate.py` verifies the final-run seal, separately recorded `FinalRunAudit`, exact formal job-map hash/row, refit-ledger cell, formal-input-ledger dependency, and unconsumed claim before loading formal-test inputs. It atomically records metric/prediction hashes into the claim on success. Aggregation starts only when all 100 claims are complete and valid.

Report:

- candidate/control Top-1 and Top-5;
- paired deltas and two-way cluster-bootstrap CI;
- positive subjects and worst subject delta;
- difference from 89.02%/98.87%;
- whether 91.30% is numerically exceeded;
- trainable parameters, FLOPs, peak GPU memory, training time, encoder count, and inference latency.

- [ ] **Step 3: Run the single formal 10 × 5 paired grid only after the second explicit audit**

After an explicit final-seal audit, record `FinalRunAudit` separately and submit `formal_array.slurm` against the seal-bound immutable 100-row job map. Every array task explicitly sets `PYTHONPATH=experiments/samga_brain_rw`, activates `test`, uses one row only, and cannot be resubmitted after its completion claim validates.

- [ ] **Step 4: Update bilingual READMEs from generated results**

State “numerically above the SAMGA paper headline under our locked local protocol” only if the final absolute Top-1 exceeds 91.30%; never claim an exact paper-protocol superiority result.

## Execution Checkpoints

1. **Checkpoint A — after Tasks 1–4:** protocol, manifests, seal guards, and provenance are implemented; formal test remains inaccessible.
2. **Checkpoint B — after Task 17:** Tasks 6–16 implementation/unit tests and all reviewed SLURM/job-map tests pass; Task 18 may now begin GPU smokes and stage execution.
3. **Checkpoint C — after Task 13:** online/cache parity is proven before true InternViT LoRA training.
4. **Checkpoint D — after Tasks 14–16:** applicable expensive pilots are gated and at most one survivor per stage is locked.
5. **Checkpoint E — during Task 19:** the chosen pair is refit train-only, the preparation seal is audited before formal-input materialization, and the final seal/map require a second explicit audit before evaluation.
6. **Checkpoint F — after Task 20:** the single formal paired result is complete and documented.

## Immediate First Execution Batch

The first implementation session executes Tasks 1–4 CPU work only. It creates the protocol/config package, immutable split registry, fail-closed artifact guards, provenance checks, and target-resolution tests.

It submits no GPU job until Task 17's reviewed launcher exists, and it must not load any formal-test manifest or cache until Checkpoint A passes.
