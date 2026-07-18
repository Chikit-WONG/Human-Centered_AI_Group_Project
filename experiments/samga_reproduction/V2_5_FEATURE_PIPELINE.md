# InternViT-6B-448px-V2_5 feature pipeline

This pipeline is isolated from the older exploratory InternViT cache. Its
default root is
`artifacts/samga_reproduction/features/internvit_v2_5_multi_variant`.
Downloaded model files live outside the repository under
`EEG_Project/models`. Use the fail-closed downloader documented in
[`DOWNLOADER_SAFETY.md`](DOWNLOADER_SAFETY.md), then set
`INTERNVIT_V2_5_MODEL_PATH` to the pinned local
`OpenGVLab/InternViT-6B-448px-V2_5` revision
`9d1a4344077479c93d42584b6941c64d795d508d`:

```bash
MODEL_PATH=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/InternViT-6B-448px-V2_5/9d1a4344077479c93d42584b6941c64d795d508d
python experiments/samga_reproduction/download_v2_5_safe.py \
  --output-dir "$MODEL_PATH"
export INTERNVIT_V2_5_MODEL_PATH="$MODEL_PATH"
```

Each shard contains:

- `raw_cls.npy`: `[rows, 10, 3200]`, float16;
- `patch_mean.npy`: `[rows, 10, 3200]`, float16, excluding CLS;
- `metadata.json`: manifest order/hash, exact model revision and three weight
  SHA256 values, row interval, per-array hashes, and layer routing.

The actual captured block outputs are
`20,21,24,25,28,29,32,33,36,37`. Router-facing logical layer IDs remain
`20,24,28,32,36`. Metadata routes `idx0` to actual outputs
`20,24,28,32,36` and `idx_plus_1` to `21,25,29,33,37`.

Run the debug-partition array from the project root only after creating the
log directory. Array tasks 0–7 are the eight train shards and task 8 is the
single test shard. Set `INTERNVIT_V2_5_MAX_ROWS` for a smoke run; smoke
artifacts use a separate subdirectory and cannot pass full-coverage merge.

```bash
MODEL_PATH=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/InternViT-6B-448px-V2_5/9d1a4344077479c93d42584b6941c64d795d508d
mkdir -p logs/samga_reproduction
sbatch --export=ALL,INTERNVIT_V2_5_MODEL_PATH="$MODEL_PATH" \
  experiments/samga_reproduction/extract_v2_5_debug.slurm
```

Merge and verify a complete split:

```bash
python experiments/samga_reproduction/merge_v2_5_features.py \
  --manifest artifacts/samga_lora/manifests/sub-01_train.json \
  --output-directory artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/merged/train \
  --shard-directories artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/shards/train/shard_*

python experiments/samga_reproduction/merge_v2_5_features.py \
  --manifest artifacts/samga_lora/manifests/sub-01_test.json \
  --output-directory artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/merged/test \
  --shard-directories artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/shards/test/shard_*

python experiments/samga_reproduction/verify_v2_5_features.py \
  --manifest artifacts/samga_lora/manifests/sub-01_train.json \
  --expected-artifact-kind internvit_v2_5_feature_merged \
  --feature-directories artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/merged/train

python experiments/samga_reproduction/verify_v2_5_features.py \
  --manifest artifacts/samga_lora/manifests/sub-01_test.json \
  --expected-artifact-kind internvit_v2_5_feature_merged \
  --feature-directories artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/merged/test
```

Materialize the train and test `[rows, 5, 3200]` caches without changing the
router layer IDs:

```bash
python experiments/samga_reproduction/materialize_v2_5_variant.py \
  --manifest artifacts/samga_lora/manifests/sub-01_train.json \
  --merged-directory artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/merged/train \
  --output-directory artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/variants/train_idx0_patch_mean \
  --pooling patch_mean \
  --indexing-variant idx0

python experiments/samga_reproduction/materialize_v2_5_variant.py \
  --manifest artifacts/samga_lora/manifests/sub-01_test.json \
  --merged-directory artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/merged/test \
  --output-directory artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/variants/test_idx0_patch_mean \
  --pooling patch_mean \
  --indexing-variant idx0
```

The reported reproduction uses actual block outputs 20/24/28/32/36,
patch-token mean pooling excluding CLS, and no additional per-vector
normalization. That choice is an audited reproduction assumption: SAMGA does
not publish its checkpoint, extractor, pooling rule, or layer-index semantics.
