# Exploratory inferred-InternViT reproduction

This directory is deliberately separate from the source-locked confirmatory
CLIP experiment. It tests one frozen SAMGA configuration using the plausible
`OpenGVLab/InternViT-6B-448px-V1-5` backbone at a pinned revision and the visual
hidden-state indices 20, 24, 28, 32, and 36.

It is **not an exact reproduction of the SAMGA paper**: the paper/repository
does not identify the InternViT checkpoint, provide the extraction code, or
define whether layer numbers refer to block outputs or Transformers hidden-state
indices. Here embedding output is hidden-state zero, so index 20 is the output
after the twentieth encoder block. The run uses the released `intra.sh` seed 2025, fixes the final
checkpoint at epoch 60, and opens the test set once after training. Hungarian
decoding and test-set early stopping are excluded.

Runtime caches and checkpoints are written below `artifacts/samga_lora/internvit`;
the final one-seed summary is written below
`results/samga_lora/internvit_exploratory`. These generated files are ignored by
Git.

## Verified exploratory result

All ten subject runs completed at seed 2025 and fixed epoch 60. Strict
aggregation and an independent re-derivation from all 2,000 prediction rows
agreed:

| Metric | Ten-subject mean | Correct / total | Subject sample SD |
|---|---:|---:|---:|
| Top-1 | **83.05%** | 1661/2000 | 5.55 points |
| Top-5 | **98.00%** | 1960/2000 | 1.65 points |

These values remain a one-seed inferred-model diagnostic, not the paper's
five-seed 91.3%/98.8% result.

## Execution outline

The downloader accepts concurrent HTTP ranges but verifies all three shards
against the official Hugging Face SHA-256 values before use. Configuration and
remote-code files come from the pinned official revision. After a small debug
smoke, the 16,540 train images can be divided into eight disjoint row ranges;
the merge utility rejects gaps, overlaps, incompatible provenance, or cache
hash mismatches.

```bash
python experiments/samga_lora/exploratory_internvit/download_weights.py \
  --output-dir artifacts/samga_lora/internvit/model-03e138c81d3f

# Smoke first; then submit the full eight-train-shard plus one-test array.
sbatch --array=0-0 \
  --export=ALL,INTERNVIT_MAX_ROWS=64,INTERNVIT_BATCH_SIZE=32 \
  experiments/samga_lora/exploratory_internvit/extract.slurm
sbatch --export=ALL,INTERNVIT_BATCH_SIZE=32 \
  experiments/samga_lora/exploratory_internvit/extract_sharded.slurm

python experiments/samga_lora/exploratory_internvit/merge_feature_shards.py \
  --manifest artifacts/samga_lora/manifests/sub-01_train.json \
  --output artifacts/samga_lora/internvit/feature_cache/internvit_layers_20_24_28_32_36_train.npy \
  --shards artifacts/samga_lora/internvit/feature_cache/shards/train_rows_*.npy

sbatch experiments/samga_lora/exploratory_internvit/train_array.slurm

python experiments/samga_lora/exploratory_internvit/aggregate.py \
  --run-root artifacts/samga_lora/internvit/exploratory_seed2025 \
  --train-cache artifacts/samga_lora/internvit/feature_cache/internvit_layers_20_24_28_32_36_train.npy \
  --test-cache artifacts/samga_lora/internvit/feature_cache/internvit_layers_20_24_28_32_36_test.npy \
  --output-dir results/samga_lora/internvit_exploratory
```

The ten training cells use seed 2025 and fixed epoch 60. `aggregate.py` strictly
re-derives every 200-query result and emits a separate exploratory report; it
does not modify the confirmatory Frozen/LoRA summary.
