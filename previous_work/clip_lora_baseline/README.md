# Archived CLIP-LoRA Baseline

This directory indexes the verified pre-SAMGA baseline without duplicating or
moving its implementation.

- Git commit: `a97b97a110c0fea7d4adafd5abce477c6cce525c`
- Local tag: `clip-lora-baseline-v1`
- Protocol: ten THINGS-EEG2 subjects, seeds 42--46, 17 posterior channels,
  rank-32 CLIP ViT-B/32 LoRA, fixed epoch 25
- Standard retrieval: Top-1 `86.66% +/- 0.69`, Top-5 `98.38% +/- 0.14`

The original entry points remain at the repository root so that the existing
README commands, result validators, and model-reload checks continue to work.
The new SAMGA experiments live under `experiments/samga_lora/`.
