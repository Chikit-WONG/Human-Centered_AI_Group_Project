# Presentation and report figures

Generate the complete figure package from the repository root with:

```bash
python scripts/plot_presentation_figures.py
```

The command writes these eight owned outputs:

- `training_dynamics_50_runs.png`
- `training_dynamics_50_runs.pdf`
- `lora_final_configuration.png`
- `lora_final_configuration.pdf`
- `literature_baseline_comparison_no_samga.png`
- `literature_baseline_comparison_no_samga.pdf`
- `local_baseline_comparison_sub08_seed42.png`
- `local_baseline_comparison_sub08_seed42.pdf`

## Metric provenance

The 50 training trajectories cover five seeds (42–46) and ten subjects
(01–10), with 25 logged epochs per trajectory. Their sources are split across
two roots:

- Seed 42 is read from the sibling reference checkout
  `../test/brain-rw`. Subject 08 uses
  `runs/seed42/subj08/validation_metrics.jsonl`; subjects 01–07 and 09–10 use
  `runs/all_subjects/seed42/subjXX/validation_metrics.jsonl`.
- Seeds 43–46 are read from this project under
  `runs/all_subjects/seedXX/subjXX/validation_metrics.jsonl`.

The logged validation metrics are diagnostics evaluated on the THINGS-EEG2
test split. They must not be interpreted as validation-set model selection.
Every run uses the fixed epoch-25 checkpoint; no best-epoch selection is
performed.

## Configuration and comparison caveats

The LoRA figure documents the verified final configuration: rank 32, alpha 32,
dropout 0.0, brain-encoder learning rate 5e-4, vision learning rate 5e-5,
cosine scheduling, weight decay 0.05, and 25 epochs. No recorded local LoRA
hyperparameter sweep exists, so the figure does not claim sweep-derived
optimality.

The literature comparison reports headline 200-way retrieval values collected
across papers. Visual targets, encoders, schedules, evaluation implementations,
and checkpoint rules can differ, so it is contextual rather than a controlled
local benchmark.

The local comparison is the controlled same-query, same-gallery Subject-08,
seed-42 evaluation of NICE, ATM-S, and Our Project. Its global one-to-one
assignment methods inspect the complete similarity matrix and are transductive
diagnostics, not standard independent retrieval metrics.

SAMGA is deliberately excluded: it has no method row or rendered label in this
figure package.
