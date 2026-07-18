# Bilingual README and `.gitignore` Design

Date: 2026-07-18

## Goal

Publish a clear bilingual documentation hierarchy for the full course project
and the standalone SAMGA public-code reproduction, while preventing generated
HPC artifacts from entering Git.

## Documentation structure

The repository root remains the project-level entry point:

- `README.md` is the English project overview.
- `README_ZH.md` is the complete Simplified Chinese counterpart.
- Each file links to the other at the top.
- Each root README links to the SAMGA reproduction guide in the same language.
- Existing verified metrics, protocol distinctions, leakage warnings, setup
  instructions, and references remain intact.

The SAMGA reproduction directory becomes a standalone bilingual entry point:

- `experiments/samga_reproduction/README.md` remains the English guide and
  gains a top-level link to `README_ZH.md`.
- `experiments/samga_reproduction/README_ZH.md` is a complete Chinese
  counterpart and links back to `README.md`.
- Both guides cover the same source/model/data provenance, feature
  assumptions, metric semantics, three canonical result protocols, commands,
  limitations, and claim boundaries.

## `.gitignore` policy

Keep the existing generated-output exclusions:

- `artifacts/`, `runs/`, `logs/`, `results/`, checkpoints, model weights,
  NumPy arrays, caches, and SLURM outputs remain ignored.
- `experiments/` and `previous_work/` remain eligible for version control
  because they contain implementation, tests, launchers, and documentation.
- The externally stored InternViT model remains under `EEG_Project/models`;
  no broad repository-local `models/` rule is added.

Add only `/.superpowers/` for local agent progress state. Keep
`docs/superpowers/` versionable because it contains intentional design and
implementation records rather than runtime artifacts.

The generated `results/` tree remains ignored. Essential SAMGA numbers and
their interpretation are retained in the versionable READMEs, avoiding
exceptions that might accidentally expose per-run CSV/JSON output.

## Validation

After implementation:

1. Confirm reciprocal language links in both README pairs.
2. Confirm every local Markdown and image link resolves.
3. Confirm English/Chinese SAMGA headings, tables, metrics, and warnings are
   structurally aligned.
4. Confirm Markdown code fences are paired and `git diff --check` passes.
5. Use `git check-ignore` to verify generated artifacts stay ignored while
   experiment source, tests, and both SAMGA READMEs remain visible to Git.
6. Check that no unignored experiment file exceeds 10 MiB.

## Non-goals

- Do not retrain models or change any reported metric.
- Do not move, delete, or unignore local result artifacts.
- Do not rewrite the root READMEs into shorter documents or remove existing
  reproducibility details.
- Do not modify the official SAMGA source tree.
