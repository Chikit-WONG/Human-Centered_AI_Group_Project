# Root README Matching-Fairness Summary Design

## Objective

Add a compact, bilingual summary of the completed three-baseline matching-fairness experiment to the repository-root `README.md` and `README_ZH.md`. The summary must help a reader distinguish ordinary independent EEG-to-image retrieval from global assignment-based decoding without turning either root README into a full experimental appendix.

## Authoritative source

The numerical source is the local audited artifact:

`matching_fairness_v3/aggregate/main_table.csv`

SHA-256:

`54f00400eb5c9c9c41c0a855b1d60bc3094672c842a405cd7d7cfca4af151952`

The table contains the fixed `sub-08`, seed-`42` results for NICE, ATM-S, and Our project. The complete perturbation reports remain outside the root README and are documented through the version-controlled matching-fairness guides.

## README content

Both root READMEs will contain the same information in their respective language:

1. A scope statement identifying the result as a single-subject, single-seed implementation/re-evaluation rather than a perfect paper reproduction or cross-subject significance result.
2. One main `3 baselines × 6 metrics` table containing:
   - Independent Top-1;
   - Independent Top-5;
   - Greedy assignment accuracy;
   - Hungarian assignment accuracy;
   - Stable Matching assignment accuracy;
   - Sinkhorn assignment accuracy.
3. The checkpoint/training source for every baseline.
4. A semantics note explaining that only Independent Top-1/Top-5 are standard per-query retrieval metrics; the other columns are one-choice assignment accuracies and have no assignment Top-5.
5. One compact robustness finding from the real disjoint-trial duplicate-EEG experiment: at `220 × 200`, Our-project Hungarian is `89.09%`, Independent is `90.45%`, and the hard one-to-one methods leave 20 queries unmatched.
6. A warning that the standard Sinkhorn cells did not meet the preregistered `1e-8`/500-iteration convergence criterion, so the retained Sinkhorn accuracies are diagnostic.
7. A link to the corresponding English or Chinese matching-fairness guide for all 27 standard perturbations, three real duplicate-EEG settings, provenance, and commands.

## Presentation and language

- Keep the section immediately before the existing standalone Hungarian ablation so the narrative moves from the broad three-baseline comparison to the narrower Our-project-only ablation.
- Bold the Our-project row because it is the strongest row in every displayed standard metric, while preserving the protocol caveats.
- Use fully English headers in `README.md` and fully Chinese headers in `README_ZH.md`.
- Refer to the CSV as a local audited artifact rather than implying that the ignored result file is available as a GitHub link.
- Preserve the existing bilingual navigation and the actual Chinese filename `README_ZH.md`.

## Verification

Before completion:

1. Recompute the CSV SHA-256 and compare every displayed value to the source row.
2. Verify the duplicate-EEG numbers and unmatched count against the aggregate report.
3. Check that the English and Chinese tables have identical numerical values and metric order.
4. Check all relative guide links exist.
5. Run a Markdown structure check for balanced table column counts and duplicate section headings.
6. Review `git diff` to ensure only the two READMEs and this approved spec are involved in this task.

## Out of scope

- Copying all perturbation rows into the root READMEs.
- Changing experiment outputs, score matrices, checkpoints, or matching code.
- Resuming the paused SFT+DPO experiment.
- Renaming `README_ZH.md`.
