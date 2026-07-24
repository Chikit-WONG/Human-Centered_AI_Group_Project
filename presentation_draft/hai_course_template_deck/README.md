# HAI course-template presentation

This folder is a self-contained Beamer draft for a 10-minute AIAA3800
presentation. It uses the supplied `ChPresentation` course theme and HKUST(GZ)
branding without modifying the original template directory.

## Files

- `main.tex`: editable English slide source.
- `main.pdf`: compiled deck.
- `speaker_script_zh.md`: timed Chinese speaker script and backup-slide Q&A.
- `ref.bib`: selected references.
- `beamerthemeChPresentation.sty`: course theme with one engine-compatibility
  guard so Tectonic/XeTeX skips the pdfLaTeX-only `inputenx` package. No visual
  styling is changed.
- `assets/`: project-authored figures used in the main talk and backup slides.
- `logos/`: course-template HKUST(GZ) logo.

## Fill presenter metadata

Edit the following commands near the top of `main.tex`:

```tex
\newcommand{\PresenterName}{YOUR NAME}
\newcommand{\StudentID}{YOUR STUDENT ID}
\newcommand{\TeamMembers}{YOUR TEAM MEMBERS}
```

## Compile

The required local environment is `cityflow`:

```bash
cd presentation_draft/hai_course_template_deck
conda run -n cityflow tectonic -X compile main.tex
```

The directory can also be uploaded to Overleaf as-is. Select `main.tex` as the
main document.

## Story and claim boundaries

- Main claim: parameter-efficient asymmetric brain--vision co-adaptation.
- Main evidence: fixed epoch-25 independent 200-way retrieval across ten
  subjects and five seeds.
- Controlled evidence: Frozen CLIP versus visual LoRA/TTUR inside a separate
  SAMGA-style system.
- Training trajectories are test-split diagnostics and are not checkpoint
  selection evidence.
- Hungarian assignment, public SAMGA reproduction, and RAG/reconstruction are
  backup-only tracks and are never mixed into the main score.
- The presentation does not claim open-world mind reading, cross-subject
  generalization, clinical use, or unqualified state of the art.
