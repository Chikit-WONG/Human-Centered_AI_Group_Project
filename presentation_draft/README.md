# HCAI presentation draft

This folder contains an English, 16:9 Beamer draft for a 10-minute course
presentation.

## Files

- `main.tex`: editable LaTeX source.
- `main.pdf`: compiled presentation.
- `assets/Architecture.png`: project-authored architecture figure.

## Before presenting

Edit these commands near the top of `main.tex`:

```tex
\newcommand{\PresenterName}{YOUR NAME}
\newcommand{\StudentID}{YOUR STUDENT ID}
\newcommand{\TeamMembers}{YOUR TEAM MEMBERS}
```

The source contains `% Speaker note:` comments for rehearsal. They are not
visible in the PDF. Each slide also has an auditable `% [Sources]` comment block.

## Compile locally

The verified local build uses the `cityflow` Conda environment:

```bash
cd presentation_draft
conda run -n cityflow tectonic main.tex
```

The same folder can be uploaded to Overleaf. Set `main.tex` as the main document
and use pdfLaTeX.

## Content boundaries

- The main result is standard independent 200-way retrieval.
- The controlled SAMGA comparison isolates visual LoRA/TTUR under a shared CLIP
  backbone; it is not an exact reproduction of the SAMGA paper.
- Hungarian assignment and reconstruction are not presented as main results.
