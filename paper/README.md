# RegimeLift Paper Draft

This directory contains the TeX source and generated assets for the active RegimeLift research draft. Legacy He-LMAS material is referenced only as archived context.

## Build

```bash
cd paper
python3 scripts/generate_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

A convenience `Makefile` is included:

```bash
cd paper && make
```

## CI build

The PDF is built in GitHub Actions via `.github/workflows/ci.yml`.

- Job: `paper`
- Output artifact: `regimelift-paper` (`paper/main.pdf`)

The repository tracks paper sources and generated analysis assets. LaTeX temporary build files and the compiled PDF are CI outputs, not versioned source files.

## What is included

- TeX source for the paper narrative.
- Generated figures from the completed experiment tables.
- Generated LaTeX tables pulled directly from the archived result CSVs.
- Log excerpts and an experiment timeline so the draft records how the runs were actually executed.

## Scope of the draft

The draft is written as a living research document. It now includes the completed heldout80 targeted-site validation as part of the core evidence base, alongside the completed pilot, layer-sweep, and suffix-span studies.
