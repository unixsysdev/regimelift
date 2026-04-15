# Paper Draft

This directory contains the TeX source and generated assets for the HeLMAS-3n research draft.

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

## What is included

- TeX source for the paper narrative.
- Generated figures from the completed experiment tables.
- Generated LaTeX tables pulled directly from the archived result CSVs.
- Log excerpts and an experiment timeline so the draft records how the runs were actually executed.

## Scope of the draft

The draft is written as a living research document. It uses the completed pilot, layer-sweep, suffix-span, and targeted-site evidence as the current result base, while calling out the larger held-out80 validation as the active next gate.
