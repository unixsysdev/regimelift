PAPER=main

.PHONY: all figures paper clean

all: paper

figures:
	python3 scripts/generate_assets.py

paper: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

clean:
	latexmk -C $(PAPER).tex
	rm -f generated/*.tex figures/*.png figures/*.pdf
