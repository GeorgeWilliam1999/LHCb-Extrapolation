# Build the LHCb extrapolation mini-paper.
#   make        -> main.pdf (pdflatex, bibtex, pdflatex, pdflatex)
#   make zip    -> Mini_paper.zip, ready for Overleaf "New Project -> Upload Project"
#   make clean  -> remove LaTeX build products
#   make distclean -> also remove main.pdf and the zip

LATEX  = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex
SRC    = main.tex references.bib
FIGS   = $(wildcard figures/*.png)

all: main.pdf

main.pdf: $(SRC) $(FIGS)
	$(LATEX) main.tex
	$(BIBTEX) main
	$(LATEX) main.tex
	$(LATEX) main.tex

zip: main.pdf
	rm -f Mini_paper.zip
	zip -r Mini_paper.zip main.tex references.bib figures README.md Makefile build.sh

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc

distclean: clean
	rm -f main.pdf Mini_paper.zip

.PHONY: all zip clean distclean
