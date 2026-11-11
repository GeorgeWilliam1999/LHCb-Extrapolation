# Build the LHCb self-chained extrapolation paper.
#   make        -> main.pdf (pdflatex, bibtex, pdflatex, pdflatex)
#   make zip    -> Self_chained_paper.zip, ready for Overleaf "New Project -> Upload Project"
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
	rm -f Self_chained_paper.zip
	zip -r Self_chained_paper.zip main.tex references.bib figures scripts results README.md Makefile build.sh \
	    -x 'results/cache/*' -x 'results/verify_*' -x '*/__pycache__/*'

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc

distclean: clean
	rm -f main.pdf Self_chained_paper.zip

.PHONY: all zip clean distclean
