#!/bin/sh
# Build the self-chained paper: pdflatex, bibtex, pdflatex, pdflatex.
# Usage: sh build.sh   (from inside Self_chained_paper/)
set -e
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
echo "built main.pdf"
