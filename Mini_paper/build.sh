#!/bin/sh
# Build the mini-paper: pdflatex, bibtex, pdflatex, pdflatex.
# Usage: sh build.sh   (from inside Mini_paper/)
set -e
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
echo "built main.pdf"
