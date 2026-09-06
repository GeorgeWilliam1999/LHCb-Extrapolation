#!/bin/sh
# Build the progress talk. Two pdflatex passes: no bibliography, but the
# frame count and any cross-references need the second pass.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
echo "built main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}') pages)"
