# Commit the self-chained paper

Written 23 September 2026, after the verification pass, the fold-in of the
eleven open issues, and the addition of the assembled-loss paragraph at the head
of Section 2.5 with its two schematics (Figures 1 and 2). Everything below is read-only review plus one commit and
one push; nothing in this folder or in any experiment folder needs changing
first. **No git command was run on your behalf beyond `git status`.**

## The commands

```
cd /data/bfys/gscriven/LHCb_Extrapolation_Project
git add Self_chained_paper
git status --short -- Self_chained_paper   # review
git commit -m "Self-chained paper: one network chained across the magnet, cost-weighted label-free loss, data and scheme accuracy (75 pp, 17 figs, 26 tables)" -m "Assembled 2026-09-22/23. Figures regenerated from committed result files by scripts/; every table cell verified against results/ (scripts/verify_tables.py, 0 mismatches)." -m "Output parameterisation stated plainly as a correction to the straight line (George, 2026-09-23)." -m "Design-history sentence corrected: the output form was chosen explicitly on 14 Sept 2026 for the fixed-step study, then inherited (George, 2026-09-23)." -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push origin main
```

`git add Self_chained_paper` stages **91 files, 14 MB** (including this file). It honours
`Self_chained_paper/.gitignore`, so the LaTeX by-products and the 57 MB of
cached arrays are left out on their own; you do not need any `-x` here.

## What goes in

`main.tex` (the single authoritative source), `references.bib`, `README.md`,
`Makefile`, `build.sh`, `.gitignore`, `main.pdf`, `Self_chained_paper.zip`
(the Overleaf bundle, 3.6 MB), `figures/` (17 PNGs), `scripts/` (20 Python
files: `numbers.py`, `common.py`, `fig01`--`fig17`, `verify_tables.py`) and
`results/` (46 files: `paper_numbers.json`, the `tab_*.csv` and `fig*.csv`
companions, `README.md`, `verification_report.md`).

`main.pdf` and `Self_chained_paper.zip` are in on purpose --- the README lists
both as files of this folder and the zip is what gets uploaded to Overleaf. If
you would rather not carry them in git, add them to `.gitignore` before the
`git add`.

This file, `COMMIT.md`, will also be staged. `rm Self_chained_paper/COMMIT.md`
first if you would rather it did not land in the repository.

## What is deliberately left out

| path | why |
|---|---|
| `Self_chained_paper/results/cache/` | 57 MB of regenerable `.npz` arrays (per-step residuals, single-step scans). Every figure that reads them can rebuild them with `scripts/numbers.py`. Now in `.gitignore`. |
| `Self_chained_paper/main.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.toc` | LaTeX by-products; `sh build.sh` regenerates them. In `.gitignore` (`.lot` and `.lof` are listed too, pre-emptively --- this paper produces neither). |
| `Self_chained_paper/scripts/__pycache__/` | Python bytecode. Now in `.gitignore`. |
| `Self_chained_paper/sections/` | deleted during the fold-in. `main.tex` is the only copy of the body text. |

## Do NOT `git add -A`

Three other untracked paths sit in this repository, in two groups, and are
**not** part of this paper. `git add Self_chained_paper` skips them; `git add -A` or `git add .`
would sweep them in.

| path | what it is |
|---|---|
| `single_network_chain_discrete_approach/Block_F_reweighted_loss/F3_Analysis/email/` | draft correspondence, not a result |
| `single_network_chain_discrete_approach/Block_G_low_momentum_window/G1_Training/condor/jobs_active.txt` | the **live** Block G job list, rewritten while jobs are running |
| `single_network_chain_discrete_approach/Block_G_low_momentum_window/G1_Training/results/p03-08/` | the **live** Block G results directory, still being written to |

Block G is the low-momentum window study --- Section 6 of the paper is reserved
for it and is deliberately empty. Commit it when it is finished, not now.

## State at the time of writing

```
$ git -C /data/bfys/gscriven/LHCb_Extrapolation_Project status --short -- Self_chained_paper
?? Self_chained_paper/
```

Nothing tracked under `Self_chained_paper/` is modified, because nothing under
it is tracked yet: this is the paper's first commit.

Build and checks, re-run immediately before this file was written:

- `sh build.sh` --- **75 pages** (74 before pass 4, the relabelling of the
  output parameterisation as a correction to the straight line), 0 errors, 0 overfull boxes, 0 undefined
  references, 0 undefined citations, 0 font warnings, 45 underfull hboxes (42
  before the 23 September additions; the three new ones are the two new rows of
  Appendix A's figure map, and all of them are loose spacing around long
  typewriter paths).
- `scripts/verify_tables.py` --- **1,445 numeric cells across 26 tables, 0
  mismatches.**
- `make zip` --- `Self_chained_paper.zip`, 3.6 MB, holding `main.tex`,
  `references.bib`, `figures/`, `scripts/`, `results/` (without
  `results/cache/`), `README.md`, `Makefile`, `build.sh`.
