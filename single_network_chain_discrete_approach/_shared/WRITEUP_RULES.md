# Rules every write-up in this folder follows (2026-09-22)

These bind the agent writing the page. They come from /data/bfys/gscriven/CLAUDE.md, the
/writeup skill (/data/bfys/gscriven/.claude/skills/writeup/SKILL.md) and George's standing
instructions. Read all three before writing.

## Where it goes
- Notion write-up database, data source `3265d544-b9d9-8000-8b4a-000b13a4b7c6` (fetch it first
  for the schema). Create the page with `notion-create-pages` (data_source_id parent), ONE call,
  never `update-page` for rich content (tables and images literalise).
- Read the Notion markdown spec first: fetch `notion://docs/enhanced-markdown-spec`. Tables are
  `<table header-row="true">…</table>` blocks; images `![caption](url)`; equations `$…$`.
- Properties: Title; Type = "Write up"; Status = "Done"; date:Date:start = 2026-09-22
  (date:Date:is_datetime = 0); Trust = "Provisional" (NEVER "Verified"); Provenance (text);
  Note (one-paragraph abstract); Project = ["https://app.notion.com/p/39f5d544b9d980938d27e70185e12909"];
  Todos = the relation URLs named in the plan.
- After creating: fetch the page back and check the raw markdown contains NO `\<table` and NO
  `\![` (backslash-escaped = literalised). If it does, delete nothing: report it and rebuild the
  page in a fresh create-pages call, then set Trust = Superseded and Superseded by on the bad one.
- If the page is too long for one call, split: a main page with the Intro/Aims/Method/Conclusion
  and the headline results, plus child pages (created with create-pages under the main page's
  page_id) for the long tables and figure sets, each linked from the main page. Block D's write-up
  (https://app.notion.com/p/3dc5d544b9d9813fa4a1dfba8da3cc98) is the precedent: fetch it and copy
  its conventions.

## Figures
- Inline from public GitHub raw URLs ONLY:
  `https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/<path>`
  where `<path>` is the file's path under this folder as it exists on disk today. NOTHING in this
  folder is committed yet (whole folder untracked; HEAD 1040ee9, GitHub main a062bd8), so the
  images render only after George commits and pushes. Say so in the Provenance and in a callout at
  the top of the page. Do not invent a commit hash.
- NO block letters in figure titles, axis labels, legends or table headers. Networks are named by
  what they are: "N = 64 steps, q = 2 stages, dz = 81 mm". Block letters may appear ONLY in the
  Provenance text (folder names) and in one sentence of the Intro that maps the study to its folder.
  Where an existing figure carries "Block E"/"Block F" in its title, regenerate it through a
  title-wrapping runner (F3_Analysis/run_e3_for_block_f.py shows the technique: wrap matplotlib's
  title/suptitle calls, do not edit the analysis scripts) into a `figures_writeup/` folder and use
  that file. Never compare with a comparison row/curve for another study inside a figure unless the
  plan says so.
- Every figure gets a caption that says what is plotted, on which tracks, endpoint or single step.

## Units and wording
- tx, ty are dimensionless slopes dx/dz, dy/dz (from the MC hit's exit − entry displacement over
  its dz; no arctan anywhere). Every "mrad" in the CSVs and figure labels is a slope difference
  × 10³. In the text write "×10⁻³ (slope)" and say once, in Method, that the figure labels read
  "mrad" and what that means (within 7 % of milliradians at the edge of the acceptance, within
  1 % for most tracks). Positions in µm or mm; state which.
- Say "endpoint error" (network applied N times from the real last-UT state, compared with the
  RK6 track at the first SciFi plane) or "single-step error" (one application from an RK6 state)
  every time a number appears. Radial = sqrt(dx² + dy²); "max metric" = max(|dx|, |dy|). Name the
  metric beside every number.
- Structure: Intro · Aims · Method · Results · Conclusion (+ Next steps inside Conclusion), then a
  Provenance section at the end that repeats the Provenance property in full.
- Assume zero prior knowledge: define every term the first time (Gauss–Legendre stages, collocation,
  L-BFGS, restart, round, plateau rule, lever arm, coherence, the momentum bands). Longer is fine.
  Full derivations where a formula appears. George's style: first person plural, active voice, a
  roadmap paragraph at the start of each section, numbered protocols for procedures, an analogy or a
  worked micro-example for each hard concept, symbols defined in "where …" clauses. Neutral,
  declarative results; no slogans, no drama, no "remarkable". No em-dashes.
- Every number in the page must come from a named results file (CSV/JSON) or README table in this
  folder; cite the file in the Provenance. If a number you want is not in any file, compute it with
  a small script saved beside the results (e.g. `E4_Writeup/numbers.py`) so it is reproducible, and
  cite that script. Never quote a number from memory.
- Provenance property format: repo github.com/GeorgeWilliam1999/LHCb-Extrapolation · state of the
  working tree (uncommitted, HEAD 1040ee9; figures pinned to main, render after push) · every
  script that produced a number or figure, by path · input data with row counts (sample, tracks
  file, splits) · farm clusters · the date.

## Trust
- Trust = Provisional. Only George sets Verified. Finish by reporting: the page URL, the child page
  URLs, the verification result (no literalised tables/images), the list of figures that will
  render only after the push, and any number you could not source.
