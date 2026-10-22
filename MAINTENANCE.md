# Maintaining ccao2679.com

How the site's showcase stays current. A weekly scheduled Claude task follows this file; it is
also the checklist for doing it by hand.

## Where things live

| What | File(s) |
|---|---|
| One showcased repo | `content/{research,projects,coursework}/<slug>.md` (front matter: `title`, `date`, `repo`, `link`, `lang`, `stars`, `topics`, optional `paper`) |
| Its overview figure | `assets/figs/<slug>.svg` — inlined next to the entry; drawn by `tools/figs/figures.py` |
| Publications | `data/publications.toml` — one `[[pub]]` per paper; `first = true` gets the full card |
| A paper's main figure | `assets/pubs/<id>.png` (or `.jpg`), referenced by `figure = "<file>"` |
| Citation counts | `data/citations.json` — per-paper counts, rewritten by `tools/citations.py` (OpenAlex) in the Pages workflow |
| Google Scholar totals | `data/scholar.toml` — copied by hand from the owner's CV or profile (Scholar has no API); the weekly task does not change it |
| Contact page, CV, résumé | `content/contact/_index.md` (front matter), `static/chao-cao-cv.pdf`, `static/chao-cao-resume.pdf` (phone number removed) |

## Overview figures

All figures share one vocabulary (`tools/figs/figlib.py`): a 360×270 canvas, rounded boxes, grey
bars, and one highlight colour for the thing that matters. To add or change one, edit or add a
function in `tools/figs/figures.py`, then:

```bash
cd tools/figs
python3 figures.py ../../assets/figs /tmp/figs-solo   # site copies + light/dark README copies
```

Rules: every number on a figure must come from the repo's own README or results; keep labels to a
few words (they are read at ~260 px wide); the highlight marks the main idea or result, nothing
else.

## README posters

Each showcased repo's README opens with a 1600×960 poster in the layout of anota's
`docs/figures/local-architecture.png`: headline, four pipeline steps, three fact panels. One JSON
config per repo lives in `tools/figs/posters/`; render with

```bash
cd tools/figs
python3 poster.py posters/ /tmp/posters      # writes <slug>.svg and <slug>.png, flags clipped text
```

and copy the pair to the repo as `docs/figures/overview.{png,svg}`. Accent colours follow the
homepage sections: research `#3157A4`, projects `#007F78`, coursework `#56688A`. READMEs share one
shape: centered title, one-line subtitle, badges (including a link back to the homepage entry),
the poster, then Highlights / How it works / Quick start / Repository layout, with older notes
folded into `<details>`.

## Weekly check

1. **Repo metadata.** For each entry, compare `lang`, `stars` and the summary with GitHub; update
   the front matter when they drift. A repo that went private or was deleted is flagged, not
   silently removed.
2. **New repos.** Public, non-fork, non-archived repos of HenryVarro666 that are not on the site
   are listed in the PR description as candidates with a one-line pitch. They are not added
   unless the owner says so.
3. **Figures.** When a repo's headline numbers change, update its homepage figure and its README
   poster config from the new numbers, and open a PR in that repo with the new
   `docs/figures/overview.{png,svg}`.
4. **Papers.** Look for new papers by Chao Cao with the lab's co-authors (dblp, arXiv, OpenAlex,
   IEEE Xplore, Springer). Add them to `data/publications.toml`; a new first-author paper gets a
   full card (main figure, TL;DR, problem / approach / findings, BibTeX), with any part that could
   not be verified listed in the PR.
5. **Build and look.** `hugo --gc --minify`, then check `/`, `/research/`, `/projects/`,
   `/publications/` at desktop and phone widths in light and dark.
6. **Deliver as PRs.** All changes go to a branch and a pull request; nothing is pushed to
   `master` or merged by the task. No changes, no PR.

## Standing exceptions

- **Papers left out on purpose**: GP-GPT (arXiv 2409.09825). Do not add it back.
- **Deid**: the repo holds the i2b2/n2c2 2014 de-identification files, whose data use agreement
  forbids redistribution. Leave the repo alone until the owner decides how to handle it.
- **kokoro-aws-demo**: pushes to `main` deploy to AWS; any commit or PR title that touches it
  must carry `[skip deploy]`.
