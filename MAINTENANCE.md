# Maintaining ccao2679.com

How the site's showcase stays current. A weekly scheduled Claude task follows this file; it is
also the checklist for doing it by hand.

## Where things live

| What | File(s) |
|---|---|
| One showcased repo | `content/{research,projects}/<slug>.md` (front matter: `title`, `date`, `repo`, `link`, `lang`, `stars`, `topics`, optional `paper`; course projects carry `status: "course project"`) |
| Its overview figure | `assets/figs/<slug>.svg` — inlined next to the entry; drawn by `tools/figs/figures.py` |
| Publications | `data/publications.toml` — one `[[pub]]` per paper; `first = true` gets the full card |
| A paper's main figure | `assets/pubs/<id>.png` (or `.jpg`), referenced by `figure = "<file>"` |
| Citation counts | `data/citations.json` — per-paper counts, rewritten by `tools/citations.py` (OpenAlex) in the Pages workflow |
| Google Scholar totals | `data/scholar.toml` — copied by hand from the owner's CV or profile (Scholar has no API); the weekly task does not change it |
| Contact page, CV, résumé | `content/contact/_index.md` (front matter), `static/chao-cao-cv.pdf`, `static/chao-cao-resume.pdf` (phone number removed). The in-page preview uses page images in `assets/docs/`; rerun `python3 tools/doc_pages.py` after replacing a PDF |

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
homepage sections: research `#3157A4`, projects `#007F78`, course projects `#56688A`. READMEs share one
shape: centered title, one-line subtitle, badges (including a link back to the homepage entry),
the poster, then Highlights / How it works / Quick start / Repository layout, with older notes
folded into `<details>`.

Every showcased repo has two READMEs: `README.md` (English, primary) and `README.zh-CN.md`
(Simplified Chinese, same structure), switched by `**English** · [简体中文](README.zh-CN.md)` under the
title. Both carry an `## Architecture` / `## 架构` section with a Mermaid flowchart of the real code
structure (≤ ~16 nodes, labels in each README's language, no custom colours so dark mode works).
Check diagrams before committing by rendering them with mermaid-cli; when a README changes, change
the other language in the same commit. `PROBLEM.md` has a `PROBLEM.zh.md` twin.

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
   full card (main figure, TL;DR, problem / approach / findings, BibTeX). A co-authored paper at a
   conference or journal gets a `summary`: one paragraph on what it does and why it matters,
   written from the full paper (not a reworded abstract), with numbers checked against its
   tables, plus a `rank` within its year. An arXiv-only paper is only listed. Anything that could
   not be verified is listed in the PR.
5. **Build and look.** `hugo --gc --minify`, then check `/`, `/research/`, `/projects/`,
   `/publications/` at desktop and phone widths in light and dark.
6. **Deliver as PRs.** All changes go to a branch and a pull request; nothing is pushed to
   `master` or merged by the task. No changes, no PR.

## Standing exceptions

- **Papers left out on purpose**: GP-GPT (arXiv 2409.09825). Do not add it back.
- **open-llm-deid** (formerly Deid): the n2c2 2014 notes, model outputs on them and sentences
  extracted from them were removed from the whole history in October 2026; the data use agreement
  forbids redistribution. Never add n2c2 text or anything derived from it. `examples/` is synthetic.
- **kokoro-aws-demo**: pushes to `main` deploy to AWS; any commit or PR title that touches it
  must carry `[skip deploy]`.
- **Coursework and Notes** were folded away in October 2026: course projects live in `projects/`
  (`/coursework/` redirects there) and there is no notes section.

## Authorship

Every commit in these repositories is made as **Chao Cao
<48014495+HenryVarro666@users.noreply.github.com>**. Before committing, run
`git config user.name "Chao Cao"` and
`git config user.email "48014495+HenryVarro666@users.noreply.github.com"`. Do not add
`Co-Authored-By`, `Claude-Session` or "Generated with" lines to commits or pull requests.
