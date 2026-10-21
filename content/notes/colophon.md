---
title: "How this site is built"
date: 2026-10-21
description: "Hugo, one stylesheet, a few lines of script, deployed to GitHub Pages."
topics: [web]
---
Every entry on this site is one Markdown file. The folder it sits in (`research/`, `projects/`, `coursework/`, `notes/`) is its heading; the `topics` list in its front matter gives the dotted tags underneath.

## Adding a repository

```yaml
---
title: "Project name"
date: 2026-10-01
repo: "https://github.com/HenryVarro666/project"
lang: Python
stars: 3
topics: [speech]
---
Two or three sentences on what it does and what it found.
```

## What ships to the browser

- One stylesheet, system fonts, no webfonts or icon fonts.
- About ten lines of script, only for the light/dark switch.
- Section and topic pages are generated at build time, so filtering needs no server.

## Figures and papers

Each project's overview figure is a small SVG drawn by a script from the numbers in that repo, inlined so it follows the light/dark switch; click one to enlarge it. The publications page reads a single data file, and its citation counts are refreshed from OpenAlex every week.

Hugo builds the site in GitHub Actions on every push and publishes it to GitHub Pages.
