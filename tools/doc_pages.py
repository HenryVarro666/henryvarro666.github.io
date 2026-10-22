"""Render every PDF in static/ to page images for the in-page preview on /contact/.

    python3 tools/doc_pages.py

Writes assets/docs/<pdf-name>-<page>.png (page numbers from 1). Run it again whenever a PDF in
static/ is replaced; Hugo resizes the PNGs to WebP at build time. Images are used instead of an
embedded PDF because phone browsers do not show PDFs inside a page.
"""
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "static", ROOT / "assets" / "docs"
ZOOM = 2.5  # 612 pt letter width -> 1530 px

OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*.png"):
    old.unlink()
for pdf in sorted(SRC.glob("*.pdf")):
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            target = OUT / f"{pdf.stem}-{i}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), alpha=False).save(target)
            print(target.relative_to(ROOT))
