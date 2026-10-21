"""README showcase posters, in the layout of anota's docs/figures/local-architecture.png
(scripts/render_readme_figure.py there): a 1600x960 card with a headline, four pipeline
steps and three fact panels.

    python3 poster.py posters/<slug>.json <out-dir>       # one poster
    python3 poster.py posters/ <out-dir>                  # every config in a folder

Writes <out-dir>/<slug>.svg and <slug>.png (PNG rendered in Chromium with Inter, so the
README looks the same everywhere). Any text that runs past its card is reported and the
script exits non-zero, so a poster never ships with clipped words.
"""
import json
import sys
from html import escape
from pathlib import Path

FONT = "Inter, 'Segoe UI', Arial, sans-serif"


def build(c):
    parts = []

    def text(x, y, value, size=22, color="#536272", weight=400, tracking=None, maxx=None, anchor=None):
        spacing = f' letter-spacing="{tracking}"' if tracking else ""
        lim = f' data-maxx="{maxx}"' if maxx else ""
        anc = f' text-anchor="{anchor}"' if anchor else ""
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
                     f'font-weight="{weight}"{spacing}{lim}{anc}>{escape(value)}</text>')

    def rect(x, y, w, h, fill, stroke="none", radius=20):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                     f'fill="{fill}" stroke="{stroke}"/>')

    accent, pale = c["accent"], c["pale"]
    parts.append('<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="960" '
                 'viewBox="0 0 1600 960" role="img" aria-labelledby="title desc">'
                 f'<title id="title">{escape(c["name"])}</title>'
                 f'<desc id="desc">{escape(c["title"] + " " + c["subtitle"])}</desc>'
                 '<defs><linearGradient id="paper" x2="1" y2="1">'
                 '<stop stop-color="#F7F9FC"/><stop offset="1" stop-color="#EFF3F7"/>'
                 '</linearGradient></defs><rect width="1600" height="960" fill="url(#paper)"/>'
                 f'<g font-family="{FONT}">')
    rect(0, 0, 1600, 9, accent, radius=0)
    text(64, 66, c["name"], 20, "#172837", 700, "2", maxx=900)
    text(1536, 66, c["eyebrow"], 16, accent, 600, "1", anchor="end")
    text(60, 167, c["title"], 72, "#142638", 500, maxx=1540)
    text(64, 221, c["subtitle"], 27, "#344C5F", maxx=1536)
    text(64, 262, c["tagline"], 20, "#697988", maxx=1536)
    for i, (label, title, lines) in enumerate(c["steps"]):
        x = 64 + i * 376
        rect(x, 323, 344, 272, "#FFFFFF", "#DCE4EB")
        rect(x + 24, 349, 44, 5, accent, radius=2)
        text(x + 24, 393, label, 17, accent, 700, "1.4", maxx=x + 330)
        text(x + 24, 441, title, 28, "#172C3C", 600, maxx=x + 330)
        for j, line in enumerate(lines):
            text(x + 24, 485 + j * 32, line, 18, "#5C6D7B", maxx=x + 330)
        if i < len(c["steps"]) - 1:
            parts.append(f'<path d="M{x+352} 458h16m-6-6 6 6-6 6" fill="none" '
                         f'stroke="{accent}" stroke-width="2"/>')
    for i, (label, title, lines) in enumerate(c["panels"]):
        x = 64 + i * 501
        rect(x, 635, 470, 211, pale if i == 0 else "#E8EDF3", radius=18)
        text(x + 24, 675, label, 15, accent, 700, "1.2", maxx=x + 452)
        text(x + 24, 722, title, 24, "#193346", 600, maxx=x + 452)
        for j, line in enumerate(lines):
            text(x + 24, 765 + j * 30, line, 18, "#536777", maxx=x + 452)
    parts.append('<path d="M64 888h1472" stroke="#CED9E3"/>')
    text(64, 923, c["footer"], 17, "#647788", maxx=1536)
    parts.append("</g></svg>")
    return "\n".join(parts) + "\n"


def render(configs, out):
    from playwright.sync_api import sync_playwright
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 960})
        for cfg_path in configs:
            c = json.loads(cfg_path.read_text())
            slug = cfg_path.stem
            svg = build(c)
            (out / f"{slug}.svg").write_text(svg)
            page.set_content(f'<html><body style="margin:0">{svg}</body></html>')
            page.wait_for_timeout(150)
            over = page.evaluate("""() => [...document.querySelectorAll('text[data-maxx]')]
                .map(t => ({t: t.textContent, right: t.getBBox().x + t.getBBox().width,
                            max: +t.dataset.maxx}))
                .filter(o => o.right > o.max)""")
            for o in over:
                problems.append(f"{slug}: '{o['t']}' ends at {o['right']:.0f} > {o['max']}")
            page.locator("svg").screenshot(path=str(out / f"{slug}.png"))
            print("rendered", slug)
        browser.close()
    return problems


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    configs = sorted(src.glob("*.json")) if src.is_dir() else [src]
    probs = render(configs, out)
    for pr in probs:
        print("OVERFLOW", pr)
    sys.exit(1 if probs else 0)
