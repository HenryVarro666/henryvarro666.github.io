"""Tiny SVG builder for the project overview figures.

Every figure is drawn on a 360x270 canvas (4:3) with a small vocabulary of classes,
so all figures share one look. Two kinds of output:

  site   classes only; colors come from CSS custom properties defined in the homepage's
         site.css, so the figures follow the light/dark toggle and the palette.
  solo   the same drawing with an embedded <style> holding fixed colors, for READMEs
         (one light file, one dark file, swapped with <picture> on GitHub).
"""
from __future__ import annotations

import math
import unicodedata
from html import escape

W, H = 360, 270
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, "
        "'PingFang SC', 'Noto Sans CJK SC', sans-serif")

# Fixed palettes for README copies. Keys map to the CSS custom properties used on the site.
PALETTES = {
    "light": {
        "bg": "#f5f5f5", "node": "#ffffff", "line": "#c9c9c9", "ink": "#222222",
        "muted": "#7a7a7a", "hl": "#2f6690", "hl-soft": "#e1ebf4", "on-hl": "#ffffff",
        "bar": "#cfcfcf", "bar-2": "#8f8f8f", "arrow": "#8f8f8f",
    },
    "dark": {
        "bg": "#1f2328", "node": "#2b3036", "line": "#4a5058", "ink": "#e6e6e6",
        "muted": "#9aa0a6", "hl": "#8cb8de", "hl-soft": "#2c3c4c", "on-hl": "#111418",
        "bar": "#4a5058", "bar-2": "#7d838a", "arrow": "#7d838a",
    },
}

# class -> (property, token)
CLASS_RULES = {
    "f-bg": [("fill", "bg")],
    "f-node": [("fill", "node"), ("stroke", "line")],
    "f-hl": [("fill", "hl-soft"), ("stroke", "hl")],
    "f-fillhl": [("fill", "hl")],
    "f-fillsoft": [("fill", "hl-soft")],
    "f-bar": [("fill", "bar")],
    "f-bar2": [("fill", "bar-2")],
    "f-ln": [("stroke", "arrow"), ("fill", "none")],
    "f-lnhl": [("stroke", "hl"), ("fill", "none")],
    "f-lnink": [("stroke", "ink"), ("fill", "none")],
    "f-lnrule": [("stroke", "line"), ("fill", "none")],
    "f-head": [("fill", "arrow")],
    "f-headhl": [("fill", "hl")],
    "f-t": [("fill", "ink")],
    "f-m": [("fill", "muted")],
    "f-h": [("fill", "hl")],
    "f-onhl": [("fill", "on-hl")],
    "f-dotink": [("fill", "ink")],
    "f-dotm": [("fill", "muted")],
}


def text_width(s: str, size: float) -> float:
    """Rough advance width for a sans face: good enough to catch overflow."""
    w = 0.0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 1.0
        elif ch in "il.,:;'|!·":
            w += 0.28
        elif ch in "mwMW":
            w += 0.85
        elif ch.isupper() or ch.isdigit():
            w += 0.64
        elif ch == " ":
            w += 0.28
        else:
            w += 0.53
    return w * size


class Fig:
    def __init__(self, slug: str, title: str, desc: str):
        self.slug, self.title, self.desc = slug, title, desc
        self.parts: list[str] = []
        self.warnings: list[str] = []

    # ---- primitives -------------------------------------------------------------
    def raw(self, s: str):
        self.parts.append(s)

    def rect(self, x, y, w, h, cls="f-node", rx=6, sw=1.5, extra=""):
        self.raw(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" '
                 f'class="{cls}" stroke-width="{sw:g}"{extra}/>')

    def text(self, x, y, s, size=16, cls="f-t", anchor="start", weight=None, box=None):
        if box is not None and text_width(s, size) > box:
            self.warnings.append(f"{self.slug}: '{s}' ~{text_width(s, size):.0f}px > {box}px")
        wt = f' font-weight="{weight}"' if weight else ""
        self.raw(f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" class="{cls}" '
                 f'text-anchor="{anchor}"{wt}>{escape(s)}</text>')

    def lines(self, x, y, rows, size=16, cls="f-t", anchor="middle", gap=1.22, weight=None, box=None):
        """Several centered lines whose block is vertically centered on y."""
        n = len(rows)
        top = y - (n - 1) * size * gap / 2 + size * 0.35
        for i, r in enumerate(rows):
            c = cls[i] if isinstance(cls, (list, tuple)) else cls
            wt = weight[i] if isinstance(weight, (list, tuple)) else weight
            sz = size[i] if isinstance(size, (list, tuple)) else size
            self.text(x, top + i * (size if not isinstance(size, (list, tuple)) else max(size)) * gap,
                      r, sz, c, anchor, wt, box)

    def node(self, x, y, w, h, rows, hl=False, size=16, sub=None, subsize=14):
        """Rounded box with centered label lines; optional muted sub-line(s)."""
        self.rect(x, y, w, h, "f-hl" if hl else "f-node")
        rows = [rows] if isinstance(rows, str) else rows
        sub = [] if sub is None else ([sub] if isinstance(sub, str) else sub)
        all_rows = rows + sub
        sizes = [size] * len(rows) + [subsize] * len(sub)
        classes = ["f-h" if hl else "f-t"] * len(rows) + ["f-m"] * len(sub)
        weights = [600] * len(rows) + [None] * len(sub)
        # vertical layout by real sizes
        gap = 1.2
        total = sum(s * gap for s in sizes) - (sizes[-1] * (gap - 1))
        cy = y + h / 2 - total / 2
        for r, s, c, wt in zip(all_rows, sizes, classes, weights):
            cy += s
            self.text(x + w / 2, cy - s * 0.18, r, s, c, "middle", wt, box=w - 10)
            cy += s * (gap - 1)

    def arrow(self, x1, y1, x2, y2, hl=False, sw=1.8, head=7, both=False, dash=None):
        cls, hcls = ("f-lnhl", "f-headhl") if hl else ("f-ln", "f-head")
        ang = math.atan2(y2 - y1, x2 - x1)
        ex, ey = x2 - math.cos(ang) * head * 0.9, y2 - math.sin(ang) * head * 0.9
        sx, sy = x1, y1
        if both:
            sx, sy = x1 + math.cos(ang) * head * 0.9, y1 + math.sin(ang) * head * 0.9
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" class="{cls}" '
                 f'stroke-width="{sw:g}" stroke-linecap="round"{d}/>')
        self._head(x2, y2, ang, head, hcls)
        if both:
            self._head(x1, y1, ang + math.pi, head, hcls)

    def _head(self, x, y, ang, size, cls):
        a1, a2 = ang + math.radians(152), ang - math.radians(152)
        p = [(x, y), (x + math.cos(a1) * size, y + math.sin(a1) * size),
             (x + math.cos(a2) * size, y + math.sin(a2) * size)]
        pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in p)
        self.raw(f'<polygon points="{pts}" class="{cls}"/>')

    def path(self, d, cls="f-ln", sw=1.8, extra=""):
        self.raw(f'<path d="{d}" class="{cls}" stroke-width="{sw:g}" stroke-linecap="round" '
                 f'stroke-linejoin="round"{extra}/>')

    def circle(self, cx, cy, r, cls="f-dotink", extra=""):
        self.raw(f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" class="{cls}"{extra}/>')

    # ---- output -----------------------------------------------------------------
    def _body(self) -> str:
        return (f'<title>{escape(self.title)}</title><desc>{escape(self.desc)}</desc>'
                f'<rect x="0" y="0" width="{W}" height="{H}" rx="8" class="f-bg"/>'
                + "".join(self.parts))

    def site_svg(self) -> str:
        return (f'<svg class="fig-svg" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
                f'role="img" aria-labelledby="fig-{self.slug}-t">'
                + self._body().replace("<title>", f'<title id="fig-{self.slug}-t">', 1)
                + "</svg>\n")

    def solo_svg(self, theme: str, width: int = 720) -> str:
        pal = PALETTES[theme]
        rules = []
        for cls, props in CLASS_RULES.items():
            decl = ";".join(f"{p}:{pal.get(t, t)}" for p, t in props)
            rules.append(f".{cls}{{{decl}}}")
        style = "<style>" + "".join(rules) + f"text{{font-family:{FONT}}}</style>"
        return (f'<svg viewBox="0 0 {W} {H}" width="{width}" height="{int(width * H / W)}" '
                f'xmlns="http://www.w3.org/2000/svg" role="img">'
                + style + self._body() + "</svg>\n")
