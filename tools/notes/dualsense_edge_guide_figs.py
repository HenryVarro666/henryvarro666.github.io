#!/usr/bin/env python3
"""Figures for the note content/notes/dualsense-edge-guide/.

    python3 tools/notes/dualsense_edge_guide_figs.py

writes four SVGs next to the note, name.svg (English) and name.zh.svg (Chinese). The
controller drawings are schematic: only parts whose place Sony's support pages state (front
buttons, back buttons and the L2/R2 stop sliders on the back) are drawn; the RELEASE latch
and the reset hole are described in the text instead. English labels are wider, so the
front, back and profiles figures are 60px wider in English. Standard library only.
"""
from __future__ import annotations

import sys
from pathlib import Path

from precision_vs_recall_figs import Canvas, text_width

OUT = Path(__file__).resolve().parents[2] / "content" / "notes" / "dualsense-edge-guide"
LANG = "en"


def t(en: str, zh: str):
    """The value for the language being drawn."""
    return zh if LANG == "zh" else en


# The controller outline, front or back (it is symmetric), centred on x = 360.
BODY = ("M210 110 Q360 95 510 110 Q580 118 605 190 Q635 290 620 360 Q610 400 570 395 "
        "Q540 390 525 350 Q500 290 450 290 L270 290 Q220 290 195 350 Q180 390 150 395 "
        "Q110 400 100 360 Q85 290 115 190 Q140 118 210 110 Z")


class Fig(Canvas):
    def circle(self, cx, cy, r, cls, sw=1.5):
        self.raw(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:g}" class="{cls}" stroke-width="{sw:g}"/>')

    def d(self, d, cls, sw=1.5, extra=""):
        self.raw(f'<path d="{d}" class="{cls}" stroke-width="{sw:g}" stroke-linecap="round" '
                 f'stroke-linejoin="round"{extra}/>')

    def body(self):
        self.d(BODY, "f-ln", 1.6)

    def face_symbols(self, cx, cy, gap):
        """Triangle, circle, cross and square around (cx, cy), drawn as shapes, not glyphs."""
        tx, ty = cx, cy - gap
        self.d(f"M{tx} {ty - 6} L{tx + 5.5} {ty + 4} L{tx - 5.5} {ty + 4} Z", "f-lnink", 1.3)
        self.circle(cx + gap, cy, 5, "f-lnink", 1.3)
        xx, xy = cx, cy + gap
        self.d(f"M{xx - 5} {xy - 5} L{xx + 5} {xy + 5} M{xx + 5} {xy - 5} L{xx - 5} {xy + 5}",
               "f-lnink", 1.3)
        self.rect(cx - gap - 5, cy - 5, 10, 10, "f-lnink", rx=1, sw=1.3)

    def left_label(self, y, s, x2, size=15, weight=None):
        """A label at the left margin with a leader line from its end to x2."""
        self.text(24, y + 5, s, size, weight=weight)
        self.line(24 + text_width(s, size) + 8, y, x2, y, "f-ln", 1.0)


def front() -> Fig:
    c = Fig("front", t(780, 720), 446, t(
        "DualSense Edge front view: the PS button sits below the touch pad, between the two "
        "sticks. Press it once to open the control center; hold it to go to the home screen. "
        "The Fn buttons are below the sticks; the back buttons are on the back.",
        "DualSense Edge 正面示意图：PS 键在触摸板下方、两个摇杆中间。按一下打开控制中心，"
        "按住回到主画面。Fn 键在两个摇杆下方，背面键在手柄背面。"))
    c.text(24, 37, t("The PS button sits in the middle: hold it to go home",
                     "PS 键在手柄正中间，按住它就回主页"), 22, weight=650)
    c.text(24, 60, t("DualSense Edge front view, schematic, not to scale",
                     "DualSense Edge 正面示意图，不按比例"), 14, "f-m")
    c.body()
    c.rect(284, 116, 152, 76, "f-node", rx=10)                      # touch pad
    c.rect(247, 125, 10, 22, "f-node", rx=5)                        # create
    c.rect(463, 125, 10, 22, "f-node", rx=5)                        # options
    c.d("M172 166H188V182H204V198H188V214H172V198H156V182H172Z", "f-node")   # d-pad
    for x, y in ((540, 164), (566, 190), (540, 216), (514, 190)):
        c.circle(x, y, 11, "f-node")
    c.face_symbols(540, 190, 26)
    for x in (270, 450):                                            # sticks
        c.circle(x, 236, 24, "f-node")
        c.circle(x, 236, 14, "f-node")
    c.circle(360, 214, 11, "f-hl", 2)                               # PS button
    c.rect(350, 236.5, 20, 7, "f-node", rx=3.5)                     # mute
    for x in (263, 443):                                            # Fn buttons
        c.rect(x, 272.5, 14, 7, "f-node", rx=3.5)
    for x in (130, 556):                                            # back buttons, seen through
        c.raw(f'<rect x="{x}" y="300" width="34" height="46" rx="10" class="f-ln" '
              f'stroke-width="1.3" stroke-dasharray="4 3"/>')

    for x1, y1, x2, y2 in ((252, 92, 252, 124), (360, 92, 360, 116), (468, 92, 468, 124),
                           (577, 190, 636, 190), (474, 236, 636, 236), (590, 323, 636, 323),
                           (270, 280, 270, 404)):
        c.line(x1, y1, x2, y2, "f-ln", 1.0)
    c.d("M371 214 L380 214 L380 312", "f-lnhl", 1.6)

    c.text(252, 86, t("Create", "创建键"), 15, anchor="middle")
    c.text(360, 86, t("Touch pad", "触摸板"), 15, anchor="middle")
    c.text(468, 86, t("Options", "选项键"), 15, anchor="middle")
    c.left_label(190, t("D-pad", "方向键"), 156)
    c.left_label(236, t("Left stick", "左摇杆"), 246)
    c.text(644, 181, t("Action buttons", "动作键"), 15, weight=650)
    c.text(644, 200, t("✕ select", "✕ 确定"), 13, "f-m")
    c.text(644, 217, t("○ cancel", "○ 返回"), 13, "f-m")
    c.text(644, 241, t("Right stick", "右摇杆"), 15)
    c.text(644, 320, t("Back buttons", "背面键"), 15)
    c.text(644, 339, t("on the back", "在手柄背面"), 13, "f-m")
    c.text(270, 424, t("Fn buttons (one on each side, Edge only)", "Fn 键（左右各一，精英手柄独有）"),
           15, anchor="middle")

    bw = t(208, 176)                                                # PS callout box
    c.rect(292, 312, bw, 80, "f-hl", rx=8, sw=2)
    cx = 292 + bw / 2
    c.text(cx, 338, t("PS button", "PS 键"), 17, anchor="middle", weight=650)
    c.text(cx, 361, t("Press once: control center", "按一下：打开控制中心"), 14, anchor="middle",
           box=bw - 16)
    c.text(cx, 381, t("Hold: home screen", "按住：直接回主页"), 14, anchor="middle", weight=650,
           box=bw - 16)
    return c


def back() -> Fig:
    c = Fig("back", t(780, 720), t(440, 420), t(
        "DualSense Edge back view: the two back buttons are on the back of the grips, and the "
        "L2 and R2 stop sliders are on the back below the triggers. Seen from behind, left and "
        "right are swapped.",
        "DualSense Edge 背面示意图：两个背面键在握把背面，L2 和 R2 键程滑杆在扳机下方的背面。"
        "从背面看，左右和正面相反。"))
    c.text(24, 37, t("Back: back buttons and L2/R2 stop sliders", "背面：背面键和 L2/R2 键程滑杆"),
           22, weight=650)
    c.text(24, 60, t("Seen from behind, so left and right are swapped. Schematic, not to scale",
                     "从背面看，左右和正面相反；示意图，不按比例"), 14, "f-m")
    c.body()
    for x, name in ((140, "R2"), (490, "L2")):                       # triggers, seen from behind
        c.rect(x, 84, 90, 26, "f-node", rx=10)
        c.text(x + 45, 102, name, 14, anchor="middle", weight=650)
    for x in (200, 508):                                            # stop sliders
        c.rect(x, 138, 12, 44, "f-hl", rx=6)
        for y in (148, 160, 172):
            c.line(x + 3, y, x + 9, y, "f-lnhl", 1.2)
    for x in (130, 556):                                            # back buttons
        c.rect(x, 300, 34, 46, "f-hl", rx=10)

    slider = t("R2 stop slider", "R2 键程滑杆")
    stops = t("long / medium / short", "长 / 中 / 短三档")
    c.left_label(160, slider, 196, weight=650)
    c.line(524, 160, 612, 160, "f-ln", 1.0)
    c.text(620, 165, t("L2 stop slider", "L2 键程滑杆"), 15, weight=650)
    for x in (206, 514):                                            # the three stops, under each slider
        c.text(x, 202, stops, 13, "f-m", "middle")
    c.left_label(323, t("Back button", "背面键"), 128, weight=650)
    c.line(592, 323, 636, 323, "f-ln", 1.0)
    c.text(644, 328, t("Back button", "背面键"), 15, weight=650)
    c.text(360, t(428, 405), t("The RELEASE latch and the reset hole are on the back too; see the text",
                       "RELEASE 锁扣和再启动键小孔也在背面，见正文"), 13, "f-m", "middle")
    return c


def home_flow() -> Fig:
    c = Fig("home-flow", 720, 300, t(
        "Two ways from a game back to the home screen: hold the PS button to go straight there, "
        "or press it once to open the control center, pick Home and press the cross button. "
        "On the home screen, select the game and press the cross button to return to it.",
        "从游戏回到主画面有两条路：按住 PS 键直接回去；或者按一下 PS 键打开控制中心，"
        "选「主页」再按交叉键。在主画面选中游戏按交叉键，就回到游戏。"))
    c.text(24, 37, t("Two ways back to the home screen; holding PS is quickest",
                     "两条路都能回到主画面，按住 PS 键最快"), 22, weight=650)
    c.rect(24, 124, 156, 52, "f-node", rx=8)
    c.text(102, 155, t("In a game or app", "在游戏或应用里"), 15, anchor="middle", box=140)
    c.rect(540, 124, 156, 52, "f-hl", rx=8, sw=2)
    c.text(618, 155, t("PS5 home screen", "PS5 主画面"), 16, anchor="middle", weight=650, box=140)
    c.rect(275, 212, 170, 58, "f-node", rx=8)
    c.text(360, 237, t("Control center", "控制中心"), 16, anchor="middle", weight=650)
    c.text(360, 257, t("icons along the bottom", "屏幕下方的一排图标"), 13, "f-m", "middle", box=154)

    c.path([(102, 123), (102, 76), (618, 76), (618, 102)], "f-ln", 1.5)
    c.arrow(618, 101, 618, 122)
    c.text(360, 66, t("Hold the PS button (quickest)", "按住 PS 键（最快）"), 15, anchor="middle",
           weight=650)
    c.path([(102, 177), (102, 241), (252, 241)], "f-ln", 1.5)
    c.arrow(251, 241, 273, 241)
    c.text(188, 231, t("Press PS once", "按一下 PS 键"), 14, anchor="middle", box=148)
    c.path([(446, 241), (618, 241), (618, 200)], "f-ln", 1.5)
    c.arrow(618, 201, 618, 178)
    c.text(532, 231, t("Pick Home, press ✕", "选「主页」，按 ✕"), 14, anchor="middle", box=168)
    c.line(538, 150, 196, 150, "f-ln", 1.5, ' stroke-dasharray="5 4"')
    c.arrow(197, 150, 182, 150)
    c.text(360, 140, t("Select the game and press ✕ to resume", "在主画面选中游戏，按 ✕ 接着玩"),
           13, "f-m", "middle", box=340)
    return c


def profiles() -> Fig:
    c = Fig("profiles", t(780, 720), 292, t(
        "Hold the Fn button and press the triangle button to switch back to the default profile; "
        "hold Fn and press the circle, cross or square button to switch to the profile assigned "
        "to it. The controller stores up to four profiles, including the default.",
        "按住 Fn 键再按三角键，换回默认配置文件；按住 Fn 键再按圆圈键、交叉键或正方键，"
        "换成分配给它的配置文件。手柄最多存四套配置文件，含默认的一套。"))
    c.text(24, 37, t("Hold Fn, then press an action button to switch profiles",
                     "按住 Fn 键再按一个动作键，就换一套配置文件"), 22, weight=650)
    c.text(24, 60, t("Up to 4 profiles on the controller: the default plus 3 you assign to shortcuts",
                     "手柄最多存 4 套：默认 1 套，加上你分配到快捷方式的 3 套"), 14, "f-m")
    c.rect(24, 141, 124, 60, "f-hl", rx=8, sw=2)
    c.text(86, 177, t("Hold Fn", "按住 Fn 键"), 16, anchor="middle", weight=650)
    c.text(176, 179, "+", 22, anchor="middle", weight=650)
    rows = (96, 146, 196, 246)
    for y in rows:
        c.circle(224, y, 15, "f-node")
        c.arrow(246, y, 296, y)
    c.d(f"M224 {rows[0] - 6} L229.5 {rows[0] + 4} L218.5 {rows[0] + 4} Z", "f-lnink", 1.3)
    c.circle(224, rows[1], 5, "f-lnink", 1.3)
    c.d(f"M219 {rows[2] - 5} L229 {rows[2] + 5} M229 {rows[2] - 5} L219 {rows[2] + 5}", "f-lnink", 1.3)
    c.rect(219, rows[3] - 5, 10, 10, "f-lnink", rx=1, sw=1.3)

    c.rect(300, rows[0] - 18, 300, 36, "f-hl", rx=6)
    c.text(316, rows[0] + 5, t("Default profile (can't be changed)", "默认配置文件（固定，不能改）"),
           15, weight=650, box=280)
    for y in rows[1:]:
        c.rect(300, y - 18, 300, 36, "f-node", rx=6)
        c.text(316, y + 5, t("Profile you assign to this shortcut", "你分配到这个快捷方式的配置文件"),
               15, box=280)
    c.text(612, rows[0] + 5, t("Use it if buttons act up", "按键乱了就按它"), 13, "f-m")
    return c


def main() -> int:
    global LANG
    OUT.mkdir(parents=True, exist_ok=True)
    warnings = []
    for LANG, suffix in (("en", ""), ("zh", ".zh")):
        for draw in (front, back, home_flow, profiles):
            c = draw()
            warnings += [f"[{LANG}] {w}" for w in c.warnings]
            (OUT / f"{c.name}{suffix}.svg").write_text(c.svg(), encoding="utf-8")
            print("wrote", f"{c.name}{suffix}.svg")
    LANG = "en"
    for w in warnings:
        print("WARNING:", w)
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
