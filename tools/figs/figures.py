"""One overview figure per showcased repository. Numbers come from each repo's own results.

Run: python3 figures.py <homepage_repo_dir> <readme_out_dir>
"""
import math
import sys
from pathlib import Path

from figlib import Fig, W, H

FIGS = {}


def fig(slug, title, desc):
    def deco(fn):
        FIGS[slug] = (title, desc, fn)
        return fn
    return deco


def header(f, s):
    f.text(18, 30, s, 15, "f-m", box=W - 36)


# ---------------------------------------------------------------- research
@fig("tts-inference-engine-bench", "vLLM vs TensorRT-LLM on a TTS model's LLM backbone",
     "Throughput of the LLM backbone: eager 21, TensorRT-LLM 213, vLLM 453 tokens per second. "
     "End-to-end TTS latency: flow-matching 68 percent, LLM 21 percent, other 11 percent.")
def tts_bench(f):
    header(f, "LLM backbone throughput, tokens/s")
    rows = [("eager", 21.4, "21", False), ("TRT-LLM", 212.5, "213", False), ("vLLM", 452.8, "453", True)]
    for i, (name, v, lab, hl) in enumerate(rows):
        y = 46 + i * 30
        f.text(88, y + 15, name, 15, "f-h" if hl else "f-t", "end", 600 if hl else None)
        w = max(3, v / 452.8 * 206)
        f.rect(96, y, w, 20, "f-fillhl" if hl else "f-bar2", rx=3, sw=0)
        f.text(96 + w + 7, y + 15, lab, 15, "f-h" if hl else "f-m", weight=600 if hl else None)
    f.text(18, 160, "end-to-end TTS latency", 15, "f-m")
    x, y, full = 18, 172, 324
    segs = [(0.678, "f-bar", "flow-matching 68%", "f-t"), (0.208, "f-fillhl", "LLM 21%", "f-onhl"),
            (0.114, "f-bar2", "", "")]
    for frac, cls, lab, tcls in segs:
        w = frac * full - 2
        f.rect(x, y, w, 34, cls, rx=3, sw=0)
        if lab:
            f.text(x + w / 2, y + 22, lab, 14, tcls, "middle", 600, box=w - 6)
        x += frac * full
    f.text(180, 236, "vLLM cuts end-to-end time by ~20%", 16, "f-t", "middle", 600, box=W - 30)
    f.text(180, 257, "the diffusion stage dominates; LLM engines skip it", 13.5, "f-m", "middle", box=W - 24)


@fig("dots-tts-preference-alignment", "When likelihood disagrees with quality",
     "Held-out WER: base 0.11, open-SOAR 0.09, SFT on reward winners 0.56, DPO-FM 0.89. "
     "Reward and model likelihood are anti-correlated, Spearman rho -0.22.")
def dots(f):
    header(f, "held-out WER, lower is better (n = 100)")
    base_y, scale = 176, 118 / 0.89
    bars = [("base", 0.112, "f-bar2", False), ("open-SOAR", 0.094, "f-fillhl", True),
            ("SFT", 0.556, "f-bar", False), ("DPO-FM", 0.890, "f-bar", False)]
    for i, (name, v, cls, hl) in enumerate(bars):
        cx = 62 + i * 79
        h = v * scale
        f.rect(cx - 23, base_y - h, 46, h, cls, rx=3, sw=0)
        f.text(cx, base_y - h - 7, f"{v:.2f}", 15, "f-h" if hl else "f-m", "middle", 600 if hl else None)
        f.text(cx, base_y + 20, name, 14.5, "f-h" if hl else "f-t", "middle", 600 if hl else None, box=78)
    f.path(f"M24 {base_y} H336", "f-lnrule", 1.5)
    f.text(180, 236, "reward vs. likelihood: ρ = −0.22", 17, "f-h", "middle", 600, box=W - 30)
    f.text(180, 257, "the reward favours the sample the model finds less likely", 12.5, "f-m", "middle",
           box=W - 20)


@fig("densitymap-gyralnet", "DensityMap-GyralNet",
     "A folded cortical surface with sulcal depth is reduced to its gyral crests, which form a graph "
     "whose three-way junctions are the 3-hinge gyri.")
def gyral(f):
    header(f, "cortical folding as a graph")
    pts = []
    for x in range(18, 158, 2):
        y = 112 - 32 * math.cos(2 * math.pi * (x - 45) / 50)
        pts.append((x, y))
    area = "M18 186 " + " ".join(f"L{x} {y:.1f}" for x, y in pts) + f" L{pts[-1][0]} 186 Z"
    f.path(area, "f-bar", 0, ' stroke="none"')
    f.path("M" + " L".join(f"{x} {y:.1f}" for x, y in pts), "f-lnink", 2.2)
    for px in (45, 95, 145):
        if px <= 156:
            f.circle(px, 80, 5, "f-fillhl")
    f.text(95, 66, "gyral crests", 13.5, "f-h", "middle", 600)
    f.text(87, 206, "surface + sulcal depth", 13.5, "f-m", "middle", box=148)
    f.arrow(166, 120, 192, 120)
    nodes = {"A": (214, 64), "B": (262, 92), "C": (330, 62), "D": (300, 132), "E": (246, 152),
             "F": (336, 180), "G": (210, 188), "H": (276, 190)}
    edges = [("A", "B", 8), ("B", "C", -10), ("B", "D", 6), ("D", "F", -6), ("D", "E", 8),
             ("E", "G", -8), ("E", "H", 6)]
    for a, b, bend in edges:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        nx, ny = -(y2 - y1), (x2 - x1)
        n = math.hypot(nx, ny) or 1
        cx, cy = mx + nx / n * bend, my + ny / n * bend
        f.path(f"M{x1} {y1} Q{cx:.1f} {cy:.1f} {x2} {y2}", "f-lnink", 2.4)
    for k in "ACFGH":
        f.circle(*nodes[k], 3.2, "f-dotm")
    for k in "BDE":
        f.circle(*nodes[k], 7, "f-fillhl")
    f.text(272, 216, "GyralNet + 3-hinges", 13.5, "f-h", "middle", 600, box=150)
    f.text(180, 252, "VTK surface → gyral skeleton → graph", 14, "f-t", "middle", box=W - 30)


@fig("deid", "De-identification with open LLMs",
     "A clinical note with protected health information goes through an open LLM, prompted or "
     "fine-tuned, and comes out with that information masked.")
def deid(f):
    header(f, "clinical-note de-identification")

    def card(x, masked):
        f.rect(x, 46, 98, 150, "f-node")
        layout = [[(0, 40), (46, 30, 1)], [(0, 76)], [(0, 22), (28, 34, 1), (68, 8)], [(0, 70)],
                  [(0, 54)], [(0, 18), (24, 28, 1), (58, 18)], [(0, 64)]]
        for r, segs in enumerate(layout):
            y = 62 + r * 18
            for s in segs:
                sx, sw = s[0], s[1]
                phi = len(s) > 2
                if phi and masked:
                    f.rect(x + 10 + sx, y, sw, 9, "f-hl", rx=3, sw=1.4)
                elif phi:
                    f.rect(x + 10 + sx, y, sw, 9, "f-fillhl", rx=3, sw=0)
                else:
                    f.rect(x + 10 + sx, y, sw, 9, "f-bar", rx=3, sw=0)
    card(16, False)
    card(246, True)
    f.arrow(118, 121, 136, 121)
    f.node(138, 84, 84, 74, ["open LLM"], hl=True, size=16, sub=["zero-shot", "prompts"], subsize=12.5)
    f.arrow(224, 121, 244, 121)
    f.text(65, 216, "PHI in the text", 13.5, "f-m", "middle", box=110)
    f.text(295, 216, "PHI masked", 13.5, "f-m", "middle", box=110)
    f.text(180, 250, "LLaMA 1/2 · Falcon-7B · Flan-T5", 15, "f-t", "middle", 600, box=W - 30)


# ---------------------------------------------------------------- projects
@fig("ai-medical-interpreter", "AI Medical Interpreter",
     "A Chinese-speaking patient and an English-speaking provider talk over the phone through Twilio and "
     "the OpenAI Realtime API; the call produces a live transcript and a SOAP note.")
def interpreter(f):
    header(f, "real-time medical phone interpretation")
    f.node(16, 46, 80, 56, ["patient"], size=15, sub=["中文"], subsize=14)
    f.node(116, 46, 128, 56, ["Twilio ↔", "OpenAI Realtime"], hl=True, size=14.5)
    f.node(264, 46, 80, 56, ["provider"], size=15, sub=["English"], subsize=14)
    f.arrow(96, 74, 118, 74, both=True)
    f.arrow(242, 74, 264, 74, both=True)
    f.arrow(160, 102, 118, 140)
    f.arrow(200, 102, 242, 140)
    f.node(56, 140, 112, 44, ["live transcript"], size=14.5)
    f.node(192, 140, 112, 44, ["SOAP note"], size=14.5)
    f.text(180, 228, "first audio in ~400–800 ms", 17, "f-h", "middle", 600, box=W - 30)
    f.text(180, 251, "translator and intake modes · audit log", 13.5, "f-m", "middle", box=W - 30)


@fig("anota", "Anota",
     "Machine translation and ASR outputs enter a blind review queue fed by annotators and a calibrated "
     "LLM judge; every action is logged and exports are content-hashed.")
def anota(f):
    header(f, "translation and speech review workbench")
    f.node(16, 48, 92, 58, ["MT / ASR", "outputs"], size=15)
    f.node(130, 48, 104, 58, ["blind", "review queue"], hl=True, size=15)
    f.node(256, 48, 88, 58, ["hashed", "export"], size=15)
    f.arrow(108, 77, 132, 77)
    f.arrow(232, 77, 256, 77)
    f.node(92, 138, 88, 48, ["annotators"], size=14.5, sub=["golden set"], subsize=12.5)
    f.node(188, 138, 88, 48, ["LLM judge"], size=14.5, sub=["calibrated"], subsize=12.5)
    f.arrow(146, 138, 166, 108)
    f.arrow(222, 138, 200, 108)
    f.rect(16, 202, 328, 24, "f-node", rx=5)
    f.text(180, 219, "append-only audit log", 13.5, "f-m", "middle")
    f.text(180, 254, "one Python process · SQLite · 3 dependencies", 13.5, "f-t", "middle", box=W - 24)


@fig("kokoro-whisper-fargate", "Kokoro + Whisper on AWS Fargate",
     "git push runs tests and cdk deploy through GitHub Actions with OIDC; the ARM64 image in ECR runs on "
     "ECS Fargate (Graviton) behind a load balancer, serving speech synthesis and transcription.")
def kokoro(f):
    header(f, "open-weight TTS + ASR as an AWS service")
    f.node(16, 44, 82, 40, ["git push"], size=15)
    f.node(120, 44, 118, 40, ["Actions + OIDC"], size=15)
    f.node(260, 44, 84, 40, ["ECR image"], size=15)
    f.arrow(98, 64, 120, 64)
    f.arrow(238, 64, 260, 64)
    f.arrow(302, 84, 302, 112)
    f.node(16, 120, 66, 50, ["browser"], size=14)
    f.node(98, 120, 50, 50, ["ALB"], size=14.5)
    f.node(166, 112, 178, 66, ["Fargate · Graviton"], hl=True, size=15,
           sub=["/synthesize  /transcribe"], subsize=13)
    f.arrow(80, 145, 98, 145)
    f.arrow(148, 145, 166, 145)
    f.text(180, 220, "~$74/mo running · $0 after cdk destroy", 15.5, "f-h", "middle", 600, box=W - 24)
    f.text(180, 244, "100% CDK · no stored keys · LoRA fine-tuning track", 13, "f-m", "middle", box=W - 20)


@fig("local-speaker-lab", "Local Speaker Lab",
     "Two local speech workflows: mixed Chinese-English clips transcribed with no, correct or unrelated "
     "glossaries and scored on terms; and two-speaker audio separated, matched, transcribed and summarized.")
def speaker(f):
    header(f, "two local speech workflows")
    f.node(16, 42, 74, 52, ["zh–en", "clip"], size=14.5)
    f.node(110, 42, 108, 52, ["ASR × 3"], hl=True, size=15, sub=["glossaries"], subsize=13)
    f.node(238, 42, 106, 52, ["term hits"], size=14.5, sub=["and errors"], subsize=13)
    f.arrow(90, 68, 110, 68)
    f.arrow(218, 68, 238, 68)
    f.text(164, 112, "none · correct · unrelated", 12.5, "f-m", "middle")
    f.path("M16 124 H344", "f-lnrule", 1, ' stroke-dasharray="3 5"')
    f.node(16, 136, 68, 52, ["2", "speakers"], size=14)
    f.node(100, 136, 78, 52, ["separate"], size=14, sub=["+ match"], subsize=12.5)
    f.node(194, 136, 70, 52, ["Qwen", "ASR"], size=14)
    f.node(280, 136, 64, 52, ["notes"], size=14, sub=["Ollama"], subsize=12.5)
    f.arrow(84, 162, 100, 162)
    f.arrow(178, 162, 194, 162)
    f.arrow(264, 162, 280, 162)
    f.text(180, 226, "200 clips × 3 glossaries = 600 runs", 15.5, "f-h", "middle", 600, box=W - 24)
    f.text(180, 249, "frozen protocol · bootstrap intervals · CPU", 13, "f-m", "middle", box=W - 24)


@fig("qiorazen", "Qiorazen",
     "27 screening questions are scored with the GB/T 46939-2025 standard into nine constitution types; "
     "AI drafts suggestions that a Wellness Advisor reviews.")
def qiorazen(f):
    header(f, "constitution screening with human review")
    cx, cy, r = 92, 136, 68
    vals = [0.62, 0.38, 0.82, 0.5, 0.42, 0.72, 0.3, 0.56, 0.46]
    for ring in (0.5, 1.0):
        ptsr = [(cx + r * ring * math.sin(2 * math.pi * k / 9), cy - r * ring * math.cos(2 * math.pi * k / 9))
                for k in range(9)]
        f.path("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in ptsr) + " Z", "f-lnrule", 1.2)
    for k in range(9):
        a = 2 * math.pi * k / 9
        f.path(f"M{cx} {cy} L{cx + r * math.sin(a):.1f} {cy - r * math.cos(a):.1f}", "f-lnrule", 1)
    poly = [(cx + r * v * math.sin(2 * math.pi * k / 9), cy - r * v * math.cos(2 * math.pi * k / 9))
            for k, v in enumerate(vals)]
    f.path("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in poly) + " Z", "f-hl", 2)
    f.text(cx, 226, "9 constitution types", 13.5, "f-m", "middle")
    steps = [("27 questions", False), ("GB/T 46939 score", False), ("AI suggestions", False),
             ("Wellness Advisor", True)]
    for i, (lab, hl) in enumerate(steps):
        y = 46 + i * 46
        f.node(190, y, 154, 34, [lab], hl=hl, size=14.5)
        if i:
            f.arrow(267, y - 12, 267, y)
    f.text(180, 254, "Next.js · Supabase · live at qiorazen.com", 13.5, "f-t", "middle", box=W - 24)


# ---------------------------------------------------------------- coursework
@fig("heart-disease-prediction", "Heart Disease Prediction",
     "ROC AUC on a held-out split of 61 patients: SVM 0.97, random forest 0.97, logistic regression 0.96, "
     "decision tree 0.95, k-nearest neighbours 0.89.")
def heart(f):
    header(f, "ROC AUC on a held-out split (n = 61)")
    rows = [("SVM", 0.97, True), ("RF", 0.97, True), ("LR", 0.96, False), ("DT", 0.95, False), ("KNN", 0.89, False)]
    x0, x1 = 78, 312
    for i, (name, v, hl) in enumerate(rows):
        y = 46 + i * 30
        f.text(x0 - 10, y + 15, name, 15, "f-h" if hl else "f-t", "end", 600 if hl else None)
        w = (v - 0.5) / 0.5 * (x1 - x0)
        f.rect(x0, y, w, 20, "f-fillhl" if hl else "f-bar2", rx=3, sw=0)
        f.text(x0 + w + 7, y + 15, f"{v:.2f}", 14.5, "f-h" if hl else "f-m", weight=600 if hl else None)
    f.path(f"M{x0} 40 V200", "f-lnrule", 1.5)
    f.text(x0, 218, "0.5", 12.5, "f-m", "middle")
    f.text(x1, 218, "1.0", 12.5, "f-m", "middle")
    f.text(180, 252, "UCI Cleveland data, 303 patients", 14, "f-t", "middle", box=W - 24)


@fig("movie-recommendation-system", "Movie Recommendation System",
     "A user-movie rating matrix with missing ratings is filled by user- and item-based collaborative "
     "filtering to produce a ranked list of picks.")
def movie(f):
    header(f, "collaborative-filtering recommender")
    ratings = [[2, 0, 1, 2, 0, 1], [1, 9, 2, 9, 1, 9], [0, 2, 2, 0, 1, 2], [2, 1, 0, 2, 2, 0],
               [1, 2, 1, 0, 2, 1], [0, 1, 2, 1, 0, 2]]
    for r, row in enumerate(ratings):
        for c, v in enumerate(row):
            x, y = 20 + c * 24, 48 + r * 24
            if v == 9:
                f.rect(x, y, 21, 21, "f-hl", rx=3, sw=1.4)
                f.text(x + 10.5, y + 16, "?", 14, "f-h", "middle", 700)
            elif v == 0:
                f.rect(x, y, 21, 21, "f-node", rx=3, sw=1)
            else:
                f.rect(x, y, 21, 21, "f-bar" if v == 1 else "f-bar2", rx=3, sw=0)
    f.arrow(172, 120, 202, 120)
    f.text(212, 60, "top picks", 15, "f-t", weight=600)
    for i in range(3):
        y = 74 + i * 38
        f.rect(212, y, 22, 30, "f-hl" if i == 0 else "f-node", rx=3, sw=1.4)
        f.rect(242, y + 6, 92 - i * 12, 8, "f-bar2" if i == 0 else "f-bar", rx=3, sw=0)
        f.rect(242, y + 18, 60 - i * 8, 7, "f-bar", rx=3, sw=0)
    f.text(92, 210, "ratings (seeded)", 13.5, "f-m", "middle", box=150)
    f.text(274, 210, "user-CF · item-CF", 13.5, "f-m", "middle", box=140)
    f.text(180, 248, "Django · SQLite · Pearson user-CF", 14.5, "f-t", "middle", box=W - 24)


@fig("policy-changes-prediction", "Policy Changes Prediction",
     "Daily COVID-19 cases plotted with a national policy strictness level; each change in level is "
     "marked on the case curve.")
def policy(f):
    header(f, "policy strictness vs. daily cases")
    f.path("M30 190 H340", "f-lnrule", 1.5)
    f.path("M30 52 V190", "f-lnrule", 1.5)
    curve = []
    for x in range(32, 340, 3):
        t = (x - 32) / 308
        y = 186 - 96 * math.exp(-((t - 0.28) / 0.11) ** 2) - 64 * math.exp(-((t - 0.72) / 0.09) ** 2)
        curve.append((x, y))
    f.path("M" + " L".join(f"{x} {y:.1f}" for x, y in curve), "f-ln", 2.2)
    lv = {1: 170, 2: 140, 3: 110, 4: 80}
    steps = [(32, 1), (138, 3), (166, 4), (214, 2), (276, 3), (318, 1)]
    d = f"M{steps[0][0]} {lv[steps[0][1]]}"
    for i in range(1, len(steps)):
        x, l = steps[i]
        d += f" H{x} V{lv[l]}"
    d += " H338"
    f.path(d, "f-lnhl", 2.6)
    for x, l in steps[1:]:
        f.circle(x, lv[l], 4.5, "f-fillhl")
    f.text(158, 70, "policy level", 13.5, "f-h", "end", 600)
    f.text(70, 112, "cases", 13.5, "f-m")
    f.text(180, 228, "~1-month lag, read off the plot", 14.5, "f-t", "middle", 600, box=W - 24)
    f.text(180, 251, "OxCGRT policy tracker + CDC case counts", 13.5, "f-m", "middle", box=W - 24)


@fig("public-space-tracking", "Public Space Tracking",
     "People in a street video are detected with Faster R-CNN and their centroids linked across frames; "
     "links that enter or leave a hand-drawn region are marked.")
def tracking(f):
    header(f, "people tracking in a public space")
    f.rect(16, 42, 328, 158, "f-node")
    f.path("M118 200 L162 42", "f-lnrule", 1.5)
    f.path("M260 200 L210 42", "f-lnrule", 1.5)
    f.path("M196 148 H300 V196 H196 Z", "f-lnhl", 2, ' stroke-dasharray="6 5"')
    f.text(300, 141, "region", 12.5, "f-h", "end", 600)
    people = [(146, 112, 20, 42, "M156 154 Q150 178 140 198"),
              (194, 84, 16, 34, "M202 118 Q214 150 232 198"),
              (228, 126, 22, 46, "M239 172 Q248 186 254 199"),
              (176, 56, 13, 26, "M182 82 Q170 120 176 198"),
              (64, 92, 18, 40, "M73 132 Q96 160 104 198")]
    for x, y, w, h, trail in people:
        f.path(trail, "f-ln", 1.6, ' stroke-dasharray="2 4"')
        f.rect(x, y, w, h, "f-lnhl", rx=2, sw=1.8)
        f.circle(x + w / 2, y + h / 2, 2.4, "f-fillhl")
    f.text(180, 228, "detect → link centroids → mark enter/leave", 14.5, "f-t", "middle", 600, box=W - 20)
    f.text(180, 251, "Faster R-CNN, TF2 Detection Zoo", 13.5, "f-m", "middle", box=W - 24)


def build(site_dir: Path, solo_dir: Path):
    site_dir.mkdir(parents=True, exist_ok=True)
    solo_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    for slug, (title, desc, fn) in FIGS.items():
        f = Fig(slug, title, desc)
        fn(f)
        warnings += f.warnings
        (site_dir / f"{slug}.svg").write_text(f.site_svg())
        for theme in ("light", "dark"):
            (solo_dir / f"{slug}-{theme}.svg").write_text(f.solo_svg(theme))
    return warnings


if __name__ == "__main__":
    w = build(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{len(FIGS)} figures")
    for x in w:
        print("WARN", x)
