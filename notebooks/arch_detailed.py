"""arch_detailed.py -- publication-style architecture figures for the motor zoo. Each learner
gets one wide figure showing the REAL internal wiring of every module (canonical GRU cell with
its gates and pointwise ops; the KINESIS morphological force decode; the fixed echo-state
reservoir + local three-factor readout; actor + twin critics; LoRA memory packs), tensor shapes
on every edge, and the LOSS block with the backward learning-signal path. Grounded in
motor_zoo(_monkey).py; MotorNet read-only.

variant: 'arm' (obs 16 → 6 muscles, sigmoid head) or '2d' (obs 12 → 3 raw → morphological force head → 4 muscles).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, FancyArrowPatch, Circle

GRAD, PLAUS, GREY = "#e76f51", "#2a9d8f", "#5b6472"
C_IN, C_GRU, GRU_D, C_LIN, C_RES, C_HEAD, C_PLANT, C_CRIT, C_LOSS, TEAL_D = \
    "#4a6fa5", "#e9a13b", "#9a6a10", "#8ec7c3", "#b8b3e6", "#7bc47f", "#c0d6df", "#c98bb9", "#e07a5f", "#1c6b60"
W_CANVAS = 34

VAR = {
    "arm": dict(obs=16, raw=6, out=6, hidden=96, Nr=4096, head="sigmoid",
                parts=[("goal", 2), ("fingertip", 2), ("musc. len", 6), ("musc. vel", 6)],
                plant="RigidTendonArm26\n2-joint · 6 Hill muscles"),
    "2d": dict(obs=12, raw=3, out=4, hidden=96, Nr=4096, head="morph",
               parts=[("goal", 2), ("position", 2), ("musc. len", 4), ("musc. vel", 4)],
               plant="ReluPointMass24\n2-D point mass · 4 muscles"),
}


# ============================================================ primitives
def _h_of(dim): return 0.75 + 1.95 * (np.log(max(dim, 2)) / np.log(2048))


def slab(ax, cx, cy, dim, color, top=None, bot=None, n=1, fixed=False, w=0.46, z=3):
    h = _h_of(dim); dx, dy = 0.17, 0.24; hatch = "////" if fixed else None
    for i in range(n):
        ox = cx - (n - 1) * 0.13 / 2 + i * 0.13; x0, y0 = ox - w / 2, cy - h / 2
        ax.add_patch(Polygon([(x0, y0 + h), (x0 + dx, y0 + h + dy), (x0 + w + dx, y0 + h + dy), (x0 + w, y0 + h)],
                     closed=True, fc=color, ec="#2b2b2b", lw=0.7, zorder=z))
        ax.add_patch(Polygon([(x0 + w, y0), (x0 + w + dx, y0 + dy), (x0 + w + dx, y0 + h + dy), (x0 + w, y0 + h)],
                     closed=True, fc=color + "aa", ec="#2b2b2b", lw=0.7, zorder=z))
        ax.add_patch(Rectangle((x0, y0), w, h, fc=color + "dd", ec="#2b2b2b", lw=0.9, hatch=hatch, zorder=z + 0.1))
    if fixed: ax.text(cx + dx / 2, cy + h / 2 + dy + 0.02, "❄", ha="center", va="bottom", fontsize=9, color=GREY, zorder=z + 1)
    if top: ax.text(cx + dx / 2, cy + h / 2 + dy + 0.18, top, ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#222", zorder=z + 1)
    if bot: ax.text(cx, cy - h / 2 - 0.16, bot, ha="center", va="top", fontsize=7.4, color=GREY, zorder=z + 1)
    return cx - w / 2, cx + w / 2 + dx, h


def block(ax, cx, cy, w, h, title, color, sub=None, fixed=False, tfs=8.6, fc=None):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc=fc or (color + "cc"), ec="#2b2b2b", lw=1.1, ls=(0, (4, 2)) if fixed else "solid", zorder=4))
    ax.text(cx, cy + (0.17 if sub else 0), title, ha="center", va="center", fontsize=tfs, fontweight="bold", color="#1a1a1a", zorder=5)
    if sub: ax.text(cx, cy - 0.27, sub, ha="center", va="center", fontsize=7.2, color="#333", zorder=5)
    return cx - w / 2, cx + w / 2


def sop(ax, cx, cy, txt, fc="#fff", ec="#333", w=0.8, h=0.56, fs=9):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                 fc=fc, ec=ec, lw=1.25, zorder=7))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs, zorder=8)
    return cx, cy, w


def node(ax, cx, cy, sym, color="#333", r=0.17, fs=10):
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=color, lw=1.5, zorder=8))
    ax.text(cx, cy + 0.01, sym, ha="center", va="center", fontsize=fs, color=color, zorder=9)
    return cx, cy


def flow(ax, p0, p1, color="#444", lw=1.6, ls="solid", rad=0.0, label=None, lfs=7.4, mut=12, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                 mutation_scale=mut, lw=lw, color=color, ls=ls, zorder=z))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.17, label, ha="center", va="bottom", fontsize=lfs, color=color, zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))


def _canvas(spec, variant, subtitle):
    v = dict(VAR[variant]); col = GRAD if spec["family"] == "global-gradient" else PLAUS
    # HEAD IS PER-MODEL, NOT PER-VARIANT. The 2-D variant used to hardcode the morphological
    # raw-3 head for ALL 13 diagrams, which contradicted the actual head split in motor_zoo:
    # only the biologically-plausible rules (and morphological KINESIS) wear the morphological
    # force head; every non-plausible learner wears the MotorNet raw-4 muscle-sigmoid head.
    if variant == "2d":
        if spec["family"] == "global-gradient":
            v.update(head="sigmoid", raw=4)          # MotorNet muscle head
        else:
            v.update(head="morph", raw=3)            # morphological force head (plausible / KINESIS)
    fig, ax = plt.subplots(figsize=(17, 7.0)); ax.set_xlim(0, W_CANVAS); ax.set_ylim(0, 13); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W_CANVAS, 13, fc="#fcfdff", ec="none", zorder=0))
    ax.text(0.3, 12.55, spec["name"], ha="left", va="top", fontsize=16, fontweight="bold", color=col)
    ax.text(0.3, 11.75, subtitle, ha="left", va="top", fontsize=9.6, color="#333")
    ax.text(W_CANVAS - 0.3, 12.5, spec["badge"], ha="right", va="top", fontsize=8.8, color=col,
            bbox=dict(boxstyle="round,pad=0.3", fc=col + "12", ec=col, lw=1))
    ax.text(W_CANVAS - 0.3, 11.8, spec["cite"], ha="right", va="top", fontsize=7.6, color=GREY, style="italic")
    ax.plot([0.3, W_CANVAS - 0.3], [11.35, 11.35], color=col, lw=1.1, alpha=0.45)
    return fig, ax, v, col


def _legend(ax, items, y=0.35):
    x = 0.5
    for lab, c in items:
        ax.add_patch(Rectangle((x, y), 0.42, 0.42, fc=c + "dd", ec="#2b2b2b", lw=0.8, zorder=4))
        ax.text(x + 0.56, y + 0.21, lab, ha="left", va="center", fontsize=7.6, color="#333", zorder=4)
        x += 0.72 + 0.15 * len(lab)


def _input(ax, v, cx=1.9, cy=7.6):
    ax.text(cx + 0.1, cy + 2.25, f"observation $o_t\\in\\mathbb{{R}}^{{{v['obs']}}}$", ha="center", fontsize=8.6, fontweight="bold")
    yy = cy + 1.55
    for name, d in v["parts"]:
        ax.add_patch(Rectangle((cx - 0.85, yy), 1.7, 0.36, fc=C_IN + "cc", ec="#2b2b2b", lw=0.7, zorder=3))
        ax.text(cx - 0.75, yy + 0.18, name, ha="left", va="center", fontsize=6.9, color="white", zorder=4)
        ax.text(cx + 0.78, yy + 0.18, str(d), ha="right", va="center", fontsize=6.9, color="white", zorder=4)
        yy -= 0.42
    block(ax, cx, cy - 1.15, 1.7, 0.6, r"norm $(o-\mu)/\sigma$", C_IN, tfs=7.4)
    return cx + 0.95, cy


# ============================================================ canonical GRU cell
def gru_cell(ax, cx, cy, col, hidden=96, xlab=r"$\hat o_t$"):
    W, H = 8.0, 5.0; X, Y = cx - W / 2, cy - H / 2
    ax.add_patch(FancyBboxPatch((X, Y), W, H, boxstyle="round,pad=0.03,rounding_size=0.14", fc="#fff8ee", ec=C_GRU, lw=1.8, zorder=2))
    ax.text(X + 0.22, Y + H - 0.24, f"GRU cell   (hidden {hidden})", ha="left", va="top", fontsize=9.5, fontweight="bold", color=GRU_D, zorder=5)
    yh, yx = cy + 1.55, cy - 1.75
    ax.text(X - 0.12, yh, r"$h_{t-1}$", ha="right", va="center", fontsize=9, color=GRU_D)
    ax.text(X - 0.12, yx, xlab, ha="right", va="center", fontsize=9, color=col)
    flow(ax, (X - 0.02, yh), (X + 0.55, yh), color=GRU_D, mut=9); flow(ax, (X - 0.02, yx), (X + 0.55, yx), color=col, mut=9)
    ax.plot([X + 0.55, X + 0.9], [yh, yh], "o", ms=4, color=GRU_D, zorder=6); ax.plot([X + 0.55, X + 0.9], [yx, yx], "o", ms=4, color=col, zorder=6)
    xg = X + 2.0
    rr = sop(ax, xg, cy - 0.55, r"$\sigma$", fc="#ffe6bf", ec=GRU_D); zz = sop(ax, xg, cy + 0.9, r"$\sigma$", fc="#ffe6bf", ec=GRU_D)
    ax.text(xg, cy - 0.55 - 0.46, "reset $r_t$", ha="center", fontsize=7.2, color=GRU_D); ax.text(xg, cy + 0.9 + 0.44, "update $z_t$", ha="center", fontsize=7.2, color=GRU_D)
    for s in (rr, zz):
        flow(ax, (X + 0.55, yx), (s[0] - 0.42, s[1] - 0.12), color=col, rad=0.05, mut=8); flow(ax, (X + 0.55, yh), (s[0] - 0.42, s[1] + 0.12), color=GRU_D, rad=-0.05, mut=8)
    mr = node(ax, xg + 1.35, cy - 0.55, "×", color=GRU_D)
    flow(ax, (rr[0] + 0.4, rr[1]), (mr[0] - 0.17, mr[1]), color=GRU_D, mut=8); flow(ax, (X + 0.9, yh), (mr[0], mr[1] + 0.17), color=GRU_D, rad=-0.25, mut=8)
    tt = sop(ax, xg + 2.7, cy - 0.55, "tanh", fc="#ffe0b0", ec=GRU_D, w=0.95)
    flow(ax, (mr[0] + 0.17, mr[1]), (tt[0] - 0.48, tt[1]), color=GRU_D, mut=8); flow(ax, (X + 0.9, yx), (tt[0] - 0.48, tt[1] - 0.28), color=col, rad=-0.08, mut=8)
    ax.text(tt[0] + 0.06, tt[1] - 0.52, r"$\tilde h_t$", ha="center", fontsize=8, color=GRU_D)
    om = sop(ax, xg + 1.35, cy + 0.9, "1−", fc="#fff", ec=GRU_D, w=0.66, fs=8.5)
    flow(ax, (zz[0] + 0.4, zz[1]), (om[0] - 0.33, om[1]), color=GRU_D, mut=8)
    ma = node(ax, xg + 2.7, cy + 1.35, "×", color=GRU_D)
    flow(ax, (om[0] + 0.33, om[1]), (ma[0] - 0.17, ma[1] - 0.12), color=GRU_D, mut=8); flow(ax, (X + 0.9, yh), (ma[0], ma[1] + 0.17), color=GRU_D, rad=0.12, mut=8)
    mb = node(ax, xg + 3.9, cy - 0.05, "×", color=GRU_D)
    flow(ax, (zz[0] + 0.2, zz[1] - 0.28), (mb[0] - 0.05, mb[1] + 0.17), color=GRU_D, rad=-0.3, mut=8); flow(ax, (tt[0] + 0.48, tt[1]), (mb[0] - 0.17, mb[1]), color=GRU_D, rad=0.1, mut=8)
    pl = node(ax, xg + 4.8, cy + 0.65, "+", color=GRU_D)
    flow(ax, (ma[0] + 0.17, ma[1]), (pl[0] - 0.12, pl[1] + 0.13), color=GRU_D, mut=8); flow(ax, (mb[0] + 0.17, mb[1]), (pl[0] - 0.12, pl[1] - 0.13), color=GRU_D, mut=8)
    hx = X + W; flow(ax, (pl[0] + 0.17, pl[1]), (hx, pl[1]), color=C_GRU, mut=12)
    ax.text(hx - 0.14, pl[1] + 0.3, r"$h_t$", ha="right", fontsize=9, color=GRU_D)
    ax.add_patch(FancyArrowPatch((hx - 0.15, cy + 0.65), (X + 0.12, yh), connectionstyle="arc3,rad=0.42", arrowstyle="-|>", mutation_scale=10, lw=1.1, color=C_GRU, ls="dotted", zorder=3))
    ax.text(cx, Y + H + 0.02, r"recurrence  $h_{t}\!\to\!h_{t-1}$", ha="center", fontsize=7.4, color=GRU_D, style="italic")
    ax.text(X + 0.2, Y + 0.18, r"$r_t,z_t=\sigma(W_{r,z}[\hat o_t,h_{t-1}]),\ \tilde h_t=\tanh(W_h[\hat o_t,r_t\odot h_{t-1}]),\ h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$",
            ha="left", va="bottom", fontsize=6.8, color="#7a5300")
    return X, hx, (hx, pl[1])


# ============================================================ heads
def sigmoid_head(ax, cx, cy, out):
    l, r = block(ax, cx, cy, 2.1, 1.3, "Muscle head", C_HEAD, sub=r"$a=\sigma(\mathrm{raw})$")
    return l, r, cy


def morph_head(ax, cx, cy, out=4):
    W, H = 8.0, 4.6; X, Y = cx - W / 2, cy - H / 2
    ax.add_patch(FancyBboxPatch((X, Y), W, H, boxstyle="round,pad=0.03,rounding_size=0.12", fc="#eaf7f5", ec=PLAUS, lw=1.7, zorder=2))
    ax.text(X + 0.2, Y + H - 0.24, "Morphological force head   (the body computes the muscle map)", ha="left", va="top", fontsize=9, fontweight="bold", color=TEAL_D, zorder=5)
    ax.text(X - 0.12, cy + 0.45, r"raw$\,[3]$", ha="right", va="center", fontsize=8.5, color=PLAUS)
    tf = sop(ax, X + 1.25, cy + 1.15, "tanh", fc="#cfeeea", ec=TEAL_D, w=0.92); sc = sop(ax, X + 1.25, cy - 0.2, r"$\sigma$", fc="#cfeeea", ec=TEAL_D)
    flow(ax, (X - 0.02, cy + 0.45), (tf[0] - 0.47, tf[1]), color=PLAUS, rad=0.15, mut=9); flow(ax, (X - 0.02, cy + 0.45), (sc[0] - 0.42, sc[1]), color=PLAUS, rad=-0.15, mut=9)
    ax.text(tf[0], tf[1] + 0.42, r"$\mathrm{raw}_{0:2}$", ha="center", fontsize=6.8, color=TEAL_D); ax.text(sc[0] - 0.5, sc[1] - 0.42, r"$\mathrm{raw}_{2}$", ha="center", fontsize=6.8, color=TEAL_D)
    ff = sop(ax, X + 2.65, cy + 1.15, r"$\times F_s$", fc="white", ec=TEAL_D, w=0.82); flow(ax, (tf[0] + 0.46, tf[1]), (ff[0] - 0.41, ff[1]), color=TEAL_D, mut=9)
    ax.text(X + 3.35, cy + 1.62, r"$f$  (2-D endpoint force)", ha="left", va="center", fontsize=7.0, color=TEAL_D); ax.text(sc[0] + 0.45, sc[1], r"$c$  (co-contraction)", ha="left", va="center", fontsize=7.0, color=TEAL_D)
    bg = block(ax, X + 3.0, cy - 1.5, 4.4, 0.6, r"body geometry   $d_m=(A_m-P)/l_m$", C_HEAD, tfs=7.4, fc="#dff0ed")
    ax.text(X + 3.0, cy - 2.05, r"anchors $A_m$ (fixed) · position $P$, muscle length $l_m$  (from $o_t$)", ha="center", fontsize=6.7, color="#555")
    pm = block(ax, X + 6.1, cy + 0.25, 3.1, 1.5, "", "#cfeeea", tfs=7.0)
    ax.text(X + 6.1, cy + 0.25, r"$a_m=\mathrm{clip}(\mathrm{relu}(d_m\cdot f)/F_{max}+c)$", ha="center", va="center", fontsize=7.0, zorder=6)
    flow(ax, (ff[0] + 0.41, ff[1]), (X + 6.1 - 1.55, cy + 0.7), color=TEAL_D, rad=-0.08, mut=9)
    flow(ax, (sc[0] + 0.36, sc[1]), (X + 6.1 - 1.55, cy + 0.25), color=TEAL_D, rad=-0.12, mut=9)
    flow(ax, (X + 3.0 + 2.25, cy - 1.5), (X + 6.1 - 1.0, cy - 0.35), color=TEAL_D, rad=-0.1, mut=9)
    return X, X + W, (X + W, cy + 0.25)


def synergy_head(ax, cx, cy, out=6):
    W, H = 5.8, 3.8; X, Y = cx - W / 2, cy - H / 2
    ax.add_patch(FancyBboxPatch((X, Y), W, H, boxstyle="round,pad=0.03,rounding_size=0.12", fc="#eaf7f5", ec=PLAUS, lw=1.7, zorder=2))
    ax.text(X + 0.2, Y + H - 0.24, "Fixed muscle synergies   (morphological computation)", ha="left", va="top", fontsize=9, fontweight="bold", color=TEAL_D, zorder=5)
    sg = block(ax, X + 1.2, cy, 1.8, 0.7, r"$s=\sigma(W_s h)$", "#cfeeea", tfs=8.0); ax.text(X + 1.2, cy - 0.55, "4 synergy drives", ha="center", fontsize=6.8, color="#555")
    mx, my, cw, ch = X + 2.9, cy - 0.9, 0.24, 0.3
    ax.text(mx + 2 * cw, my + 6 * ch + 0.14, r"synergy matrix $M\in\mathbb{R}^{6\times4}$ (fixed)", ha="center", fontsize=7.2, color=TEAL_D)
    M = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
    for i in range(6):
        for j in range(4):
            ax.add_patch(Rectangle((mx + j * cw, my + (5 - i) * ch), cw, ch, fc=(PLAUS if M[i][j] else "#ffffff"), ec=TEAL_D, lw=0.6, zorder=6))
    flow(ax, (X + 2.1, cy), (mx - 0.12, cy), color=TEAL_D, mut=9)
    return X, X + W, (X + W, cy)


def _head(ax, v, cx, cy):
    if v["head"] == "morph": return morph_head(ax, cx, cy, v["out"])
    return sigmoid_head(ax, cx, cy, v["out"])


def _plant(ax, v, cx, cy):
    p = v["plant"].split("\n")
    block(ax, cx, cy, 2.7, 1.7, p[0], C_PLANT, sub=p[1] if len(p) > 1 else None, tfs=7.8)
    ax.text(cx, cy - 1.15, r"$r_t=-d/d_{max}-\lambda_e\Vert a\Vert^2$", ha="center", fontsize=7.2, color="#222")
    return cx - 1.35, cx + 1.35


def _loss(ax, cx, cy, text, col):
    block(ax, cx, cy, 2.6, 1.1, "Loss", C_LOSS, tfs=9)
    ax.text(cx, cy - 0.75, text, ha="center", va="top", fontsize=8.0, color=col, fontweight="bold")


def _backward(ax, col, text, x0=W_CANVAS - 1.0, x1=1.6, y=2.5):
    flow(ax, (x0, y), (x1, y), color=col, ls=(0, (6, 3)), lw=2.0, mut=17)
    ax.text((x0 + x1) / 2, y + 0.22, text, ha="center", va="bottom", fontsize=8.2, color=col, style="italic")


# ============================================================ renderers
def draw_gru_apg(spec, variant, head="std"):
    fig, ax, v, col = _canvas(spec, variant, "gated-recurrent controller · trained end-to-end by analytic policy gradient (BPTT through the differentiable plant)")
    ix, iy = _input(ax, v)
    gl, gr, hpt = gru_cell(ax, 8.0, iy, col, hidden=v["hidden"])
    flow(ax, (ix, iy), (gl, iy), color=col, label=f"$\\hat o$ [{v['obs']}]")
    hs_l, hs_r, _ = slab(ax, gr + 1.3, iy, v["hidden"], C_GRU, top="$h_t$", bot=f"[{v['hidden']}]")
    flow(ax, (gr, iy), (hs_l, iy), color=col)
    if head == "synergy":
        shl, shr, spt = synergy_head(ax, gr + 5.4, iy, v["out"])
        flow(ax, (hs_r, iy), (shl + 0.3, iy), color=col, label=f"$h_t$[{v['hidden']}]")
        msl, msr, _ = slab(ax, shr + 1.2, iy, v["out"], C_HEAD, top="muscles", bot=f"[{v['out']}]"); flow(ax, (spt[0], spt[1]), (msl, iy), color=PLAUS, label=f"a[{v['out']}]")
        pl_, pr_ = _plant(ax, v, msr + 1.6, iy); flow(ax, (msr, iy), (pl_, iy), color=PLAUS)
        _loss(ax, W_CANVAS - 2.5, iy - 3.2, r"$L=-\sum_t r_t$", col)
        _backward(ax, col, "backward: analytic policy gradient  (plausibility is morphological — the fixed synergy map — not the update)")
    else:
        rl, rr = block(ax, gr + 3.5, iy, 2.1, 1.2, "Linear readout", C_LIN, sub=rf"$W_o h_t\to$ raw[{v['raw']}]")
        flow(ax, (hs_r, iy), (rl, iy), color=col)
        hl, hr, hpt2 = _head(ax, v, (rr + 5.2) if v["head"] == "morph" else (rr + 1.9), iy)
        flow(ax, (rr, iy), (hl, iy), color=col, label=f"raw[{v['raw']}]")
        msl, msr, _ = slab(ax, hr + 1.2, iy, v["out"], C_HEAD, top="muscles", bot=f"[{v['out']}]"); flow(ax, (hpt2[0] if v["head"] == "morph" else hr, iy), (msl, iy), color=col, label=f"a[{v['out']}]")
        pl_, pr_ = _plant(ax, v, msr + 1.6, iy); flow(ax, (msr, iy), (pl_, iy), color=col)
        _loss(ax, W_CANVAS - 2.5, iy - 3.2, r"$L=-\sum_t r_t$" + "\n(maximise return)", col)
        _backward(ax, col, "backward: backprop-through-time through the unrolled policy × the differentiable plant")
    _legend(ax, [("input", C_IN), ("GRU", C_GRU), ("linear", C_LIN), ("head", C_HEAD), ("plant", C_PLANT), ("loss", C_LOSS)])
    ax.text(0.3, 1.05, "trained: GRU {W_r,W_z,W_h} + readout · Adam, grad-clip 1.0 · 100 steps/ep, batch 256", fontsize=7.6, color="#444")
    fig.tight_layout(); return fig


def draw_shac(spec, variant):
    fig, ax, v, col = _canvas(spec, variant, "recurrent actor + value critic · short-horizon differentiable rollout (H=50) with a bootstrapped value tail")
    ix, iy = _input(ax, v)
    gl, gr, hpt = gru_cell(ax, 8.0, iy, col, hidden=v["hidden"]); flow(ax, (ix, iy), (gl, iy), color=col, label=f"$\\hat o$[{v['obs']}]")
    hs_l, hs_r, _ = slab(ax, gr + 1.3, iy, v["hidden"], C_GRU, top="$h_t$", bot=f"[{v['hidden']}]"); flow(ax, (gr, iy), (hs_l, iy), color=col)
    rl, rr = block(ax, gr + 3.3, iy, 1.9, 1.1, "actor readout", C_LIN, sub=f"raw[{v['raw']}]"); flow(ax, (hs_r, iy), (rl, iy), color=col)
    hl, hr, hpt2 = _head(ax, v, (rr + 5.2) if v["head"] == "morph" else (rr + 1.9), iy); flow(ax, (rr, iy), (hl, iy), color=col, label="raw")
    msl, msr, _ = slab(ax, hr + 1.2, iy, v["out"], C_HEAD, top="musc.", bot=f"[{v['out']}]"); flow(ax, (hpt2[0] if v["head"] == "morph" else hr, iy), (msl, iy), color=col)
    pl_, pr_ = _plant(ax, v, msr + 1.6, iy); flow(ax, (msr, iy), (pl_, iy), color=col)
    block(ax, gr + 3.3, iy - 3.2, 2.2, 1.2, "Value critic", C_CRIT, sub=r"MLP 128-128 → $V_\phi$")
    flow(ax, (ix + 0.3, iy - 0.9), (gr + 2.4, iy - 3.2), color=C_CRIT, ls="dotted", rad=-0.15, label=r"$\hat o$")
    ax.text(W_CANVAS / 2 + 3, 2.5, r"actor  $L_\pi=-(\sum_{k=0}^{H-1}\gamma^k r_{t+k}+\gamma^H V_\phi(\hat o_{t+H}))$  (truncated BPTT);   critic  $L_V=\mathrm{MSE}(V_\phi,\ \mathrm{GAE})$",
            ha="center", fontsize=8.2, color=col, fontweight="bold")
    _legend(ax, [("input", C_IN), ("GRU actor", C_GRU), ("linear", C_LIN), ("critic", C_CRIT), ("head", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig


def draw_bootstrap(spec, variant):
    flavor = spec["flavor"]
    fig, ax, v, col = _canvas(spec, variant, "off-policy actor-critic · RANDOM recurrent actor trained FROM SCRATCH on a perfect (expert) replay buffer + behaviour-cloning anchor (no weight copy)")
    ix, iy = _input(ax, v, cy=8.4)
    al, ar = block(ax, 7.0, 8.4, 3.2, 1.9, "GRUCell actor", C_GRU, sub=f"hidden {v['hidden']} · random init (scratch)")
    ax.text(7.0, 7.65, r"$h_t=\mathrm{GRUCell}(\hat o_t,h_{t-1});\ \mathrm{raw}=W_o h_t$" + (r" ; $+\log\sigma$" if flavor == "sac" else ""), ha="center", fontsize=7.0, color="#7a5300")
    flow(ax, (ix, 8.4), (al, 8.4), color=col, label=f"$\\hat o$[{v['obs']}]")
    sl, sr, _ = slab(ax, ar + 1.2, 8.4, v["raw"], C_LIN, top="raw", bot=f"[{v['raw']}]"); flow(ax, (ar, 8.4), (sl, 8.4), color=col)
    hl, hr, hpt2 = _head(ax, v, (sr + 5.0) if v["head"] == "morph" else (sr + 1.9), 8.4); flow(ax, (sr, 8.4), (hl, 8.4), color=col)
    msl, msr, _ = slab(ax, hr + 1.2, 8.4, v["out"], C_HEAD, top="musc.", bot=f"[{v['out']}]"); flow(ax, (hpt2[0] if v["head"] == "morph" else hr, 8.4), (msl, 8.4), color=col)
    pl_, pr_ = _plant(ax, v, msr + 1.6, 8.4); flow(ax, (msr, 8.4), (pl_, 8.4), color=col)
    block(ax, 4.5, 4.6, 2.4, 1.1, "Perfect replay buf.", GREY, sub="expert pre-fill\n(o,a,r,o',d,a*)", tfs=7.4)
    ct = "twin critics — SimbaNet" if flavor == "simba" else "twin critics — MLP"
    block(ax, 9.0, 4.6, 3.4, 1.5, ct, C_CRIT, sub=r"$Q_1,Q_2(\hat o,a)$ + targets · " + ("resid." if flavor == "simba" else "256-256"))
    block(ax, 14.0, 4.6, 3.0, 1.2, "Demonstrator (teacher)", PLAUS, sub=r"frozen → stores $a^*_t$", fixed=True, tfs=7.8)
    flow(ax, (5.7, 4.6), (7.3, 4.6), color=GREY, mut=10); flow(ax, (12.5, 4.6), (10.7, 4.6), color=C_CRIT, ls="dotted", mut=10)
    flow(ax, (14.0, 5.2), (8.6, 7.5), color=PLAUS, ls="dotted", rad=-0.1, label="BC target $a^*$", mut=10)
    flow(ax, (9.0, 5.35), (7.0, 7.45), color=C_CRIT, ls="dotted", rad=0.2, label=r"$-Q$", mut=10)
    ent = r"+\alpha\log\pi" if flavor == "sac" else ""
    ax.text(W_CANVAS / 2 + 4, 2.7, rf"critic  $L_Q=\mathrm{{MSE}}(Q_i,\ r+\gamma(1-d)\min_i Q_i')$", ha="center", fontsize=8.2, color=C_CRIT, fontweight="bold")
    ax.text(W_CANVAS / 2 + 4, 2.05, rf"actor  $L_\pi=\beta\Vert a-a^*\Vert^2-\rho\,Q_1(\hat o,a){ent}$   (BC anchor − RL)", ha="center", fontsize=8.4, color=col, fontweight="bold")
    ax.text(W_CANVAS / 2 + 4, 1.4, "off-policy (no plant gradient) · Polyak targets $\\tau$ · " + ("entropy-regularised" if flavor == "sac" else "TD3 twin-Q + target smoothing"), ha="center", fontsize=7.4, color="#444")
    _legend(ax, [("input", C_IN), ("GRU actor", C_GRU), ("linear", C_LIN), ("critic", C_CRIT), ("head", C_HEAD), ("plant", C_PLANT), ("teacher", PLAUS)])
    fig.tight_layout(); return fig


def reservoir_module(ax, cx, cy, v, col):
    """Fixed echo-state reservoir drawn with its input weights, recurrent weights and leaky node."""
    W, H = 5.2, 4.6; X, Y = cx - W / 2, cy - H / 2
    ax.add_patch(FancyBboxPatch((X, Y), W, H, boxstyle="round,pad=0.03,rounding_size=0.12", fc="#f3f1fb", ec=C_RES, lw=1.7, ls=(0, (5, 3)), zorder=2))
    ax.text(X + 0.2, Y + H - 0.24, f"Fixed reservoir  (N={v['Nr']})  ❄", ha="left", va="top", fontsize=8.8, fontweight="bold", color="#4a4590", zorder=5)
    win = block(ax, X + 1.15, cy, 1.3, 0.8, r"$W_{in}$", C_RES, sub="fixed", tfs=8.5, fc="#d9d3f2")
    lk = node(ax, X + 3.0, cy, "leaky", color="#4a4590", r=0.5, fs=6.5)
    ax.text(X + 3.0, cy - 0.78, r"$(1{-}a)h{+}a\,\tanh(\cdot)$", ha="center", fontsize=6.6, color="#4a4590")
    wres = block(ax, X + 3.25, cy + 1.5, 1.5, 0.66, r"$W_{res}$ ($\rho$, fixed)", C_RES, tfs=7.2, fc="#d9d3f2")
    flow(ax, (X + 1.8, cy), (lk[0] - 0.5, cy), color="#4a4590", mut=9)
    ax.add_patch(FancyArrowPatch((lk[0] + 0.25, cy + 0.45), (wres[0] + 0.45, cy + 1.16), connectionstyle="arc3,rad=-0.3", arrowstyle="-|>", mutation_scale=8, lw=1.0, color="#4a4590", zorder=6))
    ax.add_patch(FancyArrowPatch((wres[0] - 0.55, cy + 1.16), (lk[0] - 0.2, cy + 0.45), connectionstyle="arc3,rad=-0.3", arrowstyle="-|>", mutation_scale=8, lw=1.0, color="#4a4590", ls="dotted", zorder=6))
    ax.text(X + 1.75, cy + 0.95, "recurrence", ha="center", fontsize=6.4, color="#4a4590", style="italic")
    flow(ax, (lk[0] + 0.5, cy), (X + W, cy), color="#4a4590", mut=10)
    ax.text(X + W - 0.15, cy + 0.3, r"$h_t$", ha="right", fontsize=8.5, color="#4a4590")
    return X - 0.9, X + W, (X + W, cy)   # left input anchor at Win


def draw_reservoir(spec, variant):
    elig = spec["elig"]; etxt = "low-pass eligibility trace" if elig == "trace" else "instantaneous 3-factor"
    fig, ax, v, col = _canvas(spec, variant, f"FIXED echo-state reservoir + a plastic linear readout trained by a LOCAL three-factor rule ({etxt}) — no BPTT, no reward")
    ix, iy = _input(ax, v)
    rl, rr, hpt = reservoir_module(ax, 7.3, iy, v, col); flow(ax, (ix, iy), (rl + 0.4, iy), color=col, label=f"$\\hat o$[{v['obs']}]")
    zl, zr, _ = slab(ax, rr + 1.2, iy, v["Nr"] + v["obs"], C_RES, top=r"$z=[h;\hat o]$", bot=f"[{v['Nr']}+{v['obs']}]", w=0.55)
    flow(ax, (rr, iy), (zl, iy), color="#4a4590")
    ro_l, ro_r = block(ax, zr + 1.6, iy, 2.2, 1.2, "Plastic readout", C_LIN, sub=r"$\mathrm{raw}=Wz+b$"); flow(ax, (zr, iy), (ro_l, iy), color="#4a4590")
    hl, hr, hpt2 = _head(ax, v, (ro_r + 5.2) if v["head"] == "morph" else (ro_r + 1.9), iy); flow(ax, (ro_r, iy), (hl, iy), color=col, label=f"raw[{v['raw']}]")
    msl, msr, _ = slab(ax, hr + 1.2, iy, v["out"], C_HEAD, top="musc.", bot=f"[{v['out']}]"); flow(ax, (hpt2[0] if v["head"] == "morph" else hr, iy), (msl, iy), color=col)
    pl_, pr_ = _plant(ax, v, msr + 1.6, iy); flow(ax, (msr, iy), (pl_, iy), color=col)
    # teacher + local rule band
    block(ax, ro_l + 1.1, iy - 3.3, 2.8, 1.0, "Demonstrator (teacher)", PLAUS, sub=r"BPTT-GRU → $a^*_t$", fixed=True, tfs=7.6)
    nd = node(ax, ro_r + 1.1, iy - 3.3, "−", color=PLAUS)
    ax.text(ro_r + 2.7, iy - 3.3, r"error  $\delta_t=a^*_t-\mathrm{raw}_t$", ha="left", va="center", fontsize=8.2, color=PLAUS, fontweight="bold")
    trace = r"$\bar e=(1-\frac{dt}{\tau})\bar e+\frac{dt}{\tau}z$" if elig == "trace" else r"$e=z$"
    ax.text(ro_r + 2.7, iy - 3.95, "eligibility " + trace + r";   local update  $\Delta W=\eta\langle\delta\otimes e\rangle-\lambda W$   (readout only)", ha="left", va="center", fontsize=8.2, color=col, fontweight="bold")
    flow(ax, (ro_l + 2.5, iy - 3.3), (nd[0] - 0.18, iy - 3.3), color=PLAUS, mut=10)
    flow(ax, (ro_r + 1.1, iy - 0.65), (ro_r + 1.1, iy - 3.12), color=PLAUS, ls="dotted", mut=10)
    ax.text(W_CANVAS / 2, 1.1, "the reservoir is FROZEN — only the readout W learns, by a local outer product (Hebbian × error) · no backprop-through-time, no reward", ha="center", fontsize=7.8, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("fixed reservoir", C_RES), ("plastic readout", C_LIN), ("head", C_HEAD), ("plant", C_PLANT), ("teacher", PLAUS)])
    fig.tight_layout(); return fig


def draw_dendritron(spec, variant):
    fig, ax, v, col = _canvas(spec, variant, "fixed reservoir + a FROZEN base readout + per-context LoRA memory packs bound by an autonomous router → continual learning, no forgetting")
    ix, iy = _input(ax, v)
    rl, rr, hpt = reservoir_module(ax, 7.0, iy, v, col); flow(ax, (ix, iy), (rl + 0.4, iy), color=col, label=f"$\\hat o$")
    zl, zr, _ = slab(ax, rr + 1.2, iy, v["Nr"] + v["obs"], C_RES, top=r"$z$", bot=f"[{v['Nr']}+{v['obs']}]", w=0.55); flow(ax, (rr, iy), (zl, iy), color="#4a4590")
    # readout = frozen base W0 + LoRA pack (A_c down, B_c up)
    W, H = 4.6, 3.2; bx = zr + 3.4; X, Y = bx - W / 2, iy - H / 2
    ax.add_patch(FancyBboxPatch((X, Y), W, H, boxstyle="round,pad=0.03,rounding_size=0.1", fc="#eef6f4", ec=C_LIN, lw=1.5, zorder=2))
    ax.text(X + 0.15, Y + H - 0.2, "Readout: frozen base + LoRA pack", ha="left", va="top", fontsize=8.4, fontweight="bold", color="#1c6b60")
    block(ax, bx, iy + 0.75, 3.0, 0.6, r"$W_0 z$  (frozen after skill-1)", C_LIN, tfs=7.2, fc="#cfe6e0")
    ad = block(ax, bx - 1.0, iy - 0.5, 1.1, 0.7, r"$A_c z$", "#c9a227", tfs=7.6, fc="#f0e2b8"); bu = block(ax, bx + 1.0, iy - 0.5, 1.1, 0.7, r"$B_c$", "#c9a227", tfs=7.6, fc="#f0e2b8")
    ax.text(bx, iy - 1.15, "low-rank memory pack (per context $c$)", ha="center", fontsize=6.8, color="#8a6d1a")
    flow(ax, (ad[1], iy - 0.5), (bu[0], iy - 0.5), color="#8a6d1a", mut=9, label="rank $r$")
    flow(ax, (zl - 0.35, iy), (X + 0.1, iy), color="#4a4590")
    ax.text(bx, Y - 0.25, r"$\mathrm{raw}=W_0 z+(zA_c^\top)B_c^\top+b$", ha="center", fontsize=7.6, color="#1c6b60")
    hl, hr, hpt2 = _head(ax, v, (bx + W / 2 + 5.0) if v["head"] == "morph" else (bx + W / 2 + 1.9), iy); flow(ax, (X + W, iy), (hl, iy), color=col, label=f"raw[{v['raw']}]")
    msl, msr, _ = slab(ax, hr + 1.2, iy, v["out"], C_HEAD, top="musc.", bot=f"[{v['out']}]"); flow(ax, (hpt2[0] if v["head"] == "morph" else hr, iy), (msl, iy), color=col)
    _plant(ax, v, msr + 1.6, iy); flow(ax, (msr, iy), (msr + 1.6 - 1.35, iy), color=col)
    block(ax, bx, iy - 2.4, 2.6, 0.9, "Autonomous router", "#c9a227", sub="context → best-return pack (no labels)", tfs=7.8)
    flow(ax, (bx, iy - 1.95), (bx, iy - H / 2 + 0.05), color="#c9a227", ls="dotted", mut=9)
    ax.text(W_CANVAS / 2, 2.35, r"base (skill-1): $\Delta W_0=\eta\langle\delta\otimes z\rangle-\lambda W_0$;   pack (later): $\Delta B_c=\eta\langle\delta\otimes(A_c z)\rangle,\ \Delta A_c=\eta\langle(B_c^\top\delta)\otimes z\rangle$,  $\delta=a^*-\mathrm{raw}$",
            ha="center", fontsize=7.8, color=col, fontweight="bold")
    ax.text(W_CANVAS / 2, 1.6, "a new skill = a new FROZEN pack → cannot overwrite an old one (no catastrophic forgetting) · local three-factor, no BPTT", ha="center", fontsize=7.6, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("fixed reservoir", C_RES), ("readout+packs", C_LIN), ("router", "#c9a227"), ("head", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig


# ============================================================ DISTINCT plausible renderers
# Every local-plausible rule shares two explicitly-labelled crutches -- a FIXED reservoir
# (working-memory substrate) and the morphological/muscle head -- but each has its OWN plastic
# component, eligibility/trace, third factor, and characteristic unit, so no two diagrams are
# the same. (Previously e-prop..hebb3 all reused draw_reservoir with only an `elig` badge.)
def _res_front(spec, variant, subtitle, res_note):
    fig, ax2, v, col = _canvas(spec, variant, subtitle)
    ix, iy = _input(ax2, v)
    rl, rr, hpt = reservoir_module(ax2, 6.8, iy, v, col)
    flow(ax2, (ix, iy), (rl + 0.4, iy), color=col, label=f"$\\hat o$[{v['obs']}]")
    if res_note: ax2.text(6.8, iy - 2.55, res_note, ha="center", fontsize=7.0, color="#4a4590", style="italic")
    return fig, ax2, v, col, iy, rr

def _res_back(ax, v, col, iy, ro_r, zbadge=True):
    hl, hr, hpt2 = _head(ax, v, (ro_r + 5.0) if v["head"] == "morph" else (ro_r + 1.9), iy)
    flow(ax, (ro_r, iy), (hl, iy), color=col, label=f"raw[{v['raw']}]")
    msl, msr, _ = slab(ax, hr + 1.15, iy, v["out"], C_HEAD, top="musc.", bot=f"[{v['out']}]")
    flow(ax, (hpt2[0] if v["head"] == "morph" else hr, iy), (msl, iy), color=col)
    pl_, _ = _plant(ax, v, msr + 1.5, iy); flow(ax, (msr, iy), (pl_, iy), color=col)

def _zslab(ax, v, x, iy):
    zl, zr, _ = slab(ax, x, iy, v["Nr"] + v["obs"], C_RES, top=r"$z=[h;\hat o]$", bot=f"[{v['Nr']}+{v['obs']}]", w=0.55)
    return zl, zr

def _teacher_band(ax, x, iy, col, elig_txt, upd_txt, dy=-3.35):
    block(ax, x, iy + dy, 2.7, 0.95, "Demonstrator", PLAUS, sub=r"BPTT-GRU $\to a^*_t$", fixed=True, tfs=7.4)
    nd = node(ax, x + 3.0, iy + dy, "−", color=PLAUS)
    ax.text(x + 3.4, iy + dy, r"$\delta_t=a^*_t-\mathrm{raw}_t$", ha="left", va="center", fontsize=8.0, color=PLAUS, fontweight="bold")
    ax.text(x + 3.4, iy + dy - 0.62, elig_txt, ha="left", va="center", fontsize=7.8, color="#333")
    ax.text(x + 3.4, iy + dy - 1.18, upd_txt, ha="left", va="center", fontsize=8.0, color="#111", fontweight="bold")
    flow(ax, (x + 1.4, iy + dy), (nd[0] - 0.18, iy + dy), color=PLAUS, mut=9)
    return nd

def draw_eprop(spec, variant):
    fig, ax, v, col, iy, rr = _res_front(spec, variant,
        "ALIF reservoir + eligibility-trace readout with a broadcast learning signal — forward-in-time credit, no BPTT",
        r"units are ADAPTIVE (ALIF): $a_t\!=\!\rho a_{t-1}+h_t$ raises the effective threshold")
    zl, zr = _zslab(ax, v, rr + 1.15, iy); flow(ax, (rr, iy), (zl, iy), color="#4a4590")
    # adaptation tag on the reservoir
    block(ax, 6.8, iy + 2.15, 2.2, 0.6, r"adaptation $a_t$", "#d9d3f2", tfs=7.4, fc="#e6e1f7")
    flow(ax, (6.8, iy + 1.15), (6.8, iy + 1.85), color="#4a4590", ls="dotted", mut=8)
    ro_l, ro_r = block(ax, zr + 1.7, iy, 2.2, 1.2, "Plastic readout", C_LIN, sub=r"$\mathrm{raw}=Wz+b$"); flow(ax, (zr, iy), (ro_l, iy), color="#4a4590")
    _res_back(ax, v, col, iy, ro_r)
    block(ax, zr + 1.7, iy - 1.9, 2.4, 0.7, r"eligibility trace $\bar e$", PLAUS, sub=r"$\bar e=(1{-}\frac{dt}{\tau_e})\bar e+\frac{dt}{\tau_e}\psi z$", tfs=7.4, fc="#d9efec")
    flow(ax, (zr + 1.7, iy - 0.6), (zr + 1.7, iy - 1.55), color="#4a4590", ls="dotted", mut=8)
    nd = _teacher_band(ax, ro_l - 0.4, iy, col, r"learning signal $L_t=\delta_t$ (broadcast);   eligibility trace $\bar e$ (low-pass, ALIF pseudo-derivative $\psi$)",
                       r"$\Delta W=\eta\,\langle L_t\otimes \bar e_t\rangle-\lambda W$   (readout only, forward in time)")
    ax.text(W_CANVAS / 2, 1.05, "adaptive units + a temporal eligibility trace are the e-prop signature — distinct from RFLO's instantaneous rule", ha="center", fontsize=7.6, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("ALIF reservoir", C_RES), ("readout", C_LIN), ("eligibility", PLAUS), ("head", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig

def draw_rtrrl(spec, variant):
    fig, ax, v, col, iy, rr = _res_front(spec, variant,
        "reservoir + real-time local readout — instantaneous update, error routed by fixed RANDOM FEEDBACK (feedback alignment)",
        r"RFLO: no eligibility trace, no adaptation — credit is immediate")
    zl, zr = _zslab(ax, v, rr + 1.15, iy); flow(ax, (rr, iy), (zl, iy), color="#4a4590")
    ro_l, ro_r = block(ax, zr + 1.7, iy, 2.2, 1.2, "Plastic readout", C_LIN, sub=r"$\mathrm{raw}=Wz+b$"); flow(ax, (zr, iy), (ro_l, iy), color="#4a4590")
    _res_back(ax, v, col, iy, ro_r)
    # random feedback matrix B on the error path
    bb = block(ax, ro_r + 0.2, iy - 2.4, 1.9, 0.75, r"random $B$", "#c98bb9", sub="feedback align.", tfs=7.6, fc="#eccfe0")
    nd = _teacher_band(ax, ro_l - 0.4, iy, col, r"random feedback  $\tilde\delta_t=B\,\delta_t$  (fixed $B$, no weight transport);   eligibility $e=z$ (instantaneous)",
                       r"$\Delta W=\eta\,\langle \tilde\delta_t\otimes z_t\rangle-\lambda W$   (real-time, forward, no BPTT)")
    flow(ax, (nd[0], iy - 3.35 + 0.42), (bb[0] + 0.9, iy - 2.78), color="#b0559a", ls="dotted", mut=8)
    ax.text(W_CANVAS / 2, 1.05, "the fixed random-feedback pathway (feedback alignment) is the RFLO signature — no stored trace, no adaptation", ha="center", fontsize=7.6, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("fixed reservoir", C_RES), ("readout", C_LIN), ("random feedback", "#c98bb9"), ("head", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig

def draw_btsp(spec, variant):
    fig, ax, v, col, iy, rr = _res_front(spec, variant,
        "reservoir + a plastic readout written by rare dendritic PLATEAUS — a seconds-long trace bound one-shot to the target",
        r"behavioural timescale: $\tau_{slow}\!\approx\!1$ s (10$\times$ every other rule)")
    zl, zr = _zslab(ax, v, rr + 1.15, iy); flow(ax, (rr, iy), (zl, iy), color="#4a4590")
    ro_l, ro_r = block(ax, zr + 1.7, iy, 2.2, 1.2, "Plastic readout", C_LIN, sub=r"$\mathrm{raw}=Wz+b$"); flow(ax, (zr, iy), (ro_l, iy), color="#4a4590")
    _res_back(ax, v, col, iy, ro_r)
    # slow trace + plateau gate
    tb = block(ax, zr + 1.7, iy - 1.9, 2.5, 0.7, r"slow trace $\bar z$", PLAUS, sub=r"$\bar z=(1{-}\frac{dt}{\tau_{slow}})\bar z+\frac{dt}{\tau_{slow}}z$", tfs=7.2, fc="#d9efec")
    flow(ax, (zr + 1.7, iy - 0.6), (zr + 1.7, iy - 1.55), color="#4a4590", ls="dotted", mut=8)
    pg = node(ax, ro_l + 0.9, iy - 3.35, "⚡", color="#c98bb9", r=0.26, fs=11)
    ax.text(ro_l + 0.9, iy - 3.9, "dendritic\nplateau (sparse)", ha="center", va="top", fontsize=6.6, color="#b0559a")
    _teacher_band(ax, ro_l + 1.9, iy, col, r"plateau gate $g_t\!\in\!\{0,1\}$ fires rarely (prob $\propto p_{plat}$);   bind slow trace $\bar z$ to error $\delta$",
                  r"$\Delta W=\eta\,g_t\,\langle\delta_t\otimes\bar z_t\rangle$   (one-shot, behavioural-timescale)")
    ax.text(W_CANVAS / 2, 1.05, "a sparse plateau gate writing a seconds-long trace in one shot is unique to BTSP — no continuous gradient", ha="center", fontsize=7.6, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("fixed reservoir", C_RES), ("readout", C_LIN), ("plateau gate", "#c98bb9"), ("head", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig

def draw_rstdp(spec, variant):
    fig, ax, v, col, iy, rr = _res_front(spec, variant,
        "reservoir readout with a reward-modulated STDP eligibility tag — pre-spike × post-error, gated by a global DOPAMINE signal",
        r"eligibility is SPIKE-timing gated ($s=\mathbb{1}[h>v_{th}]$); the third factor is reward")
    zl, zr = _zslab(ax, v, rr + 1.15, iy); flow(ax, (rr, iy), (zl, iy), color="#4a4590")
    ro_l, ro_r = block(ax, zr + 1.7, iy, 2.2, 1.2, "Plastic readout", C_LIN, sub=r"$\mathrm{raw}=Wz+b$"); flow(ax, (zr, iy), (ro_l, iy), color="#4a4590")
    _res_back(ax, v, col, iy, ro_r)
    # spike-gated eligibility tag branch + dopamine third factor
    sg = block(ax, zr + 1.7, iy - 1.9, 2.5, 0.7, r"STDP tag $c_t$", "#c98bb9", sub=r"pre-spike $s\odot z$ $\times$ post-err $\delta$", tfs=7.2, fc="#eccfe0")
    flow(ax, (zr + 1.7, iy - 0.6), (zr + 1.7, iy - 1.55), color="#4a4590", ls="dotted", mut=8)
    dop = block(ax, ro_r + 0.4, iy - 2.5, 2.0, 0.75, r"dopamine $d_t$", "#c9a227", sub=r"reward $-\!$baseline", tfs=7.4, fc="#f0e2b8")
    _teacher_band(ax, ro_l - 0.4, iy, col, r"STDP tag  $c_t=(1{-}\frac{dt}{\tau_c})c_t+\delta_t\otimes(s\odot z)$   (spike-gated pre $\times$ post-error);   third factor = dopamine $d_t$",
                  r"$\Delta W=\eta\,d_t\,c_t$   (reward-modulated spike-timing plasticity)")
    ax.text(W_CANVAS / 2, 1.05, "a spike-gated STDP eligibility + a dopamine (reward) third factor — and spike-based SynOps energy — set R-STDP apart from the error-driven rules", ha="center", fontsize=7.6, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("fixed reservoir", C_RES), ("readout", C_LIN), ("STDP tag", "#c98bb9"), ("dopamine", "#c9a227"), ("head", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig

def draw_predcode(spec, variant):
    fig, ax, v, col, iy, rr = _res_front(spec, variant,
        "hierarchical predictive coding — a latent r predicts the reservoir feature; ERROR UNITS drive inference; motor read from r",
        r"inference iterates $r\!\leftarrow\!r+\eta(\epsilon W_{pred})$ to minimise $\epsilon$")
    # representation r + error units + generative Wpred
    rr2 = block(ax, rr + 2.0, iy + 0.65, 1.7, 0.9, r"repr. $r$", C_LIN, sub=f"[{VAR[variant].get('Nrep',128) if False else 128}]", tfs=8.0)
    eu = block(ax, rr + 2.0, iy - 1.05, 2.2, 0.8, "error units", "#c98bb9", sub=r"$\epsilon=h-W_{pred}r$", tfs=7.4, fc="#eccfe0")
    flow(ax, (rr, iy), (rr + 2.0 - 0.85, iy + 0.5), color="#4a4590")
    ax.add_patch(FancyArrowPatch((rr + 2.0, iy + 0.2), (rr + 2.0, iy - 0.65), connectionstyle="arc3,rad=0.3", arrowstyle="-|>", mutation_scale=8, lw=1.0, color="#b0559a", ls="dotted", zorder=6))
    ax.add_patch(FancyArrowPatch((rr + 2.0 + 0.5, iy - 0.65), (rr + 2.0 + 0.5, iy + 0.2), connectionstyle="arc3,rad=0.3", arrowstyle="-|>", mutation_scale=8, lw=1.0, color="#b0559a", zorder=6))
    ax.text(rr + 3.3, iy - 0.25, "top-down\nprediction", ha="left", va="center", fontsize=6.4, color="#b0559a")
    ro_l, ro_r = block(ax, rr + 5.4, iy, 2.2, 1.2, "Motor readout", C_LIN, sub=r"$\mathrm{raw}=W_{mot}[r;\hat o]$"); flow(ax, (rr + 2.85, iy + 0.65), (ro_l, iy), color="#4a4590")
    _res_back(ax, v, col, iy, ro_r)
    _teacher_band(ax, ro_l - 0.6, iy, col, r"sensory error $\epsilon$ trains the generative model; motor error $\delta$ trains the readout (Hebbian on errors)",
                  r"$\Delta W_{mot}=\eta\,\langle\delta\otimes[r;\hat o]\rangle$;   inference minimises $\epsilon$ online")
    ax.text(W_CANVAS / 2, 1.05, "explicit error units + a top-down generative prediction + iterative inference make this a generative model, not a feedforward readout", ha="center", fontsize=7.6, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("fixed reservoir", C_RES), ("repr / readout", C_LIN), ("error units", "#c98bb9"), ("head", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig

def draw_hebb3(spec, variant):
    fig, ax, v, col, iy, rr = _res_front(spec, variant,
        "reservoir + a readout trained by a local Hebbian delta GATED BY A NEUROMODULATOR — pre×post-error scaled by a reward-driven dopamine signal",
        r"dopamine gate $M=\sigma(g\,(R_t-\bar R))$ scales every Hebbian update")
    zl, zr = _zslab(ax, v, rr + 1.15, iy); flow(ax, (rr, iy), (zl, iy), color="#4a4590")
    ro_l, ro_r = block(ax, zr + 1.7, iy, 2.2, 1.2, "Plastic readout", C_LIN, sub=r"$\mu=Wz+b$"); flow(ax, (zr, iy), (ro_l, iy), color="#4a4590")
    da = node(ax, ro_r + 0.9, iy + 0.9, "M", color="#c98bb9"); ax.text(ro_r + 0.9, iy + 1.45, "dopamine gate", ha="center", fontsize=6.8, color="#b0559a")
    flow(ax, (ro_r, iy), (da[0] - 0.17, da[1] - 0.1), color=C_LIN, rad=0.15, mut=8)
    _res_back(ax, v, col, iy, ro_r)
    rw = block(ax, ro_r + 0.6, iy - 2.4, 2.2, 0.8, r"reward $R_t$", "#c9a227", sub=r"$-\Vert e\Vert^2$, baseline $\bar R$", tfs=7.2, fc="#f0e2b8")
    _teacher_band(ax, ro_l - 0.6, iy, col, r"third factor = dopamine $M=\sigma(g\,(R_t-\bar R))$;   local Hebbian pre-act $z$ $\times$ post-error $e=a^*-a$",
                  r"$\Delta W=\eta\,M\,\langle e\otimes z\rangle$   (reward-gated Hebbian delta; the gate protects mastered motor memories)")
    ax.text(W_CANVAS / 2, 1.05, "a local Hebbian outer product (pre-activity × postsynaptic error) written only as much as a global reward-driven neuromodulator permits", ha="center", fontsize=7.6, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("fixed reservoir", C_RES), ("readout", C_LIN), ("reward", "#c9a227"), ("head", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig


# ==============================================================================
# Per-model diagrams for the off-policy trio and KINESIS.
# These four used to render through TWO shared functions (draw_bootstrap for all of
# SAC/FastTD3/Simba, draw_gru_apg for both BPTT-GRU and KINESIS), so five of thirteen
# models showed essentially the same picture. Each now draws the mechanism that is
# actually in its torch class and named in its paper.
# ==============================================================================
def _offpolicy_front(spec, variant, subtitle):
    fig, ax, v, col = _canvas(spec, variant, subtitle)
    ix, iy = _input(ax, v, cy=8.4)
    al, ar = block(ax, 7.0, 8.4, 3.2, 1.9, "GRUCell actor", C_GRU, sub=f"hidden {v['hidden']} - policy 12.4k params")
    flow(ax, (ix, 8.4), (al, 8.4), color=col, label=f"$\\hat o$[{v['obs']}]")
    sl, sr, _ = slab(ax, ar + 1.2, 8.4, v["raw"], C_LIN, top="raw", bot=f"[{v['raw']}]"); flow(ax, (ar, 8.4), (sl, 8.4), color=col)
    hl, hr, hpt2 = _head(ax, v, (sr + 5.0) if v["head"] == "morph" else (sr + 1.9), 8.4); flow(ax, (sr, 8.4), (hl, 8.4), color=col)
    msl, msr, _ = slab(ax, hr + 1.2, 8.4, v["out"], C_HEAD, top="musc.", bot=f"[{v['out']}]")
    flow(ax, (hpt2[0] if v["head"] == "morph" else hr, 8.4), (msl, 8.4), color=col)
    pl_, pr_ = _plant(ax, v, msr + 1.6, 8.4); flow(ax, (msr, 8.4), (pl_, 8.4), color=col)
    block(ax, 4.5, 4.6, 2.4, 1.1, "Replay buffer", GREY, sub="on-policy fill\n(o,a,r,o',d)", tfs=7.4)
    ax.text(W_CANVAS / 2, 0.95, "reward $r=-\\Vert$fingertip$-$goal$\\Vert_1$ = MINUS MotorNet's own loss, so this maximises the SAME objective the gradient rules minimise", ha="center", fontsize=7.5, color="#444", style="italic")
    return fig, ax, v, col


def draw_sac(spec, variant):
    """SAC: STOCHASTIC squashed-Gaussian actor + entropy temperature. `self.stoch=True`,
    `self.log_std`, entropy bonus `-ent*log_std.sum()` in motor_zoo.BootstrapRL."""
    fig, ax, v, col = _offpolicy_front(spec, variant, "off-policy - STOCHASTIC actor: state-dependent $\\log\\sigma$ head, entropy-regularised exploration (the SAC signature)")
    block(ax, 7.0, 6.15, 3.0, 0.8, r"$\log\sigma$ head (learned)", "#c98bb9", sub=r"$a\sim\tanh(\mu+\sigma\epsilon)$, clamped $[\log.05,\log.6]$", tfs=7.2, fc="#eccfe0")
    flow(ax, (7.0, 7.45), (7.0, 6.55), color="#c98bb9", ls="dotted", mut=8)
    block(ax, 9.6, 4.6, 3.4, 1.5, "twin critics - MLP", C_CRIT, sub=r"$Q_1,Q_2(\hat o,a)$ 256-256 + Polyak targets")
    flow(ax, (5.7, 4.6), (7.9, 4.6), color=GREY, mut=10)
    flow(ax, (9.6, 5.35), (7.4, 7.45), color=C_CRIT, ls="dotted", rad=0.2, label=r"$-Q_1$", mut=10)
    ax.text(W_CANVAS / 2 + 3, 2.5, r"actor  $L_\pi=-Q_1(\hat o,a)-\alpha\,\mathbb{H}[\pi]$        critic  $L_Q=\mathrm{MSE}(Q_i,\ r+\gamma(1-d)\min_i Q_i')$", ha="center", fontsize=8.2, color=col, fontweight="bold")
    ax.text(W_CANVAS / 2 + 3, 1.8, "entropy term is what distinguishes SAC from the deterministic TD3 actor below", ha="center", fontsize=7.4, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("GRU actor", C_GRU), ("log-sigma", "#c98bb9"), ("critic", C_CRIT), ("head", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig


def draw_fasttd3(spec, variant):
    """FastTD3: DETERMINISTIC actor + target-policy smoothing + upd=4 update ratio."""
    fig, ax, v, col = _offpolicy_front(spec, variant, "off-policy - DETERMINISTIC actor, target-policy SMOOTHING, high update-to-data ratio (upd=4)")
    block(ax, 7.0, 6.15, 3.4, 0.8, "fixed Gaussian exploration", "#b0803a", sub=r"$a=\pi(\hat o)+\sigma\epsilon$, $\sigma$ CONSTANT (not learned)", tfs=7.2, fc="#f0dfc0")
    flow(ax, (7.0, 7.45), (7.0, 6.55), color="#b0803a", ls="dotted", mut=8)
    block(ax, 9.6, 4.6, 3.4, 1.5, "twin critics - MLP", C_CRIT, sub=r"$Q_1,Q_2$ 256-256 + Polyak targets")
    tp = block(ax, 14.3, 4.6, 3.2, 1.1, "target smoothing", "#7f8fa6", sub=r"$a'=\pi'(o')+\mathrm{clip}(\sigma\epsilon,\pm c)$", tfs=7.2, fc="#dde3ea")
    flow(ax, (5.7, 4.6), (7.9, 4.6), color=GREY, mut=10); flow(ax, (12.7, 4.6), (11.3, 4.6), color="#7f8fa6", ls="dotted", mut=10)
    flow(ax, (9.6, 5.35), (7.4, 7.45), color=C_CRIT, ls="dotted", rad=0.2, label=r"$-Q_1$", mut=10)
    ax.text(W_CANVAS / 2 + 3, 2.5, r"actor  $L_\pi=-Q_1(\hat o,\pi(\hat o))$   (no entropy term)        4 gradient updates per environment step", ha="center", fontsize=8.2, color=col, fontweight="bold")
    ax.text(W_CANVAS / 2 + 3, 1.8, "clipped noise on the TARGET action is the TD3 fix for critic overestimation - absent in SAC", ha="center", fontsize=7.4, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("GRU actor", C_GRU), ("explore", "#b0803a"), ("critic", C_CRIT), ("smoothing", "#7f8fa6"), ("plant", C_PLANT)])
    fig.tight_layout(); return fig


def draw_simba(spec, variant):
    """SimbaV2: TD3 whose critic trunk is RESIDUAL + LayerNorm (_SimbaBlock: x + f(LN(x)))."""
    fig, ax, v, col = _offpolicy_front(spec, variant, "off-policy - TD3 actor with a RESIDUAL, LAYER-NORMED critic trunk (the Simba architecture)")
    block(ax, 7.0, 6.15, 3.4, 0.8, "fixed Gaussian exploration", "#b0803a", sub=r"deterministic actor $+\ \sigma\epsilon$", tfs=7.2, fc="#f0dfc0")
    flow(ax, (7.0, 7.45), (7.0, 6.55), color="#b0803a", ls="dotted", mut=8)
    X, Y, Wd, Hd = 8.1, 3.4, 6.2, 2.7
    ax.add_patch(FancyBboxPatch((X, Y), Wd, Hd, boxstyle="round,pad=0.03,rounding_size=0.12", fc="#eaf2f6", ec=C_CRIT, lw=1.7, zorder=2))
    ax.text(X + 0.2, Y + Hd - 0.22, "twin critics - SimbaNet trunk", ha="left", va="top", fontsize=8.6, fontweight="bold", color=C_CRIT, zorder=5)
    block(ax, X + 1.35, Y + 1.15, 1.7, 0.62, "Linear in", C_CRIT, tfs=7.2, fc="#cfe3ec")
    for k in (0, 1):
        bx = X + 3.15 + k * 1.75
        block(ax, bx, Y + 1.15, 1.55, 0.92, f"block {k+1}", C_CRIT, sub=r"$x{+}f(\mathrm{LN}(x))$", tfs=7.0, fc="#cfe3ec")
        ax.add_patch(FancyArrowPatch((bx - 0.6, Y + 1.75), (bx + 0.6, Y + 1.75), connectionstyle="arc3,rad=-0.55", arrowstyle="-|>", mutation_scale=7, lw=0.9, color=C_CRIT, ls="dotted", zorder=6))
    ax.text(X + 4.0, Y + 2.05, "residual skip", ha="center", fontsize=6.4, color=C_CRIT, style="italic")
    flow(ax, (5.7, 4.6), (X + 0.5, 4.6), color=GREY, mut=10)
    flow(ax, (X + 1.0, Y + Hd), (7.4, 7.45), color=C_CRIT, ls="dotted", rad=0.2, label=r"$-Q_1$", mut=10)
    ax.text(W_CANVAS / 2 + 3, 2.4, r"LayerNorm + residual blocks let the critic scale in depth without the plasticity loss plain MLP critics suffer", ha="center", fontsize=7.8, color=col, fontweight="bold")
    ax.text(W_CANVAS / 2 + 3, 1.75, "identical actor and TD3 update to FastTD3 - ONLY the critic architecture differs", ha="center", fontsize=7.4, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("GRU actor", C_GRU), ("explore", "#b0803a"), ("SimbaNet critic", C_CRIT), ("head", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig


def draw_kinesis(spec, variant):
    """KINESIS: GRU -> (endpoint force, co-contraction) -> muscle pulls via the BODY's
    geometry. Its head reads muscle path anchors from the effector, which is the
    morphological-computation claim; BPTT-GRU has no such stage."""
    fig, ax, v, col = _canvas(spec, variant, "analytic policy gradient - the BODY does the muscle coordinate transform (morphological computation)")
    ix, iy = _input(ax, v, cy=8.0)
    al, ar = block(ax, 6.6, 8.0, 3.0, 1.8, "GRU", C_GRU, sub=f"hidden {v['hidden']}")
    flow(ax, (ix, 8.0), (al, 8.0), color=col, label=f"$\\hat o$[{v['obs']}]")
    sl, sr, _ = slab(ax, ar + 1.1, 8.0, 3, C_LIN, top="raw", bot="[3]"); flow(ax, (ar, 8.0), (sl, 8.0), color=col)
    ax.text(sr + 0.1, 9.05, r"$f=\tanh(\mathrm{raw}_{0:2})\,f_{scale}$", ha="left", fontsize=7.4, color=col)
    ax.text(sr + 0.1, 6.95, r"$c=\sigma(\mathrm{raw}_2)$  co-contraction", ha="left", fontsize=7.4, color=col)
    X, Y, Wd, Hd = sr + 1.5, 6.2, 5.4, 3.6
    ax.add_patch(FancyBboxPatch((X, Y), Wd, Hd, boxstyle="round,pad=0.03,rounding_size=0.12", fc="#e8f5f2", ec=TEAL_D, lw=1.8, ls=(0, (5, 3)), zorder=2))
    ax.text(X + 0.2, Y + Hd - 0.22, "Morphological decode  (body geometry, FIXED)", ha="left", va="top", fontsize=8.4, fontweight="bold", color=TEAL_D, zorder=5)
    block(ax, X + 1.5, Y + 1.9, 2.2, 0.8, "muscle anchors", TEAL_D, sub=r"$A$ from effector path", fixed=True, tfs=7.2, fc="#cfe9e3")
    block(ax, X + 1.5, Y + 0.75, 2.2, 0.72, "pull dirs", TEAL_D, sub=r"$d_m=(A_m-P)/\ell_m$", tfs=7.2, fc="#cfe9e3")
    block(ax, X + 4.1, Y + 1.3, 1.9, 1.1, "project", TEAL_D, sub=r"$a_m=\frac{[d_m\cdot f]_+}{F_{max}}+c$", tfs=7.0, fc="#cfe9e3")
    flow(ax, (sr, 8.0), (X + 0.55, Y + 1.9), color=col, rad=-0.1, label="$f$, $c$")
    msl, msr, _ = slab(ax, X + Wd + 1.3, 8.0, v["out"], C_HEAD, top="muscles", bot=f"[{v['out']}]")
    flow(ax, (X + Wd, Y + 1.3), (msl, 8.0), color=TEAL_D, rad=0.1)
    pl_, pr_ = _plant(ax, v, msr + 1.6, 8.0); flow(ax, (msr, 8.0), (pl_, 8.0), color=col)
    _backward(ax, col, "backward: analytic policy gradient through the plant (plausibility is the BODY, not the update rule)")
    ax.text(W_CANVAS / 2, 1.0, "the 3-D force command is decoded to 4 muscles by the arm's own geometry - BPTT-GRU emits the 4 excitations directly and has no such stage", ha="center", fontsize=7.6, color="#444", style="italic")
    _legend(ax, [("input", C_IN), ("GRU", C_GRU), ("linear", C_LIN), ("morphology (fixed)", TEAL_D), ("muscles", C_HEAD), ("plant", C_PLANT)])
    fig.tight_layout(); return fig


RENDER = {"bptt_gru": lambda s, v: draw_gru_apg(s, v),
          "kinesis": draw_kinesis,
          "shac": draw_shac, "sac": draw_sac, "fasttd3": draw_fasttd3, "simbav2": draw_simba,
          "eprop": draw_eprop, "rtrrl": draw_rtrrl, "btsp": draw_btsp,
          "rstdp": draw_rstdp, "predcode": draw_predcode, "hebb3": draw_hebb3, "dendritron": draw_dendritron}
SPECS = {
    "bptt_gru": dict(name="BPTT-GRU", family="global-gradient", badge="RNN (GRU) · analytic policy gradient", cite="Codol+ 2024 MotorNet; Werbos 1990 (BPTT)"),
    "shac": dict(name="SHAC", family="global-gradient", badge="RNN · TRUNCATED 16-step analytic gradient", cite="Xu+ 2022"),
    "sac": dict(name="SAC (demo-boot)", family="global-gradient", flavor="sac", badge="off-policy · entropy · twin-Q", cite="Haarnoja+ 2018; Ball+ 2023 (RLPD)"),
    "fasttd3": dict(name="FastTD3 (demo-boot)", family="global-gradient", flavor="td3", badge="off-policy · TD3 · twin-Q", cite="Fujimoto+ 2018; +2021 (TD3+BC)"),
    "simbav2": dict(name="Simba (demo-boot)", family="global-gradient", flavor="simba", badge="off-policy · residual critic", cite="Lee+ 2024 Simba; TD3+BC"),
    "eprop": dict(name="e-prop", family="local-plausible", badge="ALIF reservoir · eligibility trace + learning signal", cite="Bellec+ 2020 (Nat. Commun.)"),
    "rtrrl": dict(name="RTRRL / RFLO", family="local-plausible", badge="reservoir · real-time local · random feedback", cite="Murray 2019 (eLife)"),
    "btsp": dict(name="BTSP", family="local-plausible", badge="reservoir · dendritic plateau · one-shot", cite="Bittner+ 2017 (Science)"),
    "rstdp": dict(name="R-STDP", family="local-plausible", badge="spiking reservoir · STDP tag · dopamine", cite="Izhikevich 2007"),
    "predcode": dict(name="Predictive coding", family="local-plausible", badge="error units · generative model · inference", cite="Rao & Ballard 1999"),
    "hebb3": dict(name="3-factor Hebb", family="local-plausible", badge="reservoir · reward-gated Hebbian · dopamine", cite="Kuśmierz+ 2017"),
    "dendritron": dict(name="Dendritron", family="local-plausible", badge="reservoir + frozen experts + router", cite="Hu+ 2022 (LoRA)"),
    "kinesis": dict(name="KINESIS", family="morphological", badge="RNN (GRU) + fixed morphology · APG", cite="Simos+ 2025 (ICRA); Hogan 1984"),
}
ORDER = ["bptt_gru", "shac", "sac", "fasttd3", "simbav2", "eprop", "rtrrl", "btsp", "kinesis", "rstdp", "predcode", "hebb3", "dendritron"]


def draw_all(variant, order=ORDER): return [RENDER[k](SPECS[k], variant) for k in order]


if __name__ == "__main__":
    for k in ["bptt_gru", "kinesis", "sac", "eprop", "dendritron", "shac"]:
        for var in (["arm", "2d"] if k in ("bptt_gru", "kinesis") else ["arm"]):
            fig = RENDER[k](SPECS[k], var); fig.savefig(f"/tmp/det_{k}_{var}.png", dpi=95, bbox_inches="tight"); plt.close(fig)
    print("saved detailed samples")
