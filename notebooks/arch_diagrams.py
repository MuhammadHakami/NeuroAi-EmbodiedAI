"""arch_diagrams.py -- one block-diagram per learner: input -> backbone -> head -> output,
plus the backward feedback loop labelled with the learning rule and loss. Shared by
4-train-net.ipynb (2-D point-mass, force head) and 4-monkey-net.ipynb (6-muscle arm); the
input/output dims are passed in, the backbone/head/loss identity is the same in both.

Box style encodes what is TRAINED: solid = plastic (learned), dashed = FIXED (random
reservoir / fixed synergy). Colour encodes family: orange = global-gradient (backprop / TD),
teal = local-plausible (local three-factor rules, no BPTT).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FAM_COLOR = {"global-gradient": "#e76f51", "local-plausible": "#2a9d8f"}

# order: 5 global-gradient, then 8 local-plausible. backbone/head/feedback/loss are the
# method identity (same across both notebooks); input & output boxes are filled per-notebook.
SPECS = [
    dict(key="bptt_gru", name="BPTT-GRU", family="global-gradient", atype="RNN · GRU",
         backbone="GRU\nh=96", bb_fixed=False, head="Linear\n→ muscle head",
         feedback="BPTT through the differentiable plant  (analytic policy gradient)",
         loss=r"$L=-\sum_t r_t,\ \ r=-\,dist/d_{max}-effort$"),
    dict(key="shac", name="SHAC", family="global-gradient", atype="RNN actor + MLP critic",
         backbone="GRU actor\nh=96", bb_fixed=False, head="Linear\n→ muscle head",
         feedback="short-horizon BPTT (H=50) + bootstrapped value critic",
         loss=r"actor $-(\sum\gamma^k r+\gamma^H V)$;  critic MSE$(V,\mathrm{GAE})$"),
    dict(key="sac", name="SAC (demo-boot)", family="global-gradient", atype="RNN actor + twin-Q MLP",
         backbone="GRUCell actor\nh=96  (warm-start ← teacher)", bb_fixed=False, head="Linear\n→ muscle head",
         feedback="off-policy TD (replay) + behaviour-cloning anchor to teacher",
         loss=r"actor $\beta\Vert a-a_{tch}\Vert^2+\alpha\log\pi-Q$;  critic TD-MSE"),
    dict(key="fasttd3", name="FastTD3 (demo-boot)", family="global-gradient", atype="RNN actor + twin-Q MLP · TD3",
         backbone="GRUCell actor\nh=96", bb_fixed=False, head="Linear\n→ muscle head",
         feedback="off-policy TD3 (twin-Q, target-policy smoothing) + BC anchor",
         loss=r"actor $\beta\Vert a-a_{tch}\Vert^2-Q$;  critic TD-MSE"),
    dict(key="simbav2", name="Simba (demo-boot)", family="global-gradient", atype="RNN actor + residual-MLP critic",
         backbone="GRUCell actor\nh=96", bb_fixed=False, head="Linear\n→ muscle head",
         feedback="off-policy TD3+BC, LayerNorm-residual (SimbaNet) critic",
         loss=r"actor $\beta\Vert a-a_{tch}\Vert^2-Q$;  critic TD-MSE"),
    dict(key="eprop", name="e-prop", family="local-plausible", atype="reservoir RNN + local readout",
         backbone="FIXED reservoir\n2048 (echo-state)", bb_fixed=True, head="linear readout\n(plastic)",
         feedback="local 3-factor rule, low-pass eligibility trace  (no BPTT, no reward)",
         loss=r"$\Delta W=\eta\,(err\otimes \bar e)-\lambda W,\ err=a_{tch}-out$"),
    dict(key="rtrrl", name="RTRRL / RFLO", family="local-plausible", atype="reservoir RNN + local readout",
         backbone="FIXED reservoir\n2048", bb_fixed=True, head="linear readout\n(plastic)",
         feedback="real-time recurrent learning, low-pass eligibility trace",
         loss=r"$\Delta W=\eta\,(err\otimes \bar e)-\lambda W$"),
    dict(key="btsp", name="BTSP", family="local-plausible", atype="reservoir RNN + local readout",
         backbone="FIXED reservoir\n2048", bb_fixed=True, head="linear readout\n(plastic)",
         feedback="behavioural-timescale (dendritic plateau) eligibility",
         loss=r"$\Delta W=\eta\,(err\otimes \bar e_{plateau})-\lambda W$"),
    dict(key="kinesis", name="KINESIS", family="local-plausible", atype="RNN (GRU) + fixed morphology",
         backbone="GRU intent\nh=96", bb_fixed=False, head="FIXED synergy /\nbody map", head_fixed=True,
         feedback="analytic policy gradient  (plausibility = morphology, not the update)",
         loss=r"$L=-\sum_t r_t$"),
    dict(key="rstdp", name="R-STDP", family="local-plausible", atype="reservoir RNN + local readout",
         backbone="FIXED reservoir\n2048", bb_fixed=True, head="linear readout\n(plastic)",
         feedback="reward-modulated STDP, instantaneous 3-factor (dopamine gate)",
         loss=r"$\Delta W=\eta\,(err\otimes z)-\lambda W$"),
    dict(key="predcode", name="Predictive coding", family="local-plausible", atype="reservoir RNN + local readout",
         backbone="FIXED reservoir\n2048", bb_fixed=True, head="linear readout\n(plastic)",
         feedback="prediction-error minimisation (active inference), instantaneous",
         loss=r"$\Delta W\propto err\otimes z,\ err=target-out$"),
    dict(key="hebb3", name="3-factor Hebb", family="local-plausible", atype="reservoir RNN + local readout",
         backbone="FIXED reservoir\n2048", bb_fixed=True, head="linear readout\n(plastic)",
         feedback="neuromodulated Hebbian: pre × post × global 3rd factor",
         loss=r"$\Delta W=\eta\,(err\otimes z)$"),
    dict(key="dendritron", name="Dendritron", family="local-plausible", atype="reservoir + frozen experts + router",
         backbone="FIXED reservoir\n2048", bb_fixed=True, head="frozen readout +\nLoRA packs + router",
         feedback="local 3-factor on per-context packs (base frozen → no forgetting)",
         loss=r"$dB=err\otimes(Az),\ \ dA=(B^{\top}err)\otimes z$"),
]


def _box(ax, cx, w, text, color, fixed, y=0.60, h=0.34, fs=8.5):
    ls = (0, (4, 3)) if fixed else "solid"
    ax.add_patch(FancyBboxPatch((cx - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.6, linestyle=ls, edgecolor=color, facecolor=color + "1f"))
    ax.text(cx, y, text, ha="center", va="center", fontsize=fs, color="#222")


def _arrow(ax, x0, x1, y=0.60, color="#555"):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.4, color=color))


def draw_model(ax, spec, obs_dim, n_out, out_label=None):
    col = FAM_COLOR[spec["family"]]
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
    # forward pipeline: input -> backbone -> head -> output
    centers = [1.15, 3.5, 5.85, 8.15]; ws = [1.7, 2.15, 1.95, 1.7]
    BY, BH = 0.75, 0.26
    _box(ax, centers[0], ws[0], f"input\nobs {obs_dim}-D", "#264653", False, y=BY, h=BH)
    _box(ax, centers[1], ws[1], spec["backbone"], col, spec["bb_fixed"], y=BY, h=BH)
    _box(ax, centers[2], ws[2], spec["head"], col, spec.get("head_fixed", False), y=BY, h=BH)
    _box(ax, centers[3], ws[3], out_label or f"output\n{n_out} muscle acts", "#264653", False, y=BY, h=BH)
    for i in range(3):
        _arrow(ax, centers[i] + ws[i] / 2, centers[i + 1] - ws[i + 1] / 2, y=BY)
    # backward feedback loop: gentle arc BELOW the pipeline, output -> trainable modules.
    # rad is small because the x-span >> y-span (a large rad would bulge off-canvas).
    y0 = BY - BH / 2
    fb = FancyArrowPatch((centers[3], y0), (centers[1], y0), connectionstyle="arc3,rad=-0.035",
                         arrowstyle="-|>", mutation_scale=13, lw=1.6, color=col, linestyle=(0, (5, 3)))
    ax.add_patch(fb)
    ax.text(5.0, 0.30, "◀ feedback: " + spec["feedback"], ha="center", va="center",
            fontsize=7.7, color=col, style="italic")
    ax.text(5.0, 0.10, spec["loss"], ha="center", va="center", fontsize=8.4, color="#333")
    # title + architecture-type badge
    ax.text(0.05, 0.98, spec["name"], ha="left", va="top", fontsize=10.5, fontweight="bold", color=col)
    ax.text(9.95, 0.98, spec["atype"], ha="right", va="top", fontsize=8.2, color="#555",
            bbox=dict(boxstyle="round,pad=0.25", fc=col + "14", ec=col, lw=0.8))


def draw_zoo(obs_dim, n_out, out_label=None, title="Model architectures", specs=SPECS):
    n = len(specs)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 1.55 * n))
    for ax, sp in zip(axes, specs):
        draw_model(ax, sp, obs_dim, n_out, out_label)
    fig.suptitle(title + "  —  input · backbone · head · output · backward feedback loop (loss)",
                 fontsize=12.5, fontweight="bold", y=0.998)
    fig.text(0.5, 0.986, "solid box = trained   ·   dashed box = FIXED (random reservoir / synergy)      |      "
             "orange = global-gradient (BPTT / TD)   ·   teal = local-plausible (local 3-factor, no BPTT)",
             ha="center", va="top", fontsize=8.5, color="#666")
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    return fig


if __name__ == "__main__":
    fig = draw_zoo(obs_dim=16, n_out=6, title="4-monkey-net (arm)")
    fig.savefig("/tmp/arch_test.png", dpi=100, bbox_inches="tight")
    print("saved /tmp/arch_test.png with", len(SPECS), "model diagrams")
