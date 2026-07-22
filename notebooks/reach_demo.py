"""Qualitative evaluation of any trained controller on the free-reach task.

The notebook's last cell used to `from motor_zoo import (18 names...)`, so a single missing or
drifted name (or a stale kernel) killed the whole cell -- which is exactly the ImportError this
module removes. Everything is imported INTERNALLY here, the model class is inferred from the
checkpoint file name, and the cell becomes three robust lines.

    run_model("save/models/kinesis.pt")                        # 4 reaches, 1.0 kg
    run_model("save/models/shac.pt", n_runs=6, ball_weight=2.1)          # one ball weight
    run_model("save/models/eprop.pt", ball_weight=[0.5, 1.2, 2.1, 2.5])  # a list of weights
    run_model("save/models/btsp.pt", render=False)                       # metrics only
    run_all()                                                            # every checkpoint

Ball weight is meaningful HERE (2-D point mass); the monkey-net demo uses maze_demo instead.
"""
import os
import glob

_ZOO = None


def _zoo():
    """tag -> class, built once, tolerant of import order."""
    global _ZOO
    if _ZOO is None:
        import motor_zoo as mz
        import plausible_learners as pl
        _ZOO = {"motornet_ref": mz.MotorNetRef, "bptt_gru": mz.BPTTGRU, "shac": mz.SHAC,
                "sac": mz.SAC, "fasttd3": mz.FastTD3, "simbav2": mz.SimbaV2,
                "eprop": pl.EProp, "rtrrl": pl.RTRRL, "btsp": pl.BTSP, "kinesis": mz.Kinesis,
                "rstdp": pl.RSTDP, "predcode": pl.PredictiveCoding, "hebb3": pl.Hebb3,
                "dendritron": mz.Dendritron}
    return _ZOO


def _env_with_mass(mass):
    import motornet as mn
    import motor_zoo as mz
    return mz.env_to(mz.ReachEnv(effector=mn.effector.ReluPointMass24(mass=float(mass)),
                                 max_ep_duration=1.0), mz.DEVICE)


def load_any(weights_path, env=None, model=None, **ctor_kwargs):
    """Rebuild a learner from a checkpoint, inferring the class from the file name."""
    import torch as th
    import motor_zoo as mz
    if env is None:
        env = mz.make_mass_env(mz.DEVICE, mz.TRAIN_MASSES)
    tag = os.path.splitext(os.path.basename(weights_path))[0]
    cls = model or _zoo().get(tag)
    if cls is None:
        raise KeyError(f"cannot infer a model from {tag!r}; pass model=<Class>. Known: {sorted(_zoo())}")
    m = cls(env, **ctor_kwargs)
    sd = th.load(weights_path, map_location=mz.DEVICE)
    m.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd, strict=False)
    return m


def render_runs(model, n_runs=4, ball_weight=1.0, out=None, fps=20, title=None, seed=0):
    """Roll a model out for n_runs free reaches and animate the point mass + its 4 muscles."""
    import numpy as np
    import torch as th
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from matplotlib import cm
    out = out or os.path.join("save", "episode.mp4")
    weights = list(ball_weight) if isinstance(ball_weight, (list, tuple, np.ndarray)) else [ball_weight] * n_runs
    assert len(weights) == n_runs, f"ball_weight must be a scalar or a list of length n_runs={n_runs}, got {len(weights)}"

    runs = []
    for i, w in enumerate(weights):
        env = _env_with_mass(w); n = int(env.max_ep_duration / env.dt)
        th.manual_seed(seed + i)
        obs, info = env.reset(options={"batch_size": 1}); h = model.init_state(1)
        TIP, GOAL, ACT = [], [], []
        with th.no_grad():
            for _t in range(n):
                a, h = model.act(obs, h)
                TIP.append(np.asarray(obs[0, 2:4].detach().cpu())); GOAL.append(np.asarray(obs[0, 0:2].detach().cpu()))
                ACT.append(np.asarray(a[0].detach().cpu())); obs, r, term, trunc, info = env.step(a)
        TIP, GOAL = np.array(TIP), np.array(GOAL)
        runs.append(dict(w=w, tip=TIP, goal=GOAL, act=np.array(ACT), err=np.linalg.norm(TIP - GOAL, axis=1) * 100))
    _pc = env.effector._path_coordinates[0, :, 0::2].T
    anchors = np.asarray(_pc.detach().cpu() if hasattr(_pc, "detach") else _pc)
    n = len(runs[0]["tip"]); vary = len(set(weights)) > 1
    col = [cm.tab10(i % 10) for i in range(n_runs)]
    HAZE = min(0.9, 0.55 + 0.12 * n_runs)

    fig, ax = plt.subplots(figsize=(6.8, 6.4)); lim = float(np.abs(anchors).max()) + 0.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    ax.scatter(anchors[:, 0], anchors[:, 1], s=110, facecolors="none", edgecolors="#2F3E46", lw=1.4, zorder=2)
    for A in anchors:
        ax.annotate("", xy=(A[0]*.92, A[1]*.92), xytext=(0, 0), zorder=1,
                    arrowprops=dict(arrowstyle="-", color="#B0B7C3", lw=0.6, alpha=0.4))
    muscles, trails, dots, halos = [], [], [], []
    for i, r in enumerate(runs):
        ax.plot([r["goal"][-1, 0]], [r["goal"][-1, 1]], marker="*", ms=20, c=col[i], ls="", alpha=0.9, zorder=6)
        muscles.append([ax.plot([], [], "-", c=col[i], solid_capstyle="round", zorder=4)[0] for _ in range(4)])
        halos.append(ax.plot([], [], "o", c=col[i], mfc="none", zorder=4)[0])
        trails.append(ax.plot([], [], "-", c=col[i], lw=1.6, alpha=0.35 * HAZE, zorder=3)[0])
        lab = f"{r['w']:g} kg" if vary else f"reach {i+1}"
        dots.append(ax.plot([], [], "o", ms=12, c=col[i], alpha=0.95, zorder=7, label=lab)[0])
    txt = ax.text(-lim + 0.1, lim - 0.28, "", fontsize=11, fontweight="bold", color="#1F2933")
    ax.set_title(f"{title or getattr(model, 'name', 'controller')} — {n_runs} reach{'es' if n_runs>1 else ''}"
                 "\nfingertip + 4 muscles (thickness = activation); ★ = each episode's target",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower right", title=("ball weight" if vary else "episode"), fontsize=8, title_fontsize=8, framealpha=.9)

    def update(f):
        arts = [txt]
        for i, r in enumerate(runs):
            P = r["tip"][f]; a4 = r["act"][f]
            for m in range(4):
                muscles[i][m].set_data([anchors[m, 0], P[0]], [anchors[m, 1], P[1]])
                muscles[i][m].set_alpha((0.10 + 0.75 * float(a4[m])) * HAZE)
                muscles[i][m].set_linewidth(0.8 + 6 * float(a4[m]))
            cc = float(np.minimum(a4[0], a4[3]) + np.minimum(a4[1], a4[2]))
            halos[i].set_data([P[0]], [P[1]]); halos[i].set_markersize(14 + 34 * cc); halos[i].set_alpha(0.25 * HAZE)
            trails[i].set_data(r["tip"][:f + 1, 0], r["tip"][:f + 1, 1]); dots[i].set_data([P[0]], [P[1]])
            arts += muscles[i] + [halos[i], trails[i], dots[i]]
        txt.set_text(f"t = {f * 10:3d} ms   mean error = {np.mean([r['err'][f] for r in runs]):.1f} cm")
        return arts

    ani = FuncAnimation(fig, update, frames=n, interval=1000.0 / fps, blit=False)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    try:
        ani.save(out, writer=FFMpegWriter(fps=fps, bitrate=2600))
    except Exception:
        out = os.path.splitext(out)[0] + ".gif"; ani.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {out}  |  {n_runs} episode(s): " + ", ".join(f"{r['w']:g}kg->{r['err'][-1]:.1f}cm" for r in runs))
    return out


def _display(path):
    """Embed the result inline so it animates and PERSISTS in the saved notebook."""
    import base64
    from IPython.display import Video, HTML, display
    if path.endswith(".mp4"):
        display(Video(path, embed=True))
    else:
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        display(HTML(f'<img src="data:image/gif;base64,{b64}" style="max-width:100%"/>'))


def run_model(weights_path, n_runs=4, ball_weight=1.0, model=None, render=True, **ctor_kwargs):
    """Load `weights_path`, score it on the held-out ball weights, and (optionally) animate it."""
    import numpy as np
    import motor_zoo as mz
    m = load_any(weights_path, model=model, **ctor_kwargs)
    mt = mz.eval_metrics(mz.make_mass_env(mz.DEVICE, mz.VAL_MASSES), m)
    pol, aux = mz.count_params(m)
    tag = os.path.splitext(os.path.basename(weights_path))[0]
    print(f"{getattr(m,'name',tag)}\n"
          f"  weights          {weights_path}\n"
          f"  held-out error   {mt['err_cm']:.2f} cm   (balls {list(mz.VAL_MASSES)} kg, never trained on)\n"
          f"  completion       {mt['completion']:.1f}% @5cm   {mt['completion2']:.1f}% @2cm\n"
          f"  params           {pol:,} policy  +  {aux:,} auxiliary\n"
          f"  updates/episode  {mz.updates_per_episode(m, mz.make_mass_env(mz.DEVICE, mz.TRAIN_MASSES)):.4g}")
    out = dict(tag=tag, name=getattr(m, "name", tag), err_cm=mt["err_cm"], completion=mt["completion"],
               params_policy=pol, model=m)
    if render:
        w = ball_weight if isinstance(ball_weight, (list, tuple)) else [ball_weight] * n_runs
        p = render_runs(m, n_runs=n_runs, ball_weight=list(w), out=os.path.join("save", f"reach_{tag}.mp4"))
        out["video"] = p; _display(p)
    return out


def run_all(n_runs=4, ball_weight=(0.5, 1.2, 2.1, 2.5), model_dir=None, render=False):
    """Score every checkpoint in the model dir into one table."""
    import motor_zoo as mz
    model_dir = model_dir or mz.MODEL_DIR
    rows = []
    for p in sorted(glob.glob(os.path.join(model_dir, "*.pt"))):
        try:
            rows.append(run_model(p, n_runs=n_runs, ball_weight=list(ball_weight), render=render))
        except Exception as e:
            print(f"{os.path.basename(p):<16} FAILED: {type(e).__name__}: {e}")
    rows.sort(key=lambda r: -r["completion"])
    print(f"\n{'model':<34}{'err cm':>8}{'compl%':>8}{'params':>10}")
    for r in rows:
        print(f"{r['name'][:33]:<34}{r['err_cm']:>8.2f}{r['completion']:>8.1f}{r['params_policy']:>10,}")
    return rows


def demo():
    """Self-check: class inference works without loading any weights."""
    z = _zoo()
    assert z["shac"].__name__ == "SHAC" and z["eprop"].__name__ == "EProp"
    assert len(z) == 14
    print(f"reach_demo OK: {len(z)} models resolvable by file name")


if __name__ == "__main__":
    demo()
