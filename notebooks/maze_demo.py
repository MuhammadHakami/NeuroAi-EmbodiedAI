"""Qualitative side-by-side: how a monkey, a human, and our AI move the hand to a target.

Builds ONE animated grid — rows are reach directions, columns are the four datasets plus our
SHAC model — so the reader can watch, at a glance, each nervous system (and the network) solve
the same movement.

HONESTY ABOUT WHAT EACH DATASET ACTUALLY RECORDED
-------------------------------------------------
The five columns do NOT all have the same behaviour, and the plot never pretends they do:

  m1  (MC_Maze, monkey M1) : a REAL barrier maze with a REAL, curved hand trajectory. The only
                             dataset that ever solved mazes -- barriers are drawn.
  s1  (Area2_Bump, monkey S1) : REAL centre-out hand trajectory (no barriers -- this task had none).
  human_ic / human_meg      : NEURAL-ONLY datasets. No hand trajectory was recorded, so the panel
                             shows a schematic straight reach to the target for that direction and
                             says so ("direction only — no kinematics"). Nothing is fabricated as
                             if it were measured.
  SHAC                      : our trained model's REAL rollout on the matched maze (MotorNet). It
                             was trained on free reaches, so it heads straight for the goal; it is
                             labelled zero-shot on the maze.

The one axis all five genuinely share is the reach DIRECTION, so that is what a row holds fixed.
"""
import os
import numpy as np

CENTRE = np.zeros(2)


def _ang(v):
    return np.degrees(np.arctan2(v[1], v[0])) % 360


def _nearest_dir(theta, dirs):
    d = np.asarray(dirs)
    return int(np.argmin(np.abs(((d - theta + 180) % 360) - 180)))


def pick_maze_rows(cfg, n=4):
    """Choose n maze conditions spread over reach direction, preferring ones WITH barriers so
    the monkey's curved solution is visible."""
    tg = cfg["targets"]
    ang = np.array([_ang(t) for t in tg])
    nb = cfg["n_barriers"]
    order = np.lexsort((-nb, ang))            # by angle, barriered first within a bin
    picks, used = [], []
    for i in order:
        if all(abs(((ang[i] - a + 180) % 360) - 180) > 30 for a in used):
            picks.append(int(i)); used.append(ang[i])
        if len(picks) == n:
            break
    while len(picks) < n:                     # fall back if too few distinct angles
        picks.append(int(order[len(picks)]))
    return picks


def monkey_maze_path(ds, ti, cond_key, max_trials=8):
    """Mean monkey hand path (T,2) for one (maze_id, version), in metres, centred on movement start."""
    mz_id, ver = cond_key
    g = ti[(ti["maze_id"] == mz_id) & (ti["trial_version"] == ver)]
    hp = ds.data["hand_pos"]
    segs = []
    import pandas as _pd
    for _, r in g.iterrows():
        t0 = r["move_onset_time"]
        if _pd.isna(t0):
            continue
        s = hp.loc[t0:t0 + _pd.Timedelta("450ms")].values
        if len(s) > 5 and np.isfinite(s).all():
            segs.append(s[:: max(1, len(s) // 40)][:40])
        if len(segs) >= max_trials:
            break
    if not segs:
        return None
    L = min(len(s) for s in segs)
    P = np.stack([s[:L] for s in segs]).mean(0)
    return (P - P[0]) * 1e-3                  # mm -> m, start at origin


def s1_dir_path(ds, ti, theta, max_trials=8):
    """Mean monkey-S1 hand path for the centre-out direction nearest theta (metres, origin-start)."""
    dirs = sorted(ti["target_dir"].dropna().unique())
    tgt = dirs[_nearest_dir(theta, dirs)]
    g = ti[ti["target_dir"] == tgt]
    hp = ds.data["hand_pos"]
    segs = []
    import pandas as _pd
    for _, r in g.iterrows():
        t0 = r["move_onset_time"]
        if _pd.isna(t0):
            continue
        s = hp.loc[t0:t0 + _pd.Timedelta("400ms")].values
        if len(s) > 5 and np.isfinite(s).all():
            segs.append(s[:: max(1, len(s) // 40)][:40])
        if len(segs) >= max_trials:
            break
    if not segs:
        return None
    L = min(len(s) for s in segs)
    return (np.stack([s[:L] for s in segs]).mean(0) - segs[0][0])


def schematic_reach(theta, reach=0.12, T=40):
    """A straight centre->target reach for a direction-only (neural) dataset. Clearly a schematic."""
    end = reach * np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    s = (1 - np.cos(np.linspace(0, np.pi, T))) / 2      # smooth minimum-jerk-ish ramp
    return CENTRE[None] + s[:, None] * end[None]


def fit_reach(path, reach=0.12):
    """Rescale a real trajectory to a fixed display reach, preserving its SHAPE and DIRECTION.

    The four datasets record hand position in different (and here ambiguous) units, and the
    non-maze reaches do not need to sit in the maze's metric frame -- their job in this panel is
    to show the shape and direction of the movement. So we normalise the extent and keep the
    curve. Returns None untouched (missing data)."""
    if path is None:
        return None
    p = path - path[0]
    m = np.abs(p).max()
    return p * (reach / m) if m > 1e-9 else p


def shac_maze_paths(shac, env, cond_indices, to_maze_coords=False):
    """Real SHAC rollout on each maze condition -> list of (T,2) fingertip paths.

    If `to_maze_coords`, map the plant-space fingertip back through the env's own affine
    (raw = (plant - centre) / scale) so the path lines up with the barriers and target, which
    are drawn in raw maze coordinates."""
    import torch as th
    B = len(cond_indices)
    env.force_conditions(cond_indices)              # reset() will use exactly these mazes
    obs, info = env.reset(options={"batch_size": B})
    env.force_conditions(None)
    st = shac.init_state(B)
    path = [env.states["fingertip"].detach().cpu().numpy().copy()]
    for _ in range(int(env.max_ep_duration / env.dt)):
        with th.no_grad():
            a, st = shac.act(obs, st)
        obs, *_ = env.step(a)
        path.append(env.states["fingertip"].detach().cpu().numpy().copy())
    P = np.stack(path, 1)                                # (B, T, 2) plant coords
    if to_maze_coords:
        # ABSOLUTE maze coords, no origin-subtraction: the barriers and target are drawn
        # absolutely, and SHAC does not start exactly at the maze centre, so subtracting its
        # start would slide the whole path off the target it is actually reaching for.
        return [(P[i] - env.maze_centre[None]) / env.maze_scale for i in range(B)]
    return [P[i] - P[i, 0] for i in range(B)]


ZOO_LAZY = None


def _zoo():
    """tag -> class, built once. Kept here so the notebook cells stay one-liners."""
    global ZOO_LAZY
    if ZOO_LAZY is None:
        import motor_zoo as mz
        import plausible_learners as pl
        ZOO_LAZY = {"motornet_ref": mz.MotorNetRef, "bptt_gru": mz.BPTTGRU, "shac": mz.SHAC,
                    "sac": mz.SAC, "fasttd3": mz.FastTD3, "simbav2": mz.SimbaV2,
                    "eprop": pl.EProp, "rtrrl": pl.RTRRL, "btsp": pl.BTSP, "kinesis": mz.Kinesis,
                    "rstdp": pl.RSTDP, "predcode": pl.PredictiveCoding, "hebb3": pl.Hebb3,
                    "dendritron": mz.Dendritron}
    return ZOO_LAZY


def load_any(weights_path, env, model=None):
    """Rebuild a learner from a checkpoint, inferring the class from the file name."""
    import os
    import torch as th
    tag = os.path.splitext(os.path.basename(weights_path))[0]
    cls = model or _zoo().get(tag)
    if cls is None:
        raise KeyError(f"cannot infer a model from {tag!r}; pass model=<Class>. "
                       f"Known: {sorted(_zoo())}")
    m = cls(env)
    sd = th.load(weights_path, map_location=env.device)
    m.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd, strict=False)
    return m


def embed_gif(path):
    """Reliable inline display: base64-embed the GIF in an <img> so it animates and PERSISTS in
    the saved notebook (a bare Image(filename) can show a blank/first-frame or vanish on reload)."""
    import base64
    from IPython.display import HTML
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return HTML(f'<img src="data:image/gif;base64,{b64}" style="max-width:100%"/>')


def render_maze_runs(weights_path, n_runs=4, model=None, out=None, seed=7, fps=16):
    """Roll a trained model out on `n_runs` of the monkey's mazes and animate it as one GIF.

    weights_path : checkpoint (.pt); the class is inferred from the file name (or pass model=).
    n_runs       : how many mazes to show side by side.
    Returns the GIF path; display it with embed_gif(path).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import maze_env
    import motor_zoo as mz

    env = maze_env.make_maze_env(mz.DEVICE, random_cond=False)
    m = load_any(weights_path, env, model=model)
    cfg = env.cfg
    rng = np.random.default_rng(seed)
    conds = rng.choice(len(cfg["targets"]), size=n_runs, replace=False)
    paths = shac_maze_paths(m, env, conds, to_maze_coords=True)

    fig, axs = plt.subplots(1, n_runs, figsize=(3.1 * n_runs, 3.3), squeeze=False)
    axs = axs[0]
    trails, dots = [], []
    tag = os.path.splitext(os.path.basename(weights_path))[0]
    for j, ci in enumerate(conds):
        ax = axs[j]
        for cx, cy, hw, hh in cfg["barriers"][ci][:int(cfg["n_barriers"][ci])]:
            ax.add_patch(plt.Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, fc="#9AA3AD", ec="none", alpha=.7))
        ax.plot(*cfg["targets"][ci], "*", ms=15, color="#D1495B", zorder=4)
        ax.plot(*paths[j][0], "o", ms=4, color="#333", zorder=4)
        (t,) = ax.plot([], [], "-", lw=2.2, color="#B3541E"); (d,) = ax.plot([], [], "o", ms=8, color="#B3541E", zorder=5)
        trails.append(t); dots.append(d)
        ax.set_xlim(-.16, .16); ax.set_ylim(-.16, .16); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"maze {tuple(int(v) for v in cfg['keys'][ci])} · {int(cfg['n_barriers'][ci])} barriers", fontsize=8)
    fig.suptitle(f"{getattr(m, 'name', tag)} solving the monkey's mazes", fontsize=11, fontweight="bold")

    NF = 40
    RES = [p[np.linspace(0, len(p) - 1, NF).astype(int)] for p in paths]

    def update(f):
        arts = []
        for j in range(n_runs):
            k = min(f, NF - 1)
            trails[j].set_data(RES[j][:k + 1, 0], RES[j][:k + 1, 1])
            dots[j].set_data([RES[j][k, 0]], [RES[j][k, 1]])
            arts += [trails[j], dots[j]]
        return arts

    plt.tight_layout()
    ani = FuncAnimation(fig, update, frames=NF, interval=1000 / fps, blit=False)
    out = out or os.path.join("save", f"maze_{tag}.gif")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ani.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {out} | {getattr(m,'name',tag)} on mazes {[tuple(int(v) for v in cfg['keys'][c]) for c in conds]}")
    return out


def demo():
    """Self-check: geometry helpers behave, no data/network needed."""
    assert _nearest_dir(10, [0, 90, 180, 270]) == 0
    assert _nearest_dir(80, [0, 90, 180, 270]) == 1
    r = schematic_reach(90, reach=0.1, T=20)
    assert r.shape == (20, 2) and abs(r[-1, 1] - 0.1) < 1e-6 and abs(r[-1, 0]) < 1e-6
    assert np.allclose(r[0], 0)
    print("maze_demo OK: direction matching + schematic reach correct")


if __name__ == "__main__":
    demo()
