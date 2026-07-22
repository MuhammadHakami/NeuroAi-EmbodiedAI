"""The monkey's OWN maze puzzles, rebuilt inside MotorNet.

WHY
---
Until now the AI models reached to free-space targets while the monkey solved barrier mazes.
Comparing their neural activity across two different tasks confounds "different learning rule"
with "different problem". This module puts the models in the monkey's task: the SAME 108 maze
configurations Jenkins solved in MC_Maze (Churchland+ 2012 / NLB MC_Maze, DANDI 000128), with
the same barrier geometry and the same active target.

MotorNet is a read-only submodule, so nothing here edits it: `MazeReach` SUBCLASSES the
project's ReachEnv and adds the maze on top.

THE 108 PUZZLES
---------------
MC_Maze is 36 maze_ids x 3 trial_versions = 108 distinct conditions. Each carries
  target_pos   (n_targets, 2)  candidate targets, mm
  active_target                index of the one that is actually cued
  barrier_pos  (n_barriers, 4) rectangles (cx, cy, half_w, half_h), mm; 0/6/7/8/9 per maze
Positions are converted mm -> m and re-centred on the MotorNet workspace, because the monkey's
hand coordinates and the plant's coordinates have different origins and extents.

COLLISION
---------
The point mass cannot be given true rigid contact without editing the plant, so barriers act
through a DIFFERENTIABLE penetration penalty: for each rectangle, penetration depth is
    pen = relu(half_w - |dx|) * relu(half_h - |dy|)   (>0 only strictly inside)
summed over barriers. It is differentiable, so a gradient rule feels it through the plant, and
it is a plain scalar, so a local rule feels it through the same shared objective. Every model
therefore gets the identical maze cost -- see `collision_penalty`.
"""
import os
import numpy as np
import torch as th

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save", "mc_maze_configs.npz")
NWB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "000128",
                   "sub-Jenkins", "sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb")
MM_TO_M = 1e-3
# The monkey's centre-hold position in raw maze coords (its mean hand position at movement
# onset across the 108 mazes; std <=4 mm). Every maze reach starts here, so a model compared to
# the monkey must start here too -- see hold_reset_options().
MAZE_HOLD = (0.0, -0.038)


def hold_reset_options(env, batch):
    """reset() options that place the point mass at the monkey's centre-hold (in the env's plant
    frame), so a model solves the maze from the SAME start the monkey did."""
    import torch as _th
    hx = MAZE_HOLD[0] * env.maze_scale + float(env.maze_centre[0])
    hy = MAZE_HOLD[1] * env.maze_scale + float(env.maze_centre[1])
    js = _th.tensor([[hx, hy, 0.0, 0.0]], dtype=_th.float32).repeat(batch, 1)   # x,y,vx,vy (CPU)
    return {"batch_size": batch, "joint_state": js}


def _add_nlb_to_path():
    import sys
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlb_tools")
    if p not in sys.path:
        sys.path.insert(0, p)


def load_maze_nwb(nwb_path=NWB):
    """The MC_Maze NWBDataset, with nlb_tools put on the path first. Robust to the caller's
    working directory and to nlb_tools not already being importable (the qualitative demo needs
    the monkey's real hand trajectories, which live in this NWB)."""
    _add_nlb_to_path()
    from nlb_tools.nwb_interface import NWBDataset
    return NWBDataset(nwb_path)


def extract_configs(nwb_path=NWB, cache=CACHE, force=False):
    """Pull the 108 (maze_id, version) puzzles out of the NWB once and cache them."""
    if os.path.exists(cache) and not force:
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlb_tools"))
    from nlb_tools.nwb_interface import NWBDataset
    ti = NWBDataset(nwb_path).trial_info

    keys, targets, barriers, nbar = [], [], [], []
    for (mz, ver), g in ti.groupby(["maze_id", "trial_version"]):
        r = g.iloc[0]
        tp = np.asarray(r["target_pos"], dtype=np.float64).reshape(-1, 2)
        ai = int(r["active_target"])
        ai = ai if 0 <= ai < len(tp) else 0
        bp = np.asarray(r["barrier_pos"], dtype=np.float64).reshape(-1, 4) \
            if int(r["num_barriers"]) > 0 else np.zeros((0, 4))
        keys.append((int(mz), int(ver))); targets.append(tp[ai])
        barriers.append(bp); nbar.append(len(bp))

    B = max(nbar) if nbar else 0
    bar = np.zeros((len(keys), B, 4)); msk = np.zeros((len(keys), B))
    for i, b in enumerate(barriers):
        if len(b): bar[i, :len(b)] = b; msk[i, :len(b)] = 1.0
    out = dict(keys=np.array(keys), targets=np.array(targets) * MM_TO_M,
               barriers=bar * MM_TO_M, mask=msk, n_barriers=np.array(nbar))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    return out


def collision_penalty(pos, barriers, mask):
    """Summed penetration depth of `pos` into each barrier rectangle. 0 outside every barrier.

    pos      (B, 2)          fingertip
    barriers (B, K, 4)       (cx, cy, half_w, half_h)
    mask     (B, K)          1 for a real barrier, 0 for padding
    Differentiable in `pos`, so an analytic-gradient rule can descend it; a scalar, so a local
    rule can consume it through the same shared objective.
    """
    d = (pos[:, None, :] - barriers[..., :2]).abs()
    pen = th.relu(barriers[..., 2] - d[..., 0]) * th.relu(barriers[..., 3] - d[..., 1])
    return (pen * mask).sum(-1)


class MazeReach:
    """Mixin that turns any ReachEnv subclass into the monkey's maze task.

    Use `make_maze_env(...)`; this is kept separate from the plant so MotorNet is untouched.
    """
    def _init_maze(self, cfg, scale=1.0, collide_w=6.0, conditions=None):
        self.cfg = cfg
        self.collide_w = float(collide_w)
        idx = np.arange(len(cfg["targets"])) if conditions is None else np.asarray(conditions)
        self.cond_idx = idx
        dev = self.device
        # monkey hand coords -> plant workspace: centre both, then scale to fit
        lo = self.effector.pos_lower_bound.detach().cpu().numpy()
        hi = self.effector.pos_upper_bound.detach().cpu().numpy()
        ctr = (hi + lo) / 2.0
        span_plant = float(np.min((hi - lo) / 2.0))
        span_monkey = float(np.abs(cfg["targets"]).max())
        k = scale * span_plant / max(span_monkey, 1e-6)
        self.maze_scale, self.maze_centre = k, ctr
        self._tg = th.tensor(cfg["targets"][idx] * k + ctr, dtype=th.float32, device=dev)
        b = cfg["barriers"][idx].copy()
        b[..., :2] = b[..., :2] * k + ctr        # centres move with the workspace
        b[..., 2:] = b[..., 2:] * k              # half-extents only scale
        self._bar = th.tensor(b, dtype=th.float32, device=dev)
        self._msk = th.tensor(cfg["mask"][idx], dtype=th.float32, device=dev)
        self._cond = None

    def force_conditions(self, idx):
        """Pin the maze conditions used by the NEXT reset(s) to `idx` (indices into cond_idx).
        Pass None to release back to the env's default sampling."""
        self._forced_cond = None if idx is None else np.asarray(idx)

    def sample_conditions(self, batch, generator=None, fixed=None):
        n = self._tg.shape[0]
        if fixed is not None:
            c = th.as_tensor(fixed, device=self._tg.device).long().reshape(-1)
            c = c.repeat((batch + len(c) - 1) // len(c))[:batch]
        else:
            c = th.randint(0, n, (batch,), device=self._tg.device, generator=generator)
        self._cond = c
        return c

    def maze_goal(self):
        return self._tg[self._cond]

    def maze_collision(self, pos):
        if self._cond is None:
            return th.zeros(pos.shape[0], device=pos.device)
        return collision_penalty(pos, self._bar[self._cond], self._msk[self._cond])


def make_maze_env(dev, mass_set=None, conditions=None, collide_w=6.0, scale=0.85,
                  random_cond=True, **kw):
    """A MotorNet env running the monkey's 108 maze puzzles.

    Subclasses the project's MassReach (which itself subclasses MotorNet's env -- MotorNet is
    never edited). On reset it draws a maze condition and OVERRIDES the goal with that maze's
    active target; reward becomes MotorNet's L1 position error MINUS the barrier penalty, so
    the maze cost enters the one shared objective every model already optimises.
    """
    import motor_zoo as mz
    cfg = extract_configs()

    class _MazeEnv(mz.MassReach, MazeReach):
        def reset(self, *a, **k):
            obs, info = super().reset(*a, **k)
            B = self.states["fingertip"].shape[0]
            # a caller can pin the exact conditions for this reset (e.g. the qualitative demo
            # rolling a model out on chosen mazes) via `env.force_conditions(idx)`; otherwise
            # draw randomly (training) or tile deterministically (eval).
            forced = getattr(self, "_forced_cond", None)
            fixed = forced if forced is not None else (None if random_cond
                                                       else np.arange(B) % len(self.cond_idx))
            self.sample_conditions(B, fixed=fixed)
            g = self.maze_goal()
            self.goal = g if self.goal.shape[-1] == g.shape[-1] else \
                th.cat([g, th.zeros_like(g)], -1)[..., :self.goal.shape[-1]]
            info["goal"] = self.goal
            return self.get_obs(), info

        def reward(self, action):
            ft = self.states["fingertip"]
            pos = -th.sum(th.abs(self.goal[..., :ft.shape[-1]] - ft), dim=-1, keepdim=True)
            return pos - self.collide_w * self.maze_collision(ft)[:, None]

    env = _MazeEnv(effector=mz.mn.effector.ReluPointMass24(), max_ep_duration=1.0,
                   mass_set=mass_set, **kw)
    env = mz.env_to(env, dev)
    env._init_maze(cfg, scale=scale, collide_w=collide_w, conditions=conditions)
    return env


def demo():
    """Self-check: geometry loads, collision fires inside a barrier and not outside."""
    cfg = extract_configs()
    n = len(cfg["targets"])
    assert n == 108, f"expected 108 MC_Maze conditions, got {n}"
    assert cfg["barriers"].shape[-1] == 4
    # a point at a barrier's centre must collide; one far outside must not
    i = int(np.argmax(cfg["n_barriers"]))
    bar = th.tensor(cfg["barriers"][i:i + 1], dtype=th.float32)
    msk = th.tensor(cfg["mask"][i:i + 1], dtype=th.float32)
    inside = bar[0, 0, :2][None]
    outside = th.tensor([[1e3, 1e3]])
    assert collision_penalty(inside, bar, msk).item() > 0, "no penalty at a barrier centre"
    assert collision_penalty(outside, bar, msk).item() == 0, "penalty far outside a barrier"
    # differentiable
    p = inside.clone().requires_grad_(True)
    collision_penalty(p, bar, msk).sum().backward()
    assert p.grad is not None and th.isfinite(p.grad).all()
    print(f"maze_env OK: {n} puzzles, barriers/maze {cfg['n_barriers'].min()}-"
          f"{cfg['n_barriers'].max()}, penalty differentiable")


if __name__ == "__main__":
    demo()
