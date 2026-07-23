"""Reproducible FAIR trainer for the MONKEY'S OWN MAZE task (MC-Maze, 108 puzzles, WITH collision).

The monkey moved a cursor through barrier mazes to a cued target. Here every one of the 13 learners
controls the SAME point-mass cursor (obs = goal + proprioception, action = 4 muscles -- the
joystick-via-muscle interface) on the SAME 108 MC-Maze conditions, optimising the SAME composite
objective (fairness by construction, decided by the env, not per model):

    reach the target FAST + with LEAST movement + LEAST endpoint error + WITHOUT hitting the barriers.

  * gradient rules (BPTT / SHAC / KINESIS) descend the composite maze cost by BPTT,
  * local plausible rules regress onto a spinal reflex = reach reflex + obstacle-avoidance reflex,
  * model-free deep-RL optimises env reward = -(composite cost),
all from task-space signals only -- no plant Jacobian, no privileged simulator state, one objective.

Writes save_monkey/maze_models/{tag}.pt + save_monkey/maze_results.json.
Run:  MZ_BUDGET=20000 .venv/bin/python notebooks/maze_train.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlb_tools"))
import numpy as np
import torch as th
import motor_zoo as mz
import plausible_learners as pl
import maze_env

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODEL_DIR = os.path.join(REPO, "save_monkey", "maze_models")
RESULTS_JSON = os.path.join(REPO, "save_monkey", "maze_results.json")
BUDGET = int(os.environ.get("MZ_BUDGET", 20_000))
DEVICE = mz.DEVICE
REACH_CM = 3.0                                   # "reached" threshold for the reach-TIME metric

# FIXED 60/20/20 split of the 108 puzzles (seeded). Train on `train` ONLY; track the learning
# curve on `val`; report the final scorecard on `test`, which is NEVER seen during training.
TRAIN_IDX, VAL_IDX, TEST_IDX = maze_env.maze_split(seed=0)
MZ_TRAIN = maze_env.make_maze_env(DEVICE, conditions=TRAIN_IDX, random_cond=True)    # 60% train
MZ_VAL   = maze_env.make_maze_env(DEVICE, conditions=VAL_IDX,  random_cond=False)    # 20% val  (curve + tuning)
MZ_TEST  = maze_env.make_maze_env(DEVICE, conditions=TEST_IDX, random_cond=False)    # 20% test (held out)
pl.configure(mz.morph_head(MZ_TRAIN), mz.obs_norm, mz.OpCounter)

LEARNERS = [
    (mz.MotorNetRef, "motornet_ref"), (mz.BPTTGRU, "bptt_gru"), (mz.SHAC, "shac"),
    (mz.SAC, "sac"), (mz.FastTD3, "fasttd3"), (mz.SimbaV2, "simbav2"),
    (pl.EProp, "eprop"), (pl.RTRRL, "rtrrl"), (pl.BTSP, "btsp"),
    (mz.Kinesis, "kinesis"), (pl.RSTDP, "rstdp"), (pl.PredictiveCoding, "predcode"),
    (pl.Hebb3, "hebb3"), (mz.Dendritron, "dendritron"),
]


@th.no_grad()
def maze_metrics(L, env, batch=512, seed=mz.EVAL_SEED):
    """Roll a trained learner over `env`'s mazes; measure the monkey's own scorecard."""
    MZ_EVAL = env
    obs, info = MZ_EVAL.reset(seed=seed, options={"batch_size": batch, "deterministic": True})
    st = L.init_state(batch); n = int(MZ_EVAL.max_ep_duration / MZ_EVAL.dt); dt_ms = MZ_EVAL.dt * 1000.0
    prev = MZ_EVAL.states["fingertip"].clone()
    path = th.zeros(batch, device=DEVICE)                          # total fingertip travel (movement)
    in_barrier = th.zeros(batch, device=DEVICE)                    # steps spent inside a barrier
    reach_step = th.full((batch,), float(n), device=DEVICE)        # first step within REACH_CM
    effort = 0.0
    for t in range(n):
        a, st = L.act(obs, st); obs, r, term, trunc, info = MZ_EVAL.step(a)
        ft = MZ_EVAL.states["fingertip"]
        path += th.linalg.vector_norm(ft - prev, dim=-1); prev = ft.clone()
        in_barrier += (MZ_EVAL.maze_collision(ft) > 0).float()
        d = th.linalg.vector_norm(ft - MZ_EVAL.goal[:, :2], dim=-1)
        reached = (d < REACH_CM / 100.0) & (reach_step == float(n))
        reach_step = th.where(reached, th.full_like(reach_step, float(t)), reach_step)
        effort += float(a.pow(2).mean())
    d = th.linalg.vector_norm(MZ_EVAL.states["fingertip"] - MZ_EVAL.goal[:, :2], dim=-1)
    k = max(1, int(0.2 * n))
    return dict(
        err_cm=100.0 * d.mean().item(),                           # least cm
        reach5=100.0 * (d < 0.05).float().mean().item(),          # % reached (<5 cm)
        reach2=100.0 * (d < 0.02).float().mean().item(),          # % reached (<2 cm, precise)
        reach_time_ms=dt_ms * reach_step.mean().item(),           # FAST: mean ms to first get within 3 cm
        path_cm=100.0 * path.mean().item(),                       # LEAST MOVEMENT: fingertip travel
        in_barrier_pct=100.0 * (in_barrier / n).mean().item(),    # AVOID: % of the reach inside a wall
        effort=effort / n,
    )


def run_one(cls, tag, budget=BUDGET, bs=32):
    th.manual_seed(0); np.random.seed(0)
    L = cls(MZ_TRAIN)
    pr = mz.Probe(MZ_VAL, every_eps=max(1, budget // 40), budget=budget)   # learning curve on VAL (never test)
    t0 = time.perf_counter(); L.fit(MZ_TRAIN, budget, pr, batch=bs); train_s = time.perf_counter() - t0
    m = maze_metrics(L, MZ_TEST)                                            # final scorecard on HELD-OUT test
    val_err = maze_metrics(L, MZ_VAL)["err_cm"]                             # val error (matches the tuning objective)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if isinstance(L, th.nn.Module):
        th.save(L.state_dict(), os.path.join(MODEL_DIR, f"{tag}.pt"))
    return dict(name=L.name, cite=L.cite, kind=L.kind, wins=getattr(L, "wins", ""), tag=tag,
                curve=pr.curve, val_err_cm=val_err, params=mz.count_params(L)[0], train_s=train_s, **m)


def main():
    only = [t for t in os.environ.get("MZ_ONLY", "").split(",") if t]
    learners = [(c, t) for c, t in LEARNERS if (not only or t in only)]
    print(f"device {DEVICE} | budget {BUDGET:,} eps | maze obs {MZ_TRAIN.observation_space.shape[0]} "
          f"act {MZ_TRAIN.action_space.shape[0]} | 108 MC-Maze WITH collision, split "
          f"{len(TRAIN_IDX)}/{len(VAL_IDX)}/{len(TEST_IDX)} train/val/TEST (seed 0) | {len(learners)} model(s)\n",
          flush=True)
    rand = mz.RandomFloor(MZ_TRAIN.action_space.shape[0])
    print(f"random-floor (test): err={maze_metrics(rand, MZ_TEST)['err_cm']:.1f}cm\n", flush=True)
    order = [t for _, t in LEARNERS]
    prior = {r["tag"]: r for r in json.load(open(RESULTS_JSON))} if (only and os.path.exists(RESULTS_JSON)) else {}
    for cls, tag in learners:
        try:
            r = run_one(cls, tag)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"{tag:14s} FAILED: {e}", flush=True); continue
        prior[tag] = r
        print(f"{r['name']:42s} err={r['err_cm']:5.1f}cm reach5={r['reach5']:4.0f}% "
              f"time={r['reach_time_ms']:5.0f}ms move={r['path_cm']:5.1f}cm wall={r['in_barrier_pct']:4.1f}% "
              f"{r['train_s']:5.0f}s", flush=True)
        json.dump([prior[t] for t in order if t in prior], open(RESULTS_JSON, "w"), indent=1)
    print(f"\nsaved {len([t for t in order if t in prior])} models -> {MODEL_DIR}\nresults -> {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
