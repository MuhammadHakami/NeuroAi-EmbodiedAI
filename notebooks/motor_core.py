"""ONE training loop, ONE learning objective, ONE evaluation surface -- for all 13 learners.

WHY THIS FILE EXISTS
--------------------
Every learner used to carry its own `fit()` (15 of them across motor_zoo.py + plausible_learners.py),
and the `Learner` base said outright that "fit is free to update however it likes". That freedom is
exactly how the benchmark drifted into optimising DIFFERENT objectives per family:

    analytic-gradient rules -> task reward (MassReach)
    local-plausible rules   -> imitation MSE against a demonstrator
    off-policy trio         -> behaviour-cloning + Q

A plausible-vs-non-plausible comparison across three different objectives measures nothing. Here the
episode loop AND the objective live in ONE place, and a learner supplies only its CREDIT ASSIGNMENT
-- the single thing actually under study.

THE SHARED OBJECTIVE (identical for all 13) = MOTORNET'S OWN
------------------------------------------------------------
    L = mean( sum( |fingertip - goal|, dim=-1 ) )        MotorNet examples/4-train-net.ipynb

together with MotorNet's own training setup: Adam(lr=1e-3), clip_grad_norm_(max_norm=1.0),
episodes of `max_ep_duration / dt` steps, loss accumulated over the whole episode.

Every rule minimises this identical task-space quantity; they differ only in HOW credit for it
is assigned -- BPTT through the differentiable plant, truncated-horizon analytic gradient,
eligibility trace, dendritic plateau, dopamine gate, off-policy TD. That is the single variable
under study.

An EARLIER version of this file imitated a demonstrator instead. That was a mistake and it is
worth recording why: imitation has no single space all 13 share (the morphological head emits
force + co-contraction, the muscle head emits 4 excitations), so it silently became a different
objective per family, needed two teachers, forced a per-model exception for KINESIS, and
subsidised imitators with a teacher budget nobody else paid. MotorNet's task loss has none of
those problems -- it is defined in task space, so it is model-agnostic by construction, and it
needs no teacher at all.

LEARNER CONTRACT (all a rule implements)
----------------------------------------
    init_state(B)                 -> recurrent state
    forward(obs, state)           -> (raw, new_state, aux)   aux = whatever its own rule needs
    on_step(ctx)                  -> apply ITS credit-assignment rule      <-- the only difference
    on_episode_start(B)           -> optional
    on_episode_end()              -> optional (gradient rules take their optimiser step here)

`ctx` is a StepCtx: obs, raw, target, aux, state, t, n, batch, reward.
"""
from dataclasses import dataclass
from typing import Any, Optional
import torch as th

# ---- the single evaluation surface: re-exported so there is exactly ONE import point ----------
from motor_zoo import (rollout, evaluate, eval_metrics, zero_shot, Probe,     # noqa: F401
                       SUCCESS_CM, TRAIN_MASSES, VAL_MASSES, OOD_MASSES, make_mass_env,
                       detach_env_state)


@dataclass
class StepCtx:
    """Everything a credit-assignment rule may read for ONE control step."""
    obs: th.Tensor
    raw: th.Tensor            # the learner's muscle activation (post-head)
    cmd: th.Tensor            # the learner's RAW command (pre-head), what a readout writes
    target: th.Tensor         # info["goal"] -- where the fingertip should be
    pos: th.Tensor            # info["states"]["fingertip"] -- where it actually is
    vel: th.Tensor            # cartesian fingertip velocity (proprioception)
    aux: Any                  # rule-specific internals produced by forward()
    state: Any
    t: int
    n: int
    batch: int
    reward: Optional[th.Tensor] = None
    learner: Any = None
    collision_force: Optional[th.Tensor] = None    # maze only: outward obstacle-avoidance force (B,2), for the reflex
    maze_cost: Optional[th.Tensor] = None          # maze only: the composite step cost (reach + collision + effort)

    @property
    def err(self) -> th.Tensor:
        """goal - fingertip: the CARTESIAN position error. THE shared learning signal.

        This is MotorNet's own teaching signal, and it is the one quantity that is meaningful
        for all 13 rules at once: it lives in task space, so it does not depend on how a model
        parameterises its motor command. It is also the signal a real motor system actually
        has -- where my hand is versus where I want it.
        """
        return self.target - self.pos

    @property
    def err_local(self) -> th.Tensor:
        """The SAME error, mapped into the coordinates a local readout can write to.

        A local rule's readout emits R raw units, not a 2-D position, so it needs the shared
        task-space error projected into its own output space. Doing that with the true
        Jacobian of body+head would be backprop through the plant -- precisely what a local
        rule may not do -- so each rule projects through its OWN FIXED feedback matrix
        (feedback alignment, Lillicrap+16). The OBJECTIVE is untouched; only the
        credit-assignment pathway differs, which is the variable under study.
        """
        p = getattr(self.learner, "project_error", None)
        return p(self) if p is not None else self.err

    def loss(self) -> th.Tensor:
        """The training loss for ONE step. Free reach: MotorNet's exact L1 on fingertip position,
        l1(x,y) = mean(sum(|x-y|,-1)). Maze: the composite maze cost (reach + barrier + effort),
        so a gradient rule descends the SAME objective model-free RL gets from the env reward."""
        if self.maze_cost is not None:
            return self.maze_cost
        return th.mean(th.sum(th.abs(self.err), dim=-1))


def _detach(x):
    if th.is_tensor(x): return x.detach()
    if isinstance(x, (tuple, list)): return type(x)(_detach(v) for v in x)
    return x


@th.no_grad()
def _teacher_raw(teacher, obs, t):
    return teacher.raw_from(obs, t)


def train(learner, env, budget, probe, batch=256, teacher=None, grad=False):
    """THE training loop. Identical for all 13 learners.

    `grad=True` keeps autograd alive so a BPTT/APG rule can accumulate the SAME objective across the
    episode and step its optimiser in `on_episode_end()`. Local rules run under no_grad and update
    in `on_step`. Either way the objective, the rollout, the budget accounting and the probe
    schedule are shared, so nothing about the comparison depends on who wrote which fit().
    """
    n = int(env.max_ep_duration / env.dt)
    eps = 0
    ctxmgr = th.enable_grad if grad else th.no_grad
    while eps < budget:
        with ctxmgr():
            state = learner.init_state(batch)
            obs, info = env.reset(options={"batch_size": batch})
            obs = obs if grad else obs.detach()
            if hasattr(learner, "on_episode_start"):
                learner.on_episode_start(batch)
            for t in range(n):
                raw, state, aux = learner.forward(obs, state)
                action = learner.HEAD(obs, raw)
                nobs, r, term, trunc, info = env.step(action)
                # step FIRST so the reward for this action is available to rules that use a
                # neuromodulator (R-STDP's dopamine gate); obs in the ctx is still the PRE-step
                # observation the command was computed from, so no rule sees the future.
                #
                # THE objective, with NO per-model exception: MotorNet's L1 fingertip-to-goal
                # position error. Task space is the only space in which all 13 rules are
                # commensurable -- raw commands are not (morphological head emits force +
                # co-contraction, muscle head emits 4 excitations), so any command-space loss
                # is silently a different objective per family. This also removes the
                # demonstrator entirely: no teacher, no teacher budget to charge, no question
                # of which family got the better teacher.
                fingertip = info["states"]["fingertip"]
                goal = info["goal"][..., :fingertip.shape[-1]]
                vel = info["states"]["cartesian"][..., 2:4]
                # Maze task: the objective is the env's composite reward (reach + barrier + effort),
                # so a gradient rule descends -reward (identical to what model-free RL maximises), and
                # the plausible reflex gets the analytic outward avoidance force. Determined by the ENV,
                # so all 13 rules get the SAME maze objective with no per-model code path.
                is_maze = hasattr(env, "maze_collision_force")
                cforce = env.maze_collision_force(fingertip) if is_maze else None
                mcost = (-r.mean()) if is_maze else None
                ctx = StepCtx(obs=obs, raw=action, cmd=raw, target=goal, pos=fingertip, vel=vel, aux=aux,
                              state=state, t=t, n=n, batch=batch, reward=r,
                              learner=learner, collision_force=cforce, maze_cost=mcost)
                learner.on_step(ctx)                                 # <-- the ONLY per-rule difference
                obs = nobs if grad else nobs.detach()
                # A rule whose identity is TRUNCATED backprop (SHAC's short horizon) declares
                # `bptt_horizon`; the core cuts the graph there so the truncation is a property of
                # the RULE, not of who wrote the loop.
                hz = getattr(learner, "bptt_horizon", None)
                if grad and hz and (t + 1) % hz == 0:
                    # Cut the tape EVERYWHERE it is carried before backprop-ing this chunk:
                    # the recurrent state, the obs, AND the plant's internal tensors. Detaching
                    # only the first two leaves MotorNet's effector states as live graph nodes,
                    # so the next chunk's backward walks into the freed previous chunk
                    # ("backward through the graph a second time") -- see detach_env_state.
                    state = _detach(state); obs = obs.detach(); detach_env_state(env)
                    if hasattr(learner, "on_horizon_end"):
                        learner.on_horizon_end()                 # ... then backward this chunk only
            if hasattr(learner, "on_episode_end"):
                learner.on_episode_end()
        eps += batch
        probe(learner, eps)
    probe(learner, eps, force=True)
    return eps


def demonstrator_budget(teacher_budget, consumes_demonstrator: bool) -> int:
    """Episodes that must be ADDED to a method's own budget because it consumed the demonstrator.
    Reporting sample-efficiency without this credits imitation learners with a subsidy they did not
    pay for -- which is why the old Sample-eff column was not comparable across families."""
    return int(teacher_budget) if consumes_demonstrator else 0


def total_budget(own_episodes, teacher_budget, consumes_demonstrator):
    return int(own_episodes) + demonstrator_budget(teacher_budget, consumes_demonstrator)


if __name__ == "__main__":
    # self-check: the shared objective and its signal are well-formed and rule-agnostic
    b = 8
    goal = th.tensor([[3.0, 4.0]]).repeat(b, 1)      # 3-4-5 triangle: |err|_1 = 7 per row
    ctx = StepCtx(obs=th.zeros(b, 12), raw=th.zeros(b, 4), cmd=th.zeros(b, 3),
                  target=goal, pos=th.zeros(b, 2), vel=th.zeros(b, 2),
                  aux=None, state=None, t=0, n=100, batch=b)
    assert th.allclose(ctx.err, goal), "err must be goal - fingertip"
    # MotorNet's l1: mean over batch of the SUM over xy -- not a mean over xy
    assert abs(ctx.loss().item() - 7.0) < 1e-6, f"expected L1 7.0, got {ctx.loss().item()}"

    # with no learner attached, err_local falls through to the shared error untouched
    assert th.allclose(ctx.err_local, ctx.err), "err_local must default to err"

    # a learner supplying project_error gets its OWN pathway, and only that
    class _FakeRule:
        def project_error(self, c): return c.err @ th.ones(2, 3)
    ctx.learner = _FakeRule()
    assert ctx.err_local.shape == (b, 3), "project_error must map into the readout's space"
    assert abs(ctx.loss().item() - 7.0) < 1e-6, "the OBJECTIVE must not change with the rule"

    assert total_budget(100, 1000, True) == 1100 and total_budget(100, 1000, False) == 100
    print("motor_core self-check OK: MotorNet L1 objective, per-rule pathway, honest budgets")
