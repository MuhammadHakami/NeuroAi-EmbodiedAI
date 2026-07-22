# Baseline Fairness Audit — Prioritized Report

Scope note: two different failure directions matter here. Some deviations **inflate** a baseline (unfair advantage → skews the ranking); others **handicap or mislabel** a baseline (the row isn't the method it claims to be). Both are flagged per item.

---

## HIGH fairness impact

**SAC — the row is TD3, not SAC** (four coupled defects, one root cause)
- vs paper: The entropy-augmented soft Bellman target is missing — critic target is the plain TD3 `y = r + γ(1-d)·min(Q1t,Q2t)` with no `-α·logπ(a'|s')` (motor_zoo.py:1634). The next action is TD3 clipped target-policy smoothing `na = μ(s') + clamp(0.2·randn,-0.5,0.5)` (1632), not a policy sample. Actor loss is deterministic policy gradient `-(Q/‖Q‖) - ent·Σlog_std` (1643), not the reparameterized entropy loss. `log_std` (1593) is a single state-independent parameter with no tanh squash and no tanh-Jacobian correction — a fixed global exploration width, not a state-conditioned tanh-Gaussian. The critic backup is byte-identical to the FastTD3 sibling.
- Direction: **mislabel** — SAC's defining maximum-entropy mechanism is entirely absent.
- Fix: sample `(a',logp')` from a reparameterized squashed Gaussian at `s'`; set `y = r + γ(1-d)·(min(Q1t,Q2t)(s',a') - α·logp')`; delete the clipped smoothing noise. Add a state-dependent `log_std` head off the GRU trunk, `u=μ+σ·ε`, `a=tanh(u)`, correct `logp` by `-Σlog(1-a²+1e-6)`. Actor loss `la = (α·logp - min(Q1,Q2)(s,pa)).mean()`; drop the `-(Q/‖Q‖)` DPG term.

**SHAC — active path is truncated-window BPTT, not SHAC**
- vs paper: The smooth critic and the terminal value bootstrap `γ^H·V(s_H)` beyond the truncation window are absent from the live path. `fit()`→`motor_core.train(grad=True)`; `on_step` accumulates per-step L1; `on_horizon_end` backprops each 16-step window. All the critic/GAE/bootstrap code sits in `_fit_reward_unused`, never called. What is benchmarked is BPTT-GRU with a 16-step window — the window loss carries zero information past step 16, exactly the gap SHAC's critic exists to fill.
- Direction: **mislabel + handicap** — SHAC is stripped of its defining mechanism.
- Fix: treat per-step L1 as cost, fit a small value head on windowed TD-λ cost-to-go with a soft-updated target critic (α≈0.4), add `γ^H·V(s_H)` to each window's actor loss before `backward()`. Report critic params separately (as TD3/SAC/Simba already do).

**KINESIS — fixed correct muscle geometry is gifted as ground truth**
- vs paper: Policy emits a 2D endpoint force + scalar co-contraction, converted to muscle activations through a **fixed hand-coded** map `a_m = relu(d_m·f)/F_MAX + c` where `d_m` uses the arm's TRUE muscle-path anchors `env.effector._path_coordinates` (motor_zoo.py:1243). Real KINESIS outputs per-muscle activations directly and learns the map. The other 12 learners must learn the force→muscle transform from scratch; KINESIS is handed a correct inverse-coordinate prior.
- Direction: **unfair advantage** — inflates its score vs geometry-free learners.
- Fix: output per-muscle activations directly, OR make the force→muscle map a **learned (not frozen)** linear layer. If the fixed morphological head is kept as a deliberate "structural-prior entry," relabel it so — do not treat its score as like-for-like.

**Dendritron — LoRA adapter is dormant; row measures a plain readout**
- vs paper: In `fit()`, `base_frozen` flips True only *after* the budget is consumed, so during the timed run only the base readout `W0` trains; the low-rank packs `A_c/B_c` stay at init (`B_c=0` ⇒ contributes exactly 0) and never update. Packs would only train on a second `fit`/skill, which the harness never triggers. The "LoRA/Dendritron" row is a fixed reservoir + linearly-trained readout — no LoRA.
- Direction: **mislabel** — the defining low-rank update never runs.
- Fix: set `base_frozen=True` at the *start* of the benchmark `fit` so the measured learning is the BA pack update (or run ≥2 skills).

---

## MEDIUM fairness impact

**SAC/analysis-net — RL trio gets expert demonstrations**
- vs paper: In 4-analysis-net.ipynb (line 207) SAC/FastTD3/Simba are retrained with `teacher=BPTTGRU` (via `MLP_TAGS`), enabling the BC anchor + expert-replay pre-fill (motor_zoo.py:1582, 1655-71) — demonstration mechanisms not in these algorithms and not shared by the no-teacher baselines in that same notebook. (Headline train-net scoreboard uses `teacher=None`, so it's unaffected.)
- Direction: **unfair advantage**, analysis notebook only.
- Fix: instantiate the trio with `teacher=None` in 4-analysis-net (remove sac/fasttd3/simbav2 from the teacher branch), matching train-net — or explicitly disclose the bootstrap.

**KINESIS — output gain tuned on the held-out eval set**
- vs paper: `f_scale` tuned 600→650 explicitly "so the morphological policy... reaches 100% completion on the held-out set" (motor_zoo.py:1227-28,1241). Peers get no equivalent target-set gain tuning.
- Direction: **unfair advantage** (eval-set leakage).
- Fix: fix output scaling to the shared task-agnostic convention, or fold the gain into the trainable head. Pick it on a train split and freeze — never on the held-out set.

**SHAC — param-budget rationale is inconsistent with the benchmark's own rule**
- vs paper/benchmark: The critic was dropped citing a "36,610 dead params breaking equal-capacity," but the capacity-parity comment (line 540) states critics are algorithm-specific training machinery reported separately — and FastTD3 keeps twin critics. So SHAC is handicapped by a constraint its peers are exempt from.
- Direction: **handicap** (inconsistent rule).
- Fix: drop the param-budget rationale for SHAC; reinstate the critic as reported-but-not-budgeted machinery (implements the HIGH SHAC fix above).

**SAC — no entropy temperature anywhere**
- vs paper: Neither fixed-α soft target (v1) nor auto-tuned α (v2). `ent=0.01` only multiplies a `Σlog_std` proxy in the actor loss; nothing scales the critic target.
- Fix: introduce α used in **both** critic target and actor loss. Fix α (v1) or learn `log_alpha` with `-α·(logπ + H_target).detach()`, `H_target = -dim(A) = -4` (v2).

**FastTD3 — Bellman target uses the online actor, not the target actor**
- vs paper: `na,_ = self._raw(n,nh)` runs the online GRU (motor_zoo.py:1676). `gru_t/fc_t` are created and Polyak-updated but never read, so the bootstrap tracks the fast-moving online actor.
- Direction: handicap (weakens baseline).
- Fix: add a `_raw_t` forwarding `gru_t/fc_t`; compute `na,_ = self._raw_t(n,nh)` before adding smoothing noise.

**FastTD3 — the "Delayed" update is missing**
- vs paper: Actor and all target Polyak updates run every `_update` call (policy_freq=1) instead of once per 2 critic updates.
- Direction: handicap.
- Fix: gate the actor + target-net updates behind `if step % 2 == 0`; run the critic every step.

**Simba — SimBa architecture lives only in the throwaway critic**
- vs paper: The residual + pre/post-LayerNorm trunk is applied only to `_SimbaNet` (critic). The deployed policy is the plain GRUCell+Linear (byte-identical to FastTD3). Since the benchmark deploys and counts the policy, the "12.3k-param Simba" contains zero SimBa architecture at eval; the paper applies SimBa to both actor and critic.
- Direction: **mislabel** (row is architecturally TD3 at deployment).
- Fix: replace the actor's Linear readout with a small SimBa post-LN residual head on the GRU output — or at minimum document that SimBa lives only in the critic.

**e-prop — recurrent/input weights are frozen; only the readout is plastic**
- vs paper: `Wr`, `Win` are frozen buffers; only readout `W` trains. e-prop's entire point is training `W_rec`/`W_in` forward-in-time via eligibility traces. Frozen, it's reservoir/FORCE readout learning, not weight-space e-prop.
- Direction: **mislabel** (defining credit-assignment never acts).
- Fix: spend the 12.3k budget on a small **trainable** recurrent SNN (~50 ALIF units, train `Wr+Win+Wout`) and apply the eligibility+learning-signal update to `Wr/Win`. If the shared frozen reservoir is kept, downgrade the "e-prop" claim to reservoir-readout learning.

**e-prop — learning signal is a hand-designed spinal-PD reflex, not output error**
- vs paper: `err_local` is `[KP·err - KD·vel, cocontraction] - cmd` — a PD controller that alone reaches 5.2cm/59%. e-prop's learning signal is the task error broadcast through feedback `B`. (Shared across all 6 plausible rules, so no e-prop-specific edge — but it injects a strong hand-tuned prior the gradient baselines lack.)
- Direction: shared prior; affects plausible-family vs gradient-family comparison.
- Fix: drive the update with the projected task error `B @ err`. If the reflex is required to reach sub-5cm, keep it but report a **no-reflex ablation** so the reflex's contribution is separable.

**RTRRL — random-feedback matrix has the wrong shape and role**
- vs paper: `B` is 3×3 (output×output), applied to the readout error `e = err_local @ B.t()`. RFLO's `B` is N_hidden×N_output, projecting output error into the hidden layer to update recurrent weights; for output weights RFLO uses the raw error directly. The 3×3 rotation is a spurious mix of force_x/force_y/co-contraction that RFLO never applies — the signature "random feedback" is decorative and mislocated.
- Fix: remove the 3×3 `B`; use `e = err_local` directly, matching `dW_out = η·ε·h`.

**BTSP — the plateau gate is mean-preserving, so it's a continuous Hebb rule**
- vs paper: `on_step` divides by `(batch·_gp)` (plausible_learners.py:273-74); since `E[gate]=_gp`, the expected update equals the ungated continuous slow-trace Hebbian update. Real BTSP is a rare, biased, one-shot plateau that creates plasticity. As implemented the plateau injects only variance, zero mean-effect — the row measures slow-trace Hebb, not BTSP (and collapses toward e-prop/Hebb3).
- Direction: **mislabel** (undercuts the "distinct rules" premise).
- Fix: drop the `1/(batch·_gp)` normalization so a plateau is a genuinely biased sparse write; pair with a low plateau rate + larger per-plateau lr for a one-shot delta.

**BTSP — plateau rate ~8× too high for "one-shot"**
- vs paper: `p_plateau=8.0` gives ~8 plateaus per 1-s reach; Bittner 2017 induces a field from one/few.
- Fix: set `p_plateau≈1` (with `tau_slow=1s`) so ~1 plateau fires per episode.

**PredCoding — the generative model is never learned**
- vs paper: `Wpred`, `Wenc` are fixed random buffers; `on_step` updates only the motor readout. The docstring claims `dWpred ~ ε⊗r` but no such update exists. PC collapses to a delta-rule readout on a fixed nonlinear feature — the headline mechanism (learning the generative model on prediction error) is absent.
- Direction: **mislabel**.
- Fix: register `Wpred` as `nn.Parameter` with a small latent `Nrep` (choose so `Nr·Nrep + readout ≈ 12.3k`); in `on_step` add a slow Hebbian update `dWpred += (lr_gen/n)·(εᵀr/batch - λ·Wpred)`, `lr_gen ≪` readout lr.

**3-factor-Hebb — no eligibility trace**
- vs paper: Update is instantaneous `M(t)·(err_local⊗z)`; Fremaux & Gerstner's defining mechanism is a seconds-long decaying trace. (FAIR baseline — no sibling advantage — but a loose reproduction; dense per-step reward makes the practical effect small.)
- Fix: add `elig = decay·elig + (1-decay)·(err_localᵀz/batch)`, `W += lr/n·M·(elig - λ·W)`, `tau_e≈0.5–1s`. If distinctness (vs e-prop/BTSP, which own the trace) is preferred, keep it instantaneous but stop citing the eligibility-trace mechanism as implemented.

**Dendritron — adapters aren't trained by backprop**
- vs paper: `B_c` updates as `err⊗(A_c z)` (= exact LoRA gradient), but `A_c` uses a fixed random feedback matrix `F_c` in place of `B_cᵀ` (feedback alignment); no Adam, no `alpha/r` scale. Source LoRA trains both A and B by Adam on the task loss.
- Fix: train `A_c`, `B_c` with autograd/Adam (true gradient uses `B_cᵀ`), add `alpha/r` scale — or keep the feedback-alignment variant but label it a "plausible LoRA variant," not "LoRA."

---

## Faithful (no material deviations)

- **BPTT-GRU** — the only fully faithful entry. Its three deviations (fixed-affine obs norm, sum-vs-mean L1 over time, hidden width 57) are all **low**: shared identically across all 13 or required by the 12.3k-param parity constraint. The verbatim width-32 upstream is kept unmatched as `MotorNetRef`.

Fair-but-loose (verdict "minor-deviations," no unfair *advantage*, but not faithful to source): **FastTD3** (weakened by online-actor target + no delayed update), **Simba** (SimBa arch only in critic), **BTSP** and **3-factor-Hebb** (identity/faithfulness gaps, no sibling edge).

Net: only **KINESIS** (geometry prior + eval-gain tuning) and the **analysis-net RL trio** (demo bootstrap) actually *inflate* their scores. **SAC, SHAC, Dendritron, PredCoding, e-prop** are the mislabels — the row is not the algorithm named.

---

## Ordered TODO — most fairness gained first

1. **Kill KINESIS's two advantages** (skews the ranking, most direct fairness win): make the force→muscle map a learned layer (or relabel as a structural-prior entry), and move `f_scale` selection off the held-out set to a frozen train-split value.
2. **Remove the analysis-net demo bootstrap**: set `teacher=None` for SAC/FastTD3/Simba in 4-analysis-net.ipynb so they match train-net.
3. **Make "SAC" actually SAC**: soft Bellman target with `-α·logp'`, reparameterized squashed-Gaussian actor + state-dependent `log_std`, entropy temperature α in both target and actor loss. (Otherwise rename the row TD3.)
4. **Restore SHAC's critic + terminal bootstrap** on the L1-cost stream; drop the inconsistent param-budget rationale and report critic params separately.
5. **Activate Dendritron's LoRA**: `base_frozen=True` from the start of the timed `fit` so the BA packs are what's measured.
6. **Fix the two remaining FastTD3 mechanisms**: target-actor Bellman target + delayed (policy_freq=2) actor/target updates.
7. **Make PredCoding learn its generative model** and **e-prop train recurrent/input weights** (or explicitly downgrade both claims to "readout-only on a fixed reservoir").
8. **BTSP one-shot fix**: drop the mean-preserving normalization and set `p_plateau≈1`, so the row measures BTSP rather than slow Hebb.
9. **Add the e-prop/plausible-family no-reflex ablation** so the shared spinal-PD prior's contribution is separable from each learning rule.
10. Cleanups (low fairness, high honesty): remove RTRRL's 3×3 `B`, add Hebb3's eligibility trace or stop citing it, put a SimBa head on the deployed policy or document its absence, fix the RTRRL/e-prop docstrings that misdescribe the source rules.

Files to touch: `motor_zoo.py` (BPTT/SHAC/SAC/FastTD3/Simba/KINESIS/PredCoding + shared obs_norm), `plausible_learners.py` (e-prop/RTRRL/BTSP/Hebb3/Dendritron), `4-analysis-net.ipynb` (teacher branch, line ~207).
---

## Structural finding (maintainer note, added during part-3 fixes)

motor_zoo.py has TWO definitions each of SAC / FastTD3 / SimbaV2:
- standalone (lines ~801/985/1065): FAITHFUL algorithms (real SAC soft target + entropy;
  TD3 target actor) BUT hidden=256 -> policy params far exceed the 12.3k parity budget.
- BootstrapRL subclasses (lines ~1756/1760/1765): param-matched (FAIR_HIDDEN=57) BUT
  unfaithful (TD3-like, no SAC entropy). These SHADOW the standalone ones and are LIVE.

Resolving this is both a PART-1 (one definition per model) and PART-2 (faithful) task, and
they are in tension: faithful vs param-matched. The correct fix is to keep ONE definition per
RL model that is BOTH faithful AND param-matched -- resize the faithful standalone actor to
FAIR_HIDDEN (keep the critic wide, reported separately), delete the BootstrapRL shadows. This
is a careful reimplementation (verify each trains) -- staged, not rushed.

---

## Part-3 remaining: 4-monkey-net fair retrain (structural requirement)

4-monkey-net loads ARM (RigidTendonArm26, 6-muscle) checkpoints via a SEPARATE 1259-line
`motor_zoo_monkey.py` -- a full parallel implementation of all 13 learners, still on the OLD
imitation objective. This is BOTH a part-1 violation (two definitions per model: motor_zoo point-mass
+ motor_zoo_monkey arm) AND the part-3 blocker (monkey side not on the fair setup).

Root cause: motor_zoo models are point-mass-specific:
- `RAW=4` hardcoded (arm needs 6 = n_muscles).
- `muscle_head` = `sigmoid(raw[:, :4])` -- hardcoded 4.
- `force_head` uses point-mass geometry (obs[:,2:4] fingertip + fixed corner ANCHORS).

Correct fix (unifies part 1 AND enables part-3 monkey retrain -- ONE plant-agnostic definition per
model used by BOTH notebooks):
1. RAW := env.n_muscles (muscle-head models) so it adapts to plant (4 point-mass / 6 arm).
2. muscle_head := sigmoid(raw[:, :env.n_muscles]).
3. force_head / KINESIS: derive muscle anchors + fingertip index from the effector, not the global
   point-mass ANCHORS, so the morphological map works on the arm too.
4. Delete motor_zoo_monkey.py; point 4-monkey-net at motor_zoo (+ plausible_learners) on the arm env
   with motor_core's shared objective.
5. Retrain the arm models under the fair setup -> save to the monkey MODEL_DIR; 4-monkey-net loads them.

This is a real refactor (touch RAW + both heads + the morphological geometry across 13 models, then
retrain + verify on the arm) -- a dedicated session, not a low-context patch. 4-train-net + 4-analysis
are already retrained on the fair setup; this closes the monkey side.

---

## Part 3 progress — plant-agnostic unification DONE; arm retrain remaining (2026-07-22)

### Done + committed (verified)
The 5-step refactor above is IMPLEMENTED and verified. All 13 learners are now ONE plant-agnostic
definition each, running on BOTH the point mass (4 muscles) and the monkey RigidTendonArm26 (6):

- **Gradient/deep-RL family** (BPTT-GRU, SHAC, SAC, FastTD3, Simba, MotorNetRef): `muscle_head` =
  `sigmoid(raw)`; `RAW := env.action_space.shape[0]`; `obs_norm` branches on muscle count
  (`[4 vision, n_musc len, n_musc vel]`). Point-mass params/constants byte-identical (no regression).
- **Plausible family** (e-prop, RTRRL, BTSP, R-STDP, PredCoding, Hebb3): UNCHANGED — raw width stays
  3 (2-D endpoint force + co-contraction) on both plants, so the readout/Bfb/spinal reflex are
  identical. Only the HEAD differs, installed via `pl.configure(mz.morph_head(env), ...)`.
- **Kinesis + Dendritron**: route their decode through the plant-aware `morph_head(env)`.
- `make_arm_env()` builds the arm ReachEnv; `eval_metrics` co-contraction is plant-aware.

Verified: all 13 construct + train on the arm; a spinal reflex through the arm head reaches 88-90%
within 5 cm; on a 4096-ep smoke budget e-prop already reaches 60% within 5 cm on the arm.

### Fairness: NO data leakage (user requirement, 2026-07-22)
The **arm morphological head** is the force_head analog: endpoint force --J(q)^T--> joint torque
--(-M(q)^+)--> least-effort muscle tensions. It uses ONLY the OBSERVATION (fingertip = obs[:,2:4],
from which q is recovered by inverse kinematics) + FIXED body anatomy (link lengths; the moment-arm
function M(q1), which is q0-independent and calibrated ONCE from the plant geometry, exactly like the
point-mass fixed `anchors`). It NEVER reads MotorNet's internal per-step state (`env.states` joint /
moment arms). The first cut DID read env.states (leak) and was rebuilt; leak-free costs ~nothing
(90% vs 92% reflex). Audit result: **the only policy-path leak was the arm head (fixed); every other
`env.states` read is in the shared learning signal (the L1 objective / reflex target), which uses
ground-truth state identically for all 13 — the task definition, not privileged input.**

Constraint verified — **4-train-net**: the ball mass is NOT in the observation (obs dim = 12 with or
without a `mass_set`); MassReach sets `skeleton.mass` (physics) only. The model feels the mass through
the dynamics, exactly like the original MotorNet, which never tells the policy the mass.

### Remaining (the actual retrain — a focused next pass)
1. Write a reproducible **arm trainer** (the old one that produced `save_monkey/models` is not in the
   repo): for each of 13, build on `make_arm_env`, `pl.configure(morph_head(arm))`, train at full
   budget, `eval_metrics` + `zero_shot`, assemble `results.json` (keys the notebook reads: name, cite,
   kind, wins, tag, curve, acc, completion, completion5, params, zs_mean), save `{tag}.pt`.
2. Recompute `linking.pkl` (neuro_link model<->S1/M1/human correlations) from fair-model rollouts.
3. Migrate notebook cells 3/5/7 off `motor_zoo_monkey` -> `motor_zoo` + `plausible_learners`:
   `z.make_env`->`mz.make_arm_env`, drop the `z.BPTTGRU` teacher (fair setup has NO demonstrator:
   `cls(env)` not `cls(env, teacher=...)`), `z.N_MUSCLES`->6. Keep input/output experiment-faithful:
   obs goal == the reach target the monkey/human saw; output = muscle->joint->fingertip (comparable to
   monkey arm EMG/kinematics), so the model<->brain comparison is input-to-input, output-to-output.
4. Delete `motor_zoo_monkey.py` (1259-line duplicate) once the notebook no longer imports it.
5. Re-run/re-bake 4-monkey-net.

---

## Part 2 — model-to-paper faithfulness audit (3 independent agents, 2026-07-22)

Fairness result (all 3 agents agree): **NO policy-path leakage in any of the 13.** Every `act`/
`forward` reads only obs -> fixed reservoir/GRU -> readout -> head(obs). `arm_force_head` confirmed
clean (obs fingertip + once-calibrated fixed moment-arm table, no live `env.states`). The only
`env.states` reads are in the shared learning signal (L1 objective / spinal-reflex target), identical
for all models — the task definition, not privileged input. This independently confirms the arm-head
leak fix.

Paper-faithfulness verdicts (| model | faithful? | gap |):

- MotorNetRef / BPTT-GRU — **faithful** (MotorNet tutorial Policy verbatim; obs-norm is a shared benign transform).
- **SHAC** — **partial, genuine break**: short-horizon truncated differentiable-sim gradient present
  (horizon 16), but the **critic + TD(λ) value bootstrap is dead** (in `_fit_reward_unused`, never
  called). Runtime SHAC = criticless 16-step BPTT. Fix: restore the value bootstrap (costs params,
  breaks parity) OR relabel the row "truncated-window BPTT (SHAC-style, criticless)". [Xu+22]
- SAC — **faithful soft AC** (reparameterized sample, entropy in soft target + actor loss, twin-Q);
  minor: entropy on the unsquashed action (no tanh/Jacobian correction — the dead standalone SAC had
  it), state-independent log_std. [Haarnoja+18]
- FastTD3 — **faithful vanilla TD3** (twin-Q, target smoothing, deterministic actor); missing the
  distributional C51 critic that defines FastTD3 (breaks param budget) + no delayed policy update ->
  name overclaims; cite already says "TD3". [Fujimoto+18 / Seo+25]
- SimbaV2 — critic **is** a residual/LayerNorm SimbaNet (Simba-v1, faithful; cite says "Simba");
  class NAME "V2" overclaims (V2 = hyperspherical norm, absent). Display name already reads "Simba". [Lee+24]
- e-prop — **partial**: ALIF + pseudo-derivative eligibility + learning-signal x eligibility present,
  but recurrence is **frozen** (shared reservoir) and the adaptation-variable eligibility term is
  missing. [Bellec+20]
- **RTRRL / RFLO** — **NOT the algorithm**: the random-feedback matrix B is **dead code**, there is
  **no eligibility trace**, and `on_step` is a bare delta rule structurally identical to
  PredictiveCoding's. The "instantaneous / RTRL-truncation" comment misreads Murray 2019 (RFLO keeps a
  LEAKY trace). Fix (HIGH): implement `p=(1-dt/tau)p + (dt/tau)phi'(presyn)*presyn` and route error
  through the registered B (`dW ~ (B eps) (x) p`), or relabel as a delta baseline. [Murray 2019 eLife]
- BTSP — **faithful** (1 s plateau eligibility, stochastic plateau gate, one-shot biased write). [Bittner+17]
- R-STDP — **partial**: spiking presynaptic gate + eligibility tag + dopamine third factor present,
  but it is a reward-modulated Hebbian tag, not a true asymmetric pre-post STDP window; dopamine floored
  (no LTD). Low. [Izhikevich 2007]
- **PredictiveCoding** — **partial**: iterative inference + top-down generative pathway are real, but
  the **generative weights never learn** (Wpred/Wenc fixed random; only the readout has a delta rule).
  Docstring overclaims. Fix (HIGH): add `dWpred ~ eps (x) r` OR downgrade the docstring. [Rao&Ballard 1999]
- Hebb3 — **faithful** three-factor (reward-minus-baseline neuromodulator gates a pre x post-error
  eligibility trace). [Fremaux&Gerstner 2016]
- **Kinesis** — the morphological force head + LEARNED f_scale is faithfully implemented, but the real
  KINESIS (arXiv:2503.14637) is model-free RL motion imitation on an 80-muscle MyoSuite model. This is
  a **labeling** issue: present the row as morphological-computation control (Hogan 84 / Wochner+23),
  with KINESIS as thematic inspiration (the code header is already candid). [Simos+25]
- Dendritron — **faithful** (frozen base + active per-context LoRA packs A@B, feedback-alignment A
  update, router; no forgetting). [Hu+22 LoRA]

Prioritized remediation (fairness-compatible code fixes first, then honest relabels):
1. HIGH — RFLO/RTRRL: implement the leaky eligibility trace + random-feedback readout (plausible_learners.py 214-238).
2. HIGH — PredictiveCoding: learn the generative weights, or correct the docstring (plausible_learners.py 356-403).
3. MED  — e-prop: add the adaptation-variable eligibility term (plausible_learners.py 187-197).
4. MED  — SAC: tanh-squash + Jacobian log-prob correction (motor_zoo.py ~1794-1819).
5. LOW  — R-STDP: allow dopamine dip (LTD); e-prop/others per notes.
6. RELABEL (honest, no code): SHAC (criticless), FastTD3 (=TD3), Kinesis (morphological-control), so the
   scoreboard never claims an algorithm it does not implement. Delete dead standalone SAC/FastTD3/SimbaV2
   + shadowed Reservoir-based EProp/.../Hebb3 in motor_zoo.py (confusing for a faithfulness reader).

### Part 2 remediation APPLIED (2026-07-22)
- [x] RFLO/RTRRL — implemented the leaky eligibility trace + random-feedback readout (now genuinely RFLO, honestly weaker).
- [x] PredictiveCoding — generative weights Wpred now learn (Rao-Ballard second half), renormalized to stay contractive.
- [x] SHAC — relabeled honestly (16-step truncated diff-sim BPTT; critic dropped for parity).
- [ ] SAC tanh-squash + Jacobian correction (medium); e-prop adaptation-eligibility term (medium);
      R-STDP dopamine-dip/LTD (low); FastTD3/SimbaV2 name refinement (minor) — partial-not-wrong, deferred.
NOTE: RTRRL + PredictiveCoding changed -> 4-train-net's point-mass results for those two are stale;
re-retrain 4-train-net (and re-bake 4-monkey-net after the arm retrain) to refresh both scoreboards.

---

## Verification of the retrained fair scoreboard (5-agent adversarial workflow, 2026-07-22)

**Verdict: PUBLISHABLE — trustworthy, leak-free, paper-faithful.** Every value in save_monkey/
results.json reproduces on rebuild; no dishonest or buggy figure across all 5 checks.
- **Leakage: SOUND.** Proven at runtime: trashing env.states + goal leaves arm_force_head's output
  identical (th.equal True) -> the head is a pure obs+fixed-anatomy function. obs[:,2:4]==fingertip
  (maxdiff 0.0). Only the shared L1 objective/reflex target uses ground truth, identical for all.
- **Fair invariants: hold in substance.** count_params reproduces exactly for all 14 (gradient
  ~13.2k, plausible 12339, MotorNetRef 4998 as a deliberate unmatched baseline). Plausible plastic
  readout (12339) is ~6% BELOW gradient (13173) -> if anything disadvantages the plausible family.
  All obs=16/act=6, one budget (curves len 41), boot=obs=False (no demonstrator).
- **RFLO + PredictiveCoding fixes: confirmed correct** against Murray 2019 / Rao&Ballard 1999.

Two principled improvements applied post-verification:
- **SAC**: the 52.8cm sub-floor score was the fixed-alpha entropy-dominance pitfall (entropy 7.3x
  the tiny reward -> actor collapses to one posture), NOT a bug. Implemented entropy-constrained
  temperature auto-tuning (Haarnoja+18 v2). Retrained: SAC 11.5cm/11% (was 0.6%), uses all 6 muscles
  -- a fair, faithful deep-RL baseline. FastTD3 (same class, entropy off) already reached 14.2cm,
  confirming TD3 learns and only SAC's temperature was the issue.
- **BTSP**: the /_gp plateau-frequency compensation was tried and REJECTED (blows up one-shot
  variance -> worse than the floor). BTSP's low completion is an honest cost of sparse one-shot
  plasticity on equal episode budget, reported as-is.

Faithful retrain (40k budget) final reaching (complete@5cm): BPTT/SHAC 100%, MotorNetRef 99%;
PredCoding 68%, e-prop 55%, Dendritron 52%, Hebb3 46%, RFLO 31%, R-STDP 25%, BTSP 6%; SAC 11%,
FastTD3 9%, Simba 2%; KINESIS 2%. Story: credit-assignment quality (not plausibility per se) sets
the reaching ceiling. NOTE: 4-train-net's point-mass rows for RTRRL/PredictiveCoding/SAC are now
stale (those models changed) -> re-retrain 4-train-net to refresh them.
