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