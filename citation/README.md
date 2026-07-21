# References

All papers behind the models, data, environment, and analysis metrics used in this capstone.
BibTeX is in [`references.bib`](references.bib); each notebook ends with a **References** cell that
cites these inline. The per-model table below is the index; the sections after it group every
reference by role, with its citation key and where it is used.

## The thirteen models

Every model below controls the **same** MotorNet plant, minimises the **same** objective —
MotorNet's own training loss `L = mean Σₜ ‖fingertipₜ − goalₜ‖₁` — under the **same** setup taken
from [MotorNet's `examples/4-train-net.ipynb`](https://github.com/OlivierCodol/MotorNet):
`Adam(lr=1e-3)`, `clip_grad_norm_(1.0)`, batch 32, and a **~12.3 k-parameter policy**. There is
**no demonstrator**: nothing here imitates a trained network. The only thing that varies is the
**credit-assignment rule** — the single variable under study — and that rule's step size.

*Policy params* counts the network that produces behaviour. Auxiliary machinery is listed
separately because it is not part of the controller: critics for the off-policy learners, and the
**fixed, untrained** reservoir for the local rules.

| # | Model | Family | Credit assignment (what actually differs) | Head | Policy params | Auxiliary | Code | Paper |
|---|---|---|---|---|---|---|---|---|
| 1 | **BPTT-GRU** | global-gradient | Backprop-through-time over the full 100-step episode, through the differentiable plant. The reference point. | muscle (4·σ) | 12,373 | — | [`motor_zoo.py`](../notebooks/motor_zoo.py) | [Werbos 1990](https://doi.org/10.1109/5.58337) · [Cho+ 2014](https://arxiv.org/abs/1406.1078) · [Codol+ 2024](https://doi.org/10.7554/eLife.88591) |
| 2 | **SHAC** | global-gradient | Same analytic gradient, **truncated to a 16-step window** — one update per window, not per episode. | muscle (4·σ) | 12,373 | — | [`motor_zoo.py`](../notebooks/motor_zoo.py) | [Xu+ 2022](https://arxiv.org/abs/2204.07137) |
| 3 | **SAC** | global-gradient | Off-policy TD. **Stochastic** squashed-Gaussian actor with a learned `log σ` head + entropy bonus. No plant gradient. | muscle (4·σ) | 12,377 | twin MLP critics + targets (294 k) | [`motor_zoo.py`](../notebooks/motor_zoo.py) | [Haarnoja+ 2018](https://arxiv.org/abs/1801.01290) |
| 4 | **FastTD3** | global-gradient | Off-policy TD. **Deterministic** actor, fixed exploration σ, **clipped noise on the target action**, 4 updates per env step. | muscle (4·σ) | 12,373 | twin MLP critics + targets (294 k) | [`motor_zoo.py`](../notebooks/motor_zoo.py) | [Fujimoto+ 2018](https://arxiv.org/abs/1802.09477) |
| 5 | **Simba** | global-gradient | TD3's actor and update, but the critic trunk is **residual + LayerNorm** (`x + f(LN(x))`). | muscle (4·σ) | 12,373 | SimbaNet critics + targets (1.09 M) | [`motor_zoo.py`](../notebooks/motor_zoo.py) | [Lee+ 2024](https://arxiv.org/abs/2410.09754) |
| 6 | **KINESIS** | morphological | Analytic policy gradient like ①, but the command is an **endpoint force + co-contraction** that the *body's own geometry* decodes into muscle pulls. Plausibility is the body, not the update. | morphological (3) | 12,315 | — | [`motor_zoo.py`](../notebooks/motor_zoo.py) | [Simos+ 2025](https://arxiv.org/abs/2503.14637) · [Hogan 1984](https://doi.org/10.1109/TAC.1984.1103644) |
| 7 | **e-prop** | local-plausible | **ALIF** (adaptive) units + a forward-in-time **eligibility trace** combined with a broadcast learning signal. No BPTT. | morphological (3) | 12,327 | fixed reservoir (~1.7 M, untrained) | [`plausible_learners.py`](../notebooks/plausible_learners.py) | [Bellec+ 2020](https://doi.org/10.1038/s41467-020-17236-y) |
| 8 | **RTRRL / RFLO** | local-plausible | **Instantaneous** update (no trace, no adaptation), error routed through a **fixed random feedback** matrix — feedback alignment. | morphological (3) | 12,327 | fixed reservoir (~1.7 M, untrained) | [`plausible_learners.py`](../notebooks/plausible_learners.py) | [Murray 2019](https://doi.org/10.7554/eLife.43299) · [Williams & Zipser 1989](https://doi.org/10.1162/neco.1989.1.2.270) |
| 9 | **BTSP** | local-plausible | A seconds-long trace bound to the learning signal only at a **sparse, stochastic dendritic plateau** — one-shot, behavioural-timescale. | morphological (3) | 12,327 | fixed reservoir (~1.7 M, untrained) | [`plausible_learners.py`](../notebooks/plausible_learners.py) | [Bittner+ 2017](https://doi.org/10.1126/science.aan3846) |
| 10 | **R-STDP** | local-plausible | **Spike-gated** eligibility tag (post-error × pre-spike) consolidated by a **dopamine** third factor read from reward. | morphological (3) | 12,327 | fixed reservoir (~1.7 M, untrained) | [`plausible_learners.py`](../notebooks/plausible_learners.py) | [Izhikevich 2007](https://doi.org/10.1093/cercor/bhl152) |
| 11 | **Predictive coding** | local-plausible | Explicit **error units**; the latent is settled by iterative inference each step, and only the top-level prediction error descends onto the readout. | morphological (3) | 12,327 | fixed reservoir (~1.7 M, untrained) | [`plausible_learners.py`](../notebooks/plausible_learners.py) | [Rao & Ballard 1999](https://doi.org/10.1038/4580) · [Friston 2010](https://doi.org/10.1038/nrn2787) |
| 12 | **3-factor Hebb** | local-plausible | Local pre × post-error product **gated by a neuromodulator** that bursts when reward beats its slow baseline. | morphological (3) | 12,327 | fixed reservoir (~1.7 M, untrained) | [`plausible_learners.py`](../notebooks/plausible_learners.py) | [Kuśmierz+ 2017](https://doi.org/10.1016/j.conb.2017.08.020) · [Frémaux & Gerstner 2016](https://doi.org/10.3389/fncir.2015.00085) |
| 13 | **Dendritron** | local-plausible | Frozen low-rank expert "packs" (LoRA-style) selected by a router — context isolation against forgetting. | morphological (3) | 12,327 | fixed reservoir (~1.7 M, untrained) | [`motor_zoo.py`](../notebooks/motor_zoo.py) | [Hu+ 2021](https://arxiv.org/abs/2106.09685) |

**Heads.** Non-plausible learners wear MotorNet's native **muscle head** (4 excitations via sigmoid)
— no morphological advantage, they coordinate muscles the hard way. The plausible rules and KINESIS
wear the **morphological force head** (`aₘ = [dₘ·f]₊ / F_max + c`), where the arm's own geometry does
the muscle coordinate transform. That 3-vs-4 split is why the policy counts differ by ~0.5%: it is
the independent variable, not an accident.

**Shared substrate.** All six local rules sit on one **fixed** echo-state reservoir
([Jaeger 2001](https://doi.org/10.1162/089976602760407955), [Maass+ 2002](https://doi.org/10.1162/089976602760407955))
— a declared memory crutch, never trained by BPTT or the plant gradient — so what is compared
between them is strictly the plasticity rule. Because a local rule cannot backprop the plant's
Jacobian, it converts the shared task error into a weight change through a **fixed feedback
projection** ([Lillicrap+ 2020](https://doi.org/10.1038/s41583-020-0277-3)) rather than weight
transport.


## Simulator / plant
| key | paper | used for |
|---|---|---|
| `codol2024motornet` | Codol et al. 2024, *eLife* — **MotorNet** | the differentiable biomechanical arm/point-mass all models control |
| `hogan1984impedance` | Hogan 1984, *IEEE TAC* | co-contraction → endpoint impedance (KINESIS morphological head) |

## Neural datasets & tools
| key | paper | used for |
|---|---|---|
| `chowdhury2020area2` | Chowdhury, Glaser & Miller 2020, *eLife* | **Area2_Bump** monkey S1 center-out (DANDI 000127) — primary neural target |
| `pei2021nlb` | Pei, Ye et al. 2021, *NeurIPS D&B* | **Neural Latents Benchmark** + `nlb_tools` loader; MC_Maze/MC_RTT |
| `churchland2012neural` | Churchland et al. 2012, *Nature* | source reaching/maze task behind **MC_Maze** monkey M1/PMd (DANDI 000128) |
| `meg2023centerout` | *Scientific Data* 2023 | **human MEG** 4-direction center-out (figshare 6431021) |
| `yinfei2025humanbmi` | Yin Fei et al. 2025, *Zenodo* | **human intracortical M1** 8-dir center-out cursor BMI (Zenodo 19445138) |
| `dandi` | DANDI Archive | hosting of 000127 / 000128 |

## Standard ANNs / deep-RL learners (global-gradient family)
| key | paper | model |
|---|---|---|
| `cho2014gru` | Cho et al. 2014 | **GRU** backbone (BPTT-GRU, SHAC, KINESIS, RL actors) |
| `werbos1990bptt` | Werbos 1990 | **BPTT** / analytic policy gradient |
| `xu2022shac` | Xu et al. 2022 | **SHAC** short-horizon actor-critic |
| `haarnoja2018sac` | Haarnoja et al. 2018 | **SAC** |
| `ball2023rlpd` | Ball et al. 2023 | **RLPD** — demonstration-bootstrapped RL (our SAC/TD3/Simba fix) |
| `fujimoto2018td3` | Fujimoto et al. 2018 | **TD3** (FastTD3) |
| `fujimoto2021td3bc` | Fujimoto & Gu 2021 | **TD3+BC** behaviour-cloning anchor |
| `lee2024simba` | Lee et al. 2024 | **Simba** residual/LayerNorm RL |

## Biologically-plausible NeuroAI learners (local-plausible family)
| key | paper | model / mechanism |
|---|---|---|
| `bellec2020eprop` | Bellec et al. 2020, *Nat. Commun.* | **e-prop** eligibility propagation |
| `murray2019rflo` | Murray 2019, *eLife* | **RFLO / RTRRL** real-time recurrent learning |
| `williams1989rtrl` | Williams & Zipser 1989 | RTRL (background for RTRRL) |
| `bittner2017btsp` | Bittner et al. 2017, *Science* | **BTSP** behavioural-timescale plasticity |
| `simos2025kinesis` | Simos, Chiappa & Mathis 2025 | **KINESIS** morphological muscle control |
| `wochner2023embodiment` | Wochner et al. 2023, *CoRL* | morphological computation / muscle embodiment |
| `izhikevich2007rstdp` | Izhikevich 2007, *Cereb. Cortex* | **R-STDP** reward-modulated STDP |
| `rao1999predcoding` | Rao & Ballard 1999, *Nat. Neurosci.* | **predictive coding** |
| `friston2010freeenergy` | Friston 2010, *Nat. Rev. Neurosci.* | active inference (predictive coding) |
| `kusmierz2017threefactor` | Kuśmierz et al. 2017 | **3-factor Hebb** learning rules |
| `fremaux2016threefactor` | Frémaux & Gerstner 2016 | neuromodulated three-factor STDP |
| `maass2002lsm` | Maass et al. 2002 | liquid-state machine (the fixed **reservoir** substrate) |
| `jaeger2001esn` | Jaeger 2001 | echo-state networks (reservoir) |
| `hu2021lora` | Hu et al. 2021 | **LoRA** memory packs (Dendritron) |

## Model ↔ brain "neural-linking" metrics
| key | paper | metric |
|---|---|---|
| `marinvargas2024proprioception` | Marín Vargas, Bisi et al. 2024, *Cell* | field-standard model↔S1 comparison recipe |
| `schrimpf2020brainscore` | Schrimpf et al. 2020 | Brain-Score neural predictivity |
| `kornblith2019cka` | Kornblith et al. 2019 | **CKA** |
| `williams2021shape` | Williams et al. 2021 | **Procrustes / shape** metric (`netrep`) |
| `kriegeskorte2008rsa` | Kriegeskorte et al. 2008 | **RSA** |
| `nili2014rsatoolbox` | Nili et al. 2014 | `rsatoolbox` |
| `khosla2024softmatch` | Khosla & Williams 2024, *NeurIPS* | **soft-matching** (single-neuron identity) |
| `georgopoulos1982tuning` | Georgopoulos et al. 1982 | **cosine directional tuning** |
| `huh2024platonic` | Huh et al. 2024 | cross-system representational convergence (interpretation) |

## Background
| key | paper | topic |
|---|---|---|
| `mccloskey1989forgetting` | McCloskey & Cohen 1989 | catastrophic forgetting (continual-learning motivation) |
| `lillicrap2020backprop` | Lillicrap et al. 2020, *Nat. Rev. Neurosci.* | biological plausibility of credit assignment |

---
*A few community artifacts (the `Dendritron v0.4.2` Colab; the exact MEG/Zenodo author lists) have no
formal citation and are referenced by their archive DOI. DOIs/arXiv IDs are given where verified.*
