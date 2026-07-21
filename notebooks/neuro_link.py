"""neuro_link.py -- model<->brain "neural linking", data-source agnostic.

Every function operates on plain condition-averaged tensors, so the SAME code compares
a trained network to monkey S1 (Area2_Bump), monkey M1, or human reaching data. The
model side is captured with `grab_activity`; the neural side is any (n_dir, T, U) PSTH.

Metrics implemented (field-standard, Marin Vargas & Bisi et al. Cell 2024 / Brain-Score):
  linear_predictivity  cross-validated ridge model->neuron encoding (median held-out r),
                       leave-one-direction-out so temporal autocorrelation can't inflate it;
                       optional noise-ceiling normalisation.
  linear_cka           debiased linear CKA (Kornblith 2019 + Nguyen/Murphy debiased).
  procrustes_sim       orthogonal-Procrustes angular shape similarity (Williams 2021,
                       netrep if installed, else a self-contained numpy fallback).
  rsa                  RDM (1-corr over directions) compared by Spearman of upper triangles.
  tuning               Georgopoulos cosine fit per unit -> preferred direction + depth.
  pca_trajectories     top-k PCs of the condition-averaged population (the "3D" plots).
  brain_match          combine the similarity metrics into one 0..1 rank score.

MotorNet / nlb_tools are read-only; this module only imports + subclasses.
"""
import numpy as np
import torch as th
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

CANON_ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)


# ============================================================ model activity capture
def _flat_state(st, B):
    """Flatten whatever recurrent state a learner returns into (B, U): collect every
    float tensor with a batch-sized axis, move that axis to front, flatten the rest."""
    outs = []
    def rec(x):
        if th.is_tensor(x) and x.is_floating_point():
            t = x.detach()
            dims = [i for i, s in enumerate(t.shape) if s == B]
            if dims:
                outs.append(t.movedim(dims[0], 0).reshape(B, -1))
        elif isinstance(x, (list, tuple)):
            for e in x: rec(e)
        elif isinstance(x, dict):
            for e in x.values(): rec(e)
    rec(st)
    return th.cat(outs, 1) if outs else None


@th.no_grad()
def grab_activity(env, learner, n_dirs=8, batch=1024, seed=20260717, resample_T=None,
                  canon_dirs=None):
    """Roll out `learner`, capture its recurrent population state every control step,
    and condition-average by reach direction into a (n_dirs, T, U) PSTH -- the model's
    analogue of a neural PSTH. `canon_dirs` (degrees) fixes the direction bin centres so
    the model's directions line up 1:1 with the neural dataset's (default 0,45,..)."""
    obs, info = env.reset(seed=seed, options={"batch_size": batch, "deterministic": True})
    st = learner.init_state(batch)
    reach_deg = env.reach_dir().detach().cpu().numpy()             # (B,)
    n = int(env.max_ep_duration / env.dt)
    H = []
    for t in range(n):
        a, st = learner.act(obs, st, explore=False)
        H.append(_flat_state(st, batch))
        obs, r, term, trunc, info = env.step(a, deterministic=True)
    A = th.stack(H, 1).detach().cpu().numpy()                      # (B, T, U)
    if canon_dirs is None:
        canon_dirs = np.arange(n_dirs) * (360.0 / n_dirs)
    canon_dirs = np.asarray(canon_dirs, float)
    # nearest canonical direction (circular)
    d = np.abs(((reach_deg[:, None] - canon_dirs[None, :] + 180) % 360) - 180)
    lab = d.argmin(1)
    T, U = A.shape[1], A.shape[2]
    psth = np.zeros((len(canon_dirs), T, U), np.float32)
    for i in range(len(canon_dirs)):
        m = lab == i
        psth[i] = A[m].mean(0) if m.any() else np.nan
    psth = np.nan_to_num(psth)
    if resample_T is not None and resample_T != T:
        psth = resample_time(psth, resample_T)
    return dict(psth=psth, dirs=canon_dirs, T=psth.shape[1], U=U, n_used=len(reach_deg))


# ============================================================ shape helpers
def resample_time(psth, T2):
    """(D,T,U) -> (D,T2,U) by linear interpolation along the time axis."""
    D, T, U = psth.shape
    xs, xt = np.linspace(0, 1, T), np.linspace(0, 1, T2)
    out = np.empty((D, T2, U), np.float32)
    for dd in range(D):
        for u in range(U):
            out[dd, :, u] = np.interp(xt, xs, psth[dd, :, u])
    return out


def align_time(a, b):
    """Resample two (D,T,U) PSTHs to a common T (the smaller of the two)."""
    Ta, Tb = a.shape[1], b.shape[1]
    T = min(Ta, Tb)
    return (a if Ta == T else resample_time(a, T)), (b if Tb == T else resample_time(b, T))


def feature_matrix(psth):
    """(D,T,U) -> (D*T, U) design matrix: each row is one (direction,time) population state."""
    D, T, U = psth.shape
    return psth.reshape(D * T, U)


def _center_scale(X):
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.norm(X) + 1e-12
    return X / s


# ============================================================ 1. linear predictivity
def linear_predictivity(model_psth, neural_psth, alphas=CANON_ALPHAS,
                        ceiling=None, standardize=True):
    """Cross-validated ridge encoding: model units -> each neuron, leave-one-DIRECTION-out
    (so within-direction temporal autocorrelation can't leak into the test fold). Returns
    median (and mean) held-out Pearson r across neurons; `norm_r` divides by `ceiling`
    (per-neuron split-half noise ceiling) when supplied. This is the primary Cell-2024
    "how brain-like" score."""
    m, n = align_time(model_psth, neural_psth)
    X, Y = feature_matrix(m), feature_matrix(n)                    # (D*T, Um), (D*T, Un)
    D, T = m.shape[0], m.shape[1]
    groups = np.repeat(np.arange(D), T)
    n_splits = min(D, 8) if D > 1 else 2
    if standardize:
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Yhat = np.zeros_like(Y)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, groups=groups):
        # RidgeCV selects alpha by efficient SVD-based generalized CV in ONE fit -- exact and
        # ~50x cheaper than refitting a dense Ridge per alpha (which is O(p^3) for the 2048-unit
        # reservoirs). Falls back to a fixed mid alpha if the tiny fold degenerates.
        try:
            r = RidgeCV(alphas=alphas).fit(X[tr], Y[tr])
        except Exception:
            r = Ridge(alpha=alphas[len(alphas) // 2]).fit(X[tr], Y[tr])
        Yhat[te] = r.predict(X[te])
    rs = np.array([_pearson(Yhat[:, j], Y[:, j]) for j in range(Y.shape[1])])
    rs = np.nan_to_num(rs)
    out = dict(median_r=float(np.median(rs)), mean_r=float(rs.mean()), per_neuron_r=rs)
    if ceiling is not None:
        c = np.asarray(ceiling); ok = c > 0.05
        nr = np.zeros_like(rs); nr[ok] = np.clip(rs[ok] / c[ok], -1, 1.5)
        out["norm_median_r"] = float(np.median(nr[ok])) if ok.any() else float("nan")
    return out


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-12 else 0.0


def noise_ceiling(trial_psths):
    """Split-half noise ceiling per neuron (Spearman-Brown corrected). `trial_psths` is a
    list of (D,T,U) PSTHs from independent trial splits (>=2). Returns (U,) ceiling r."""
    arrs = [feature_matrix(align_time(trial_psths[0], p)[1]) for p in trial_psths]
    half = len(arrs) // 2
    A = np.mean(arrs[:half], 0); B = np.mean(arrs[half:], 0)
    r = np.array([_pearson(A[:, j], B[:, j]) for j in range(A.shape[1])])
    return np.clip(2 * r / (1 + np.abs(r) + 1e-9), 0, 1)          # Spearman-Brown


# ============================================================ 2. CKA (debiased linear)
def linear_cka(model_psth, neural_psth, debiased=True):
    """Debiased linear Centered Kernel Alignment (Kornblith 2019; Nguyen/Murphy debiased
    HSIC). Invariant to rotation/isotropic scale, so it compares representational geometry
    without fitting an alignment. 0..1."""
    m, n = align_time(model_psth, neural_psth)
    X, Y = feature_matrix(m).astype(np.float64), feature_matrix(n).astype(np.float64)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    K, L = X @ X.T, Y @ Y.T
    hsic = _hsic1 if debiased else _hsic0
    denom = np.sqrt(max(hsic(K, K), 1e-12) * max(hsic(L, L), 1e-12))
    return float(hsic(K, L) / denom)


def _hsic0(K, L):
    n = K.shape[0]; H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)


def _hsic1(K, L):
    """Unbiased HSIC estimator (Song 2012), used for debiased CKA."""
    n = K.shape[0]
    K = K.copy(); L = L.copy(); np.fill_diagonal(K, 0); np.fill_diagonal(L, 0)
    ones = np.ones(n)
    t = np.trace(K @ L)
    a = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    b = 2.0 / (n - 2) * (ones @ K @ L @ ones)
    return float((t + a - b) / (n * (n - 3)))


# ============================================================ 3. Procrustes shape similarity
def procrustes_sim(model_psth, neural_psth):
    """Rotation-invariant shape similarity in [0,1] = cos(angular Procrustes distance).
    Uses netrep's generalized shape metric if installed (Williams 2021), else an equivalent
    orthogonal-Procrustes numpy fallback (pad to common dim, whiten, best rotation)."""
    m, n = align_time(model_psth, neural_psth)
    X, Y = feature_matrix(m), feature_matrix(n)
    try:
        from netrep.metrics import LinearMetric
        metric = LinearMetric(alpha=1.0, center_columns=True, score_method="angular")
        ang = metric.fit(X, Y).score(X, Y)                        # radians in [0, pi/2]
        return float(np.cos(ang))
    except Exception:
        return _procrustes_np(X, Y)


def _procrustes_np(X, Y):
    X = _center_scale(X); Y = _center_scale(Y)
    p = max(X.shape[1], Y.shape[1])
    X = np.pad(X, ((0, 0), (0, p - X.shape[1]))); Y = np.pad(Y, ((0, 0), (0, p - Y.shape[1])))
    # nuclear norm of X^T Y = max sum of singular values under orthogonal alignment
    s = np.linalg.svd(X.T @ Y, compute_uv=False)
    corr = float(s.sum())                                          # in [0,1] after unit-scaling
    return float(np.clip(corr, 0, 1))


# ============================================================ 4. RSA
def _rdm(psth):
    """Representational dissimilarity matrix over directions (time-averaged pop vectors)."""
    V = psth.mean(1)                                               # (D, U)
    V = V - V.mean(0, keepdims=True)
    C = np.corrcoef(V)
    return 1.0 - C


def rsa(model_psth, neural_psth):
    """Spearman correlation of the two RDMs' upper triangles (Kriegeskorte 2008). Both
    RDMs are over the shared directions, so model/neural unit counts need not match."""
    Rm, Rn = _rdm(model_psth), _rdm(neural_psth)
    iu = np.triu_indices(Rm.shape[0], 1)
    rho, _ = spearmanr(Rm[iu], Rn[iu])
    return float(np.nan_to_num(rho))


# ============================================================ 5. directional tuning
def tuning(psth, dirs):
    """Georgopoulos cosine tuning per unit on time-averaged rate: fit r = b0 + g*cos(th-PD)
    by least squares on [1, cos th, sin th]. Returns preferred directions (deg), tuning
    depth g, and fraction of units significantly tuned (depth over a small threshold)."""
    V = psth.mean(1)                                               # (D, U)
    th_ = np.deg2rad(np.asarray(dirs, float))
    Xd = np.stack([np.ones_like(th_), np.cos(th_), np.sin(th_)], 1)  # (D, 3)
    beta, *_ = np.linalg.lstsq(Xd, V, rcond=None)                 # (3, U)
    b0, b1, b2 = beta
    pd = (np.rad2deg(np.arctan2(b2, b1)) % 360)
    depth = np.sqrt(b1 ** 2 + b2 ** 2)
    rng = (V.max(0) - V.min(0)) + 1e-9
    tuned = depth / rng                                            # normalized modulation
    return dict(pd=pd, depth=depth, frac_tuned=float((tuned > 0.15).mean()),
                mean_depth=float(np.median(depth)))


def tuning_match(model_psth, neural_psth, dirs):
    """Similarity of tuning STRUCTURE: how close the model's preferred-direction distribution
    is to the neural one (circular-histogram cosine over 8 bins). 0..1."""
    tm = tuning(model_psth, dirs)["pd"]; tn = tuning(neural_psth, dirs)["pd"]
    b = np.linspace(0, 360, 9)
    hm, _ = np.histogram(tm, b); hn, _ = np.histogram(tn, b)
    hm = hm / (hm.sum() + 1e-9); hn = hn / (hn.sum() + 1e-9)
    return float(hm @ hn / (np.linalg.norm(hm) * np.linalg.norm(hn) + 1e-12))


# ============================================================ 5b. soft matching (SOTA 2024)
def _topk_units(A, k):
    """Keep the k highest-variance columns (units) of A (samples, units)."""
    if A.shape[1] <= k:
        return A
    return A[:, np.argsort(A.var(0))[-k:]]


def soft_matching_sim(model_psth, neural_psth, cap=128):
    """Soft Matching similarity (Khosla & Williams, NeurIPS 2024) -- the ONE metric here
    sensitive to single-NEURON identity (not rotation-invariant): match model units to
    neurons by optimal assignment and measure the leftover distance. Answers "do individual
    model units look like individual neurons?", the microcircuit question. Uses POT's OT for
    unequal unit counts if installed, else a zero-padded Hungarian assignment. Populations
    larger than `cap` are reduced to their top-`cap` highest-variance units first, so the
    O(n^3) assignment stays cheap (the 2048-unit reservoirs would otherwise dominate runtime).
    Returns 1 - distance in [0,1]."""
    from scipy.optimize import linear_sum_assignment
    m, n = align_time(model_psth, neural_psth)
    X, Y = _topk_units(feature_matrix(m), cap), _topk_units(feature_matrix(n), cap)
    Xn = X - X.mean(0); Yn = Y - Y.mean(0)
    Xn = Xn / (np.linalg.norm(Xn) + 1e-12); Yn = Yn / (np.linalg.norm(Yn) + 1e-12)
    xi, yj = Xn.T, Yn.T                                            # (Um, S), (Un, S) neurons-as-points
    C = (np.sum(xi ** 2, 1)[:, None] + np.sum(yj ** 2, 1)[None, :] - 2 * xi @ yj.T)
    C = np.clip(C, 0, None)
    try:
        import ot
        a = np.full(xi.shape[0], 1 / xi.shape[0]); b = np.full(yj.shape[0], 1 / yj.shape[0])
        d = float(ot.emd2(a, b, C))
    except Exception:
        p = max(xi.shape[0], yj.shape[0])                          # zero-pad to equal counts
        Cp = np.full((p, p), C.max() if C.size else 1.0)
        Cp[:C.shape[0], :C.shape[1]] = C
        ri, ci = linear_sum_assignment(Cp); d = float(Cp[ri, ci].mean())
    return float(np.clip(1.0 - np.sqrt(max(d, 0.0)), 0, 1))


# ============================================================ leaderboard (rank-averaged)
def leaderboard(per_model, metrics=("predictivity", "cka", "procrustes", "rsa", "tuning")):
    """Combine several metrics into one ranking the SOTA way (Brain-Score / Cell 2024):
    rank models within each metric (higher=better) and average the ranks -- avoids averaging
    raw metrics that live on different scales. `per_model` = {name: {metric: value}}.
    Returns list of (name, mean_rank, row) sorted best-first (lowest mean rank)."""
    names = list(per_model)
    ranks = {nm: [] for nm in names}
    for k in metrics:
        vals = [(nm, per_model[nm].get(k, float("-inf"))) for nm in names]
        order = sorted(vals, key=lambda t: t[1], reverse=True)     # best first
        for rank, (nm, _) in enumerate(order, 1):
            ranks[nm].append(rank)
    scored = [(nm, float(np.mean(ranks[nm])), per_model[nm]) for nm in names]
    return sorted(scored, key=lambda t: t[1])


# ============================================================ 6. PCA trajectories
def pca_trajectories(psth, k=3):
    """Top-k PCs of the condition-averaged population -> (D, T, k) trajectories for the 3D
    neural-trajectory plots. PCA is fit on the stacked (D*T, U) states."""
    D, T, U = psth.shape
    X = psth.reshape(D * T, U); X = X - X.mean(0, keepdims=True)
    Uu, Sv, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = (X @ Vt[:k].T).reshape(D, T, k)
    var = (Sv[:k] ** 2 / (Sv ** 2).sum())
    return pcs, var


# ============================================================ combined brain-match score
SIM_METRICS = ("cka", "procrustes", "rsa", "tuning")

def brain_match(model_psth, neural_psth, dirs, predictivity_ceiling=None):
    """All similarity metrics + a single combined 0..1 brain-match score (mean of the
    rotation/scale-robust similarities). Predictivity is reported alongside but not folded
    into the score by default because its scale depends on the neural noise ceiling."""
    pred = linear_predictivity(model_psth, neural_psth, ceiling=predictivity_ceiling)
    out = dict(
        predictivity=pred["median_r"],
        predictivity_norm=pred.get("norm_median_r", float("nan")),
        cka=linear_cka(model_psth, neural_psth),
        procrustes=procrustes_sim(model_psth, neural_psth),
        rsa=rsa(model_psth, neural_psth),
        tuning=tuning_match(model_psth, neural_psth, dirs),
    )
    out["brain_match"] = float(np.mean([out[k] for k in SIM_METRICS]))
    return out


if __name__ == "__main__":
    # self-check: a rotated+noisy copy of a synthetic PSTH must score near-perfect on the
    # rotation-invariant metrics, and a shuffled one must score much lower.
    rng = np.random.RandomState(0)
    D, T, U = 8, 25, 40
    base = rng.randn(D, T, U)
    Q, _ = np.linalg.qr(rng.randn(U, U))                          # random rotation
    rot = (base.reshape(D * T, U) @ Q).reshape(D, T, U) + 0.01 * rng.randn(D, T, U)
    shuf = base[rng.permutation(D)]                              # break direction correspondence
    dirs = np.arange(D) * 45.0
    good = brain_match(base, rot, dirs); bad = brain_match(base, shuf, dirs)
    print("rotated  :", {k: round(v, 3) for k, v in good.items()})
    print("shuffled :", {k: round(v, 3) for k, v in bad.items()})
    assert good["cka"] > 0.9 and good["procrustes"] > 0.9, "rotation invariance broken"
    assert good["brain_match"] > bad["brain_match"], "match score not discriminative"
    print("OK neuro_link self-check")
