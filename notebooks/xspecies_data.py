"""xspecies_data.py -- cross-species neural loaders on ONE common schema, so neuro_link's
metrics compare a trained model to any of them with identical code.

Every loader returns:
    dict(psth   = float array (n_dir, n_time, n_units)   condition-averaged rate,
         dirs   = float array (n_dir,)  target directions in degrees,
         times  = float array (n_time,) ms relative to movement onset,
         units  = int  n_units/channels,
         region, species, modality, name = strings,  bin_ms = int)

Datasets (all FULLY OPEN):
  Area2_Bump  macaque S1 / area 2 sorted spikes   (Chowdhury/Miller 2020, DANDI 000127)
  MC_Maze     macaque M1 / PMd sorted spikes       (Churchland/NLB 2021,   DANDI 000128)
  Human MEG   human sensorimotor 306-ch MEG        (Scientific Data 2023, figshare 6431021)

nlb_tools / MotorNet are read-only submodules -- import + subclass only. The MC_Maze NWB
ships a non-monotonic time index that trips nlb_tools' pandas slicing on newer pandas; we
sort the index (a local copy) and tolerate the cosmetic index.freq stamping failure.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.abspath("../nlb_tools"))
import monkey_data  # noqa: F401  -- applies the read-only np.nan_to_num patch on import


def _canon_bin(reach_deg, n_dirs=8):
    """Assign each reach angle (deg) to the nearest of n_dirs canonical directions."""
    canon = np.arange(n_dirs) * (360.0 / n_dirs)
    d = np.abs(((reach_deg[:, None] - canon[None, :] + 180) % 360) - 180)
    return canon, d.argmin(1)


def _psth_from_trials(spikes, align_time, reach_deg, n_dirs, bin_ms, window):
    """Condition-average (rows, units) spikes into (n_dir, n_time, units) rate (sp/s),
    dropping directions that no trial visited (returns the kept directions too)."""
    canon, lab = _canon_bin(reach_deg, n_dirs)
    tb = np.array(sorted(np.unique(align_time)))
    U = spikes.shape[1]
    psth = np.zeros((n_dirs, len(tb), U), np.float32)
    cnt = np.zeros(n_dirs, int)
    for i in range(n_dirs):
        cnt[i] = int((lab == i).sum())
        for j, tv in enumerate(tb):
            m = (lab == i) & (align_time == tv)
            if m.any(): psth[i, j] = spikes[m].mean(0)
    keep = cnt > 0
    psth = psth[keep] * (1000.0 / bin_ms)
    times = tb if np.issubdtype(tb.dtype, np.number) else np.arange(len(tb)) * bin_ms + window[0]
    return psth, canon[keep], np.asarray(times, float)


# ----------------------------------------------------------- monkey S1 (Area2_Bump)
def load_s1(bin_ms=20):
    d = monkey_data.load_area2bump(bin_ms=bin_ms)
    return dict(psth=d["psth"].astype(np.float32), dirs=d["dirs"].astype(float),
                times=d["times"], units=d["units"], bin_ms=bin_ms,
                region="S1 (area 2)", species="macaque", modality="sorted spikes",
                name="Area2_Bump (monkey S1)", trials=d.get("trials"))


# ----------------------------------------------------------- monkey M1/PMd (MC_Maze)
def _mcmaze_dataset(bin_ms):
    from nlb_tools.nwb_interface import NWBDataset
    _orig = NWBDataset.resample
    def _safe(self, tb):
        try:
            _orig(self, tb)
        except ValueError:                       # pandas refused to stamp index.freq
            self.data = self.data.sort_index(); self.data.index.freq = None; self.bin_width = tb
    NWBDataset.resample = _safe
    f = glob.glob("../data/000128/**/*train*.nwb", recursive=True) or \
        glob.glob("data/000128/**/*train*.nwb", recursive=True)
    if not f:
        raise FileNotFoundError("MC_Maze not found -- run `dandi download DANDI:000128` in ./data")
    ds = NWBDataset(f[0])
    ds.data = ds.data.sort_index()               # NWB index is non-monotonic -> sort a local copy
    ds.resample(bin_ms); ds.data = ds.data.sort_index()
    return ds


def load_m1(bin_ms=20, window=(-100, 400), n_dirs=8):
    ds = _mcmaze_dataset(bin_ms)
    ti = ds.trial_info
    def ang(r):
        try:
            v = np.asarray(r["target_pos"])[int(r["active_target"])]
            return np.rad2deg(np.arctan2(v[1], v[0])) % 360
        except Exception:
            return np.nan
    tmap = dict(zip(ti["trial_id"].values, ti.apply(ang, axis=1).values))
    td = ds.make_trial_data(align_field="move_onset_time", align_range=window)
    tid = td[("trial_id", "")].values.astype(int)
    at = td[("align_time", "")].values
    reach = np.array([tmap.get(t, np.nan) for t in tid])
    ok = ~np.isnan(reach)
    psth, dirs, times = _psth_from_trials(td["spikes"].values[ok], at[ok], reach[ok],
                                          n_dirs, bin_ms, window)
    return dict(psth=psth, dirs=dirs, times=times, units=psth.shape[2], bin_ms=bin_ms,
                region="M1 / PMd", species="macaque", modality="sorted spikes",
                name="MC_Maze (monkey M1/PMd)")


# ----------------------------------------------------------- human M1 intracortical (Zenodo 19445138)
def load_human_ic(session="20241105", array="MC-MED", gt="0.248", smooth="gaussian_0.1_10",
                  win_pct=25):
    """Human INTRACORTICAL motor-cortex firing rates during 8-direction centre-out cursor BMI
    (Zenodo 19445138, two tetraplegic participants). Spikes-vs-spikes with the monkeys -- the
    strong cross-species partner. Per-trial DataArrays (time, unit) are cropped to a common
    window (win_pct-th percentile of reach durations) and condition-averaged by target_angle.
    Caveats: preprocessed RATES (not raw spike times); attempted BMI movement in tetraplegia;
    M1 (not homologous with the monkey S1 set -- a cross-AREA as well as cross-species pairing)."""
    import pickle, glob, pandas as pd
    base = "../data/human_ic" if glob.glob("../data/human_ic/*trials.csv") else "data/human_ic"
    def _list(fn):
        f = sorted(glob.glob(f"{base}/*{session}*firing_rates_{array}*gt_{gt}*{smooth}_{fn}.pkl"))
        if not f:
            raise FileNotFoundError(f"human IC {session}/{array}/gt_{gt}/{fn} not found -- unzip "
                                    "sub-N2_RadialGrid.zip in ./data/human_ic (Zenodo 19445138)")
        return pickle.load(open(f[0], "rb"))[0]
    trials = _list("far") + _list("near")
    csv = pd.read_csv(sorted(glob.glob(f"{base}/*{session}*trials.csv"))[0])
    lens = [d.shape[0] for d in trials]
    Tw = int(np.percentile(lens, win_pct))                        # common time window
    X, deg, t0 = [], [], None
    for d in trials:
        tid = int(d.attrs["trial_id"])                            # trial_id indexes CSV row order
        if tid >= len(csv) or d.shape[0] < Tw:
            continue
        X.append(np.asarray(d.values)[:Tw]); deg.append(float(csv.iloc[tid].target_angle_deg) % 360.0)
        if t0 is None: t0 = np.asarray(d.coords["time"].values)[:Tw]
    X = np.stack(X).astype(np.float32); deg = np.array(deg)
    psth, dirs, _ = _psth_from_trials_ratelike(X, deg, 8)
    times = (t0 - t0[0]) * 1000.0 if t0 is not None else np.arange(Tw) * 10.0
    return dict(psth=psth, dirs=dirs, times=times, units=psth.shape[2], bin_ms=10,
                region="M1 (motor)", species="human", modality="intracortical firing rate",
                name=f"Human M1 intracortical ({array}, N2)")


def _psth_from_trials_ratelike(X, reach_deg, n_dirs):
    """Condition-average already-binned per-trial rates X (n_trials, T, U) by direction."""
    canon, lab = _canon_bin(reach_deg, n_dirs)
    T, U = X.shape[1], X.shape[2]
    psth = np.zeros((n_dirs, T, U), np.float32); cnt = np.zeros(n_dirs, int)
    for i in range(n_dirs):
        m = lab == i; cnt[i] = int(m.sum())
        if m.any(): psth[i] = X[m].mean(0)
    keep = cnt > 0
    return psth[keep], canon[keep], None


# ----------------------------------------------------------- human MEG (non-invasive)
def load_human_meg(*a, **k):
    from human_meg import load_human_meg as _l
    return _l(*a, **k)


LOADERS = {"s1": load_s1, "m1": load_m1, "human_ic": load_human_ic, "human": load_human_meg}

if __name__ == "__main__":
    for k in ("s1", "m1"):
        d = LOADERS[k]()
        print(f"{k:4s} {d['name']:28s} psth={d['psth'].shape} dirs={np.round(d['dirs']).astype(int)} "
              f"| {d['species']}/{d['region']}/{d['modality']}")
