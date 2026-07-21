# ==============================================================================
# Monkey Area2_Bump loader (Chowdhury 2020, DANDI 000127) -- arm-independent.
# Validated: 49 S1 (area 2) units, 8 target directions, condition-averaged PSTHs.
# nlb_tools is read-only; we work around its numpy>=2 read-only issue by patching
# np.nan_to_num to tolerate read-only inputs (no edit to the submodule).
# ==============================================================================
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.abspath("../nlb_tools"))

# --- read-only workaround for nlb_tools' `np.nan_to_num(arr, copy=False)` ------
_ORIG_NAN_TO_NUM = np.nan_to_num
def _nan_to_num_rw(x, copy=False, **kw):
    x = np.asarray(x)
    if not x.flags.writeable:
        try: x.setflags(write=True)
        except Exception: x = x.copy()
    return _ORIG_NAN_TO_NUM(x, copy=copy, **kw)
np.nan_to_num = _nan_to_num_rw


def _find(train=True):
    key = "train" if train else "test"
    for pat in (f"../data/000127/**/*{key}*.nwb", f"data/000127/**/*{key}*.nwb"):
        hits = glob.glob(pat, recursive=True)
        if hits: return hits[0]
    raise FileNotFoundError(f"Area2_Bump {key} NWB not found -- run `dandi download DANDI:000127` in ./data")


def load_area2bump(bin_ms=20, align="move_onset_time", window=(-100, 400)):
    """Returns dict with:
      psth   : (n_dir=8, n_time, n_units=49) condition-averaged S1 firing rate (spikes/s)
      dirs   : (8,) target directions in degrees
      units  : n S1 units;  bin_ms, times (ms rel. to alignment)
      trials : the raw per-trial aligned spikes (rows, units) + align_time + target_dir
      ds     : the NWBDataset (for behavior: hand_pos/vel, joint_ang x7, muscle_len x39)
    """
    from nlb_tools.nwb_interface import NWBDataset
    ds = NWBDataset(_find(True)); ds.resample(bin_ms)
    td = ds.make_trial_data(align_field=align, align_range=window)
    tid = td[("trial_id", "")].values.astype(int)
    at = td[("align_time", "")].values
    tmap = ds.trial_info.set_index("trial_id")["target_dir"]
    tdir = np.array([tmap.get(t, np.nan) for t in tid])
    sp = td["spikes"].values                                  # (rows, units)
    dirs = sorted(np.unique(tdir[~np.isnan(tdir)])); tb = sorted(np.unique(at)); U = sp.shape[1]
    psth = np.zeros((len(dirs), len(tb), U))
    for i, d in enumerate(dirs):
        for j, tv in enumerate(tb):
            m = (tdir == d) & (at == tv)
            if m.any(): psth[i, j] = sp[m].mean(0)
    psth *= (1000.0 / bin_ms)                                  # -> spikes/s
    times = (np.array(tb) if np.issubdtype(np.array(tb).dtype, np.number)
             else np.arange(len(tb)) * bin_ms + window[0])
    return dict(psth=psth, dirs=np.array(dirs), units=U, bin_ms=bin_ms,
                times=np.asarray(times, float), trials=dict(spikes=sp, align_time=at, target_dir=tdir), ds=ds)


if __name__ == "__main__":
    d = load_area2bump()
    print(f"S1 area2: {d['units']} units | {len(d['dirs'])} dirs | psth {d['psth'].shape} "
          f"| mean {d['psth'].mean():.1f} peak {d['psth'].max():.0f} sp/s")
