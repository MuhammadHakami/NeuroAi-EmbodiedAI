"""human_meg.py -- loader for the open human center-out reaching MEG dataset
(Scientific Data 2023, doi:10.1038/s41597-023-02454-y; figshare collection 6431021).

306-channel whole-head MEG (here 319 incl. a few non-MEG/flat channels), sensorimotor +
parietal coverage, while a human reaches from a centre hold to one of 4 diagonal-corner
targets -> a 4-direction planar centre-out. Files are MATLAB v7.3 (HDF5): a (4,1) cell
`epoched_data`, each element (n_trials, n_samples, n_chan), plus an MNE `info` struct.

Returned on the shared xspecies schema (see xspecies_data.py). Honest caveats, stated in
the notebook: (1) MEG is non-invasive field data, not spikes -- coarser than the monkey
intracortical sets; (2) 4 directions vs the monkey's 8; (3) the corner<->condition-index
mapping is not shipped in the .mat, so directions are assigned in file order. A fixed
unknown permutation of the 4 conditions shifts every model's score the same way, so the
plausible-vs-non-plausible RANKING is robust even though absolute values may move.
"""
import os, glob
import numpy as np
import h5py

# 4 diagonal corner targets -> directions in degrees, assigned in the file's condition order.
MEG_DIRS = np.array([45.0, 135.0, 225.0, 315.0])
SFREQ = 600.615


def _find(subject, session):
    for base in ("../data/human_meg", "data/human_meg"):
        hits = glob.glob(os.path.join(base, "**", f"Sub_{subject}_ses_{session}_ICA.mat"), recursive=True)
        if hits: return hits[0]
    raise FileNotFoundError(f"Human MEG Sub_{subject}_ses_{session} not found -- unzip ICA_MEG_data.zip "
                            "in ./data/human_meg (from figshare 6431021)")


def load_human_meg(subject=1, session=1, out_T=50, move_frac=0.5):
    """Condition-averaged human MEG evoked response per reach direction.

    out_T     : number of output time bins (the epoch is data-driven cropped then resampled).
    move_frac : fraction of the epoch kept around the peak cross-direction-variance window
                (isolates the movement-informative part, dropping quiet baseline)."""
    f = _find(subject, session)
    with h5py.File(f, "r") as h:
        ed = h["epoched_data"]
        conds = [np.asarray(h[ed[i, 0]][()]) for i in range(ed.shape[0])]   # each (trials, S, C)
    evoked = np.stack([c.mean(0) for c in conds], 0).astype(np.float32)      # (4, S, C)
    # drop flat / non-MEG channels
    ch_var = evoked.reshape(-1, evoked.shape[-1]).std(0)
    keep = ch_var > 1e-20
    evoked = evoked[:, :, keep]                                             # (4, S, U)
    # data-driven movement window: the contiguous slice of length move_frac*S with the most
    # variance ACROSS the 4 directions (i.e. where reach direction most modulates the field)
    S = evoked.shape[1]
    cross_dir_var = evoked.var(0).mean(1)                                   # (S,) var over dirs
    w = max(1, int(move_frac * S))
    csum = np.concatenate([[0], np.cumsum(cross_dir_var)])
    win_energy = csum[w:] - csum[:-w]
    s0 = int(win_energy.argmax())
    evoked = evoked[:, s0:s0 + w, :]                                        # (4, w, U)
    # resample time -> out_T
    xs, xt = np.linspace(0, 1, evoked.shape[1]), np.linspace(0, 1, out_T)
    psth = np.stack([[np.interp(xt, xs, evoked[d, :, u]) for u in range(evoked.shape[2])]
                     for d in range(4)], 0).transpose(0, 2, 1).astype(np.float32)  # (4, out_T, U)
    times = np.linspace(s0, s0 + w, out_T) / SFREQ * 1000.0                # ms into epoch
    return dict(psth=psth, dirs=MEG_DIRS.copy(), times=times, units=psth.shape[2],
                bin_ms=int(1000 / SFREQ * (evoked.shape[1] / out_T)),
                region="sensorimotor + parietal", species="human", modality="MEG (306-ch, field)",
                name=f"MEG center-out (human, Sub{subject})")


if __name__ == "__main__":
    d = load_human_meg()
    print(f"{d['name']}: psth={d['psth'].shape} dirs={d['dirs'].astype(int)} "
          f"units={d['units']} | {d['species']}/{d['region']}/{d['modality']}")
    v = d["psth"]; print(f"cross-direction signal: per-dir mean field std={v.std():.2e}")
