"""Fetch the trained model zoo on demand.

Checkpoints are not in git (the reservoir rules carry a fixed 4096x4096 recurrent matrix, so
several are 60-130 MB). `ensure_weights()` checks for them locally and, only if they are
missing, downloads the archive from Google Drive and unpacks it. Import it and call it at the
top of any cell that loads a checkpoint -- it is a no-op once the files are present.
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(HERE, "save")
MODEL_DIR = os.path.join(SAVE_DIR, "models")
DRIVE_ID = "1uTLkwmwvKdKBLV17L8RWymiGFaRvVEg0"
ARCHIVE = os.path.join(SAVE_DIR, "save.zip")


def _have_weights(model_dir=MODEL_DIR):
    return os.path.isdir(model_dir) and any(f.endswith(".pt") for f in os.listdir(model_dir))


def _download_drive(file_id, dest):
    """Google Drive direct download, handling the large-file confirmation interstitial."""
    import requests
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = "https://drive.usercontent.google.com/download"
    with requests.Session() as s:
        r = s.get(url, params={"id": file_id, "export": "download", "confirm": "t"},
                  stream=True, timeout=60)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:                     # still the interstitial -> parse the token
            import re
            m = re.search(r'name="confirm"\s+value="([^"]+)"', r.text)
            if m:
                r = s.get(url, params={"id": file_id, "export": "download",
                                       "confirm": m.group(1)}, stream=True, timeout=60)
                r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if not chunk:
                    continue
                f.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  downloading {done/2**20:6.0f} / {total/2**20:.0f} MB", end="")
        print()
    if zipfile.is_zipfile(dest):
        return dest
    # not a zip -> almost always Drive returning an HTML quota/permission page
    head = open(dest, "rb").read(400)
    os.remove(dest)
    raise RuntimeError(
        "Google Drive did not return a zip (quota, permissions, or the link changed).\n"
        f"First bytes: {head[:200]!r}\n"
        f"Download it manually and unzip into {SAVE_DIR}:\n"
        f"  https://drive.google.com/file/d/{file_id}/view")


def ensure_weights(model_dir=MODEL_DIR, file_id=DRIVE_ID, quiet=False):
    """Make sure trained checkpoints exist locally; download + unzip only if they do not.

    Returns the directory the .pt files live in. Safe to call repeatedly.
    """
    if _have_weights(model_dir):
        if not quiet:
            n = len([f for f in os.listdir(model_dir) if f.endswith(".pt")])
            print(f"weights present: {n} checkpoints in {model_dir}")
        return model_dir

    print(f"no checkpoints in {model_dir} -- fetching the model zoo from Google Drive")
    _download_drive(file_id, ARCHIVE)
    print(f"  unzipping {ARCHIVE} -> {SAVE_DIR}")
    with zipfile.ZipFile(ARCHIVE) as z:
        z.extractall(SAVE_DIR)

    if not _have_weights(model_dir):
        # the archive may wrap everything in a top-level folder; find the real one
        for root, _dirs, files in os.walk(SAVE_DIR):
            if any(f.endswith(".pt") for f in files):
                print(f"  checkpoints found at {root}")
                return root
        raise RuntimeError(f"archive unpacked but no .pt files under {SAVE_DIR}")
    n = len([f for f in os.listdir(model_dir) if f.endswith(".pt")])
    print(f"ready: {n} checkpoints in {model_dir}")
    return model_dir


def demo():
    """Self-check: the predicate is honest about what is on disk (no network needed)."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert not _have_weights(d), "empty dir must report no weights"
        open(os.path.join(d, "x.pt"), "wb").close()
        assert _have_weights(d), "a .pt file must count as weights"
    print(f"weights.py OK | local model dir {'HAS' if _have_weights() else 'is MISSING'} checkpoints")


if __name__ == "__main__":
    demo()
