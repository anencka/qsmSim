"""Validate the Python port of the RC2 metrics against the official leaderboard.

Scores every downloaded Stage 1 submission with qsmsim.rc2 and compares with
master_stage1_anonymized.csv (computed by the organizers' MATLAB code).

    python scripts/rc2_validate_metrics.py [--variant stage1|vs2] [--limit N]
"""
import argparse
import gzip
import io
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from qsmsim import rc2


def index_submissions(root):
    """basename -> loader for every .nii/.nii.gz in the submission zips and loose files."""
    idx = {}
    for p in sorted(Path(root).glob("*")):
        if p.suffix == ".zip":
            try:
                z = zipfile.ZipFile(p)
            except zipfile.BadZipFile:
                continue
            for name in z.namelist():
                base = Path(name).name
                if base.startswith("._") or not (base.endswith(".nii") or base.endswith(".nii.gz")):
                    continue
                idx.setdefault(base, (lambda z=z, name=name: z.read(name)))
        elif p.name.endswith((".nii", ".nii.gz")):
            idx.setdefault(p.name, (lambda p=p: p.read_bytes()))
    return idx


def load_bytes(raw):
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    img = nib.Nifti1Image.from_bytes(raw)
    # the organizers' load_nii() reorients by the header; the ground truth is RAS
    if int(img.header["sform_code"]) > 0 or int(img.header["qform_code"]) > 0:
        img = nib.as_closest_canonical(img)
    return np.asarray(img.dataobj, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="stage1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sims = {s: rc2.load_sim(s) for s in ("Sim1", "Sim2")}
    idx = index_submissions(rc2.ROOT / "submissions" / "stage1")
    rows = [r for r in rc2.leaderboard() if r["Sim1"] in idx and r["Sim2"] in idx]
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} leaderboard rows with both files downloaded")
    diffs = {k: [] for k in rc2.METRICS}
    print(f"{'submission':12s} " + " ".join(f"{k[:14]:>15s}" for k in rc2.METRICS))
    for r in rows:
        try:
            ms = [rc2.evaluate(load_bytes(idx[r[s]]()), sims[s], a.variant) for s in ("Sim1", "Sim2")]
        except Exception as e:  # malformed submissions exist (wrong size/orientation)
            print(f"{r['Submission Identifier']:12s} ERROR {type(e).__name__}: {e}")
            continue
        m = rc2.combine(*ms)
        cells = []
        for k in rc2.METRICS:
            ref = r[k]
            d = m[k] - ref
            if np.isfinite(ref) and np.isfinite(m[k]):
                diffs[k].append(abs(d) / max(abs(ref), 1e-9))
            cells.append(f"{m[k]:7.3g}/{ref:<7.3g}")
        print(f"{r['Submission Identifier']:12s} " + " ".join(f"{c:>15s}" for c in cells), flush=True)
    print("\nmedian / 90th-pct relative |port - official| per metric:")
    for k, v in diffs.items():
        if v:
            v = np.array(v)
            print(f"  {k:26s} n={len(v):3d}  median {100 * np.median(v):7.3f}%   p90 {100 * np.percentile(v, 90):7.3f}%"
                  f"   within 0.1%: {np.mean(v < 1e-3) * 100:5.1f}%")


if __name__ == "__main__":
    main()
