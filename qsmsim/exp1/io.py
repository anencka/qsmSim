"""Small I/O helpers shared by the Experiment 1 stages."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np


def res_tag(r):
    return f"res{float(r):.1f}"


def snr_tag(s):
    return f"snr{s:g}"


def proto_tag(dte, n):
    return f"dte{float(dte):g}_n{int(n):02d}"


def crop_box(mask, margin=1):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + margin, mask.shape)
    return [[int(a), int(b)] for a, b in zip(lo, hi)]


def crop(a, box):
    return a[tuple(slice(lo, hi) for lo, hi in box)]


def save_nifti_cropped(d, vols, box, voxel_mm, dtype=np.float32):
    """NIfTI (RAS+, diagonal voxel size) of each volume cropped to box; the affine's
    translation keeps cropped maps aligned with each other and with the full grid."""
    import nibabel as nib
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    aff = np.diag([voxel_mm] * 3 + [1.0])
    aff[:3, 3] = [lo * voxel_mm for lo, _ in box]
    for k, v in vols.items():
        v = np.asarray(v)
        if v.dtype.kind == "f":
            v = v.astype(dtype)
        nib.save(nib.Nifti1Image(np.ascontiguousarray(crop(v, box)), aff), d / f"{k}.nii.gz")


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable))
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class AtomicDir:
    """Write into <final>.tmp, then rename to <final> on success (resumable outputs)."""

    def __init__(self, final):
        self.final = Path(final)
        self.tmp = self.final.with_name(self.final.name + ".tmp")

    def __enter__(self):
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True)
        return self.tmp

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            if self.final.exists():
                shutil.rmtree(self.final)
            os.replace(self.tmp, self.final)
        return False
