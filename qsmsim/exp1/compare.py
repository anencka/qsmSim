"""Compare two Experiment 1 output directories (reproducibility / resume checks).

    python -m qsmsim.exp1.compare REF OTHER

For every production dataset in OTHER: the source SHA-256 must equal REF's, and
every chi / field map must be array-identical. Preparation hashes of phantoms
present in both must match. Exit code 0 only if everything matches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _arr(p):
    import nibabel as nib
    return np.asarray(nib.load(p).dataobj)


def compare(ref, other, log=print):
    ref, other = Path(ref), Path(other)
    ok, n_ds, n_maps = True, 0, 0
    for pj in sorted((other / "prep").glob("*/prep.json")):
        rj = ref / "prep" / pj.parent.name / "prep.json"
        if not rj.exists():
            continue
        a, b = json.loads(rj.read_text())["hashes"], json.loads(pj.read_text())["hashes"]
        common = set(a) & set(b)
        bad = [k for k in common if a[k] != b[k]]
        ok &= not bad
        log(f"prep {pj.parent.name}: {len(common) - len(bad)}/{len(common)} hashes equal")
    for m in sorted((other / "runs").glob("*/*/*/*/rep*/metrics.json")):
        rel = m.parent.relative_to(other)
        rm = ref / rel / "metrics.json"
        if not rm.exists():
            log(f"missing in reference: {rel}")
            ok = False
            continue
        a, b = json.loads(rm.read_text()), json.loads(m.read_text())
        n_ds += 1
        if a["source_sha256"] != b["source_sha256"]:
            log(f"SOURCE HASH DIFFERS: {rel}")
            ok = False
        for f in sorted(m.parent.glob("*.nii.gz")):
            n_maps += 1
            if not np.array_equal(_arr(f), _arr(ref / rel / f.name)):
                log(f"MAP DIFFERS: {rel}/{f.name}")
                ok = False
    log(f"compare: {n_ds} datasets, {n_maps} maps -> {'IDENTICAL' if ok else 'DIFFERENCES FOUND'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if compare(sys.argv[1], sys.argv[2]) else 1)
