"""Baked-in inputs: the RC2 Sim1 files P3 needs, checked against the SHA-256 sums
recorded in rc2_sha256.json (generated from the downloaded challenge data)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..rc2 import ROOT as RC2_ROOT
from ..rc2_3t import FILES

SUMS = Path(__file__).with_name("rc2_sha256.json")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_sums(root=RC2_ROOT):
    SUMS.write_text(json.dumps({f: file_sha256(Path(root) / f) for f in FILES}, indent=1) + "\n")


def check_rc2(root=RC2_ROOT, verbose=False):
    want = json.loads(SUMS.read_text())
    ok = True
    for f, h in want.items():
        p = Path(root) / f
        good = p.exists() and file_sha256(p) == h
        ok &= good
        if verbose:
            print(f"{'ok      ' if good else 'MISMATCH'} {f}")
    return ok
