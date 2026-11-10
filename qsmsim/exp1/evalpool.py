"""CPU evaluation pool for GPU jobs (GPU_Plan.md, phase 0 decisions and phase 3).

On the GPU an inversion takes seconds while scoring its χ map (exp1/evaluate.py) takes ~15 s of irregular CPU work,
so a GPU job hands each map to a small pool of CPU processes and goes on to the next inversion. Each dataset's
evaluator inputs (truth, evaluation mask, ROIs, regions, tissue fractions, vein network path) are written once to a
scratch file; a worker builds the Evaluator from it on first use and keeps the last two. Results are identical to
scoring in the job process (same numpy code; the workers run single-threaded BLAS like the jobs).

QSMSIM_EVAL_WORKERS=<n> (set by the runner for GPU jobs) turns the pool on; without it every map is scored in the job
process, as on the CPU.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

_POOL = None
_CACHE = OrderedDict()


def workers():
    return int(os.environ.get("QSMSIM_EVAL_WORKERS", "0") or 0)


def get(scratch_root=None):
    """The job's pool (created on first use), or None when QSMSIM_EVAL_WORKERS is 0 / unset."""
    global _POOL
    if _POOL is None and workers() > 0:
        _POOL = Pool(workers(), scratch_root)
    return _POOL


def close():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


def done(value):
    f = Future()
    f.set_result(value)
    return f


class Pool:
    def __init__(self, n, scratch_root=None):
        self.n = n
        self.ex = ProcessPoolExecutor(n, mp_context=get_context("spawn"), initializer=_init)
        self.dir = Path(tempfile.mkdtemp(prefix="qsmsim_eval_", dir=scratch_root))
        self._k = 0

    def register(self, truth, eval_mask, rois, regions, net_path, b0, fractions, extra=None):
        """Write one dataset's evaluator inputs; returns its key (a path)."""
        self._k += 1
        path = self.dir / f"ds{self._k}.npz"
        rn = list(rois)
        gn = list(regions or {})
        fn = list(fractions or {})
        xn = list(extra or {})
        np.savez(path, chi=truth["chi"], dmv_dist=truth["dmv_dist"], dmv_r=truth["dmv_r"],
                 voxel_mm=float(truth["voxel_mm"]), eval_mask=eval_mask,
                 roi_names=np.array(rn), rois=np.stack([rois[k] for k in rn]) if rn else np.zeros(0),
                 region_names=np.array(gn), regions=np.stack([regions[k] for k in gn]) if gn else np.zeros(0),
                 frac_names=np.array(fn), fractions=np.stack([fractions[k] for k in fn]) if fn else np.zeros(0),
                 extra_names=np.array(xn), extra=np.stack([extra[k] for k in xn]) if xn else np.zeros(0),
                 net=str(net_path), b0=np.asarray(b0, float), has_regions=regions is not None,
                 has_fractions=fractions is not None)
        return str(path)

    def release(self, key):
        """The dataset is finished: drop its scratch file (workers drop their copy when it leaves their cache)."""
        try:
            os.remove(key)
        except OSError:
            pass

    def submit(self, key, chi):
        return self.ex.submit(_score, key, np.ascontiguousarray(chi))

    def close(self):
        self.ex.shutdown(wait=True)
        shutil.rmtree(self.dir, ignore_errors=True)


def _init():
    os.environ["QSMSIM_BACKEND"] = "cpu"


def _evaluator(key):
    from ..veins import VeinNetwork
    from .evaluate import Evaluator
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    z = np.load(key)
    truth = dict(chi=z["chi"], dmv_dist=z["dmv_dist"], dmv_r=z["dmv_r"], voxel_mm=float(z["voxel_mm"]))
    rois = {str(k): m for k, m in zip(z["roi_names"], z["rois"])}
    regions = {str(k): m for k, m in zip(z["region_names"], z["regions"])} if bool(z["has_regions"]) else None
    fractions = ({str(k): f for k, f in zip(z["frac_names"], z["fractions"])}
                 if bool(z["has_fractions"]) else None)
    extra = {str(k): m for k, m in zip(z["extra_names"], z["extra"])} if "extra_names" in z.files else None
    ev = Evaluator(truth, z["eval_mask"], rois, regions, VeinNetwork.from_npz(str(z["net"])),
                   tuple(z["b0"]), fractions=fractions, extra_regions=extra or None)
    _CACHE[key] = ev
    while len(_CACHE) > 2:
        _CACHE.popitem(last=False)
    return ev


def _score(key, chi):
    t = time.time()
    m, per = _evaluator(key)(chi)
    return m, per, time.time() - t
