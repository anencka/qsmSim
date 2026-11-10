"""Deterministic seed tree (numpy SeedSequence). Every random draw in Experiment 1
comes from rng(cfg, component, *keys), so any dataset can be regenerated exactly.

Keys:
  veins / phi0  (seed_key)                    P1 and P2 share seed_key -> identical veins and phi0
  texture       (phantom index)
  noise         (seed_key, res code, replicate, echo index)
                - no SNR key: the same unit draws are scaled by sigma at every SNR level
                - no echo-spacing key: echo k gets the same draws at 5.4 and 2.4 ms
                - P1 and P2 share noise (same grids), so P1 vs P2 is paired as well
"""
from __future__ import annotations

import hashlib

import numpy as np

COMPONENTS = {"veins": 1, "phi0": 2, "texture": 3, "noise": 4}


def seq(cfg, component, *keys):
    return np.random.SeedSequence(cfg.root_seed, spawn_key=(COMPONENTS[component], *[int(k) for k in keys]))


def rng(cfg, component, *keys):
    return np.random.default_rng(seq(cfg, component, *keys))


def int_seed(cfg, component, *keys):
    return int(seq(cfg, component, *keys).generate_state(1)[0])


def res_code(res):
    return int(round(res * 100))


def unit_noise(cfg, seed_key, res, rep, echo, shape):
    """Complex standard-normal draws (each part N(0,1)) for one echo, complex64."""
    g = rng(cfg, "noise", seed_key, res_code(res), rep, echo)
    re = g.standard_normal(shape, dtype=np.float32)
    im = g.standard_normal(shape, dtype=np.float32)
    return (re + 1j * im).astype(np.complex64)


def sha256(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()
