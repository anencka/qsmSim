"""Reproducibility checks (Experiment_1.md, plan section 4).

1. cache integrity: every noiseless signal in out/cache matches its hash in prep.json
2. noise determinism: for a sample of finished production datasets, regenerate the
   noisy 13-echo source from cache + seeds and compare its SHA-256 with the one
   recorded when it was reconstructed
3. --full: re-run the whole preparation of each phantom into a scratch directory
   and compare every truth / noiseless-signal hash. Bit-identity is expected on
   the same image and CPU class; otherwise the maximum relative difference is
   reported and must be < 1e-6.
Writes out/verify/verify_<timestamp>.json.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import seeds
from .io import res_tag, write_json

REL_TOL = 1e-6


def verify(cfg, out, full=False, sample=6, phantoms=None, log=print):
    out = Path(out)
    report = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), cache=[], noise=[], full=[])
    ok = True
    # 1. cache integrity
    for ph in cfg.active_phantoms():
        pj = out / "prep" / ph / "prep.json"
        if not pj.exists():
            continue
        info = json.loads(pj.read_text())
        for key, h in info["hashes"].items():
            if "/S0_" not in key:
                continue
            f = out / "cache" / ph / (key + ".npy")
            if not f.exists():
                report["cache"].append(dict(phantom=ph, key=key, status="missing"))
                continue
            good = seeds.sha256(np.load(f)) == h
            ok &= good
            report["cache"].append(dict(phantom=ph, key=key, status="ok" if good else "MISMATCH"))
    # 2. noise determinism on finished datasets
    from .jobs import Prep, noisy_signal
    metr = sorted((out / "runs").glob("*/*/*/*/rep*/metrics.json"))
    rng = np.random.default_rng(0)
    pick = [metr[i] for i in sorted(rng.choice(len(metr), min(sample, len(metr)), replace=False))] if metr else []
    preps = {}
    for m in pick:
        d = json.loads(m.read_text())
        key = (d["phantom"], d["res"])
        if key not in preps:
            preps[key] = Prep(cfg, out, *key)
        if not (out / "cache" / d["phantom"] / res_tag(d["res"]) / f"S0_dte{d['dte']:g}.npy").exists():
            report["noise"].append(dict(dataset=str(m.parent.relative_to(out)), status="cache missing"))
            continue
        _, sha, _ = noisy_signal(cfg, preps[key], d["dte"], d["snr"], d["rep"])
        good = sha == d["source_sha256"]
        ok &= good
        report["noise"].append(dict(dataset=str(m.parent.relative_to(out)), status="ok" if good else "MISMATCH"))
    # 3. full regeneration
    if full:
        from .prepare import prepare
        for ph in phantoms or cfg.active_phantoms():
            ref = out / "prep" / ph / "prep.json"
            if not ref.exists():
                continue
            scratch = out / "verify" / f"regen_{ph}"
            if scratch.exists():
                shutil.rmtree(scratch)
            info = prepare(cfg, ph, scratch, log=lambda m: None)
            want = json.loads(ref.read_text())["hashes"]
            for key, h in want.items():
                got = info["hashes"].get(key)
                entry = dict(phantom=ph, key=key, exact=got == h)
                if got != h and "/S0_" in key:
                    a = np.load(out / "cache" / ph / (key + ".npy"))
                    b = np.load(scratch / "cache" / ph / (key + ".npy"))
                    rel = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30))
                    entry["max_rel_diff"] = rel
                    entry["within_tol"] = rel < REL_TOL
                    ok &= rel < REL_TOL
                elif got != h:
                    ok = False
                report["full"].append(entry)
            shutil.rmtree(scratch)
    report["ok"] = bool(ok)
    vdir = out / "verify"
    vdir.mkdir(parents=True, exist_ok=True)
    write_json(vdir / f"verify_{time.strftime('%Y%m%d_%H%M%S')}.json", report)
    n = lambda k, st: sum(1 for r in report[k] if r.get("status") == st)
    log(f"verify: cache {n('cache', 'ok')}/{len(report['cache'])} ok; noise {n('noise', 'ok')}/{len(report['noise'])} ok"
        + (f"; full regeneration {sum(r['exact'] for r in report['full'])}/{len(report['full'])} bit-identical, "
           f"{sum(r.get('within_tol', r['exact']) for r in report['full'])}/{len(report['full'])} within tolerance"
           if full else "") + f" -> {'OK' if ok else 'FAILED'}")
    return ok
