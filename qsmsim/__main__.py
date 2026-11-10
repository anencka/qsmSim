"""Command line: python -m qsmsim [options]   (run from projects/qsmSim)

Examples
  python -m qsmsim                                  # default 1.5 mm, 3 T, SNR 100
  python -m qsmsim --quick                          # 3 mm smoke test (~30 s)
  python -m qsmsim --name snr40 --snr 40 --sweep    # noisier, oracle-tuned regularization
  python -m qsmsim --name tilt --tilt 20 0          # head pitched 20 deg relative to B0
  python -m qsmsim --single                         # skip TFI (single-step total field inversion)
  python -m qsmsim --oversample 2 --pad 1.5         # forward-simulate at 0.75 mm (no inverse crime)
  python -m qsmsim --config my.json                 # any SimConfig field, incl. per-method params
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from .pipeline import BG_METHODS, INV_METHODS, SINGLE_STEP, SimConfig, run


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m qsmsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="JSON file with SimConfig fields (CLI flags override it)")
    p.add_argument("--name")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "outputs")
    p.add_argument("--voxel", type=float, dest="voxel_mm")
    p.add_argument("--oversample", type=int)
    p.add_argument("--pad", type=float, dest="pad_factor")
    p.add_argument("--b0", type=float, dest="b0_tesla")
    p.add_argument("--tilt", type=float, nargs=2, dest="tilt_deg", metavar=("PITCH", "ROLL"))
    p.add_argument("--te1", type=float, dest="te1_ms")
    p.add_argument("--dte", type=float, dest="dte_ms")
    p.add_argument("--necho", type=int, dest="n_echoes")
    p.add_argument("--snr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--texture", type=float, dest="texture_ppm", help="smooth chi texture amplitude (ppm, std)")
    p.add_argument("--texture-sigma", type=float, dest="texture_sigma_mm", help="texture spatial scale (mm)")
    p.add_argument("--bg", nargs="+", choices=BG_METHODS, dest="background")
    p.add_argument("--inv", nargs="+", choices=INV_METHODS, dest="inversion")
    p.add_argument("--single", nargs="*", choices=SINGLE_STEP, dest="single_step",
                   help="single-step methods on the total field (default: tfi; pass no value to skip)")
    p.add_argument("--reference", choices=["mean", "csf"])
    p.add_argument("--sweep", action="store_true", default=None, help="oracle-tune each inversion's regularization")
    p.add_argument("--no-nifti", action="store_false", dest="save_nifti", default=None)
    p.add_argument("--quick", action="store_true", help="3 mm grid, fewer ADMM iterations")
    a = p.parse_args(argv)

    cfg = SimConfig()
    if a.config:
        known = {f.name for f in fields(SimConfig)}
        data = json.loads(a.config.read_text())
        bad = set(data) - known
        if bad:
            p.error(f"unknown config keys: {sorted(bad)}")
        for k, v in data.items():
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    if a.quick:
        cfg.name, cfg.voxel_mm = "quick", 3.0
        for m in ("tv", "medi"):
            cfg.params.setdefault(m, {}).update(maxiter=60)
        cfg.params.setdefault("tfi", {}).update(outer=10, cg_iters=40, pad_factor=1.25)
        cfg.params.setdefault("vsharp", {}).update(max_radius_mm=9.0)
    for f in fields(SimConfig):
        v = getattr(a, f.name, None)
        if v is not None:
            setattr(cfg, f.name, tuple(v) if isinstance(v, list) else v)

    outdir = a.out / cfg.name
    print(f"output: {outdir}")
    run(cfg, outdir)
    print(f"done - see {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
