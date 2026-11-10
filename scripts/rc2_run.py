"""Run qsmSim's dipole inversions on QSM Reconstruction Challenge 2.0 data and
place them on the official leaderboard (metrics: qsmsim.rc2, a validated
port of the challenge's MATLAB evaluation).

    python scripts/rc2_run.py                      # defaults (Stage-1 analogue) + oracle-tuned (Stage-2 analogue)
    python scripts/rc2_run.py --methods tv medi --inputs fit

Inputs ("--inputs"):
  freq  the provided frequency map (Hz) / (gamma * 7 T); data weights = 1 in the mask
  fit   our multi-echo complex fit of the 4 echoes (echofit.fit_field): field + noise weights
Protocols:
  defaults  qsmSim's default parameters, fixed a priori (no ground truth) - like Stage 1
  tuned     one parameter per method chosen to minimise the combined Sim1/Sim2 NRMSE
            against the ground truth (same value for both sims) - like Stage 2 (at SNR 100)
Writes data/rc2/ours/<protocol>_<input>_<method>_Sim{1,2}.nii.gz and outputs/rc2/report.md.
"""
import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from qsmsim import echofit, inversion, rc2
from qsmsim.pipeline import DEFAULT_PARAMS
from qsmsim.signal import hz_per_ppm
from qsmsim.tfi import tfi

ROOT = Path(__file__).resolve().parent.parent
B0, VOX, BDIR = 7.0, (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)
GRIDS = {"tkd": ("thr", [0.1, 0.15, 0.19, 0.25]),
         "cfl2": ("lam", [3e-3, 1e-2, 3e-2, 1e-1]),
         "tv": ("alpha", [3e-5, 1e-4, 3e-4, 1e-3]),
         "medi": ("alpha", [3e-5, 1e-4, 3e-4, 1e-3]),
         "tfi": ("lam", [3e-4, 1e-3, 3e-3])}
LABEL = {"tkd": "TKD", "cfl2": "CF-L2", "tv": "TV-ADMM", "medi": "MEDI-like", "tfi": "TFI"}


def prepare(sim, inputs):
    m = sim["mask"] > 0
    S = (sim["magnitude"] * np.exp(1j * sim["phase"])).astype(np.complex64)
    fit = echofit.fit_field(S, np.array(rc2.TE_MS) * 1e-3, m, VOX, B0)
    if inputs == "freq":
        field = np.where(m, sim["frequency"] / hz_per_ppm(B0), 0).astype(np.float32)
        weights = m.astype(np.float32)
    else:
        field, weights = fit.field_ppm, fit.weights
    return dict(field=field, weights=weights, mask=m, magnitude=fit.magnitude)


def reconstruct(method, d, value=None):
    p = dict(DEFAULT_PARAMS[method])
    if value is not None:
        p[GRIDS[method][0]] = value
    f, m, w = d["field"], d["mask"], d["weights"]
    if method == "tkd":
        return inversion.tkd(f, m, VOX, BDIR, thr=p["thr"])
    if method == "cfl2":
        return inversion.cfl2(f, m, VOX, BDIR, lam=p["lam"])
    if method == "tv":
        p.pop("alpha")
        return inversion.tv_admm(f, m, VOX, BDIR, weights=w, alpha=value or DEFAULT_PARAMS["tv"]["alpha"], **p)[0]
    if method == "medi":
        a = value or DEFAULT_PARAMS["medi"]["alpha"]
        p.pop("alpha")
        return inversion.medi_like(f, m, VOX, d["magnitude"], BDIR, weights=w, alpha=a, **p)[0]
    if method == "tfi":
        p.pop("data")
        # RC2 has no background sources outside the brain: tight padding, and a reduced
        # iteration budget (the full 25 x 80 takes ~1 h per sim on this 6.9M-voxel grid)
        p.update(pad_factor=1.1, outer=10, cg_iters=40)
        return tfi(f, m, VOX, d["magnitude"], BDIR, weights=w, **p)[0]
    raise ValueError(method)


def score(recons, sims):
    return rc2.combine(*[rc2.evaluate(recons[s], sims[s]) for s in ("Sim1", "Sim2")])


def save(arr, name, ref):
    out = ROOT / "data" / "rc2" / "ours"
    out.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), ref.affine, ref.header), out / f"{name}.nii.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["tkd", "cfl2", "tv", "medi", "tfi"])
    ap.add_argument("--inputs", nargs="+", default=["freq", "fit"])
    ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    sims = {s: rc2.load_sim(s) for s in ("Sim1", "Sim2")}
    ref = nib.load(rc2.ROOT / "Sim1GT" / "Chi.nii.gz")
    results = []
    for inp in a.inputs:
        data = {s: prepare(sims[s], inp) for s in sims}
        log(f"inputs '{inp}' ready")
        for meth in a.methods:
            key, grid = GRIDS[meth]
            default = DEFAULT_PARAMS[meth].get(key)
            protocols = [("defaults", default)]
            if not a.no_tune:
                sweep = []
                for v in grid:
                    rec = {s: reconstruct(meth, data[s], v) for s in sims}
                    sc = score(rec, sims)
                    sweep.append((sc["rmse"], v, rec, sc))
                    log(f"  {inp:4s} {meth:4s} {key}={v:g}: NRMSE {sc['rmse']:.1f}")
                best = min(sweep, key=lambda x: x[0])
                protocols.append(("tuned", best[1]))
            for proto, v in protocols:
                if proto == "tuned":
                    rec, sc = best[2], best[3]
                else:
                    hit = [x for x in (sweep if not a.no_tune else []) if x[1] == v]
                    rec = hit[0][2] if hit else {s: reconstruct(meth, data[s], v) for s in sims}
                    sc = hit[0][3] if hit else score(rec, sims)
                for s in sims:
                    save(rec[s], f"{proto}_{inp}_{meth}_{s}", ref)
                results.append(dict(protocol=proto, inputs=inp, method=meth, param=key, value=v, **sc))
                log(f"{proto:8s} {inp:4s} {LABEL[meth]:9s} {key}={v:g}: NRMSE {sc['rmse']:.1f}  "
                    f"dNRMSE {sc['rmse_detrend']:.1f}  DGM {sc['rmse_detrend_DGM']:.1f}  "
                    f"Blood {sc['rmse_detrend_Blood']:.1f}  CalcStreak {sc['CalcStreak']:.3f}")
    out = ROOT / "outputs" / "rc2"
    out.mkdir(parents=True, exist_ok=True)
    rj = out / "results.json"                     # merge with earlier runs (other methods/inputs)
    if rj.exists():
        new = {(r["protocol"], r["inputs"], r["method"]) for r in results}
        results = [r for r in json.loads(rj.read_text())
                   if (r["protocol"], r["inputs"], r["method"]) not in new] + results
    rj.write_text(json.dumps(results, indent=2, default=float))
    write_report(results, out / "report.md")
    log(f"wrote {out / 'report.md'}")


def write_report(results, path):
    board = [r for r in rc2.leaderboard() if np.isfinite(r["rmse"])]
    nr = np.array(sorted(r["rmse"] for r in board))
    L = ["# qsmSim on QSM Reconstruction Challenge 2.0", "",
         f"Metrics: Python port of the official evaluation (validated against master_stage1_anonymized.csv). "
         f"Leaderboard: {len(board)} Stage 1 submissions; best NRMSE {nr[0]:.1f}, median {np.median(nr):.1f}.", "",
         "| protocol | input | method | param | NRMSE | rank (Stage 1) | dNRMSE | Tissue | Blood | DGM | slope dev. | CalcStreak | Calc moment dev. |",
         "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(results, key=lambda r: (r["protocol"], r["rmse"])):
        rank = int(np.searchsorted(nr, r["rmse"])) + 1
        L.append(f"| {r['protocol']} | {r['inputs']} | {LABEL[r['method']]} | {r['param']}={r['value']:g} | "
                 f"{r['rmse']:.1f} | {rank}/{len(nr) + 1} | {r['rmse_detrend']:.1f} | {r['rmse_detrend_Tissue']:.1f} | "
                 f"{r['rmse_detrend_Blood']:.1f} | {r['rmse_detrend_DGM']:.1f} | {r['DeviationFromLinearSlope']:.3f} | "
                 f"{r['CalcStreak']:.3f} | {r['DeviationFromCalcMoment']:.1f} |")
    L += ["", "Top 5 Stage 1 entries (official):", "", "| NRMSE | algorithm |", "|---:|---|"]
    for r in sorted(board, key=lambda r: r["rmse"])[:5]:
        L.append(f"| {r['rmse']:.1f} | {r.get('Preferred Acronym', '')} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
