"""Experiment 1 jobs: tuning sweeps (stage 1 / stage 2) and production
reconstructions. One job = one phantom x resolution x SNR (x protocol for
stage 2) or phantom x resolution x SNR x echo spacing x replicate (production).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import dataclasses

from .. import backend as be
from .. import echofit, metrics
from .. import pipeline as pl
from ..veins import VeinNetwork
from . import evalpool, seeds
from .evaluate import Evaluator
from .io import AtomicDir, crop, proto_tag, res_tag, save_nifti_cropped, snr_tag, write_json
from .prepare import tes_for

PARAM_KEY = {"tkd": "thr", "cfl2": "lam", "tv": "alpha", "medi": "alpha", "tfi": "lam"}


# ---------------------------------------------------------------------- inputs
class Prep:
    """Truth bundle + vein network + prep info for one phantom x resolution."""

    def __init__(self, cfg, out, phantom, res, cache_root=None):
        self.cfg, self.out, self.phantom, self.res = cfg, Path(out), phantom, float(res)
        self.cache_root = Path(cache_root) if cache_root else self.out
        pdir = self.out / "prep" / phantom
        self.info = json.loads((pdir / "prep.json").read_text())
        z = np.load(pdir / res_tag(res) / "truth.npz")
        self.truth = {k: z[k] for k in z.files}
        self.truth["voxel_mm"] = float(self.truth["voxel_mm"])
        names = [str(n) for n in self.truth["roi_names"]]
        self.rois = {n: m for n, m in zip(names, self.truth["rois"])}
        self.box = [list(map(int, b)) for b in self.truth["crop"]]
        self.net_path = pdir / "veins.npz"
        self.net = VeinNetwork.from_npz(self.net_path)
        fr = pdir / res_tag(res) / "fractions.npz"                 # forward model v2: tissue fractions per voxel
        if fr.exists():
            z = np.load(fr)
            self.fractions = {str(n): f for n, f in zip(z["names"], z["fractions"])}
        else:
            self.fractions = None
        self.spec = cfg.phantom(phantom)
        self.analytic = self.spec.kind == "analytic"
        vox = (self.res,) * 3
        self.pl_truth = pl.Truth(self.truth["chi"], self.truth["labels"], self.truth["mask"],
                                 self.truth["total_field"], self.truth["local_field"], None, self.rois, vox,
                                 np.asarray(cfg.b0_dir, float))

    def rc2_metrics(self, chi):
        from .. import rc2, rc2_3t
        if not hasattr(self, "_rc2_sim"):
            self._rc2_sim = rc2_3t.sim_for_eval()
        off = self.info["rc2_offset_1mm"]
        rec = rc2_3t.crop_to_rc2(chi, off, self._rc2_sim["chi"].shape).astype(np.float64)
        return {k: float(v) for k, v in rc2.evaluate(rec, self._rc2_sim).items()}

    def wm_first_echo(self, dte):
        return self.info["resolutions"][res_tag(self.res)][f"wm_first_echo_dte{dte:g}"]

    def snr_reference(self, dte):
        """Magnitude that SNR is defined against: this run's WM first echo, or (Experiment 2) the 3T reference's."""
        if self.cfg.exp2:
            from .exp2 import snr_reference
            return snr_reference(self.cfg, self.out, self.phantom, self.res, dte)
        return self.wm_first_echo(dte)

    def S0(self, dte):
        return np.load(self.cache_root / "cache" / self.phantom / res_tag(self.res) / f"S0_dte{dte:g}.npy", mmap_mode="r")


def sigma_for(cfg, prep, dte, snr):
    s = prep.snr_reference(dte) / snr
    if cfg.bandwidth_penalty:
        s *= np.sqrt(cfg.bandwidth_ref_dte_ms / dte)
    return float(s)


def noisy_signal(cfg, prep, dte, snr, rep, sigma=None):
    """13-echo noisy source dataset and its SHA-256 (the data that is discarded). sigma: the noise level to use
    instead of the one the SNR defines (the control run takes each v1 dataset's, control.py)."""
    S0 = prep.S0(dte)
    sigma = np.float32(sigma_for(cfg, prep, dte, snr) if sigma is None else sigma)
    S = np.empty(S0.shape, np.complex64)
    for e in range(S0.shape[-1]):
        S[..., e] = S0[..., e] + sigma * seeds.unit_noise(cfg, prep.spec.seed_key, prep.res, rep, e, S0.shape[:-1])
    return S, seeds.sha256(S), float(sigma)


def sim_config(cfg, prep):
    params = {k: dict(v) for k, v in cfg.recon_params.items()}
    if not prep.analytic:
        params["tfi"] = {**params.get("tfi", {}), **cfg.p3_tfi}
    return pl.SimConfig(name="exp1", voxel_mm=prep.res, b0_tesla=cfg.b0_tesla, phase_sign=cfg.phase_sign,
                        params=params, mask_erode_mm=cfg.mask_erode_mm)


# ------------------------------------------------------------------ one dataset
def _fit_to_host(fit):
    return echofit.FieldFit(*(be.to_host(getattr(fit, f.name)) for f in dataclasses.fields(fit)))


class Dataset:
    """Field fit + local fields + evaluator for one n-echo dataset.

    gpu (default: QSMSIM_BACKEND=gpu): the field fit, background removal and inversions run on the GPU
    (backend.py). self.fit / self.locals / self.masks are host copies either way; the device copies are
    kept for the inversions and every chi map comes back to the host."""

    def __init__(self, cfg, prep, S, dte, n, gpu=None, defer_eval=False):
        self.cfg, self.prep, self.dte, self.n = cfg, prep, dte, n
        self.gpu = be.use_gpu() if gpu is None else bool(gpu)
        tes = tes_for(cfg, dte)[:n]
        mask = prep.truth["mask"]
        vox = (prep.res,) * 3
        t = time.time()
        if self.gpu:
            dmask = be.to_device(mask)
            self._dfit = echofit.fit_field(be.to_device(np.ascontiguousarray(S[..., :n])), tes, dmask, vox,
                                           cfg.b0_tesla, cfg.phase_sign)
            self.fit = _fit_to_host(self._dfit)
            tr = prep.pl_truth
            self._dtruth = dataclasses.replace(tr, mask=dmask, total_field=be.to_device(tr.total_field),
                                               local_field=be.to_device(tr.local_field))
        else:
            self.fit = echofit.fit_field(S[..., :n], tes, mask, vox, cfg.b0_tesla, cfg.phase_sign)
            self._dfit, self._dtruth = self.fit, prep.pl_truth
        half = 0.5 / (cfg.b0_tesla * 42.577478 * dte * 1e-3)        # ppm per half aliasing cycle
        cycles = 0
        if cfg.exp2 or cfg.remove_global_cycles:
            # The unwrap fixes the field only up to a global whole aliasing cycle. A scanner sets its reference
            # frequency on the brain's water, so the true global level is known; the simulation has no such
            # adjustment (P1's mean field is -0.31 ppm, a half cycle at 7 T / 5.4 ms), so remove whole cycles
            # of global offset here. Sub-cycle errors and local cycle slips are left as they are.
            cycles = int(np.rint(np.median((self.fit.field_ppm - prep.truth["total_field"])[mask]) / (2 * half)))
            if cycles:
                self.fit.field_ppm[mask] -= np.float32(cycles * 2 * half)
                if self.gpu:
                    self._dfit.field_ppm[self._dtruth.mask] -= np.float32(cycles * 2 * half)
        ferr = (self.fit.field_ppm - prep.truth["total_field"])[mask]
        self.field_metrics = dict(median_abs_err_ppb=float(np.median(np.abs(ferr)) * 1e3),
                                  p99_abs_err_ppb=float(np.percentile(np.abs(ferr), 99) * 1e3),
                                  median_pred_std_ppb=float(np.median(self.fit.field_std_ppm[mask]) * 1e3),
                                  voxels_cycle_error=int(np.count_nonzero(np.abs(ferr) > half)),
                                  seconds=time.time() - t)
        if cfg.exp2 or cfg.remove_global_cycles:
            self.field_metrics["global_cycles_removed"] = cycles
        self.simcfg = sim_config(cfg, prep)
        self.locals, masks = {}, {}
        self._dlocals, self._dmasks = {}, {}
        self.bg_seconds, self.bg_info = {}, {}
        if prep.analytic:
            for b in cfg.backgrounds:
                t = time.time()
                inf = {}
                loc, msk = pl.remove_background(b, self._dfit, self._dtruth, self.simcfg, info=inf)
                if inf:
                    self.bg_info[b] = inf                # PDF: CG iterations and final residual (GPU_Plan.md phase 3)
                self._dlocals[b], self._dmasks[b] = loc, msk
                self.locals[b], masks[b] = be.to_host(loc), be.to_host(msk)
                self.bg_seconds[b] = time.time() - t
            self.eval_mask = np.logical_and.reduce([masks[b] for b in cfg.backgrounds])
            regions = metrics.regions(prep.truth["labels"], mask, self.eval_mask, prep.res)
        else:
            self.locals["direct"], masks["direct"] = (self.fit.field_ppm * mask).astype(np.float32), mask
            if self.gpu:
                self._dlocals["direct"] = (self._dfit.field_ppm * self._dtruth.mask).astype(np.float32)
                self._dmasks["direct"] = self._dtruth.mask
            else:
                self._dlocals["direct"], self._dmasks["direct"] = self.locals["direct"], mask
            self.eval_mask = mask
            regions = None
        self.masks = masks
        self.regions = regions
        self.pool, self.evaluator, self.eval_key = evalpool.get(), None, None
        if not defer_eval:
            self.set_evaluator(prep.truth, self.eval_mask, prep.rois, regions, prep.net, prep.net_path, prep.fractions)

    def set_evaluator(self, truth, mask, rois, regions, net, net_path, fractions, extra=None):
        """Score this dataset's chi maps against `truth` on `mask` (the control run sets a shared mask and voxel
        classes for both forward models). GPU jobs score in a CPU pool (evalpool.py, QSMSIM_EVAL_WORKERS)."""
        if self.pool is None:
            self.evaluator = Evaluator(truth, mask, rois, regions, net, self.cfg.b0_dir, fractions=fractions,
                                       extra_regions=extra)
        else:
            self.eval_key = self.pool.register(truth, mask, rois, regions, net_path, self.cfg.b0_dir, fractions, extra)

    def score(self, chi):
        """Future of (metrics, per-vein table, evaluation seconds): in the CPU pool, or computed now."""
        if self.pool is None:
            t = time.time()
            m, per = self.evaluator(chi)
            return evalpool.done((m, per, time.time() - t))
        return self.pool.submit(self.eval_key, chi)

    def release(self):
        """All of this dataset's maps are scored: drop its evaluator inputs from the pool's scratch."""
        if self.pool is not None:
            self.pool.release(self.eval_key)

    def invert(self, bg, inv, value=None, extra=None):
        over = dict(extra or {})
        if value is not None:
            over[PARAM_KEY[inv]] = value
        if inv == "tfi":
            chi, _, info = pl.run_single_step("tfi", self._dfit, self._dtruth, self.simcfg, None, over)
            return be.to_host(chi), dict(outer=info["outer"], cg=int(sum(info["cg"])))
        chi, info = pl.invert(inv, self._dlocals[bg], self._dmasks[bg], self._dfit, self._dtruth, self.simcfg, over)
        return be.to_host(chi), info

    def chains(self):
        bgs = self.cfg.backgrounds if self.prep.analytic else ["direct"]
        out = [(b, i) for b in bgs for i in self.cfg.inversions]
        if self.cfg.tfi:
            out.append(("total", "tfi"))
        return out

    def ref_background(self):
        return "oracle" if self.prep.analytic else "direct"


def chain_name(bg, inv):
    return "tfi" if inv == "tfi" else f"{bg}_{inv}"


# ---------------------------------------------------------------------- tuning
def stage1_path(out, ph, res, snr):
    return Path(out) / "tune" / "stage1" / ph / res_tag(res) / f"{snr_tag(snr)}.json"


def stage2_path(out, ph, res, snr, dte, n):
    return Path(out) / "tune" / "stage2" / ph / res_tag(res) / snr_tag(snr) / f"{proto_tag(dte, n)}.json"


def _sweep(ds, inv, values, sweep_dir, cfg, done=None):
    """Evaluate one inversion at each value (on the oracle/direct field, or TFI)."""
    rows = list(done or [])
    have = {r["value"] for r in rows}
    rows += _sweep_many(ds, inv, sweep_dir, cfg)([v for v in values if v not in have])
    rows.sort(key=lambda r: r["value"])
    return rows


def _sweep_many(ds, inv, sweep_dir, cfg):
    """[v, ...] -> sweep rows in order (and their maps when kept). The inversions run one after another; the maps
    are scored as they come (in the CPU pool on a GPU job, so the next inversion does not wait)."""
    bg = ds.ref_background()
    # sweeps cap TFI's outer iterations (they only rank lambda) - never above the production budget
    extra = ({"outer": min(cfg.sweep_tfi_outer, ds.simcfg.method_params("tfi")["outer"])}
             if inv == "tfi" else None)
    keep = cfg.keep_sweep_maps and sweep_dir is not None

    def many(values):
        pend = []
        for v in values:
            t = time.time()
            chi, info = ds.invert(bg, inv, v, extra)
            tr = time.time() - t
            if keep:
                save_nifti_cropped(sweep_dir, {f"chi_{chain_name(bg, inv)}_{PARAM_KEY[inv]}{v:g}": chi},
                                   ds.prep.box, ds.prep.res)
            pend.append((v, info, tr, ds.score(chi)))
        rows = []
        for v, info, tr, fut in pend:
            m, per, te = fut.result()
            rows.append(dict(value=v, rmse=m["rmse"], metrics=m, info=_jsonable_info(info), seconds=tr + te))
            if keep and per is not None:
                np.savez_compressed(sweep_dir / f"veins_{chain_name(bg, inv)}_{PARAM_KEY[inv]}{v:g}.npz", **per)
        return rows
    return many


def _sweep_edge(ds, inv, values, sweep_dir, cfg):
    """Stage 1 under a λ scheme: the grid, then the edge rule on the whole-brain nRMSE (lamgrid.py); extensions on
    the lattice of the grid's first value, up to edge_cap_decades beyond the grid (TKD: the scheme's limits)."""
    from . import lamgrid as LG
    sc = cfg.lam_scheme
    if inv == "tkd":
        anchor, lim = None, LG.limits(sc, "tkd", None)
    else:
        c = 10 ** sc["edge_cap_decades"]
        anchor, lim = values[0], (min(values) / c * (1 - 1e-3), max(values) * c * (1 + 1e-3))
    return LG.sweep(_sweep_many(ds, inv, sweep_dir, cfg), sc, inv, anchor, values, lim, ["rmse"])


def _best(rows):
    return min(rows, key=lambda r: r["rmse"])["value"]


def run_tune(cfg, out, ph, res, snr, stage, dte=None, n=None, log=print):
    """stage 1: full grids on the reference protocol; stage 2: narrow grids around
    the stage-1 optimum (with one edge extension) on another protocol. Rep 0."""
    prep = Prep(cfg, out, ph, res)
    if stage == 1:
        dte, n = cfg.tune_ref
        path = stage1_path(out, ph, res, snr)
    else:
        path = stage2_path(out, ph, res, snr, dte, n)
        s1 = json.loads(stage1_path(out, ph, res, snr).read_text())
    S, sha, sigma = noisy_signal(cfg, prep, dte, snr, 0)
    ds = Dataset(cfg, prep, S, dte, n)
    del S
    sweep_dir = path.with_suffix("").parent / (path.stem + "_maps")
    result = dict(phantom=ph, res=res, snr=snr, dte=dte, n=n, stage=stage, source_sha256=sha, sigma=sigma,
                  field=ds.field_metrics, methods={})
    with AtomicDir(sweep_dir) as tmp:
        for inv in cfg.inversions + (["tfi"] if cfg.tfi else []):
            t = time.time()
            edge = None
            if stage == 1:
                if cfg.lam_scheme:
                    rows, edge = _sweep_edge(ds, inv, cfg.full_grids[inv], tmp, cfg)
                else:
                    rows = _sweep(ds, inv, cfg.full_grids[inv], tmp, cfg)
                stage1_value = None
            else:
                b1 = s1["methods"][inv]["best"]
                stage1_value = b1
                if inv == "tkd":
                    grid = list(cfg.full_grids["tkd"])
                else:
                    grid = [_r(b1 / cfg.narrow_step), b1, _r(b1 * cfg.narrow_step)]
                rows = _sweep(ds, inv, grid, tmp, cfg)
                if inv != "tkd":                                     # one edge extension
                    best = _best(rows)
                    if best == min(grid):
                        rows = _sweep(ds, inv, [_r(best / cfg.narrow_step)], tmp, cfg, rows)
                    elif best == max(grid):
                        rows = _sweep(ds, inv, [_r(best * cfg.narrow_step)], tmp, cfg, rows)
            best = _best(rows)
            result["methods"][inv] = dict(param=PARAM_KEY[inv], best=best, stage1_value=stage1_value,
                                          at_grid_edge=best in (rows[0]["value"], rows[-1]["value"]),
                                          rows=rows, seconds=time.time() - t, **({"edge": edge} if edge else {}))
            log(f"tune s{stage} {ph} {res_tag(res)} {snr_tag(snr)} {proto_tag(dte, n)} {inv}: best "
                f"{PARAM_KEY[inv]}={best:g} RMSE {min(r['rmse'] for r in rows):.1f}%")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    return result


def _r(v):
    return float(f"{v:.4g}")


def lambdas(cfg, out, ph, res, snr, dte, n):
    """Regularization per inversion for one protocol (None = pipeline defaults)."""
    if not cfg.tuning:
        return {}
    if (float(dte), int(n)) == (float(cfg.tune_ref[0]), int(cfg.tune_ref[1])):
        d = json.loads(stage1_path(out, ph, res, snr).read_text())
    else:
        d = json.loads(stage2_path(out, ph, res, snr, dte, n).read_text())
    return {inv: m["best"] for inv, m in d["methods"].items()}


# ------------------------------------------------------------------ production
def run_dir(out, ph, res, snr, dte, n, rep):
    return Path(out) / "runs" / ph / res_tag(res) / snr_tag(snr) / proto_tag(dte, n) / f"rep{rep}"


def run_production(cfg, out, ph, res, snr, dte, rep, log=print):
    """All echo counts of one (phantom, res, SNR, dTE, replicate): every chain."""
    prep = Prep(cfg, out, ph, res)
    t0 = time.time()
    S, sha, sigma = noisy_signal(cfg, prep, dte, snr, rep)
    t_noise = time.time() - t0
    summary = []
    for n in cfg.counts(dte):
        lam = lambdas(cfg, out, ph, res, snr, dte, n)
        t = time.time()
        ds = Dataset(cfg, prep, S, dte, n)
        t_prep_ds = time.time() - t
        final = run_dir(out, ph, res, snr, dte, n, rep)
        res_m = dict(phantom=ph, res=res, snr=snr, dte=dte, n=n, rep=rep, source_sha256=sha, sigma=sigma,
                     lambdas=lam, field=ds.field_metrics, bg_seconds=ds.bg_seconds, bg_info=ds.bg_info,
                     eval_mask_voxels=int(ds.eval_mask.sum()), chains={},
                     seed_keys=dict(noise=[prep.spec.seed_key, seeds.res_code(res), rep]), crop=prep.box)
        per_vein = {}
        with AtomicDir(final) as tmp:
            pend = []
            for bg, inv in ds.chains():
                t = time.time()
                chi, info = ds.invert(bg, inv, lam.get(inv))
                tr = time.time() - t
                name = chain_name(bg, inv)
                save_nifti_cropped(tmp, {f"chi_{name}": chi}, prep.box, prep.res)
                rc2 = (prep.rc2_metrics(chi) if not prep.analytic and abs(prep.res - 1.0) < 1e-9   # official RC2
                       else None)                                                                    # metrics, RC2 grid
                pend.append((name, inv, info, tr, rc2, ds.score(chi)))
            for name, inv, info, tr, rc2, fut in pend:
                m, per, te = fut.result()
                if rc2 is not None:
                    m["rc2"] = rc2
                m.update(seconds=tr, eval_seconds=te, info=_jsonable_info(info),
                         param={PARAM_KEY[inv]: lam.get(inv)})
                res_m["chains"][name] = m
                if per is not None:
                    for k, v in per.items():
                        per_vein[f"{name}/{k}"] = v
            ds.release()
            save_nifti_cropped(tmp, {"field": ds.fit.field_ppm, "field_std": ds.fit.field_std_ppm}, prep.box, prep.res)
            if per_vein:
                np.savez_compressed(tmp / "veins.npz", **per_vein)
            res_m["seconds"] = dict(noise=t_noise, dataset=t_prep_ds, total=time.time() - t0)
            write_json(tmp / "metrics.json", res_m)
        summary.append(dict(n=n, rmse={k: v["rmse"] for k, v in res_m["chains"].items()}))
        log(f"run {ph} {res_tag(res)} {snr_tag(snr)} {proto_tag(dte, n)} rep{rep}: "
            + " ".join(f"{k}={v['rmse']:.0f}" for k, v in res_m["chains"].items()))
    return dict(source_sha256=sha, summary=summary, seconds=time.time() - t0)


def _jsonable_info(info):
    return {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in (info or {}).items()}
