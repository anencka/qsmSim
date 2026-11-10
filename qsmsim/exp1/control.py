"""Phase 4 control run (GPU_Plan.md 2.4): forward model v2 against v1 on the same phantoms, seeds and noise.

One control run per field, against the finished v1 run of that field (`src`: Experiment 1 production at 3 T,
Experiment 2 at 7 T). The control run's own config is the v1 config with forward = "v2" (the accepted phase 2
geometry); everything else (resolutions, noise levels, protocols, replicates, chains) is the v1 run's, on P1 and P3.

  cache:P          the v1 noiseless signals: src/cache when intact, else regenerated into <out>/v1regen and checked
                   against src/prep/P/prep.json (resweep.ensure_cache)
  prep:P           the v2 preparation (GPU). At 7 T its truth volumes must equal the 3T control's (--v2-ref) bit for bit
  cmp:P:res        noiseless differences (CPU): truth chi, total field and signals, v2 - v1, by voxel class
  ctl:P:res:snr:dte:rep   (GPU) both datasets with the v1 dataset's sigma and the same unit noise (the v1 source
                   SHA-256 must reproduce), every chain at the v1 λ on both, scored on the voxels both evaluate
                   (common evaluation mask, v2 ROIs, regions, tissue fractions and vein distances for both arms,
                   each against its own truth chi); short MEDI / TV sweeps on the reference protocol

Voxel classes (disjoint, from the v2 fractions and vein distances; also evaluator regions):
  vein      within NEAR_VEIN_MM of a deep medullary vein axis, or a voxel with any labelled-vein fraction ('veins':
            the analytic veins, ICVs) or next to one (26-neighbourhood). The second part was added 2026-10-24: an
            axis-aligned 1.2 mm vein on voxel corners gives 94 % / 9 % partial volumes at 1 mm, both on the pure
            side of PV_PURE, so neither the boundary rule nor the DMV distance saw it (control diagnostic)
  boundary  a boundary voxel (no tissue >= 90 %) or next to one (26-neighbourhood), not vein
  skull     within NEAR_SKULL_MM of the intracranial edge (where v1 and v2 fields differ most, G.3), the rest of it
  agree     everything else: pure tissue away from veins, tissue boundaries and the skull, where v1 and v2 should agree

λ: 3 T, each production dataset's tuned λ (its metrics.json); 7 T, Experiment 2's stage-1 λ (Experiment 2 has no
production chains). Pass condition (plan 2.4), per phantom x resolution x SNR x protocol x chain over the replicates:
the mean paired change v2 - v1 of the agree-class RMS error and bias is within PASS_REL_TOL (5 %) of the v1
agree-class RMS error, or within the v1 replicate 95 % CI half-width when that is larger; and the change of the error
map is concentrated outside the agree class (its share of the squared change against its share of the voxels). The
noise is identical in both arms, so the change is deterministic while the v1 CI shrinks as 1/SNR: the CI alone failed
any change at high SNR (control run 2026-10-23, GPU_Plan.md), hence the relative tolerance, agreed 2026-10-23.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from .. import metrics
from .evaluate import PV_NOT_TISSUE, PV_PURE
from .io import AtomicDir, proto_tag, res_tag, save_nifti_cropped, snr_tag, write_json

DEFAULT_PHANTOMS = ("P1", "P3")
NEAR_VEIN_MM = 3.0
NEAR_SKULL_MM = 5.0
CLASSES = ("agree", "boundary", "skull", "vein")
SWEEP_METHODS = ("medi", "tv")
SWEEP_FACTORS = (0.01, 10 ** -1.5, 0.1, 10 ** -0.5, 1.0, 10 ** 0.5)
ECHOES = (0, 4, 12)                       # noiseless signal differences at these echoes (first, 5th, 13th)
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}     # t(0.975, n-1) by n


# ---------------------------------------------------------------------- config
def make_config(src_cfg, out, src, v2_ref=None, reps=None, snrs=None, phantoms=None, resolutions=None,
                fine_mm=None):
    """Freeze a control config against the finished v1 run `src` (Experiment 1 production or Experiment 2)."""
    src, out = Path(src).resolve(), Path(out).resolve()
    if src_cfg.forward != "v1":
        raise ValueError(f"{src} is not a forward-model v1 run")
    if src_cfg.control:
        raise ValueError(f"{src} is itself a control run")
    if src_cfg.resweep and not src_cfg.exp2:
        raise ValueError(f"{src} is a resweep output: use the production run it swept")
    cfg = copy.deepcopy(src_cfg)
    cfg.name = f"control_{cfg.b0_tesla:g}t"
    cfg.forward = "v2"
    if fine_mm is not None:
        cfg.fine_mm = float(fine_mm)
    cfg.resweep, cfg.lam_scheme, cfg.tuning = None, None, False
    if cfg.exp2:
        if v2_ref is None:
            raise ValueError("a 7 T control needs --v2-ref: the 3T control run, whose v2 truth volumes the 7 T "
                             "preparation must reproduce")
        v2_ref = Path(v2_ref).resolve()
        rc = _load_cfg(v2_ref)
        if not rc.control or rc.b0_tesla != cfg.exp2["ref_b0"]:
            raise ValueError(f"{v2_ref} is not a {cfg.exp2['ref_b0']:g} T control run")
        cfg.exp2 = dict(cfg.exp2, ref=os.path.relpath(v2_ref, out))
    if reps is not None:
        cfg.only_reps = [int(r) for r in reps]
    if snrs is not None:
        cfg.only_snrs = [float(s) for s in snrs]
    # P1 and P3 (plan 2.4; Experiment 2 and the reruns drop P2)
    cfg.only_phantoms = list(phantoms) if phantoms else [p for p in cfg.active_phantoms() if p in DEFAULT_PHANTOMS]
    if resolutions is not None:
        cfg.only_resolutions = [float(r) for r in resolutions]
    cfg.control = dict(src=os.path.relpath(src, out), lam="stage1" if src_cfg.exp2 else "runs",
                       sweep=dict(methods=list(SWEEP_METHODS), factors=[float(f) for f in SWEEP_FACTORS],
                                  protocol=[float(cfg.tune_ref[0]), int(cfg.tune_ref[1])]),
                       maps_rep=0, near_vein_mm=NEAR_VEIN_MM, near_skull_mm=NEAR_SKULL_MM)
    cfg.check_v2()
    return cfg


def _load_cfg(d):
    from .config import Config
    f = Path(d) / "config.json"
    if not f.exists():
        raise ValueError(f"{f} not found")
    return Config.from_json(f.read_text())


def src_dir(cfg, out):
    return (Path(out) / cfg.control["src"]).resolve()


def v1_config(cfg, out):
    """The v1 run's config, its Experiment 2 reference made absolute (a v1 regeneration checks against it)."""
    c = _load_cfg(src_dir(cfg, out))
    if c.exp2:
        c.exp2 = dict(c.exp2, ref=str((src_dir(cfg, out) / c.exp2["ref"]).resolve()))
    for k in ("workers", "mem_gb", "prep_threads", "gpu_lanes", "eval_workers", "gpu_mem_gb"):
        setattr(c, k, getattr(cfg, k))
    return c


def v1_regen(out):
    return Path(out) / "v1regen"


def v1_cache_root(cfg, out, ph):
    r = v1_regen(out)
    return r if (r / "cache" / ph).exists() else src_dir(cfg, out)


def ensure_v1_cache(cfg, out, ph, log=print):
    from .resweep import ensure_cache
    return ensure_cache(v1_config(cfg, out), out, ph, log, src=src_dir(cfg, out), regen=v1_regen(out))


def prepare_v2(cfg, out, ph, log=print):
    from .prepare import prepare_v2 as P2
    if cfg.exp2:
        ref = (Path(out) / cfg.exp2["ref"]).resolve() / "prep" / ph / "prep.json"
        if not ref.exists():
            raise RuntimeError(f"{ref} not found: run the {cfg.exp2['ref_b0']:g} T control's preparation first "
                               "(the 7 T v2 truth is checked against it)")
    return P2(cfg, ph, out, log=log)


# ---------------------------------------------------------------------- v1 records
def v1_record(cfg, out, ph, res, snr, dte, n, rep):
    from .jobs import run_dir
    from .resweep import sweep_dir
    src = src_dir(cfg, out)
    d = (sweep_dir if cfg.exp2 else run_dir)(src, ph, res, snr, dte, n, rep)
    f = d / "metrics.json"
    if not f.exists():
        raise RuntimeError(f"v1 record {f} not found")
    return json.loads(f.read_text())


def v1_lambdas(cfg, out, ph, res, snr, rec):
    if cfg.control["lam"] == "runs":
        return dict(rec["lambdas"])
    from .jobs import stage1_path
    d = json.loads(stage1_path(src_dir(cfg, out), ph, res, snr).read_text())
    return {inv: m["best"] for inv, m in d["methods"].items()}


# ---------------------------------------------------------------------- voxel classes
def class_maps(prep2):
    """The class ingredients of one phantom x resolution (independent of the evaluation mask)."""
    fr = prep2.fractions
    if fr is None:
        raise RuntimeError("the v2 preparation has no tissue fractions")
    tissue = np.stack([v for k, v in fr.items() if k not in PV_NOT_TISSUE])
    pure = tissue.max(0) >= PV_PURE
    edge = ndimage.binary_dilation(~pure, structure=np.ones((3, 3, 3), bool))
    vein = prep2.truth["dmv_dist"] <= NEAR_VEIN_MM
    if "veins" in fr:
        vein = vein | ndimage.binary_dilation(fr["veins"] > 0, structure=np.ones((3, 3, 3), bool))
    dsk = ndimage.distance_transform_edt(fr["intracranial"] >= 1 - 1e-3) * prep2.res
    return dict(edge=edge, vein=vein, skull=dsk < NEAR_SKULL_MM)


def classes(cm, mask):
    """Disjoint voxel classes of `mask` (module docstring)."""
    vein = mask & cm["vein"]
    bnd = mask & cm["edge"] & ~cm["vein"]
    sk = mask & cm["skull"] & ~cm["edge"] & ~cm["vein"]
    return dict(agree=mask & ~(vein | bnd | sk), boundary=bnd, skull=sk, vein=vein)


def _rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if a.size else float("nan")


def change_stats(x1, x2, t1, t2, mask, C):
    """Where the reconstruction changed (v2 - v1) and where its error changed ((x2 - t2) - (x1 - t1)), by class:
    RMS (ppb) and share of the squared change, against the class's share of the voxels."""
    r = lambda a: metrics.reference(a, mask)
    d = r(x2) - r(x1)
    e = d - (r(t2) - r(t1))
    nd, ne, nv = float((d[mask].astype(np.float64) ** 2).sum()), float((e[mask].astype(np.float64) ** 2).sum()), int(mask.sum())
    out = dict(brain=dict(n=nv, change_rms_ppb=1e3 * _rms(d[mask]), error_change_rms_ppb=1e3 * _rms(e[mask])))
    for k, m in C.items():
        if not m.any():
            continue
        out[k] = dict(n=int(m.sum()), voxel_share=float(m.sum() / max(nv, 1)),
                      change_rms_ppb=1e3 * _rms(d[m]), error_change_rms_ppb=1e3 * _rms(e[m]),
                      change_share=float((d[m].astype(np.float64) ** 2).sum() / max(nd, 1e-30)),
                      error_change_share=float((e[m].astype(np.float64) ** 2).sum() / max(ne, 1e-30)))
    return out


# ---------------------------------------------------------------------- noiseless comparison
def cmp_path(out, ph, res):
    return Path(out) / "cmp" / ph / f"{res_tag(res)}.json"


def run_compare(cfg, out, ph, res, log=print):
    """Truth chi, total field and noiseless signals, v2 - v1, by voxel class of the brain both masks hold."""
    from .jobs import Prep
    c1 = v1_config(cfg, out)
    p1 = Prep(c1, src_dir(cfg, out), ph, res, cache_root=v1_cache_root(cfg, out, ph))
    p2 = Prep(cfg, out, ph, res)
    m1, m2 = p1.truth["mask"], p2.truth["mask"]
    if m1.shape != m2.shape:
        raise RuntimeError(f"{ph} {res_tag(res)}: v1 grid {m1.shape} != v2 grid {m2.shape}")
    mask = m1 & m2
    C = classes(class_maps(p2), mask)
    rec = dict(phantom=ph, res=res, b0=cfg.b0_tesla,
               mask=dict(v1=int(m1.sum()), v2=int(m2.sum()), both=int(mask.sum()),
                         dice=float(2 * mask.sum() / max(m1.sum() + m2.sum(), 1))),
               classes={k: int(m.sum()) for k, m in C.items()}, truth={}, field={}, signal={})
    t1, t2 = metrics.reference(p1.truth["chi"], mask), metrics.reference(p2.truth["chi"], mask)
    dt = (t2 - t1).astype(np.float64)
    df = (p2.truth["total_field"] - p1.truth["total_field"]).astype(np.float64)
    for k, m in [("brain", mask), *C.items()]:
        if not m.any():
            continue
        rec["truth"][k] = dict(n=int(m.sum()), mean_ppb=1e3 * float(dt[m].mean()), rms_ppb=1e3 * _rms(dt[m]),
                               p99_abs_ppb=1e3 * float(np.percentile(np.abs(dt[m]), 99)))
        rec["field"][k] = dict(mean_ppb=1e3 * float(df[m].mean()), rms_ppb=1e3 * _rms(df[m]),
                               p99_abs_ppb=1e3 * float(np.percentile(np.abs(df[m]), 99)))
    for dte in cfg.dtes_ms:
        S1, S2 = p1.S0(dte), p2.S0(dte)
        if S1.shape != S2.shape:
            raise RuntimeError(f"{ph} {res_tag(res)} dTE {dte:g}: v1 signal {S1.shape} != v2 {S2.shape}")
        ref = p1.wm_first_echo(dte)             # differences in units of the v1 WM first-echo magnitude (x SNR = sigma)
        sd = dict(wm_first_echo_v1=ref, wm_first_echo_v2=p2.wm_first_echo(dte), echoes={})
        for e in ECHOES:
            if e >= S1.shape[-1]:
                continue
            a, b = np.asarray(S1[..., e]), np.asarray(S2[..., e])
            dc = np.abs(b - a) / ref
            dm = (np.abs(b) - np.abs(a)) / ref
            ok = np.abs(a) > 0.1 * ref
            dp = np.angle(b * np.conj(a))
            ee = {}
            for k, m in [("brain", mask), *C.items()]:
                if not m.any():
                    continue
                mo = m & ok
                ee[k] = dict(complex_median=float(np.median(dc[m])), complex_p99=float(np.percentile(dc[m], 99)),
                             magnitude_rms=_rms(dm[m]),
                             phase_rms_rad=_rms(dp[mo]) if mo.any() else float("nan"),
                             phase_p99_rad=float(np.percentile(np.abs(dp[mo]), 99)) if mo.any() else float("nan"))
            sd["echoes"][str(e)] = ee
        rec["signal"][f"dte{dte:g}"] = sd
    path = cmp_path(out, ph, res)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, rec)
    b = rec["truth"].get("agree", {})
    log(f"cmp {ph} {res_tag(res)}: masks dice {rec['mask']['dice']:.4f}; truth change RMS agree "
        f"{b.get('rms_ppb', float('nan')):.2f} ppb, boundary {rec['truth'].get('boundary', {}).get('rms_ppb', float('nan')):.2f} ppb")
    return dict(dice=rec["mask"]["dice"], classes=rec["classes"])


# ---------------------------------------------------------------------- paired datasets
def ctl_dir(out, ph, res, snr, dte, n, rep):
    return Path(out) / "runs" / ph / res_tag(res) / snr_tag(snr) / proto_tag(dte, n) / f"rep{rep}"


def _r(v):
    return float(f"{v:.4g}")


def _slim(m):
    """Sweep points keep the whole-brain, per-region and summary vein metrics (not ROIs, PV bins or vein bins)."""
    v = m.get("veins") or {}
    return dict({k: m[k] for k in ("rmse", "hfen", "ssim", "deep_gm_slope", "region") if k in m},
                veins={k: v[k] for k in ("auc_core_vs_shell", "moment_ratio_median", "peak_ratio_median") if k in v})


def run_control(cfg, out, ph, res, snr, dte, rep, log=print):
    """Every echo count of one (phantom, res, SNR, dTE, replicate) on both forward models (module docstring)."""
    from . import jobs as J
    c1 = v1_config(cfg, out)
    p1 = J.Prep(c1, src_dir(cfg, out), ph, res, cache_root=v1_cache_root(cfg, out, ph))
    p2 = J.Prep(cfg, out, ph, res)
    cm = class_maps(p2)
    t0 = time.time()
    counts = cfg.counts(dte)
    rec0 = v1_record(cfg, out, ph, res, snr, dte, counts[0], rep)
    sigma = float(rec0["sigma"])
    S1, sha1, _ = J.noisy_signal(c1, p1, dte, snr, rep, sigma=sigma)
    S2, sha2, _ = J.noisy_signal(cfg, p2, dte, snr, rep, sigma=sigma)
    sha_match = sha1 == rec0["source_sha256"]
    if not sha_match:
        log(f"ctl {ph} {res_tag(res)} {snr_tag(snr)} dte{dte:g} rep{rep}: v1 SOURCE SHA-256 DIFFERS from the v1 run")
    t_noise = time.time() - t0
    sw = cfg.control["sweep"]
    ref_proto = (float(sw["protocol"][0]), int(sw["protocol"][1]))
    truth1 = dict(chi=p1.truth["chi"], dmv_dist=p2.truth["dmv_dist"], dmv_r=p2.truth["dmv_r"], voxel_mm=p2.res)
    summary = []
    for n in counts:
        rec = v1_record(cfg, out, ph, res, snr, dte, n, rep)
        lam = v1_lambdas(cfg, out, ph, res, snr, rec)
        t = time.time()
        ds1 = J.Dataset(c1, p1, S1, dte, n, defer_eval=True)
        ds2 = J.Dataset(cfg, p2, S2, dte, n, defer_eval=True)
        t_ds = time.time() - t
        mask = ds1.eval_mask & ds2.eval_mask
        C = classes(cm, mask)
        regions = metrics.regions(p2.truth["labels"], p2.truth["mask"], mask, p2.res) if p2.analytic else None
        for ds, tr in ((ds1, truth1), (ds2, p2.truth)):
            ds.set_evaluator(tr, mask, p2.rois, regions, p2.net, p2.net_path, p2.fractions, extra=C)
        keep = rep == cfg.control["maps_rep"] and (float(dte), int(n)) == ref_proto
        res_m = dict(phantom=ph, res=res, snr=snr, dte=dte, n=n, rep=rep, b0=cfg.b0_tesla, sigma=sigma,
                     source_sha256_v1=sha1, source_sha256_v1_run=rec["source_sha256"],
                     sha_match=sha1 == rec["source_sha256"], source_sha256_v2=sha2, lambdas=lam,
                     eval_mask_voxels=dict(v1=int(ds1.eval_mask.sum()), v2=int(ds2.eval_mask.sum()),
                                           common=int(mask.sum()), v1_run=rec.get("eval_mask_voxels")),
                     classes={k: int(m.sum()) for k, m in C.items()},
                     field=dict(v1=ds1.field_metrics, v2=ds2.field_metrics),
                     bg_info=dict(v1=ds1.bg_info, v2=ds2.bg_info), chains={}, sweeps={})
        with AtomicDir(ctl_dir(out, ph, res, snr, dte, n, rep)) as tmp:
            pend = []
            for bg, inv in ds1.chains():
                name = J.chain_name(bg, inv)
                tt = time.time()
                x1, i1 = ds1.invert(bg, inv, lam.get(inv))
                x2, i2 = ds2.invert(bg, inv, lam.get(inv))
                tr = time.time() - tt
                ch = change_stats(x1, x2, truth1["chi"], p2.truth["chi"], mask, C)
                if keep:
                    save_nifti_cropped(tmp, {f"chi_v1_{name}": x1, f"chi_v2_{name}": x2}, p2.box, p2.res)
                pend.append((name, inv, ch, tr, J._jsonable_info(i1), J._jsonable_info(i2), ds1.score(x1),
                             ds2.score(x2)))
            spend = []
            if (float(dte), int(n)) == ref_proto:
                bg = ds1.ref_background()
                for inv in sw["methods"]:
                    base = lam.get(inv) or ds2.simcfg.method_params(inv)[J.PARAM_KEY[inv]]
                    for f in sw["factors"]:
                        v = _r(base * f)
                        y1, _ = ds1.invert(bg, inv, v)
                        y2, _ = ds2.invert(bg, inv, v)
                        spend.append((inv, v, f, ds1.score(y1), ds2.score(y2)))
            for name, inv, ch, tr, i1, i2, f1, f2 in pend:
                (m1, _, e1), (m2, _, e2) = f1.result(), f2.result()
                res_m["chains"][name] = dict(v1=m1, v2=m2, change=ch, param={J.PARAM_KEY[inv]: lam.get(inv)},
                                             info=dict(v1=i1, v2=i2), seconds=tr, eval_seconds=e1 + e2)
            for inv, v, f, f1, f2 in spend:
                (m1, _, _), (m2, _, _) = f1.result(), f2.result()
                res_m["sweeps"].setdefault(inv, []).append(dict(value=v, factor=f, v1=_slim(m1), v2=_slim(m2)))
            ds1.release()
            ds2.release()
            res_m["seconds"] = dict(noise=t_noise, datasets=t_ds, total=time.time() - t0)
            write_json(tmp / "metrics.json", res_m)
        del ds1, ds2
        d = {k: v["v2"]["rmse"] - v["v1"]["rmse"] for k, v in res_m["chains"].items()}
        summary.append(dict(n=n, sha_match=res_m["sha_match"], d_rmse=d))
        log(f"ctl {ph} {res_tag(res)} {snr_tag(snr)} {proto_tag(dte, n)} rep{rep}: sha {'ok' if res_m['sha_match'] else 'MISMATCH'}; "
            "nRMSE v2-v1 " + " ".join(f"{k}={v:+.2f}" for k, v in d.items()))
    return dict(source_sha256_v1=sha1, sha_match=sha_match, summary=summary, seconds=time.time() - t0)


# ---------------------------------------------------------------------- aggregate
KEY = ("rmse", "hfen", "ssim", "deep_gm_slope",
       *(f"region.{c}.{m}" for c in ("brain", *CLASSES, "dmv_territory", "deep_gm") for m in ("nrmse", "rms_ppb", "bias_ppb")),
       "veins.auc_core_vs_shell", "veins.moment_ratio_median", "veins.peak_ratio_median")
PASS_METRICS = ("region.agree.rms_ppb", "region.agree.bias_ppb")
PASS_SCALE = "region.agree.rms_ppb"        # the tolerance is a share of the v1 agree-class RMS error, for both metrics
PASS_REL_TOL = 0.05


def _get(d, key):
    for k in key.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _ci(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan")), float("nan"), len(v)
    return float(v.mean()), float(T975.get(len(v), 1.96) * v.std(ddof=1) / math.sqrt(len(v))), len(v)


def aggregate(cfg, out, flatten, write, log=print):
    out = Path(out)
    adir = out / "aggregate"
    adir.mkdir(exist_ok=True)
    rows, change, sweeps, manifest = [], [], [], []
    for f in sorted((out / "runs").glob("*/*/*/*/rep*/metrics.json")):
        d = json.loads(f.read_text())
        key = {k: d[k] for k in ("phantom", "res", "snr", "dte", "n", "rep")}
        manifest.append(dict(**key, b0=d["b0"], sigma=d["sigma"], sha_match=d["sha_match"],
                             **{f"mask.{k}": v for k, v in d["eval_mask_voxels"].items()},
                             **{f"class.{k}": v for k, v in d["classes"].items()},
                             **flatten(d["field"], "field."), **flatten(d.get("bg_info", {}), "bg."),
                             **flatten(d["seconds"], "seconds."), path=str(f.parent.relative_to(out))))
        for ch, c in d["chains"].items():
            row = dict(**key, chain=ch)
            for k in KEY:
                a, b = _get(c["v1"], k), _get(c["v2"], k)
                row[f"v1.{k}"], row[f"v2.{k}"] = a, b
                row[f"d.{k}"] = (b - a) if a is not None and b is not None else None
            rows.append(row)
            cr = dict(**key, chain=ch)
            cr.update(flatten(c["change"]))
            change.append(cr)
        for inv, rr in d.get("sweeps", {}).items():
            for r in rr:
                sweeps.append(dict(**key, method=inv, value=r["value"], factor=r["factor"],
                                   **{f"{arm}.{k}": _get(r[arm], k) for arm in ("v1", "v2")
                                      for k in ("rmse", "region.agree.rms_ppb", "veins.auc_core_vs_shell")}))
    cmp = [json.loads(f.read_text()) for f in sorted((out / "cmp").glob("*/*.json"))]
    write(adir / "control.csv", rows)
    write(adir / "change.csv", change)
    write(adir / "sweeps.csv", sweeps)
    write(adir / "manifest.csv", manifest)
    write(adir / "noiseless.csv", [dict(phantom=c["phantom"], res=c["res"], **flatten({k: c[k] for k in
                                        ("mask", "classes", "truth", "field", "signal")})) for c in cmp])
    L = summary(cfg, rows, change, sweeps, manifest, cmp, out)
    (adir / "summary.md").write_text("\n".join(L) + "\n")
    log(f"aggregate (control): {len(manifest)} datasets, {len(rows)} chain pairs, {len(sweeps)} sweep points, "
        f"{len(cmp)} noiseless comparisons -> {adir}")
    return adir


def _fmt(x, p=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}"


def pass_table(rows):
    """Per phantom x res x SNR x protocol x chain over replicates: mean paired change of each PASS_METRICS entry
    against max(v1 replicate CI half-width, PASS_REL_TOL x the mean v1 agree-class RMS error)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["phantom"], r["res"], r["snr"], r["dte"], r["n"], r["chain"]), []).append(r)
    out = []
    for k, rr in sorted(groups.items(), key=lambda t: tuple(str(x) for x in t[0])):
        u = dict(zip(("phantom", "res", "snr", "dte", "n", "chain"), k), reps=len(rr))
        ok = True
        scale, _, _ = _ci([r[f"v1.{PASS_SCALE}"] for r in rr])
        tol = PASS_REL_TOL * scale
        u["tol"] = tol
        for m in PASS_METRICS:
            dm, dci, nn = _ci([r[f"d.{m}"] for r in rr])
            _, v1ci, _ = _ci([r[f"v1.{m}"] for r in rr])
            if not (np.isfinite(dm) and np.isfinite(v1ci) and np.isfinite(tol)):
                p, margin = None, float("nan")                 # fewer than 2 replicates, or no agree voxels
            else:
                margin = max(v1ci, tol)
                p = bool(abs(dm) <= margin)
            u.update({f"{m}.d_mean": dm, f"{m}.d_ci": dci, f"{m}.v1_ci": v1ci, f"{m}.margin": margin, f"{m}.pass": p})
            ok = None if (ok is None or p is None) else (ok and p)
        u["pass"] = ok
        out.append(u)
    return out


def summary(cfg, rows, change, sweeps, manifest, cmp, out):
    c = cfg.control
    L = [f"# Control run at {cfg.b0_tesla:g} T: forward model v2 against v1 (GPU_Plan.md phase 4)", "",
         f"- v1 run: `{c['src']}` (λ from its {'production datasets' if c['lam'] == 'runs' else 'stage-1 tuning'}); "
         f"same seeds, each v1 dataset's sigma, both arms on the GPU",
         f"- datasets: {len(manifest)}; chain pairs: {len(rows)}; sweep points: {len(sweeps)}; "
         f"v1 source SHA-256 reproduced: {sum(1 for m in manifest if m['sha_match'])} of {len(manifest)}",
         f"- voxel classes: vein ≤ {c['near_vein_mm']:g} mm of a DMV axis or at / next to a labelled vein; boundary = boundary voxel or next to one; "
         f"skull < {c['near_skull_mm']:g} mm of the intracranial edge; agree = the rest", ""]
    if cmp:
        L += ["## Noiseless differences, v2 − v1", "",
              "| phantom | res | mask Dice | truth χ RMS agree / boundary / skull / vein (ppb) | total field RMS agree / skull (ppb) "
              "| signal p99 \\|ΔS\\| / WM, first echo, agree / boundary / skull / vein |", "|---|---|---:|---|---|---|"]
        for x in cmp:
            tr, fl = x["truth"], x["field"]
            sg = next(iter(x["signal"].values()))["echoes"].get("0", {})
            L.append(f"| {x['phantom']} | {x['res']:g} | {x['mask']['dice']:.4f} | "
                     + " / ".join(_fmt(tr.get(k, {}).get("rms_ppb")) for k in CLASSES) + " | "
                     + " / ".join(_fmt(fl.get(k, {}).get("rms_ppb")) for k in ("agree", "skull")) + " | "
                     + " / ".join(_fmt(sg.get(k, {}).get("complex_p99"), 4) for k in CLASSES) + " |")
        L.append("")
    if rows:
        L += ["## Paired change at fixed λ (v2 − v1, mean over datasets)", "",
              "| phantom | chain | whole-brain nRMSE (pp) | agree RMS (ppb) | boundary RMS (ppb) | vein RMS (ppb) | skull RMS (ppb) | vein AUC |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        g = {}
        for r in rows:
            g.setdefault((r["phantom"], r["chain"]), []).append(r)
        mean = lambda rr, k: (lambda v: float(np.mean(v)) if v else None)([r[k] for r in rr if r[k] is not None])
        for (ph, ch), rr in sorted(g.items()):
            L.append(f"| {ph} | {ch} | {_fmt(mean(rr, 'd.rmse'))} | {_fmt(mean(rr, 'd.region.agree.rms_ppb'))} | "
                     f"{_fmt(mean(rr, 'd.region.boundary.rms_ppb'))} | {_fmt(mean(rr, 'd.region.vein.rms_ppb'))} | "
                     f"{_fmt(mean(rr, 'd.region.skull.rms_ppb'))} | {_fmt(mean(rr, 'd.veins.auc_core_vs_shell'), 4)} |")
        L.append("")
        P = pass_table(rows)
        out_dir = Path(out) / "aggregate"
        from .aggregate import _write
        _write(out_dir / "pass.csv", P)
        npass = sum(1 for p in P if p["pass"] is True)
        nt = sum(1 for p in P if p["pass"] is not None)
        L += ["## Pass condition (plan 2.4)", "",
              f"Per phantom x resolution x SNR x protocol x chain over the replicates: |mean paired change| of the agree-class "
              f"error ({', '.join(PASS_METRICS)}) within {PASS_REL_TOL:.0%} of the v1 agree-class RMS error, or the v1 "
              f"replicate 95 % CI half-width when larger. "
              f"**{npass} of {nt} testable units pass** ({len(P) - nt} not testable: fewer than 2 replicates or no "
              f"agree voxels).", "",
              "| phantom | SNR | units | testable | pass |", "|---|---|---:|---:|---:|"]
        g2 = {}
        for p in P:
            g2.setdefault((p["phantom"], p["snr"]), []).append(p)
        for (ph, s), pp in sorted(g2.items(), key=lambda t: (t[0][0], float(t[0][1]))):
            L.append(f"| {ph} | {s:g} | {len(pp)} | {sum(1 for p in pp if p['pass'] is not None)} | "
                     f"{sum(1 for p in pp if p['pass'] is True)} |")
        def excess(p):
            return max(abs(p[f"{m}.d_mean"]) / max(p[f"{m}.margin"], 1e-9) for m in PASS_METRICS
                       if np.isfinite(p[f"{m}.d_mean"]))
        fails = [p for p in P if p["pass"] is False]
        if fails:
            g3 = {}
            for p in fails:
                g3[(p["phantom"], p["chain"])] = g3.get((p["phantom"], p["chain"]), 0) + 1
            L += ["", "Failures by phantom and chain: " + ", ".join(
                f"{ph} {ch} {k}" for (ph, ch), k in sorted(g3.items(), key=lambda t: -t[1]))]
            L += ["", "Largest failures (change / margin; margin = max(v1 CI half-width, tolerance)):"]
            L += [f"- {p['phantom']} {p['res']:g} mm SNR {p['snr']:g} dTE {p['dte']:g} n {p['n']} {p['chain']}: "
                  f"agree RMS {_fmt(p['region.agree.rms_ppb.d_mean'], 3)} ± {_fmt(p['region.agree.rms_ppb.d_ci'], 3)} ppb "
                  f"vs {_fmt(p['region.agree.rms_ppb.margin'], 3)}; bias {_fmt(p['region.agree.bias_ppb.d_mean'], 3)} "
                  f"vs {_fmt(p['region.agree.bias_ppb.margin'], 3)}" for p in sorted(fails, key=excess, reverse=True)[:15]]
        L.append("")
    if change:
        L += ["## Where the error changed (share of the squared change of the error map, median over chain pairs)", "",
              "| phantom | class | voxel share | error-change share | change RMS (ppb) |", "|---|---|---:|---:|---:|"]
        for ph in sorted({r["phantom"] for r in change}):
            rr = [r for r in change if r["phantom"] == ph]
            for k in CLASSES:
                vs = [r.get(f"{k}.voxel_share") for r in rr if r.get(f"{k}.voxel_share") is not None]
                es = [r.get(f"{k}.error_change_share") for r in rr if r.get(f"{k}.error_change_share") is not None]
                cr = [r.get(f"{k}.error_change_rms_ppb") for r in rr if r.get(f"{k}.error_change_rms_ppb") is not None]
                if vs:
                    L.append(f"| {ph} | {k} | {np.median(vs):.3f} | {np.median(es):.3f} | {np.median(cr):.2f} |")
        L.append("")
    if sweeps:
        L += ["## λ optimum on the short sweeps (reference protocol, every replicate)", ""]
        g = {}
        for s in sweeps:
            g.setdefault((s["phantom"], s["res"], s["snr"], s["rep"], s["method"]), []).append(s)
        same = {"rmse": 0, "veins.auc_core_vs_shell": 0}
        moved = {"rmse": [], "veins.auc_core_vs_shell": []}
        for k, ss in g.items():
            for m, pick in (("rmse", min), ("veins.auc_core_vs_shell", max)):
                v = [(s[f"v1.{m}"], s[f"v2.{m}"], s["factor"]) for s in ss if s[f"v1.{m}"] is not None]
                if not v:
                    continue
                o1 = pick(v, key=lambda t: t[0])[2]
                o2 = pick(v, key=lambda t: t[1])[2]
                if o1 == o2:
                    same[m] += 1
                else:
                    moved[m].append((k, o1, o2))
        for m in same:
            tot = same[m] + len(moved[m])
            L.append(f"- {m}: optimum unchanged in {same[m]} of {tot} sweeps"
                     + (f"; moved: " + ", ".join(f"{k[0]} {k[1]:g} mm SNR {k[2]:g} rep{k[3]} {k[4]} ×{a:.3g}→×{b:.3g}"
                                                for k, a, b in moved[m][:12]) if moved[m] else ""))
        L.append("")
    return L
