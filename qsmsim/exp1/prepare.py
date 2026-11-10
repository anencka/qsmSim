"""Preparation stage: one phantom on the master grid -> truth bundles and noiseless
multi-echo signals at every acquisition resolution.

Writes (under out/):
  prep/<P>/veins.npz, veins_table.json, prep.json (hashes, timings, DMV stats)
  prep/<P>/res<r>/truth.npz (+ NIfTI of chi_true / masks / DMV ROI, cropped to the brain box)
  cache/<P>/res<r>/S0_dte<d>.npy   noiseless 13-echo signal (transient, regenerable)
Forward model v2 (cfg.forward == "v2", qsmsim/forward2.py) writes the same files, plus
  prep/<P>/res<r>/fractions.npz   tissue fractions per acquisition voxel (label groups, 'dmv' intravascular)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .. import dipole as dp
from .. import finegrid as fg
from .. import phantom as ph
from .. import rc2_3t
from .. import signal as sg
from . import config as C
from .. import veins as vn
from . import seeds
from .io import crop_box, res_tag, save_nifti_cropped

DMV_MAX_DIST = 6.0          # mm; DMV distance maps are capped here


def tes_for(cfg, dte):
    return (cfg.te1_ms + dte * np.arange(cfg.n_echo_sim)) * 1e-3


def build_master(cfg, spec, log):
    cfg.check_tissues()
    h = cfg.master_mm
    seed_key = spec.seed_key
    if spec.kind == "analytic":
        tex_seed = seeds.int_seed(cfg, "texture", [p.name for p in cfg.phantoms].index(spec.name))
        lut = ph.tissue_lut(cfg.tissues, cfg.b0_tesla, cfg.r2s_scale) if cfg.tissues != "v1" else None
        P = ph.build_phantom(spec.fov_mm, h, 1, spec.texture_ppm, spec.texture_sigma_mm, seed=tex_seed, lut=lut)
        grid, labels = P.grid, P.labels
        chi, m0, r2s = P.chi, P.m0, P.r2s
        if lut is None and cfg.r2s_scale != 1.0:          # another field: R2* linear in B0 (Experiment 2 v1)
            r2s = (r2s * np.float32(cfg.r2s_scale)).astype(np.float32)
        del P
        support = np.isin(labels, ph.INTRACRANIAL)
        g1 = ph.Grid(tuple(int(round(f)) for f in spec.fov_mm), (1.0,) * 3)
        lab1 = ph.build_labels(g1)
        wm1, vent1, brain1 = lab1 == ph.L.WM, ph.ventricle_mask(g1), np.isin(lab1, ph.INTRACRANIAL)
        chi_out = float(ph.TISSUES[ph.L.AIR][0])
        shim = (sg.smooth_poly_field(grid, cfg.shim_hz) / np.float32(sg.hz_per_ppm(cfg.shim_b0_tesla or cfg.b0_tesla))
                if spec.background_field and cfg.shim_hz else None)
        if not spec.background_field:
            raise NotImplementedError("analytic phantoms always carry their air/bone background field")
    elif spec.kind == "rc2":
        d = rc2_3t.build(h, spec.fov_mm, **rc2_r2s(cfg))
        grid, labels, chi, m0, r2s, support = d["grid"], d["labels"], d["chi"], d["m0"], d["r2s"], d["mask"]
        g1 = d["grid1"]
        wm1 = (d["seg1"] == rc2_3t.WM) & d["mask1"]
        vent1 = rc2_3t.lateral_ventricles(d["seg1"], d["mask1"])
        brain1 = d["mask1"]
        chi_out, shim = 0.0, None
        M_extra = dict(rc2_offset_1mm=[int(v) for v in d["rc2_offset_1mm"]])
    else:
        raise ValueError(spec.kind)
    phi0 = sg.transceive_phase(grid, seeds.rng(cfg, "phi0", seed_key), cfg.transceive_amp)
    net = vn.generate(wm1, vent1, g1.axes, 1.0, cfg.veins, seeds.rng(cfg, "veins", seed_key),
                      brain=brain1, b0=cfg.b0_dir)
    log(f"{spec.name}: master grid {grid.shape} @ {h} mm; veins {len(net)} "
        f"({net.kinds.count('dmv')} DMV, {net.kinds.count('collector')} collector)")
    M = fg.Master(grid, labels, chi, m0, r2s, support, phi0, shim, chi_out)
    M.extra = M_extra if spec.kind == "rc2" else {}
    return M, net


def rc2_r2s(cfg):
    """rc2_3t.build arguments for the configured field (the RC2 magnitudes are 7 T: at 7 T the scale is 1)."""
    if cfg.tissues == "sourced":
        b0 = float(cfg.b0_tesla)
        return dict(r2s_scale=C.RC2_SOURCED_SCALE[b0], r2s_max=C.RC2_SOURCED_MAX[b0])
    if cfg.r2s_scale == 1.0:
        return {}
    sc = 3.0 / 7.0 * cfg.r2s_scale
    return dict(r2s_scale=1.0 if abs(sc - 1) < 1e-9 else sc, r2s_max=150.0 * cfg.r2s_scale)


def dmv_maps(net, axes, voxel_mm):
    """Distance (mm) to the nearest vein axis, its radius and vein id, per voxel (capped)."""
    shape = tuple(len(a) for a in axes)
    dist = np.full(shape, np.float32(99.0))
    vid = np.full(shape, -1, np.int16)
    rad = np.zeros(shape, np.float32)
    if len(net) == 0:
        return dist, vid, rad
    xs, rs, _, ids = net.samples(0.1)
    tree = cKDTree(xs)
    lo, hi = xs.min(0) - DMV_MAX_DIST, xs.max(0) + DMV_MAX_DIST
    sl = [np.flatnonzero((a >= l) & (a <= u)) for a, l, u in zip(axes, lo, hi)]
    box = tuple(slice(s[0], s[-1] + 1) for s in sl)
    pts = np.stack(np.meshgrid(*[axes[d][sl[d]] for d in range(3)], indexing="ij"), -1).reshape(-1, 3)
    d, j = tree.query(pts, distance_upper_bound=DMV_MAX_DIST)
    ok = np.isfinite(d)
    bshape = tuple(len(s) for s in sl)
    dd = np.full(len(pts), 99.0, np.float32); dd[ok] = d[ok]
    ii = np.full(len(pts), -1, np.int16); ii[ok] = ids[j[ok]]
    rr = np.zeros(len(pts), np.float32); rr[ok] = rs[j[ok]]
    dist[box], vid[box], rad[box] = dd.reshape(bshape), ii.reshape(bshape), rr.reshape(bshape)
    return dist, vid, rad


def roi_masks(spec, labels_master, factor):
    if spec.kind == "analytic":
        return ph.roi_masks(labels_master, factor)
    out = {}
    for name, lab in rc2_3t.LABELS.items():
        frac = ph.block_mean((labels_master == lab).astype(np.float32), factor)
        out[name] = frac >= 0.9
    return out


def prepare(cfg, name, out, log=print):
    if cfg.forward == "v2":
        return prepare_v2(cfg, name, out, log)
    if cfg.forward != "v1":
        raise ValueError(f"unknown forward model {cfg.forward!r}")
    out = Path(out)
    spec = cfg.phantom(name)
    t0 = time.time()
    T = {}
    M, net = build_master(cfg, spec, log)
    T["master"] = time.time() - t0
    pdir = out / "prep" / name
    pdir.mkdir(parents=True, exist_ok=True)
    net.to_npz(pdir / "veins.npz")
    (pdir / "veins_table.json").write_text(json.dumps(net.table(cfg.b0_dir)))

    t = time.time()
    dchi, nz, dmv_info = fg.build_dmv(M, net, cfg.fine_factor, cfg.tile_core_mm, cfg.tile_halo_mm, cfg.b0_dir,
                                      cfg.near_mm, log)
    T["dmv_tiles"] = time.time() - t
    t = time.time()
    mesh, chi_tot = fg.mesh_field(M, dchi, cfg.b0_dir, cfg.mesh_pad)
    del dchi
    fg.finalize(nz, mesh)
    T["mesh"] = time.time() - t
    log(f"{name}: mesh field + sub-sample fields ({T['mesh']:.0f}s)")

    hashes, res_info = {}, {}
    resolutions = cfg.active_resolutions()
    factors = [cfg.factor(r) for r in resolutions]
    for r, f in zip(resolutions, factors):
        t = time.time()
        rdir = pdir / res_tag(r)
        rdir.mkdir(exist_ok=True)
        vox = (r,) * 3
        chi_r = ph.block_mean(chi_tot, f).astype(np.float32)
        c = f // 2
        labels_r = M.labels[c::f, c::f, c::f]
        sup_r = ph.block_mean(M.support.astype(np.float32), f) >= 0.5
        if spec.kind == "analytic":
            mask_r = ph.brain_mask(sup_r, r, cfg.mask_erode_mm)
        else:
            mask_r = sup_r
        total_r = ph.block_mean(mesh, f).astype(np.float32)
        local_r = (dp.forward_field(np.where(mask_r, chi_r, 0), vox, cfg.b0_dir, 2.0) * mask_r).astype(np.float32)
        rois = roi_masks(spec, M.labels, f)
        grid_r = ph.Grid(chi_r.shape, vox)
        dist, vid, rad = dmv_maps(net, grid_r.axes, r)
        box = crop_box(mask_r)
        names = sorted(rois)
        np.savez_compressed(rdir / "truth.npz", chi=chi_r, labels=labels_r, mask=mask_r, total_field=total_r,
                            local_field=local_r, rois=np.stack([rois[n] for n in names]), roi_names=np.array(names),
                            dmv_dist=dist, dmv_id=vid, dmv_r=rad, crop=np.array(box), voxel_mm=r)
        save_nifti_cropped(rdir / "nifti", dict(chi_true=chi_r, mask=mask_r.astype(np.uint8), labels=labels_r,
                                                dmv_dist=dist, dmv_territory=(mask_r & (dist <= 5)).astype(np.uint8)),
                           box, r)
        hashes[f"{res_tag(r)}/truth"] = seeds.sha256(chi_r, mask_r, total_r)
        res_info[res_tag(r)] = dict(shape=list(chi_r.shape), mask_voxels=int(mask_r.sum()), crop=box,
                                    dmv_territory_voxels=int((mask_r & (dist <= 5)).sum()))
        T[f"truth_{res_tag(r)}"] = time.time() - t
    del chi_tot
    exp2_info = None
    if cfg.exp2:                                          # same phantom as the 3T reference, bit for bit
        from .exp2 import check_truth
        exp2_info = check_truth(cfg, out, name, hashes)
        log(f"{name}: truth volumes identical to the 3T reference ({len(exp2_info['checked'])} resolutions)"
            + (f"; {', '.join(exp2_info['unanchored'])}: no 3T truth, vein network identical" if exp2_info.get("unanchored") else ""))

    cdir = out / "cache" / name
    for dte in cfg.dtes_ms:
        t = time.time()
        tes = tes_for(cfg, dte)
        sig = fg.synthesize(M, mesh, nz, net.params, tes, factors, cfg.b0_tesla, cfg.phase_sign, cfg.resample,
                            log=lambda m: None)
        for r, f in zip(resolutions, factors):
            S0 = sig.pop(f)
            (cdir / res_tag(r)).mkdir(parents=True, exist_ok=True)
            np.save(cdir / res_tag(r) / f"S0_dte{dte:g}.npy", S0)
            hashes[f"{res_tag(r)}/S0_dte{dte:g}"] = seeds.sha256(S0)
            # SNR reference: mean noiseless first-echo magnitude in white matter
            tr = np.load(pdir / res_tag(r) / "truth.npz")
            wm = tr["rois"][list(tr["roi_names"]).index("white_matter")] & tr["mask"]
            res_info[res_tag(r)][f"wm_first_echo_dte{dte:g}"] = float(np.abs(S0[..., 0])[wm].mean())
        T[f"signal_dte{dte:g}"] = time.time() - t
        log(f"{name}: noiseless signal dTE {dte} ms ({T[f'signal_dte{dte:g}']:.0f}s)")
    info = dict(phantom=name, master_shape=list(M.grid.shape), master_mm=cfg.master_mm, dmv=dmv_info, **M.extra,
                veins=dict(n=len(net), dmv=net.kinds.count("dmv"), collector=net.kinds.count("collector")),
                resolutions=res_info, hashes=hashes, timings_s=T, total_s=time.time() - t0)
    if exp2_info:
        info["exp2"] = dict(exp2_info, b0_tesla=cfg.b0_tesla, r2s_scale=cfg.r2s_scale, dmv_r2s=net.params.r2s)
    (pdir / "prep.json").write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------------
# forward model v2 (GPU_Plan.md, phase 2)
# ---------------------------------------------------------------------------

def fraction_groups(spec):
    """Label groups whose fractions v2 records per acquisition voxel (ROI names as in roi_masks)."""
    if spec.kind == "analytic":
        g = {k: [int(v) for v in labs] for k, labs in ph.ROIS.items()}
        g["intracranial"] = [int(v) for v in ph.INTRACRANIAL]
    else:
        g = {k: [int(v)] for k, v in rc2_3t.LABELS.items()}
        g["intracranial"] = []                            # the brain mask (source support)
    return g


def v2_model(cfg, spec, M, net, log=print):
    """forward2.Model for one phantom: geometry from the config, tissue source, vein store."""
    from .. import forward2 as f2
    cfg.check_v2()
    F = cfg.fine_per_master()
    cells = lambda mm: int(round(mm / cfg.fine_mm))
    geom = f2.Geom(M.grid.shape, cfg.master_mm, F, core=cells(cfg.v2_core_mm), halo=cells(cfg.v2_halo_mm))
    groups = fraction_groups(spec)
    if spec.kind == "analytic" and cfg.tissue_sub:
        tex = None
        if spec.texture_ppm > 0:
            tex_seed = seeds.int_seed(cfg, "texture", [p.name for p in cfg.phantoms].index(spec.name))
            tex = ph.texture_field(M.grid.shape, cfg.master_mm, spec.texture_ppm, spec.texture_sigma_mm, tex_seed)
        src = f2.AnalyticSource(tex, M.phi0, cfg.r2s_scale,
                                 lut=ph.tissue_lut(cfg.tissues, cfg.b0_tesla, cfg.r2s_scale) if cfg.tissues != "v1" else None)
        src.groups = groups
    elif spec.kind == "rc2" and cfg.tissue_sub:
        kw = rc2_r2s(cfg)
        d = rc2_3t.build(1.0, spec.fov_mm, **kw)
        vols = dict(seg=d["seg1"], mask=d["mask1"], chi=d["chi1"], m0=d["m01"], r2s=d["r2s1"])
        src = f2.SharpenedSource(vols, 1.0, M.phi0, groups, r2s_max=kw.get("r2s_max", 150.0), log=log)
    else:
        src = f2.MasterSource(M, groups)
    R = int(cfg.vein_refine)
    store = f2.VeinStore(net, geom, R, log=log)
    return f2.Model(geom, src, store, net.params, acq=[cfg.fine_per_voxel(r) for r in cfg.active_resolutions()],
                    b0_dir=tuple(cfg.b0_dir), refine_core=cells(cfg.refine_core_mm) if R > 1 else 0,
                    refine_halo=cells(cfg.refine_halo_mm) if R > 1 else 0)


class _EchoSplit:
    """Signal sink for one acquisition grid: echo e of the combined echo list goes to its spacing's array."""

    def __init__(self, parts):
        self.parts = parts
        self.where = [(i, e) for i, a in enumerate(parts) for e in range(a.shape[-1])]

    def __setitem__(self, key, val):
        i, e = self.where[key[-1]]
        self.parts[i][key[:-1] + (e,)] = val


def prepare_v2(cfg, name, out, log=print):
    from .. import backend as be
    from .. import forward2 as f2
    out = Path(out)
    spec = cfg.phantom(name)
    t0 = time.time()
    T = {}
    M, net = build_master(cfg, spec, log)
    T["master"] = time.time() - t0
    pdir = out / "prep" / name
    pdir.mkdir(parents=True, exist_ok=True)
    net.to_npz(pdir / "veins.npz")
    (pdir / "veins_table.json").write_text(json.dumps(net.table(cfg.b0_dir)))
    t = time.time()
    model = v2_model(cfg, spec, M, net, log)
    T["veins"] = time.time() - t
    xp = be.cupy() if be.use_gpu() else np
    if xp is not np:
        be.setup()
    log(f"{name}: v2 on the {'GPU' if xp is not np else 'CPU'}: {len(model.geom.tiles())} tiles of "
        f"{cfg.v2_core_mm:g} + 2 x {cfg.v2_halo_mm:g} mm at {cfg.fine_mm:g} mm, veins x{model.R}")
    chi_m, truth = model.mesh_pass(xp, log)
    T["mesh_pass"] = model.stats["mesh_pass_s"]
    t = time.time()
    mesh = f2.mesh_field(chi_m, cfg.master_mm, cfg.b0_dir, cfg.mesh_pad, M.chi_outside, M.shim_ppm)
    T["mesh"] = time.time() - t
    del chi_m

    resolutions = cfg.active_resolutions()
    cdir = out / "cache" / name
    tes_all, sinks, files = [], {}, {}
    for r, n in zip(resolutions, model.acq):
        (cdir / res_tag(r)).mkdir(parents=True, exist_ok=True)
        parts = []
        for dte in cfg.dtes_ms:
            p = cdir / res_tag(r) / f"S0_dte{dte:g}.npy"
            parts.append(np.lib.format.open_memmap(p, mode="w+", dtype=np.complex64,
                                                   shape=model.acq_shape(n) + (cfg.n_echo_sim,)))
            files[(r, dte)] = p
        sinks[n] = _EchoSplit(parts)
    for dte in cfg.dtes_ms:
        tes_all += list(tes_for(cfg, dte))
    _, fld = model.signal_pass(mesh, tes_all, cfg.b0_tesla, cfg.phase_sign, sinks=sinks, xp=xp, log=log)
    T["signal_pass"] = model.stats["signal_pass_s"]
    for s in sinks.values():
        for a in s.parts:
            a.flush()
    del sinks

    hashes, res_info = {}, {}
    for r, n in zip(resolutions, model.acq):
        t = time.time()
        rdir = pdir / res_tag(r)
        rdir.mkdir(exist_ok=True)
        vox = (r,) * 3
        tr = truth.pop(n)
        chi_r, labels_r, frac = tr["chi"], tr["labels"], tr["frac"]
        sup_r = frac["intracranial"] >= 0.5
        mask_r = ph.brain_mask(sup_r, r, cfg.mask_erode_mm) if spec.kind == "analytic" else sup_r
        total_r = fld[n]
        local_r = (dp.forward_field(np.where(mask_r, chi_r, 0), vox, cfg.b0_dir, 2.0) * mask_r).astype(np.float32)
        names = sorted(k for k in frac if k not in ("intracranial", "dmv"))
        rois = {k: frac[k] >= 0.9 for k in names}
        dist, vid, rad = dmv_maps(net, ph.Grid(chi_r.shape, vox).axes, r)
        box = crop_box(mask_r)
        np.savez_compressed(rdir / "truth.npz", chi=chi_r, labels=labels_r, mask=mask_r, total_field=total_r,
                            local_field=local_r, rois=np.stack([rois[k] for k in names]), roi_names=np.array(names),
                            dmv_dist=dist, dmv_id=vid, dmv_r=rad, crop=np.array(box), voxel_mm=r)
        fnames = sorted(frac)
        np.savez_compressed(rdir / "fractions.npz", fractions=np.stack([frac[k] for k in fnames]),
                            names=np.array(fnames), voxel_mm=r)
        save_nifti_cropped(rdir / "nifti", dict(chi_true=chi_r, mask=mask_r.astype(np.uint8), labels=labels_r,
                                                dmv_dist=dist, dmv_territory=(mask_r & (dist <= 5)).astype(np.uint8)),
                           box, r)
        hashes[f"{res_tag(r)}/truth"] = seeds.sha256(chi_r, mask_r, total_r)
        res_info[res_tag(r)] = dict(shape=list(chi_r.shape), mask_voxels=int(mask_r.sum()), crop=box,
                                    dmv_territory_voxels=int((mask_r & (dist <= 5)).sum()),
                                    boundary_voxels=int((mask_r & (np.max(np.stack([frac[k] for k in names]), 0)
                                                                    < 0.9)).sum()))
        wm = rois["white_matter"] & mask_r
        for dte in cfg.dtes_ms:
            S0 = np.load(files[(r, dte)], mmap_mode="r")
            hashes[f"{res_tag(r)}/S0_dte{dte:g}"] = seeds.sha256(np.asarray(S0))
            res_info[res_tag(r)][f"wm_first_echo_dte{dte:g}"] = float(np.abs(S0[..., 0])[wm].mean())
            del S0
        T[f"truth_{res_tag(r)}"] = time.time() - t
    exp2_info = None
    if cfg.exp2:
        from .exp2 import check_truth
        exp2_info = check_truth(cfg, out, name, hashes)
        log(f"{name}: truth volumes identical to the 3T reference ({len(exp2_info['checked'])} resolutions)"
            + (f"; {', '.join(exp2_info['unanchored'])}: no 3T truth, vein network identical" if exp2_info.get("unanchored") else ""))
    g = model.geom
    info = dict(phantom=name, forward="v2", master_shape=list(M.grid.shape), master_mm=cfg.master_mm, **M.extra,
                v2=dict(fine_mm=cfg.fine_mm, fine_shape=list(g.shape_f), tiles=len(g.tiles()), core_mm=cfg.v2_core_mm,
                        halo_mm=cfg.v2_halo_mm, vein_refine=model.R, refined_subtiles=model.stats.get("refined_subtiles"),
                        tissue=type(model.source).__name__, vein_volume_mm3=model.veins.volume_mm3,
                        backend="gpu" if xp is not np else "cpu", **({"device": be.device_info()} if xp is not np else {})),
                veins=dict(n=len(net), dmv=net.kinds.count("dmv"), collector=net.kinds.count("collector")),
                resolutions=res_info, hashes=hashes, timings_s=T, total_s=time.time() - t0)
    if exp2_info:
        info["exp2"] = dict(exp2_info, b0_tesla=cfg.b0_tesla, r2s_scale=cfg.r2s_scale, dmv_r2s=net.params.r2s)
    (pdir / "prep.json").write_text(json.dumps(info, indent=2))
    log(f"{name}: v2 preparation done ({info['total_s']:.0f}s)")
    return info
