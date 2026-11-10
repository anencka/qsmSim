"""Experiment 2 (Experiment_2.md): the Experiment 1 phantoms at another field strength (7 T), with the
noise anchored to a finished 3T reference run, so every 7T dataset pairs with a 3T one.

  * Same seeds as the reference: identical veins, chi, M0, transceive phase and unit noise. Only R2*
    (x r2s_scale, DMV blood included), B0 and the shim's Hz values change; the truth volumes must hash
    identically to the reference prep.json (checked in the prep job).
  * Noise levels are labeled snr = s3T * k: sigma = (3T reference WM first-echo magnitude) / snr, so at
    k = 1 sigma is exactly the reference sigma at s3T, and k is the intrinsic SNR gain.
  * Jobs: prep -> stage-1 tuning (reference protocol, replicate 0) -> the resweep's λ sweep with the run
    itself as the source (no stage 2, no production chains). V-SHARP and PDF still run on P1 so the
    evaluation mask is the reference's (checked per dataset against the paired 3T record).
  * A resolution the reference lacks (the 7 T 0.8 mm arm, Experiments_v2.md 4): exp2["noise_res"] maps it to a
    reference resolution whose noise it takes (option (a): the same label, sigma_0.8 = sigma_1.0). Its truth has no
    3T hash to match, so the vein network (veins_table.json) must equal the reference's instead; its datasets have
    no 3T partner.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from . import config as C
from .io import res_tag

# geometry and seeding must match the reference run exactly
SAME = ("root_seed", "master_mm", "fine_factor", "tile_core_mm", "tile_halo_mm", "near_mm", "mesh_pad", "resample",
        "b0_dir", "phase_sign", "te1_ms", "transceive_amp", "shim_hz", "mask_erode_mm", "backgrounds",
        "forward", "fine_mm", "v2_core_mm", "v2_halo_mm", "v2_align_mm", "vein_refine", "refine_core_mm",
        "refine_halo_mm", "tissue_sub", "tissues")
VEIN_GEOMETRY_ONLY = ("r2s",)          # VeinParams fields allowed to differ from the reference


def make_config(cfg, out, ref, hpg=False, tfi_reps=(), reps=None):
    """Freeze an exp2 preset against the reference run `ref` (an Experiment 1 output directory)."""
    ref = Path(ref).resolve()
    if not (ref / "config.json").exists():
        raise ValueError(f"{ref}/config.json not found: --ref must be a finished Experiment 1 run")
    rc = C.Config.from_json((ref / "config.json").read_text())
    if rc.resweep or rc.exp2:
        raise ValueError(f"{ref} is a resweep / exp2 output, not an Experiment 1 run")
    if not cfg.exp2:
        raise ValueError(f"config '{cfg.name}' is not an Experiment 2 preset")
    if hpg:
        cfg.dtes_ms = [5.4, 2.4]
        cfg.echo_counts = {**cfg.echo_counts, "2.4": list(rc.echo_counts["2.4"])}
    if tfi_reps:
        cfg.tfi = True                                   # stage 1 also tunes TFI (the sweep's base value)
        cfg.resweep["tfi_reps"] = [int(r) for r in tfi_reps]
    if reps is not None:
        cfg.only_reps = [int(r) for r in reps]
    check_compatible(cfg, rc, ref)
    cfg.exp2["ref"] = os.path.relpath(ref, Path(out).resolve())
    return cfg


def check_compatible(cfg, rc, ref):
    bad = [k for k in SAME if getattr(cfg, k) != getattr(rc, k)]
    va, vb = cfg.veins.__dict__, rc.veins.__dict__
    bad += [f"veins.{k}" for k in va if k not in VEIN_GEOMETRY_ONLY and va[k] != vb[k]]
    if float(rc.b0_tesla) != float(cfg.exp2["ref_b0"]):
        bad.append(f"reference B0 {rc.b0_tesla} T, expected {cfg.exp2['ref_b0']} T")
    if rc.bandwidth_penalty or cfg.bandwidth_penalty:
        bad.append("bandwidth_penalty (sigma anchoring assumes it off in both runs)")
    for p in cfg.active_phantoms():
        mine, theirs = cfg.phantom(p), next((q for q in rc.phantoms if q.name == p), None)
        if theirs is None or mine.__dict__ != theirs.__dict__:
            bad.append(f"phantom {p}")
        elif not (ref / "prep" / p / "prep.json").exists():
            bad.append(f"{ref}/prep/{p}/prep.json missing")
    for r in cfg.active_resolutions():
        if anchor_res(cfg, r) not in [float(x) for x in rc.resolutions]:
            bad.append(f"resolution {r:g} mm (not in the reference, and no noise_res anchor there)")
    if not set(cfg.exp2["s3t"]) <= {float(s) for s in rc.snrs}:
        bad.append(f"3T SNR anchors {cfg.exp2['s3t']} not all in the reference {rc.snrs}")
    for d in cfg.dtes_ms:
        if f"{float(d):g}" not in rc.echo_counts:
            bad.append(f"echo spacing {d} ms not in the reference")
        elif not set(cfg.counts(d)) <= set(rc.counts(d)):
            bad.append(f"echo counts at {d} ms")
    if cfg.n_echo_sim > rc.n_echo_sim:
        bad.append("n_echo_sim")
    if bad:
        raise ValueError(f"the reference run {ref} does not match this Experiment 2 config: {', '.join(bad)}")


def ref_dir(cfg, out):
    return (Path(out) / cfg.exp2["ref"]).resolve()


@lru_cache(maxsize=None)
def _json(path):
    return json.loads(Path(path).read_text())


def ref_prep(cfg, out, ph):
    return _json(str(ref_dir(cfg, out) / "prep" / ph / "prep.json"))


def level(cfg, snr):
    """(s3T, k) of a noise label."""
    s, k = cfg.exp2["levels"][f"{float(snr):g}"]
    return float(s), float(k)


def anchor_res(cfg, res):
    """The reference resolution whose noise `res` takes: itself, or its exp2["noise_res"] anchor."""
    return float((cfg.exp2.get("noise_res") or {}).get(f"{float(res):g}", res))


def snr_reference(cfg, out, ph, res, dte):
    """The 3T reference's mean noiseless first-echo WM magnitude (sigma = this / snr), at the anchor resolution."""
    return ref_prep(cfg, out, ph)["resolutions"][res_tag(anchor_res(cfg, res))][f"wm_first_echo_dte{float(dte):g}"]


def check_truth(cfg, out, ph, hashes):
    """The truth volumes (chi, mask, total field) do not depend on B0: they must be bit-identical. A resolution
    the reference lacks has no hash to match: there the vein network must be the reference's, file for file."""
    want = ref_prep(cfg, out, ph)["hashes"]
    keys = [k for k in hashes if k.endswith("/truth")]
    checked = [k for k in keys if k in want]
    unanchored = [k for k in keys if k not in want]
    diff = [k for k in checked if want[k] != hashes[k]]
    if diff:
        raise RuntimeError(f"{ph}: truth volumes differ from the 3T reference ({', '.join(diff)}): "
                           "the phantoms would not be the same object")
    info = dict(truth_match=True, checked=checked)
    if unanchored:
        mine, theirs = Path(out) / "prep" / ph / "veins_table.json", ref_dir(cfg, out) / "prep" / ph / "veins_table.json"
        if not theirs.exists() or not mine.exists() or mine.read_bytes() != theirs.read_bytes():
            raise RuntimeError(f"{ph}: {', '.join(unanchored)} has no 3T truth to match and the vein network differs "
                               "from the 3T reference's (veins_table.json)")
        info.update(unanchored=unanchored, veins_table_match=True)
    return info


def partner(cfg, out, ph, res, snr, dte, n, rep):
    """The paired 3T reference record (same phantom, res, s3T, protocol, replicate), or None."""
    from .jobs import run_dir
    s3t, _ = level(cfg, snr)
    f = run_dir(ref_dir(cfg, out), ph, res, s3t if s3t != int(s3t) else int(s3t), dte, n, rep) / "metrics.json"
    return _json(str(f)) if f.exists() else None
