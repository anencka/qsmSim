"""Figures for the RC2 comparison: leaderboard placement and side-by-side maps.

    python scripts/rc2_figures.py      (after scripts/rc2_run.py)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from qsmsim import plotting as pl
from qsmsim import rc2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "rc2"
LABEL = {"tkd": "TKD", "cfl2": "CF-L2", "tv": "TV-ADMM", "medi": "MEDI-like", "tfi": "TFI"}
WINNER = "5e0YBJKZv1"          # wL1L2TV, best Stage 1 NRMSE


def placement(results):
    board = [r for r in rc2.leaderboard() if np.isfinite(r["rmse"]) and r["rmse"] < 150]
    ours = [r for r in results if r["protocol"] == "defaults"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = np.array([r["rmse"] for r in board])
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(x))
    ax.plot(x, jitter, "o", color=pl.MUTED, alpha=0.55, markersize=6, markeredgecolor=pl.SURFACE,
            markeredgewidth=0.8, zorder=2)
    order = {"fit": 0, "freq": 1}
    for i, r in enumerate(sorted(ours, key=lambda r: (order[r["inputs"]], r["rmse"]))):
        col = pl.SERIES[0] if r["inputs"] == "fit" else pl.SERIES[1]
        mk = "D" if r["inputs"] == "fit" else "s"
        yy = 0.45 + 0.2 * (i % 3) if r["inputs"] == "fit" else -0.45 - 0.2 * (i % 3)
        ax.plot([r["rmse"]], [yy], marker=mk, color=col, markersize=8, markeredgecolor=pl.SURFACE,
                markeredgewidth=1.2, zorder=3)
        ax.plot([r["rmse"], r["rmse"]], [0, yy], color=col, lw=1, alpha=0.6, zorder=1)
        ax.annotate(LABEL[r["method"]], (r["rmse"], yy), textcoords="offset points",
                    xytext=(6, -3), color=pl.INK2, fontsize=8)
    ax.axhline(0, color=pl.GRID, lw=0.8, zorder=0)
    ax.set_yticks([0.65, 0, -0.65], ["qsmSim, multi-echo fit", "98 Stage 1 entries", "qsmSim, provided freq. map"])
    ax.tick_params(axis="y", length=0, labelcolor=pl.INK2)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("NRMSE (%) - official metric, mean of Sim1 and Sim2 - lower is better")
    ax.set_title("QSM Challenge 2.0 Stage 1: qsmSim at its a-priori default parameters vs all submissions",
                 loc="left", color=pl.INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_leaderboard.png", dpi=120)
    plt.close(fig)


def maps(results):
    sim = rc2.load_sim("Sim2")
    m = sim["mask"] > 0
    t = np.where(m, sim["chi"] - sim["chi"][m].mean(), np.nan)
    best = min((r for r in results if r["protocol"] == "defaults" and r["inputs"] == "fit"), key=lambda r: r["rmse"])
    load = lambda p: np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=np.float64)
    ours = load(ROOT / "data" / "rc2" / "ours" / f"defaults_fit_{best['method']}_Sim2.nii.gz")
    cfl2 = load(ROOT / "data" / "rc2" / "ours" / "tuned_fit_cfl2_Sim2.nii.gz")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from rc2_validate_metrics import index_submissions, load_bytes
    idx = index_submissions(ROOT / "data" / "rc2" / "submissions" / "stage1")
    win = load_bytes(idx[f"{WINNER}_Sim2_Step1.nii"]())
    ref = lambda x: np.where(m, x - x[m].mean(), np.nan)
    cols = [("ground truth", t), (f"qsmSim {LABEL[best['method']]} (defaults)", ref(ours)),
            ("wL1L2TV (Stage 1 winner)", ref(win)), ("qsmSim CF-L2 (tuned)", ref(cfl2))]
    calc = np.argwhere(sim["seg"] == 16).mean(0).astype(int)
    dgm = np.argwhere(sim["seg"] == 2).mean(0).astype(int)          # globus pallidus
    rows = [("axial, deep GM", lambda v: v[:, :, dgm[2]].T), ("axial, calcification", lambda v: v[:, :, calc[2]].T)]
    fig, axs = plt.subplots(4, 4, figsize=(13, 13.5), constrained_layout=True)
    for r, (rname, sl) in enumerate(rows):
        for c, (cname, v) in enumerate(cols):
            im = pl._img(axs[2 * r, c], sl(v), "gray", -0.1, 0.2, cname if r == 0 else None)
            if c == 0:
                axs[2 * r, 0].text(-0.05, 0.5, rname, rotation=90, va="center", ha="right",
                                   transform=axs[2 * r, 0].transAxes, color=pl.INK, fontsize=10)
                axs[2 * r + 1, 0].set_axis_off()
                continue
            imd = pl._img(axs[2 * r + 1, c], sl(v - t), pl.DIVERGING, -0.1, 0.1, "error" if r == 0 and c == 1 else None)
        pl._cbar(fig, im, axs[2 * r, :].tolist(), "chi (ppm)")
        pl._cbar(fig, imd, axs[2 * r + 1, 1:].tolist(), "error (ppm)")
    fig.suptitle("QSM Challenge 2.0, Sim2 (mean-referenced in the evaluation mask)", color=pl.INK, fontsize=12)
    fig.savefig(OUT / "fig_maps_sim2.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    res = json.loads((OUT / "results.json").read_text())
    placement(res)
    maps(res)
    print("wrote", OUT / "fig_leaderboard.png", OUT / "fig_maps_sim2.png")
