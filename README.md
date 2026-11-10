# qsmSim

Numerical simulation of quantitative susceptibility mapping (QSM), from phantom to reconstruction, with a known
ground truth at every step. qsmSim builds head phantoms with sub-voxel deep medullary veins, simulates a
multi-echo gradient-echo acquisition, runs the standard reconstruction chains, and scores each susceptibility
map against the truth. It is built to answer "how does QSM performance change if I change this?" for field
strength, voxel size, echo times, SNR, background removal, inversion and regularization.

```
phantom: analytic head (P1, P2) or the QSM Challenge 2.0 head (P3), plus a deep medullary vein network
   └─► forward model: chi -> field (dipole kernel, sub-voxel veins) -> multi-echo GRE signal with T2* decay,
       intravoxel dephasing, background and shim fields, complex Gaussian noise
         └─► field map: echo-difference init -> masked least-squares unwrap -> Gauss-Newton complex fit
               └─► background removal: oracle | SHARP | V-SHARP | PDF | LBV
                     └─► dipole inversion: TKD | CF-L2 | TV-ADMM | MEDI-like
               └─► or one step on the total field: TFI (preconditioned total field inversion)
                     └─► scoring: nRMSE, HFEN, SSIM, regional errors, deep-GM slope, vein detectability (AUC, d'),
                         per-diameter vein CNR and chi recovery, line-spread width
```

## How it was made and checked

qsmSim was written with [Claude Code](https://claude.com/claude-code) using Claude Opus 5.5, under the direction
and review of the author, who chose the questions, the models and the validation. It has been checked by
extensive unit testing (about 80 tests of the physics, each method, the scoring and the GPU backend) and by
comparison against existing open QSM code and data: a cross-check against
[QSM.jl](https://github.com/kamesy/QSM.jl) on identical inputs (agreement to floating-point precision where the
methods and FFT grids match, within 4 RMSE points for every whole chain), and a port of the QSM Reconstruction
Challenge 2.0 evaluation that reproduces the published leaderboard for 77 of 78 scorable submissions. The CPU and
GPU backends agree to within 0.2 % on every metric. Details and the commands to reproduce each check are in
[docs/VALIDATION.md](docs/VALIDATION.md).

## Install

Python 3.11 or later on Linux (macOS works for the CPU backend).

```bash
git clone https://github.com/anencka/qsmSim && cd qsmSim
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"                 # CPU backend: numpy, scipy, nibabel, matplotlib
pip install -e ".[test,gpu-cuda13]"      # + GPU backend: CuPy with CUDA 13 wheels (NVIDIA driver >= 580)
pip install -e ".[test,gpu-cuda12]"      #   or CUDA 12 wheels for older drivers
pytest                                   # ~80 tests, 1-2 min; GPU tests skip without a CUDA device
```

Or with Docker (no local Python; the GPU image needs the NVIDIA container toolkit):

```bash
docker/qsmsim.sh build cpu               # image qsmsim:cpu
docker/qsmsim.sh build gpu               # image qsmsim:gpu (CuPy 14.2, CUDA 13.2 libraries)
docker/qsmsim.sh test [--gpu]
docker/qsmsim.sh help
```

Exact library versions are pinned in `docker/requirements-exp1.lock` (CPU) and `docker/requirements-gpu.lock`.

**Challenge data (phantom P3 only).** P3 is built from the QSM Reconstruction Challenge 2.0 Sim1 head (CC-BY 4.0;
Marques et al., MRM 2021; Zenodo 10.5281/zenodo.4559541), which is not redistributed here.
`qsmsim-fetch-rc2 --p3` (or `docker/qsmsim.sh fetch-rc2`) downloads the six files it needs (~150 MB) and verifies
their checksums, into `data/rc2/` in a checkout, `~/.cache/qsmsim/rc2` otherwise, or `$QSMSIM_RC2_DIR`.

## Quick start: one demo per knob

`examples/knobs/` has one small spec per parameter (toy preset: 2 mm voxels, a 20-vein network). On an 8-core CPU
each takes 1-2 minutes, except `forward_model` (~11 minutes, for its v2 arm). `examples/demo.sh` runs a demo and
draws its plots:

```bash
examples/demo.sh list                    # the knobs
examples/demo.sh snr                     # -> runs/demo/snr/{results.csv, summary.csv, plot_*.png}
examples/demo.sh echo_trains lambda      # several
examples/demo.sh all                     # every demo except phantom_p3 (~30 min on 8 CPU cores)
docker/qsmsim.sh demo snr                # the same in Docker ([--gpu] for the GPU image)
```

| knob | demo | spec keys |
|---|---|---|
| **Phantom** | | |
| head model: analytic (P1), textured (P2), challenge head (P3) | `phantom`, `phantom_p3` | `phantoms` |
| vein network: diameters, count, length, blood chi and R2* | `veins` | `[[variants]]` with `config.veins` |
| tissue relaxation table: v1 (3 T scaled) or literature-sourced per field | `tissue_model` | `config.tissues` |
| **Acquisition** | | |
| field strength | `field_strength` | `b0_tesla` |
| head orientation to B0 | `field_direction` | `config.b0_dir` |
| voxel size | `voxel_size` | `resolution_mm` |
| first echo time | `te1` | `te1_ms` |
| echo spacing and number of echoes | `echo_trains` | `protocols` |
| SNR and noise replicates | `snr` | `snr`, `replicates` |
| noise defined against a reference field, per-field SNR gain | `snr_reference` | `[noise] reference_b0`, `gain` |
| readout bandwidth penalty for short echo spacing | `bandwidth` | `[noise] bandwidth_penalty` |
| forward model: v1 (two-scale) or v2 (every voxel at 0.1 mm) | `forward_model` | `config.forward` |
| **Reconstruction** | | |
| background removal | `background` | `backgrounds` |
| dipole inversion, or TFI | `inversion` | `inversions` |
| regularization strength | `lambda` | `lambda` |
| solver settings (iterations, kernel radii, ...) | `iterations` | `config.recon_params` |
| compute backend | `cpu_vs_gpu.toml` | `backend` |

Every list in a spec is a sweep dimension and the run covers every combination. `[[variants]]` makes any other
model setting a sweep dimension: each variant is a named set of `[config]` overrides
(any field of `qsmsim.exp1.config.Config`, or `qsmsim.veins.VeinParams` under `veins`).

## Your own sweeps: `qsmsim-explore`

```bash
qsmsim-explore example > my.toml                       # a commented spec to edit
qsmsim-explore run my.toml --out runs/my --workers 2   # resumable: finished datasets are kept
qsmsim-explore plot runs/my --x snr --y vein_auc --group protocol --facet b0_tesla
qsmsim-explore plot runs/my --results --x lam --y rmse --group chain    # every regularization value
qsmsim-explore summarize runs/my --select rmse vein_auc cnr_0.2-0.3mm
docker/qsmsim.sh run my.toml runs/my [--gpu] [--workers 2]
```

**Outputs** in `--out`: `results.csv` (one row per dataset × chain × regularization value), `summary.csv` (per
condition and chain, the value each selection metric picks, metrics averaged over replicates, with min/max and
`lam_at_grid_edge` flagging a choice at the end of the grid), `plot_*.png`, optional NIfTI χ maps
(`save_maps = true`), and `prep/` (the phantom truth, also as NIfTI, shared by every SNR, replicate and method).

**Metrics** (columns): whole-brain `rmse`, `hfen`, `ssim`, `deep_gm_slope`; per region (`dmv_territory`,
`deep_gm`, `vein_core`, ...) `nrmse`, `rms_ppb`, `ssim`, `hfen`; field-map error `field_err_median_ppb`; veins:
`vein_auc` (vein-core voxels against the 1–3 mm shell around them) and `vein_dprime` = √2 Φ⁻¹(AUC), and per
diameter bin (`0-0.2mm`, `0.2-0.3mm`, `0.3-0.5mm`, `0.5-9.9mm`) the cross-section `cnr_*`, `moment_ratio_*`
(integrated χ over the cross-section ÷ truth), `peak_ratio_*` and `lsf_fwhm_mm_*`.

**Noise** is seeded from the replicate, voxel size and echo index, so protocols, fields and methods are compared on
identical noise, and a rerun reproduces the same numbers (`source_sha256` identifies each noisy dataset).

### Full-size phantoms

Set `base` to a larger preset. The specs `examples/echo_trains.toml`, `field_strength.toml`, `voxel_size.toml` and
`p3_gpu.toml` are full-size versions of the demos.

| base | model | voxel sizes | phantom preparation | memory per dataset |
|---|---|---|---|---|
| `toy` | 1 mm master grid, 20 veins, forward model v1 | 2, 3, 4 mm | ~3 GB, ~30 s | < 2 GB |
| `explore` | 0.5 mm master grid, 300 veins, v1 | 1, 1.5, 2 mm | ~19 GB peak, ~7 min on 8 cores | 10 GB at 1 mm |
| `exp1_v2` | every voxel sampled at 0.1 mm, sourced relaxation per field | 1, 1.5, 2 mm | ~19 GB host, ~16 min on an A5000 GPU | 7–10 GB at 1 mm |

Reconstruction time at 1 mm on one CPU core: TKD and CF-L2 a few seconds, TV and MEDI-like ~6–7 min, TFI
10–35 min, plus ~15 s of scoring per map. The GPU runs the iterative inversions ~75–95× faster per iteration than
one core (RTX A5000). Forward model v2 is slow on the CPU; use a GPU for `exp1_v2`.

### CPU or GPU

The CPU backend is the default. `backend = "gpu"` in a spec, `--backend gpu`, `QSMSIM_BACKEND=gpu`, or
`docker/qsmsim.sh run ... --gpu` runs the field fit, background removal, inversions and forward model v2 on a
CUDA GPU with CuPy; scoring stays on the CPU. Results agree with the CPU to floating-point tolerance, not bit for
bit. `docker/qsmsim.sh compare` runs `examples/cpu_vs_gpu.toml` on both and checks every metric.

## The single-run pipeline: `qsmsim`

A one-shot run on the analytic head with a full report, figures and NIfTI volumes of every stage:

```bash
qsmsim --quick                                   # 3 mm smoke test, ~15 s
qsmsim                                           # 1.5 mm default, ~3 min -> outputs/default/report.md
qsmsim --sweep                                   # + regularization tuned against the truth per inversion
qsmsim --name tilt20 --tilt 20 0                 # head pitched 20 degrees to B0
qsmsim --name 7T --b0 7 --dte 5 --te1 3          # 7 T
qsmsim --name os2 --oversample 2 --pad 1.5       # forward simulation at 0.75 mm, complex-averaged
qsmsim --bg oracle pdf lbv --inv tv medi         # a subset of methods
qsmsim --config my.json                          # any SimConfig field (qsmsim/pipeline.py)
```

## Conventions

* Arrays are `(x, y, z)` = (L→R, P→A, I→S); NIfTI outputs are RAS+ with a diagonal voxel-size affine.
* χ is in ppm relative to water (air ≈ +9.4, bone −2.0, CSF 0). Fields are ΔB/B0 in ppm.
* QSM is relative: metrics are computed after removing the mean over the evaluation mask.
* Regularization weights use unitless per-voxel gradients, so they depend on resolution and SNR; sweep them.
* "Oracle" background removal passes the true local field plus the real field-mapping error, which isolates the
  inversion from background removal. "MEDI-like" is TV with a magnitude-edge prior and a linear data term, not
  the MEDI toolbox.

## Package layout

| path | what it does |
|---|---|
| `qsmsim/phantom.py` | analytic head (tissues, deep grey matter, ventricles, sinuses, calcification, veins) and tissue tables |
| `qsmsim/veins.py`, `finegrid.py`, `forward2.py` | deep medullary vein network; forward models v1 (two-scale) and v2 (fine tiles) |
| `qsmsim/rc2_3t.py`, `rc2.py`, `rc2_download.py` | phantom P3 from the challenge head; the challenge metrics; `qsmsim-fetch-rc2` |
| `qsmsim/dipole.py`, `signal.py` | dipole kernel and forward field; multi-echo GRE signal and noise |
| `qsmsim/unwrap.py`, `echofit.py` | phase unwrapping; complex multi-echo field fit with noise weights |
| `qsmsim/background.py`, `inversion.py`, `tfi.py` | SHARP, V-SHARP, PDF, LBV; TKD, CF-L2, TV-ADMM, MEDI-like; TFI |
| `qsmsim/metrics.py`, `exp1/evaluate.py` | whole-brain, regional and vein metrics |
| `qsmsim/explore.py` | `qsmsim-explore` |
| `qsmsim/exp1/` | the batch machinery behind it: presets (`config.py`), preparation, datasets, runner (`qsmsim-exp`) |
| `qsmsim/backend.py` | CPU (numpy/scipy) or GPU (CuPy) arrays |
| `qsmsim/pipeline.py`, `__main__.py`, `plotting.py` | the single-run pipeline `qsmsim` |
| `qsmsim/qsmjl.py`, `julia/` | the QSM.jl cross-check |
| `scripts/rc2_*.py` | challenge metric validation, runs and figures |
| `examples/` | the knob demos and full-size specs |
| `docker/` | CPU and GPU images, the `qsmsim.sh` wrapper, the GPU check |

Comments in `qsmsim/exp1/` refer to the design notes of the experiments qsmSim was built for (`Experiment_1.md`,
`Experiments_v2.md`, `GPU_Plan.md`); those notes are not part of this release.

## Citing

See `CITATION.cff`. If you use phantom P3, also cite the QSM Reconstruction Challenge 2.0 papers (Marques et al.,
MRM 2021;86:526–542; Bilgic et al., MRM 2021;86:1241–1255).

## License

Apache License 2.0 (`LICENSE`). `qsmsim/rc2.py` adapts the challenge's CC-BY 4.0 evaluation code; see `NOTICE`.
