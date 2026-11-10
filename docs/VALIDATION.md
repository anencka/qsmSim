# Validation

qsmSim was checked in four ways: unit tests of the physics and of each method, a cross-check against an
independent open QSM implementation (QSM.jl) on identical inputs, a port of the QSM Reconstruction Challenge
2.0 evaluation that reproduces the published leaderboard, and a CPU against GPU comparison of every
reconstruction. The numbers below were measured during development; the commands to reproduce each one are
given with it.

## 1. Unit tests

```bash
pip install -e ".[test]" && pytest          # or: docker/qsmsim.sh test [--gpu]
```

About 80 tests, 1–2 minutes on a laptop; the GPU group skips without a CUDA device. Among them:

| area | what is checked |
|---|---|
| dipole model | kernel range and DC term; rFFT kernel equals the full kernel; field of a sphere equals the analytic solution; oblique B0 |
| vein model | two-scale field equals brute-force fine-grid convolution and the analytic cylinder field; sub-voxel rasterization conserves vein volume; network counts, orientation and containment |
| forward model v2 | reduces to v1 exactly when tissue sub-sampling is off; centre samples equal v1; refined blocks average back; tile geometry |
| resampling | k-space truncation aligned with block averaging |
| signal and noise | first-echo SNR definition; noise deterministic and keyed by replicate, voxel size and echo |
| field mapping | unwrapping recovers smooth phase; the complex fit recovers the field through aliasing when noiseless; its predicted standard deviation matches the empirical noise |
| reconstruction | background removal on a known field; every inversion on noiseless data; gradient operator adjoint; TFI handles the background comparably to PDF |
| scoring | metric identities (a perfect map scores perfectly); partial-volume vein metrics; effective FWHM; challenge metrics identity and scaling |
| explorer | spec parsing, per-field relaxation scaling, variants, selection direction, an end-to-end small run that resumes to identical output |
| GPU backend | CG against scipy; unwrapping, field fit, background removal, inversions, TFI and v2 tiles on the GPU against the CPU |

## 2. Cross-check against QSM.jl 0.5.4

[QSM.jl](https://github.com/kamesy/QSM.jl) was run on qsmSim's own fitted field, mask, noise weights and edge
weights (`vsharp`, `pdf`, `tkd`, `tikh(reg=:gradient)`, `tv`, and `tv` with MEDI edge weights), each
implementation tuned against the truth on its own parameter scale.

```bash
qsmsim                                       # outputs/default (1.5 mm analytic head, SNR 100)
python -m qsmsim.qsmjl outputs/default       # ~15 min; needs Julia 1.10 and the julia/ environment
```

Julia setup (user level): install juliaup, `juliaup add 1.10` (QSM.jl 0.5.4 does not load on 1.13), then
`julia +1.10 --project=julia -e 'using Pkg; Pkg.instantiate()'`. `julia/Manifest.toml` pins FFTW.jl to 1.8.1,
because QSM.jl 0.5.4's `__init__` fails with FFTW.jl 1.9 and later.

On the same FFT grid the implementations agree to floating-point precision:

| step | qsmSim vs QSM.jl, same grid |
|---|---|
| V-SHARP | 0.001 % |
| TKD | 0.05 % |
| CF-L2 | 0.00 % |
| PDF (matched settings) | 3.1 % |

Whole-chain RMSE (%), qsmSim / QSM.jl, every pairing within 4 points:

| background | TKD | CF-L2 | TV | MEDI-like |
|---|---|---|---|---|
| oracle | 44.1 / 44.5 | 43.5 / 43.2 | 13.2 / 17.1 | 9.2 / 10.8 |
| V-SHARP | 74.7 / 74.6 | 74.0 / 77.5 | 66.2 / 65.4 | 63.9 / 62.4 |
| PDF | 67.2 / 66.9 | 64.7 / 67.8 | 53.9 / 52.5 | 50.4 / 48.5 |

The cross-check also changed qsmSim: it now uses QSM.jl's crop grid for SHARP/V-SHARP (the FFT domain is part of
the method: full FOV vs mask crop moved the local-field error from 71.5 % to 62.8 %), pads inversions by 25 % of
the brain extent, and pads MEDI-like edge weights with 1 (padding with 0 had flattered it by ~20 RMSE points).

Four QSM.jl 0.5.4 behaviours were found and are worked around in `julia/qsmjl_run.jl`: `tv(...; Wtv)` uses `Wtv`
directly as the shrinkage threshold instead of λ/ρ·Wtv; a 5-D per-direction `Wtv` with one echo throws
`UndefVarError`; weighted `tv` fits ‖W·Dχ − f‖² rather than the documented ‖W(Dχ − f)‖²; and `lbv` returns its
input unchanged. qsmSim's own LBV is used everywhere instead.

## 3. QSM Reconstruction Challenge 2.0

**Metric port.** `qsmsim/rc2.py` is a line-by-line port of the challenge's MATLAB evaluation (see NOTICE).
Scoring the downloaded stage-1 submissions and comparing with the organizers' leaderboard CSV: of 98 leaderboard
rows, 80 matched a downloaded file, 78 were scorable, and **77 reproduce all eight official metrics to the CSV's
printed precision**. The exceptions are submissions the organizers' notes say were re-centred by hand.

```bash
qsmsim-fetch-rc2 --submissions               # inputs, ground truth, evaluation code, leaderboard, submissions (~4.3 GB)
python scripts/rc2_validate_metrics.py
```

**Placement of qsmSim's inversions** on the challenge data (official NRMSE %, mean of Sim1 and Sim2; ranks among
the 98 stage-1 entries plus ours; parameters fixed a priori on qsmSim's own phantom, i.e. no ground truth used):

| input | method | NRMSE % | rank | comparable published entry |
|---|---|---:|---:|---|
| 4-echo complex fit + noise weights | MEDI-like | 35.1 | 3 / 99 | MEDI 40.5, mcTFI 37.7 |
| | TV-ADMM | 35.4 | 4 / 99 | FANSI 36.9, L1-QSM 34.0 |
| | CF-L2 | 57.5 | 41 / 99 | CF-L2 57.0 |
| | TKD | 74.3 | 80 / 99 | TKD 98–102 |
| | TFI | 76.3 | 82 / 99 | TFI_autoP 40.6 |

Like-for-like methods land where the published entries do. TFI does not transfer: the challenge head has no
background field, and its published entries use R2*-derived preconditioners and nonlinear data terms.

```bash
qsmsim-fetch-rc2 --submissions
python scripts/rc2_run.py && python scripts/rc2_figures.py      # outputs/rc2/report.md and figures
```

## 4. CPU against GPU

The GPU backend (CuPy) runs the field fit, background removal, inversions and forward model v2; scoring stays on
the CPU. The results agree to floating-point tolerance, not bit for bit (iterative solvers can stop one
iteration apart on a plateau).

```bash
docker/qsmsim.sh build cpu && docker/qsmsim.sh build gpu
docker/gpu_check.sh runs/gpu_check           # unit tests with the GPU, CPU vs GPU, a v2 run on P3 (~30 min)
```

On an RTX A5000 (driver 595, CUDA 13.2, CuPy 14.2): all GPU unit tests pass; on `examples/cpu_vs_gpu.toml`
(3 background removals × 4 inversions plus TFI, two λ each, two echo trains: 52 reconstructions) every metric
agrees within 0.2 % of its largest value in the run, the largest difference being 1.8e-3 in the χ moment ratio
of the smallest veins.
