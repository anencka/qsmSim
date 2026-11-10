# Run QSM.jl (reference implementation) on the inputs of a qsmSim run.
#
#   julia +1.10 -t auto --project=julia julia/qsmjl_run.jl outputs/<run>
#
# Reads   <run>/nifti/*.nii.gz and <run>/qsmjl/inputs/{config.toml, weights, edge_*}
# Writes  <run>/qsmjl/local_<bg>.nii.gz, chi_<bg>_<inv>.nii.gz, sweep.csv, timing.csv
#
# Background removal: V-SHARP and PDF, each with QSM.jl defaults ("*_default")
# and with parameters matched to qsmSim (used for the full chain).
# Inversion: tkd, tikh(reg=:gradient) (= CF-L2), tv (weighted), and tv with
# per-direction MEDI edge weights (Wtv). Each regularization parameter is
# oracle-tuned (lowest nRMSE vs truth on the oracle local field), exactly like
# `python -m qsmsim --sweep`, then applied to every background method.
#
# Two QSM.jl 0.5.4 issues are worked around here:
#   1. Wtv is used directly as the shrinkage threshold (not multiplied by
#      lambda/rho), so weights are passed pre-scaled by lambda/rho.
#   2. A 5-D Wtv with one echo hits an UndefVarError (lambda*Wy/Wz never set),
#      so the MEDI-style call passes the field as two identical echoes.
#   3. With data weights W, tv() solves ||W.*(D x) - f||^2 (its z-update uses
#      W*f, not W^2*f), not the documented ||W.*(D x - f)||^2. The field is
#      passed pre-weighted (W .* f) so the documented problem is solved.

using QSM, NIfTI, TOML, Printf, Statistics

const RUN = abspath(ARGS[1])
const OUT = joinpath(RUN, "qsmjl")
const INP = joinpath(OUT, "inputs")
cfg = TOML.parsefile(joinpath(INP, "config.toml"))

rd(p) = copy(niread(p).raw)
nii(name) = rd(joinpath(RUN, "nifti", name * ".nii.gz"))
const VSZ = Tuple(Float64.(cfg["vsz"]))
const BDIR = Tuple(Float64.(cfg["bdir"]))
wr(name, a) = niwrite(joinpath(OUT, name * ".nii.gz"), NIVolume(Float32.(a); voxel_size=Float32.(VSZ)))

f      = Float32.(nii("total_field_est"))
mask   = nii("mask") .!= 0
evm    = nii("eval_mask") .!= 0
chit   = Float32.(nii("chi_true"))
loc_or = Float32.(nii("local_field_oracle"))
W      = Float32.(rd(joinpath(INP, "weights.nii.gz")))
G      = cat((Float32.(rd(joinpath(INP, "edge_$a.nii.gz"))) for a in ("x", "y", "z"))...; dims=4)

function nrmse(x, t, m)
    a = x[m] .- mean(x[m]); b = t[m] .- mean(t[m])
    100 * sqrt(sum(abs2, a .- b) / sum(abs2, b))
end

timing = Tuple{String, Float64}[]
timed(name, fn) = (t0 = time(); r = fn(); push!(timing, (name, time() - t0)); @printf("  %-28s %6.1f s\n", name, timing[end][2]); r)

# ---- inversions ---------------------------------------------------------------
medi_tv(l, m; lambda) = begin
    s = lambda / (100 * lambda)                       # lambda / rho with QSM.jl's default rho = 100*lambda
    Wtv = Array{Float32}(undef, size(l)..., 2, 3)
    for e in 1:2, d in 1:3
        Wtv[:, :, :, e, d] .= s .* G[:, :, :, d]
    end
    lw = W .* l
    x = tv(cat(lw, lw; dims=4), m, VSZ; W=W, Wtv=Wtv, bdir=BDIR, lambda=lambda)
    x[:, :, :, 1]
end

INV = Dict(
    "tkd"  => (:thr,    (l, m, v) -> tkd(l, m, VSZ; bdir=BDIR, thr=v)),
    "cfl2" => (:lambda, (l, m, v) -> tikh(l, m, VSZ; bdir=BDIR, lambda=v, reg=:gradient)),
    "tv"   => (:lambda, (l, m, v) -> tv(W .* l, m, VSZ; W=W, bdir=BDIR, lambda=v)),
    "medi" => (:lambda, (l, m, v) -> medi_tv(l, m; lambda=v)),
)
order = ["tkd", "cfl2", "tv", "medi"]

let n = 24, c = (n + 1) / 2                          # JIT warm-up so timings exclude compilation
    m = BitArray([(i - c)^2 + (j - c)^2 + (k - c)^2 < 64 for i in 1:n, j in 1:n, k in 1:n])
    l = randn(Float32, n, n, n) .* m
    Wsmall, Gsmall = W, G
    global W = ones(Float32, n, n, n); global G = ones(Float32, n, n, n, 3)
    for k in order
        INV[k][2](l, m, Float64(cfg["grid_$k"][1]))
    end
    vsharp(l, m, VSZ; r=[3.0, 2.0]); pdf(l, m, VSZ; W=W, maxit=3); pdf(l, m, VSZ; W=W, Dkernel=:k, maxit=3)
    global W, G = Wsmall, Gsmall
end

# ---- background removal -----------------------------------------------------
println("background removal")
bg = Dict{String, Tuple{Array{Float32, 3}, BitArray{3}}}()
bg["oracle"] = (loc_or, mask)
let (l, m) = timed("vsharp_default", () -> vsharp(f, mask, VSZ))
    bg["vsharp_default"] = (l, BitArray(m))
end
let (l, m) = timed("vsharp", () -> vsharp(f, mask, VSZ; r=Float64.(cfg["vsharp_r"]), thr=cfg["vsharp_thr"]))
    bg["vsharp"] = (l, BitArray(m))
end
bg["pdf_default"] = (timed("pdf_default", () -> pdf(f, mask, VSZ; W=W, bdir=BDIR)), BitArray(mask))
padn = Tuple(Int.(cfg["pdf_pad"]))
bg["pdf"] = (timed("pdf", () -> pdf(f, mask, VSZ; W=W, bdir=BDIR, Dkernel=:k, pad=padn,
                                    tol=cfg["pdf_tol"], maxit=cfg["pdf_maxit"])), BitArray(mask))
for (k, (l, m)) in bg
    k == "oracle" && continue
    wr("local_$k", l .* m)
    wr("mask_$k", m)
end

println("oracle-tuning sweeps")
best = Dict{String, Float64}()
open(joinpath(OUT, "sweep.csv"), "w") do io
    println(io, "inversion,param,value,rmse")
    for k in order
        p, fn = INV[k]
        bestv, beste = NaN, Inf
        for v in Float64.(cfg["grid_$k"])
            x = timed("$k $p=$v", () -> fn(loc_or, mask, v))
            e = nrmse(x, chit, evm)
            println(io, "$k,$p,$v,$e")
            @printf("      RMSE %.1f%%\n", e)
            if e < beste
                bestv, beste = v, e
            end
            if k == "tkd" && v == cfg["tkd_same_thr"]
                wr("chi_oracle_tkd_samethr", x .* mask)    # identical-parameter agreement check
            end
        end
        best[k] = bestv
    end
end

println("full chain")
for b in ("oracle", "vsharp", "pdf"), k in order
    l, m = bg[b]
    x = timed("$b + $k", () -> INV[k][2](l, m, best[k]))
    wr("chi_$(b)_$(k)", x .* m)
end

open(joinpath(OUT, "timing.csv"), "w") do io
    println(io, "step,seconds")
    foreach(((n, t),) -> println(io, "\"$n\",$t"), timing)
end
open(joinpath(OUT, "best.toml"), "w") do io
    TOML.print(io, Dict(k => v for (k, v) in best))
end
println("done")
