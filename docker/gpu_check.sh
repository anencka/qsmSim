#!/usr/bin/env bash
# Check the GPU image on a machine with an NVIDIA GPU: unit tests with the device attached (the GPU tests then
# run), CPU against GPU on examples/cpu_vs_gpu.toml, and a forward-model-v2 run on the challenge head (P3).
#   docker/gpu_check.sh [OUT]          (OUT default runs/gpu_check; needs the qsmsim:gpu image loaded)
# Environment: GPU_DEVICE (default 0), MEM (container memory cap, default 48g), RC2_DIR (default ./data/rc2; fetched
# when missing). Writes OUT/gpu_check.log and packs OUT into OUT.tar.gz (+ .sha256).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-runs/gpu_check}
export MEM=${MEM:-48g} GPU_DEVICE=${GPU_DEVICE:-0} RC2_DIR=${RC2_DIR:-$PWD/data/rc2}
mkdir -p "$OUT"
LOG=$OUT/gpu_check.log
exec > >(tee -a "$LOG") 2>&1
say() { echo "== $(date '+%F %T') $*"; }
fail=0
say "gpu_check: image ${IMAGE_GPU:-qsmsim:gpu}, GPU $GPU_DEVICE, OUT=$OUT"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || say "nvidia-smi not available"
docker image inspect "${IMAGE_GPU:-qsmsim:gpu}" --format '{{index .Config.Env}}' | tr ' ' '\n' | grep QSMSIM_GIT_COMMIT

say "1/3 unit tests with the GPU attached"
"$HERE/qsmsim.sh" test --gpu || { fail=1; say "unit tests FAILED"; }

say "2/3 CPU against GPU (examples/cpu_vs_gpu.toml)"
"$HERE/qsmsim.sh" compare "$OUT/cpu_vs_gpu" || { fail=1; say "CPU/GPU comparison FAILED"; }

say "3/3 forward model v2 on P3 (GPU, 2 mm, SNR 20 and 100, two echo trains)"
if [ ! -f "$RC2_DIR/Sim1GT/Chi.nii.gz" ]; then
    say "fetching the challenge files into $RC2_DIR"
    "$HERE/qsmsim.sh" fetch-rc2 "$RC2_DIR" || say "fetch FAILED: put the files in $RC2_DIR (qsmsim-fetch-rc2 --p3)"
fi
cat > "$OUT/p3_v2.toml" <<'SPEC'
name = "p3_v2_check"
base = "exp1_v2"
backend = "gpu"
phantoms = ["P3"]
resolution_mm = [2.0]
snr = [20, 100]
protocols = [ { dte_ms = 5.4, echoes = 5 }, { dte_ms = 2.4, echoes = 11 } ]
inversions = ["medi", "tfi"]
lambda = { medi = [1e-4, 3e-4], tfi = [1e-3] }
SPEC
"$HERE/qsmsim.sh" run "$OUT/p3_v2.toml" "$OUT/p3_v2" --gpu || { fail=1; say "P3 v2 run FAILED"; }

say "gpu_check: $([ $fail = 0 ] && echo PASSED || echo FAILED)"
tar czf "$OUT.tar.gz" --exclude='*/prep/*/cache' "$OUT" && sha256sum "$OUT.tar.gz" > "$OUT.tar.gz.sha256"
say "packed $OUT.tar.gz"
exit $fail
