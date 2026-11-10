#!/usr/bin/env bash
# qsmSim in Docker, CPU or GPU. Run from anywhere; paths are relative to where you run it.
#
#   docker/qsmsim.sh build [cpu|gpu]                 build the image qsmsim:cpu / qsmsim:gpu
#   docker/qsmsim.sh test [--gpu]                    unit tests inside the image (GPU tests need --gpu)
#   docker/qsmsim.sh fetch-rc2 [DIR]                 the QSM Challenge 2.0 files phantom P3 needs (default ./data/rc2)
#   docker/qsmsim.sh run SPEC OUT [--gpu] [ARGS...]  qsmsim-explore run SPEC --out OUT (ARGS: --workers N --threads N)
#   docker/qsmsim.sh demo KNOB... [--gpu]           the one-knob demos (examples/demo.sh; "demo list" names them)
#   docker/qsmsim.sh summarize OUT [ARGS...]         rebuild results.csv / summary.csv
#   docker/qsmsim.sh plot OUT [ARGS...]              e.g. plot runs/x --x snr --y vein_auc --group protocol
#   docker/qsmsim.sh compare [OUT]                   CPU against GPU on examples/cpu_vs_gpu.toml (needs both images)
#   docker/qsmsim.sh shell [--gpu]                   a shell in the image, the current directory at /work
#
# Environment: MEM (container memory cap, e.g. 32g; default none), GPU_DEVICE (default 0), RC2_DIR (default
# ./data/rc2 if it exists), IMAGE_CPU / IMAGE_GPU (default qsmsim:cpu / qsmsim:gpu).
# Outputs are written as the calling user. Containers get no network except fetch-rc2.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
IMAGE_CPU=${IMAGE_CPU:-qsmsim:cpu}
IMAGE_GPU=${IMAGE_GPU:-qsmsim:gpu}
GPU_DEVICE=${GPU_DEVICE:-0}

die() { echo "qsmsim.sh: $*" >&2; exit 1; }
abspath() { mkdir -p "$1" && (cd "$1" && pwd); }

docker_args() {   # $1 = cpu|gpu
    local a=(--rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work)
    [ -n "${MEM:-}" ] && a+=(--memory "$MEM" --memory-swap "$MEM")
    local rc2=${RC2_DIR:-$PWD/data/rc2}
    [ -d "$rc2" ] && a+=(-v "$(cd "$rc2" && pwd):/data/rc2:ro")
    [ "$1" = gpu ] && a+=(--gpus "\"device=$GPU_DEVICE\"")
    [ -t 0 ] && [ -t 1 ] && a+=(-it)
    printf '%s\n' "${a[@]}"
}

in_image() {   # $1 = cpu|gpu, $2 = network (none|bridge), rest: command for the entrypoint
    local be=$1 net=$2; shift 2
    local img=$IMAGE_CPU; [ "$be" = gpu ] && img=$IMAGE_GPU
    docker image inspect "$img" >/dev/null 2>&1 || die "image $img not found: run docker/qsmsim.sh build $be"
    mapfile -t args < <(docker_args "$be")
    docker run "${args[@]}" --network "$net" "${ENTRY[@]}" "$img" "$@"
}

inside() {   # a path for the container: must be under the current directory (mounted at /work)
    local p; p=$(realpath -m "$1")
    case "$p" in "$PWD"|"$PWD"/*) echo "/work${p#"$PWD"}" ;; *) die "$1 must be under the current directory ($PWD)" ;; esac
}

cmd=${1:-help}; shift || true
ENTRY=()
case "$cmd" in
    build)
        be=${1:-cpu}; [ "$be" = cpu ] || [ "$be" = gpu ] || die "build cpu|gpu"
        commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)
        dirty=$(git -C "$ROOT" status --porcelain -- qsmsim tests examples pyproject.toml docker 2>/dev/null | head -1)
        docker build --platform linux/amd64 -f "$HERE/Dockerfile.explore" --build-arg BACKEND="$be" \
            --build-arg GIT_COMMIT="${commit}${dirty:+-dirty}" -t "qsmsim:$be" -t "qsmsim:$be-$commit${dirty:+-dirty}" "$ROOT"
        [ "$be" = cpu ] && [ "$IMAGE_CPU" != qsmsim:cpu ] && docker tag qsmsim:cpu "$IMAGE_CPU"
        [ "$be" = gpu ] && [ "$IMAGE_GPU" != qsmsim:gpu ] && docker tag qsmsim:gpu "$IMAGE_GPU"
        echo "built qsmsim:$be (commit ${commit}${dirty:+-dirty})" ;;
    test)
        be=cpu; [ "${1:-}" = --gpu ] && be=gpu
        ENTRY=(--entrypoint python -w /opt/qsmsim)
        in_image "$be" none -m pytest -q -p no:cacheprovider tests ;;
    fetch-rc2)
        dest=$(abspath "${1:-$PWD/data/rc2}")
        img=$IMAGE_CPU; docker image inspect "$img" >/dev/null 2>&1 || img=$IMAGE_GPU
        docker run --rm --user "$(id -u):$(id -g)" -v "$dest:/data/rc2" --entrypoint qsmsim-fetch-rc2 "$img" --p3 --out /data/rc2 ;;
    run)
        [ $# -ge 2 ] || die "run SPEC OUT [--gpu] [ARGS...]"
        spec=$1 out=$2; shift 2
        be=cpu; rest=()
        for x in "$@"; do if [ "$x" = --gpu ]; then be=gpu; else rest+=("$x"); fi; done
        [ -f "$spec" ] || die "spec $spec not found"
        mkdir -p "$out"
        in_image "$be" none run "$(inside "$spec")" --out "$(inside "$out")" --backend "$be" "${rest[@]}" ;;
    demo)
        be=cpu; rest=()
        for x in "$@"; do if [ "$x" = --gpu ]; then be=gpu; rest+=(--gpu); else rest+=("$x"); fi; done
        ENTRY=(--entrypoint bash)
        in_image "$be" none /opt/qsmsim/examples/demo.sh "${rest[@]}" ;;
    summarize|plot)
        [ $# -ge 1 ] || die "$cmd OUT [ARGS...]"
        out=$1; shift
        in_image cpu none "$cmd" "$(inside "$out")" "$@" ;;
    compare)
        out=${1:-runs/cpu_vs_gpu}; mkdir -p "$out"
        # the CPU leg runs in the CPU image, or in the GPU image on the CPU backend when only that is loaded
        docker image inspect "$IMAGE_CPU" >/dev/null 2>&1 || export IMAGE_CPU=$IMAGE_GPU
        cp "$ROOT/examples/cpu_vs_gpu.toml" "$out/spec.toml"
        "$0" run "$out/spec.toml" "$out/cpu" --workers 2
        "$0" run "$out/spec.toml" "$out/gpu" --gpu
        in_image cpu none compare "$(inside "$out/cpu")" "$(inside "$out/gpu")" ;;
    shell)
        be=cpu; [ "${1:-}" = --gpu ] && be=gpu
        ENTRY=(--entrypoint bash)
        in_image "$be" none ;;
    help|-h|--help)
        sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) die "unknown command $cmd (try: docker/qsmsim.sh help)" ;;
esac
