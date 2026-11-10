#!/usr/bin/env bash
# Run the one-knob demos in examples/knobs/ and draw their plots.
#
#   examples/demo.sh list                 the knobs and what each one varies
#   examples/demo.sh snr lambda           run these demos
#   examples/demo.sh all                  every demo except phantom_p3 (needs the challenge data)
#   examples/demo.sh snr --gpu            on the GPU backend
#
# Each demo is a small spec on the toy preset (2 mm voxels, a 20-vein network): 1-2 minutes on an 8-core CPU,
# ~11 minutes for forward_model (its v2 arm), ~30 minutes for all. Results go to $OUT/<knob>/ (default
# runs/demo): results.csv, summary.csv and plot_*.png, made from the "# plot:" lines at the top of each spec.
# Copy a spec and change base = "explore" for full-size phantoms.
#
# Environment: OUT (output root), WORKERS (datasets in parallel, default 2), EXPLORE (the command, default
# qsmsim-explore, else python -m qsmsim.explore).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KNOBS=$HERE/knobs
OUT=${OUT:-runs/demo}
WORKERS=${WORKERS:-2}
if [ -n "${EXPLORE:-}" ]; then read -r -a EX <<<"$EXPLORE"
elif command -v qsmsim-explore >/dev/null; then EX=(qsmsim-explore)
else EX=(python -m qsmsim.explore); fi

backend=(); names=()
for a in "$@"; do
    case "$a" in
        --gpu) backend=(--backend gpu) ;;
        all) for f in "$KNOBS"/*.toml; do n=$(basename "$f" .toml); [ "$n" = phantom_p3 ] || names+=("$n"); done ;;
        list|-h|--help|help) names=() ; break ;;
        *) [ -f "$KNOBS/$a.toml" ] || { echo "demo.sh: no knob '$a' (try: examples/demo.sh list)" >&2; exit 1; }
           names+=("$a") ;;
    esac
done

if [ ${#names[@]} -eq 0 ]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    echo; echo "Knobs:"
    for f in "$KNOBS"/*.toml; do
        printf '  %-16s %s\n' "$(basename "$f" .toml)" "$(sed -n '1s/^# Knob: //p' "$f")"
    done
    exit 0
fi

for n in "${names[@]}"; do
    spec=$KNOBS/$n.toml out=$OUT/$n
    echo "== $n: $(sed -n '1s/^# Knob: //p' "$spec")"
    "${EX[@]}" run "$spec" --out "$out" --workers "$WORKERS" "${backend[@]}"
    while IFS= read -r line; do
        read -r -a args <<<"$line"
        "${EX[@]}" plot "$out" "${args[@]}"
    done < <(sed -n 's/^# plot: //p' "$spec")
done
echo "== done: ${names[*]} -> $OUT/"
