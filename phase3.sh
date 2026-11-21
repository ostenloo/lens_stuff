#!/usr/bin/env bash
# Phase 3: fit the three gibberish conditions, then the C3 router ablation.
#
#   gibberish/phase3.sh
#
# About 23h: six shard fits at 3.2h each, plus the ablation's one, plus merges and
# weight fits. Every stage is resumable, so a killed run resumes by being re-run.
# Each condition writes its own log under logs/, since a shared log makes progress
# greps match an earlier run's text.
#
# The ablation fits C3's Jacobians against C0's router, which separates bad routing
# from bad Jacobians. rlens mode only, as the doc specifies: routing affects the
# expert Jacobians, and the J++ headline comes from the rlens fit.
set -uo pipefail

cd "$(dirname "$0")/.."
LOGS=/mnt/data/lens_stuff/logs
mkdir -p "$LOGS"
export ROWS=${ROWS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The ablation reuses C3's prompts under its own artifacts directory.
ln -sf c3_random.json corpora/c3_random_c0router.json

run() {
    local name=$1; shift
    printf '\n######## %s | starting %s ########\n' "$(date +'%F %H:%M:%S')" "$name"
    if stdbuf -oL -eL ./gibberish/fit_condition.sh "$@" > "$LOGS/phase3_$name.log" 2>&1; then
        printf '%s | %s done\n' "$(date +'%F %H:%M:%S')" "$name"
    else
        printf '%s | %s FAILED (exit %s) - see %s\n' \
            "$(date +'%F %H:%M:%S')" "$name" "$?" "$LOGS/phase3_$name.log"
        tail -5 "$LOGS/phase3_$name.log"
    fi
    df -h /mnt/data | tail -1
}

run c1_selfgen c1_selfgen
run c2_shuffled c2_shuffled
run c3_random c3_random
run c3_random_c0router c3_random_c0router --router-from c0_wikitext --modes rlens

printf '\n######## %s | Phase 3 finished ########\n' "$(date +'%F %H:%M:%S')"
for c in c1_selfgen c2_shuffled c3_random c3_random_c0router; do
    printf '%-22s %s lenses\n' "$c" "$(ls artifacts/$c/*.pt 2>/dev/null | grep -cE 'pooled_|jpp_')"
done
