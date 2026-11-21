#!/usr/bin/env bash
# Phase 4: score every lens, build the tables, run the diagnostics, write the report.
#
#   gibberish/phase4.sh
#
# Resumable: a condition whose ranks CSV already exists is skipped.
#
# Each condition is scored in its own evaluate call for two reasons. evaluate names
# a lens by its file stem, so c0/pooled_none.pt and c1/pooled_none.pt would collide
# in one call; and four 2.08 GB fp32 lenses plus 19.3 GB of weights is already
# 27.6 GB of a 32 GB card.
set -uo pipefail

cd "$(dirname "$0")/.."
LAYERS=${LAYERS:-$(seq -s, 0 30)}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
CORRECTNESS=${CORRECTNESS:-data/model_correctness_9b.csv}
RELEASED=${RELEASED:-/mnt/data/lens_stuff/reference_lenses/qwen3.5-9b/lens.pt}
CONDITIONS=${CONDITIONS:-"c0_wikitext c1_selfgen c2_shuffled c3_random c3_random_c0router"}
KINDS=${KINDS:-"pooled_none pooled_rlens jpp_none jpp_rlens"}
CLI="uv run python scripts/jpp_cli.py"
mkdir -p results

step() { printf '\n=== %s | %s ===\n' "$(date +%H:%M:%S)" "$*"; }

for CONDITION in $CONDITIONS; do
    OUT=results/ranks_$CONDITION.csv
    if [ -f "$OUT" ]; then
        step "exists, skipping: $OUT"
        continue
    fi
    LENSES=""
    for KIND in $KINDS; do
        [ -f "artifacts/$CONDITION/$KIND.pt" ] && LENSES="$LENSES artifacts/$CONDITION/$KIND.pt"
    done
    if [ -z "$LENSES" ]; then
        step "no lenses for $CONDITION, skipping"
        continue
    fi
    step "evaluate $CONDITION ($(echo "$LENSES" | wc -w) lenses)"
    $CLI evaluate --hf-model-name "$MODEL" --layers "$LAYERS" --lens $LENSES \
        --items held-out:0 --fit-fraction 0.2 --ks 1,10 \
        --correctness-csv "$CORRECTNESS" --out "$OUT" || step "FAILED: $CONDITION"
done

# The released lens and the logit lens belong to no condition, so they are scored
# separately and excluded from the gap bootstrap.
if [ ! -f results/ranks_reference.csv ]; then
    step "evaluate the released lens and the logit lens"
    ln -sf "$RELEASED" artifacts/released_jpp_9b.pt
    $CLI evaluate --hf-model-name "$MODEL" --layers "$LAYERS" \
        --lens artifacts/released_jpp_9b.pt logit \
        --items held-out:0 --fit-fraction 0.2 --ks 1,10 \
        --correctness-csv "$CORRECTNESS" --out results/ranks_reference.csv
fi

# The paper scored the J-lens and R-lens over the full vocabulary; this run applies
# Readout Filtering to all three, so one unfiltered C0 run keeps both conventions
# on the record.
if [ ! -f results/nofilter_c0.csv ]; then
    step "evaluate C0 over the full vocabulary"
    $CLI evaluate --hf-model-name "$MODEL" --layers "$LAYERS" \
        --lens artifacts/c0_wikitext/pooled_none.pt artifacts/c0_wikitext/pooled_rlens.pt \
               artifacts/c0_wikitext/jpp_rlens.pt \
        --items held-out:0 --fit-fraction 0.2 --ks 1,10 --no-readout-filter \
        --correctness-csv "$CORRECTNESS" --out results/nofilter_c0.csv
fi

step "recall tables and paired-bootstrap intervals"
uv run python gibberish/metrics.py results/ranks_c*.csv --out-dir results

for DIAGNOSTIC in weights matrix echo align; do
    step "diagnostic: $DIAGNOSTIC"
    uv run python gibberish/diagnostics.py "$DIAGNOSTIC" || step "FAILED: $DIAGNOSTIC"
done

step "report"
# matplotlib is not a dependency of the pipeline; --with keeps it out of pyproject.
uv run --with matplotlib python gibberish/make_report.py || step "FAILED: report"

step "Phase 4 finished"
ls -la results/*.csv results/*.md 2>/dev/null | tail -20
