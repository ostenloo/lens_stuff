#!/usr/bin/env bash
# Fit one condition's four lenses, resumably: router, then for each LRP mode a
# shard fit, a merge (which also writes the pooled lens) and an expert-weight fit.
#
#   gibberish/fit_condition.sh c0_wikitext
#   gibberish/fit_condition.sh c3_random --router-from c0_wikitext --modes rlens
#
# Every stage is skipped when its output already exists, so a killed run resumes
# by being re-run. Run it from the jpp_lens root, inside tmux: the fits are hours
# long and must not depend on an ssh connection.
#
# Settings follow the released Qwen3.5-9B lens's own recorded config: all 31
# source layers (0-30), the final block as the target, 64 fit prompts at
# max_seq_len 128, 16 skipped positions. The one exception is ROWS: the released
# config's 48 Jacobian rows per backward pass needs H200-class memory. On a 32 GB
# 5090 it OOMs at 30.4 GB allocated, inside the Gated DeltaNet torch fallback,
# which materialises a chunk x chunk attention matrix per layer. ROWS only sets
# how many rows of J are computed per backward pass, so the estimator is
# identical at any value; the spec says to lower it and never to change the
# sequence length or the prompt count.
# ROUTER_PROMPTS is the one departure from the spec's recipe and the reason is
# memory: the router pools fp32 CPU activations of [positions, 4096] per layer,
# which at the spec's 1000 prompts is 56 GB over 31 layers, twice that at the
# concatenate. It is identical across conditions, so the comparison holds.
set -euo pipefail

COND=${1:?usage: fit_condition.sh <condition> [--router-from <cond>] [--modes "none rlens"]}
shift

ROUTER_FROM=$COND
MODES="none rlens"
while [ $# -gt 0 ]; do
    case $1 in
        --router-from) ROUTER_FROM=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
LAYERS=${LAYERS:-$(seq -s, 0 30)}
ROWS=${ROWS:-16}
ROUTER_PROMPTS=${ROUTER_PROMPTS:-256}
FIT_PROMPTS=${FIT_PROMPTS:-64}
CORRECTNESS=${CORRECTNESS:-data/model_correctness_9b.csv}
CACHE=${CACHE:-artifacts/readout_cache.pt}

CORPUS=corpora/$COND.json
ARTIFACTS=artifacts/$COND
ROUTER=artifacts/$ROUTER_FROM/router.pt
CLI="uv run python scripts/jpp_cli.py"

mkdir -p "$ARTIFACTS" "artifacts/$ROUTER_FROM"
step() { printf '\n=== %s | %s ===\n' "$(date +%H:%M:%S)" "$*"; }

step "condition $COND | layers $LAYERS | rows/pass $ROWS | modes $MODES | router from $ROUTER_FROM"

if [ -f "$ROUTER" ]; then
    step "router exists, skipping: $ROUTER"
else
    step "fit-router on $ROUTER_FROM ($ROUTER_PROMPTS prompts)"
    $CLI fit-router --hf-model-name "$MODEL" --layers "$LAYERS" \
        --prompts-json "corpora/$ROUTER_FROM.json" --num-prompts "$ROUTER_PROMPTS" \
        --num-clusters 8 --projection-dim 64 --max-seq-len 128 \
        --skip-first-n-positions 16 --seed 0 --out "$ROUTER"
fi

for MODE in $MODES; do
    if [ -f "$ARTIFACTS/experts_$MODE.pt" ]; then
        step "experts exist, skipping fit-shard and merge: $ARTIFACTS/experts_$MODE.pt"
    else
        step "fit-shard $COND $MODE"
        $CLI fit-shard --hf-model-name "$MODEL" --layers "$LAYERS" \
            --router-path "$ROUTER" --prompts-json "$CORPUS" \
            --num-prompts "$FIT_PROMPTS" --lrp-mode "$MODE" \
            --jacobian-rows-per-pass "$ROWS" --max-seq-len 128 \
            --skip-first-n-positions 16 --target-offset -1 \
            --checkpoint-every-n-prompts 4 \
            --artifacts-base-dir "$ARTIFACTS" --checkpoint-name "$MODE"

        SHARD=$(ls -d "$ARTIFACTS"/*/"$MODE"/shard0of1 | tail -1)
        step "merge-experts $COND $MODE from $SHARD"
        # merge-experts never loads the model, so 16.6 GB of sums would fit on the
        # GPU; cpu is the conservative choice and the stage is not the bottleneck.
        $CLI merge-experts --checkpoint-dirs "$SHARD" --num-clusters 8 \
            --min-kept-positions 50 --checkpoint-name "${COND}_${MODE}" --device cpu \
            --out "$ARTIFACTS/experts_$MODE.pt" \
            --pooled-lens-out "$ARTIFACTS/pooled_$MODE.pt"
    fi

    if [ -f "$ARTIFACTS/jpp_$MODE.pt" ]; then
        step "lens exists, skipping fit-weights: $ARTIFACTS/jpp_$MODE.pt"
    else
        step "fit-weights $COND $MODE"
        $CLI fit-weights --hf-model-name "$MODEL" --layers "$LAYERS" \
            --experts "$ARTIFACTS/experts_$MODE.pt" --items fit:0 --fit-fraction 0.2 \
            --correctness-csv "$CORRECTNESS" --cache-path "$CACHE" \
            --checkpoint-name "${COND}_jpp_${MODE}" --out "$ARTIFACTS/jpp_$MODE.pt"
    fi
done

step "done with $COND"
ls -la "$ARTIFACTS"/*.pt
