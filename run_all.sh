#!/usr/bin/env bash
# Reproduce every result from the corpora, which is what Acceptance requires.
#
#   gibberish/run_all.sh                 # the whole thing, about 40 GPU-hours
#   STAGES="phase4" gibberish/run_all.sh # just the scoring, tables and report
#
# Run from the jpp_lens root inside tmux. Every stage is resumable: finished work
# is detected and skipped, so a killed run resumes by being re-invoked.
#
# Setup, once:
#   git clone https://github.com/safety-research/jpp_lens && cd jpp_lens
#   git checkout d7efc0d
#   git clone git@github.com:ostenloo/lens_stuff.git gibberish
#   git apply gibberish/patches/0001-fit-router-prompts-json.patch
#   uv sync --extra cu128 --extra dev
set -uo pipefail

cd "$(dirname "$0")/.."
STAGES=${STAGES:-"correctness corpora phase2 gate2 phase3 phase4"}
export ROWS=${ROWS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

banner() { printf '\n%s\n%s | %s\n%s\n' "########################" "$(date +'%F %H:%M:%S')" "$*" "########################"; }
have() { [ -f "$1" ]; }

for STAGE in $STAGES; do
    case $STAGE in
        correctness)
            banner "model correctness column"
            if have data/model_correctness_9b.csv; then
                echo "exists, skipping"
            else
                uv run python gibberish/build_correctness.py
            fi
            ;;
        corpora)
            banner "fitting corpora and Gate 1"
            if have corpora/c3_random.json; then
                echo "exists, skipping"
            else
                uv run python gibberish/build_corpora.py
            fi
            ;;
        phase2)
            banner "Phase 2: the C0 baseline"
            ./gibberish/fit_condition.sh c0_wikitext
            ;;
        gate2)
            banner "Gate 2"
            # gate2.sh waits for a tmux session named phase2; with the fits already
            # done there is none, so it proceeds straight to scoring.
            ./gibberish/gate2.sh
            ;;
        phase3)
            banner "Phase 3: the gibberish conditions"
            ./gibberish/phase3.sh
            ;;
        phase4)
            banner "Phase 4: scoring, diagnostics and the report"
            ./gibberish/phase4.sh
            ;;
        *)
            echo "unknown stage: $STAGE" >&2
            exit 2
            ;;
    esac
done

banner "run_all finished"
ls -la results/REPORT.md results/*.csv 2>/dev/null | tail -15
