#!/usr/bin/env bash
# Wait for the C0 fits to finish, then score them for Gate 2.
#
# Gate 2 (spec): C0 J++ beats C0 R-lens, C0 J-lens and the logit lens on held-out
# recall@10, and is within 5pp of the released Qwen3.5-9B J++ lens scored by the
# same harness. If it fails, the gibberish comparison is meaningless and Phase 3
# must not run.
#
# Lenses are split over two evaluate calls: each 31-layer fp32 lens is 2.08 GB on
# top of 19.3 GB of weights, so six at once would not fit in 32 GB.
set -uo pipefail

A=artifacts/c0_wikitext
RELEASED=/mnt/data/lens_stuff/reference_lenses/qwen3.5-9b/lens.pt
LAYERS=$(seq -s, 0 30)
M=Qwen/Qwen3.5-9B
CORRECTNESS=data/model_correctness_9b.csv
CLI="uv run python scripts/jpp_cli.py"
mkdir -p results

step() { printf '\n=== %s | %s ===\n' "$(date +%H:%M:%S)" "$*"; }

step "waiting for Phase 2 to finish (tmux session phase2 to end, $A/jpp_rlens.pt to exist)"
while tmux has-session -t phase2 2>/dev/null; do sleep 60; done
step "phase2 session ended"

MISSING=0
for f in pooled_none pooled_rlens jpp_none jpp_rlens; do
    [ -f "$A/$f.pt" ] || { echo "MISSING: $A/$f.pt"; MISSING=1; }
done
if [ "$MISSING" = 1 ]; then
    step "Phase 2 did not produce all four C0 lenses - Gate 2 cannot be scored"
    echo "Check the tail of logs/phase2_c0.log for the failure."
    exit 1
fi

# A readable name for the released lens in the results tables (its file stem names it).
ln -sf "$RELEASED" "$A/released_jpp_9b.pt" 2>/dev/null || true

step "evaluate: our four C0 lenses"
$CLI evaluate --hf-model-name "$M" --layers "$LAYERS" \
    --lens "$A/pooled_none.pt" "$A/pooled_rlens.pt" "$A/jpp_none.pt" "$A/jpp_rlens.pt" \
    --items held-out:0 --fit-fraction 0.2 --ks 1,10 \
    --correctness-csv "$CORRECTNESS" --out results/gate2_ours.csv

step "evaluate: released lens and the logit lens"
$CLI evaluate --hf-model-name "$M" --layers "$LAYERS" \
    --lens "$A/released_jpp_9b.pt" logit \
    --items held-out:0 --fit-fraction 0.2 --ks 1,10 \
    --correctness-csv "$CORRECTNESS" --out results/gate2_reference.csv

step "evaluate: C0 lenses over the full vocabulary (the paper's convention for J-/R-lens)"
$CLI evaluate --hf-model-name "$M" --layers "$LAYERS" \
    --lens "$A/pooled_none.pt" "$A/pooled_rlens.pt" "$A/jpp_rlens.pt" \
    --items held-out:0 --fit-fraction 0.2 --ks 1,10 --no-readout-filter \
    --correctness-csv "$CORRECTNESS" --out results/gate2_nofilter.csv

step "Gate 2 verdict"
uv run python - <<'PY'
import pandas as pd

NAMES = {
    "pooled_none": "C0 J-lens",
    "pooled_rlens": "C0 R-lens",
    "jpp_none": "C0 J++-noLRP",
    "jpp_rlens": "C0 J++",
    "released_jpp_9b": "released J++ 9B",
    "logit": "logit lens",
}


def recall(path):
    df = pd.read_csv(path)
    df = df[(df.weighting == "item") & (df["eval"] == "macro")]
    return {NAMES.get(r.lens, r.lens): (r.k, r.pass_at_k) for r in df.itertuples()}


scores = {}
for path in ("results/gate2_ours_pass_at_k.csv", "results/gate2_reference_pass_at_k.csv"):
    df = pd.read_csv(path)
    df = df[(df.weighting == "item") & (df["eval"] == "macro") & (df.k == 10)]
    for row in df.itertuples():
        scores[NAMES.get(row.lens, row.lens)] = row.pass_at_k

print("\nheld-out recall@10, macro, item-weighted, Readout Filtering on, 31 layers")
for name, value in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f"  {name:<20} {value * 100:6.2f}%")

jpp = scores.get("C0 J++")
released = scores.get("released J++ 9B")
checks = []
for other in ("C0 R-lens", "C0 J-lens", "logit lens"):
    if jpp is not None and other in scores:
        checks.append((f"C0 J++ beats {other}", jpp > scores[other]))
if jpp is not None and released is not None:
    gap = abs(jpp - released) * 100
    checks.append((f"C0 J++ within 5pp of the released lens (gap {gap:.2f}pp)", gap <= 5.0))

print()
for label, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
passed = all(ok for _, ok in checks)
print("\nGATE 2 PASSED" if passed else "\nGATE 2 FAILED - Phase 3 must not run; write up the failure")
PY
step "done"
