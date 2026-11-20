# Gibberish-fit workspace lenses

How much do the J-lens, R-lens and J++ lens depend on the text they are fitted on?
This refits all three on `Qwen/Qwen3.5-9B` over four fitting corpora — WikiText,
self-generated text, shuffled WikiText, random tokens — and scores every lens on
the same held-out readout items.

Spec: *Gibberish-Fit Workspace Lenses: Coding-Agent Spec*.

## Layout

This repository is the experiment's own code. It lives as `gibberish/` inside a
checkout of the released J++ pipeline, which is a pinned dependency and is never
vendored here:

```
jpp_lens/              safety-research/jpp_lens @ d7efc0d  (not in this repo)
  gibberish/           this repository
    patches/           the one library change, applied to the pinned checkout
  corpora/             <cond>.json, <cond>_stats.json      (not in this repo)
  artifacts/<cond>/    router.pt, experts_*.pt, pooled_*.pt, jpp_*.pt  (not in this repo)
  results/             ranks + recall CSVs, figures, REPORT.md
```

Fitted lenses and corpora are large and reproducible, so only code and the text
results are tracked. `results/` is snapshotted into `results/` here at report time.

## Setup

```sh
git clone https://github.com/safety-research/jpp_lens
cd jpp_lens && git checkout d7efc0d
git clone git@github.com:ostenloo/lens_stuff.git gibberish
git apply gibberish/patches/0001-fit-router-prompts-json.patch
uv sync --extra cu128 --extra dev
```

On a machine whose home partition is small, keep the caches elsewhere — see
`env.sh` in this repo, sourced before any `uv` or `python` command.

## The one library change

`patches/0001-fit-router-prompts-json.patch` adds `--prompts-json` to
`fit-router`, which `fit-shard` already had. Without it the router is always
fitted on WikiText, so the router would not be part of the condition. It reuses
the existing `load_fit_prompts_from_json`, and adds two tests on the tiny CPU
model. No other library code changes.

## Stages

```sh
uv run python gibberish/build_correctness.py   # data/model_correctness_9b.csv
uv run python gibberish/build_corpora.py       # corpora/<cond>.json + stats
gibberish/run_all.sh                           # fits, evaluation, diagnostics, report
```

`build_correctness.py` grades the 9B model on every readout item: the shipped
`model_correctness.csv` has a column for `Qwen/Qwen3.6-27B` only, and both
`fit-weights` and `evaluate` fail without a column for the model in use.
