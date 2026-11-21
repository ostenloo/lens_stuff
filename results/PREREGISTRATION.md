# Pre-registration

Written 2026-11-20, before any lens was fitted on any condition. The spec requires
these predictions to be committed before Phase 3 and not edited afterwards. Results
go in `REPORT.md`; every hypothesis gets a verdict there — supported, not supported,
or inconclusive — with the number it rests on.

## What was already known when this was written

Disclosed so that nothing here can be re-framed later as a prediction made in
ignorance:

- **Gate 0 passed.** The released Qwen3.5-9B J++ lens scores 58.6% held-out
  recall@10 against the logit lens's 32.4%, scored at 7 layers. Everything below is
  scored over all 31 layers, so that number is not the comparison point.
- **Gate 1 passed.** All four corpora hold 1,000 prompts of at least 128 tokens, and
  C0 and C2 match on the word multiset for 1,000/1,000 records.
- **Corpus statistics.** Mean per-token loss over the 64 fitting prompts:
  C1 2.256 < C0 2.697 < C2 6.264 < C3 13.836.
- **C1's loss is lower than C0's.** The model finds its own samples more predictable
  than WikiText. This makes the `C0 >= C1` half of H1 the risky half, and it was
  known before the prediction was made.
- No lens has been fitted on any condition, so no recall number exists for C0-C3.

## Predictions

**H1 (ordering).** Held-out recall@10, any of the 31 readout layers, is ordered
C0 >= C1 > C2 > C3. This holds for all three lens types.

**H2 (depth).** The C0-C3 gap is largest at the earliest readout layers and under
5 percentage points at the deepest readout layer.

**H3 (current-token collapse).** C3 lenses read out the token at the readout
position itself more often than C0 lenses: position echo rate, as defined in the
spec's Evaluation section, is higher for C3 than for C0.

**H4 (J++ rescue).** J++'s expert weights are fitted on real readout items in every
condition, so the C0-C3 gap is smaller for J++ than for the J-lens and the R-lens.

**H0 (null).** If C3 recall@10 is within 3pp of C0 for a lens type, conclude that
that lens is set mostly by the model's weights rather than by the fitting text.
Reported either way. The 3pp threshold is the spread already seen between natural
fitting corpora: on Pythia-410M, changing only the corpus moved the readout score
from 22.8 to 26.0 (the NeurIPS 2026 fitting-corpus audit).

## Analysis plan

- Primary metric: held-out recall@10, macro-averaged over the five recipe evals,
  item-weighted, any of the 31 readout layers, Readout Filtering on for every lens.
  Secondary: recall@1, and the pair-weighted variants.
- Items: `--items held-out:0 --fit-fraction 0.2`, the same split for every condition
  and lens. Expert weights are fitted on `fit:0` only.
- Gap to C0: `recall(C0) - recall(Cx)` per lens and layer, with 95% intervals from a
  paired bootstrap over items, 10,000 resamples, seed 0, since every condition is
  scored on the same items.
- No condition is dropped and no metric is added after seeing results. Anything that
  changes goes in REPORT.md under deviations, with its reason.

## Settings fixed before Phase 3

Locked in now so they cannot be tuned toward a result:

| setting | value | why not the spec's value |
|---|---|---|
| source layers | 0-30, all 31 | the released lens's own config; costs 1.3 GB more GPU than 7 |
| `jacobian_rows_per_pass` | 4 | 48 and 16 both OOM at ~30.5 GB on a 32 GB card; a pure batching knob, so the estimator is identical |
| router pool | 256 prompts | 1,000 needs 56 GB of fp32 CPU activations over 31 layers, twice that at the concatenate |
| C1 seed token | newline | this checkpoint has no BOS token; the endoftext token degenerates into repetition and leaks control tokens |
| C1 special tokens | suppressed | a control token in a fitting corpus is an artifact of unconditional sampling |
| C2 shuffle seed | the record's index in the stream | the keep decision depends on the shuffled text, so a seed from the kept index would be circular |
| kernels | PyTorch fallback for Gated DeltaNet | flash-linear-attention is not installed; the same path is used for every condition |

All four conditions are fitted and scored under identical settings, so the
comparison between them is unaffected by any of the above.
