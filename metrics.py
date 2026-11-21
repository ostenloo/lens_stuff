"""Recall tables and paired-bootstrap intervals from the ranks CSVs that
``evaluate`` writes.

A hit is a target token reaching rank <= k at *any* readout layer, which is the
paper's headline form. Item weighting averages each item's hit fraction over its
intermediates and then over items; pair weighting averages over all
(item, intermediate) pairs. Macro is the mean over the five recipe evals.

    uv run python gibberish/metrics.py results/ranks_*.csv --out-dir results
    uv run python gibberish/metrics.py results/gate2_ours.csv --check

``--check`` recomputes the recall table and compares it against the
``_pass_at_k.csv`` the CLI wrote beside the input, which verifies that these
definitions match the pipeline's own before any of it is reported.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MACRO_EVALS = ("multihop", "multilingual", "typo", "association", "poetry")
REFERENCE_CONDITION = "c0_wikitext"


def condition_of(path: Path) -> str:
    """The condition a ranks CSV belongs to, from ``ranks_<cond>.csv``."""
    match = re.fullmatch(r"ranks_(.+)", path.stem)
    return match.group(1) if match else path.stem


def load_ranks(paths: list[Path]) -> pd.DataFrame:
    """One frame of every ranks CSV, labelled by condition.

    ``evaluate`` names a lens by its file stem, so two conditions scored in one
    call would collide; one CSV per condition is what keeps them apart.
    """
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["condition"] = condition_of(path)
        frames.append(frame)
    ranks = pd.concat(frames, ignore_index=True)
    missing = set(MACRO_EVALS) - set(ranks["eval"])
    if missing:
        raise SystemExit(f"{sorted(missing)} missing from {[str(p) for p in paths]}")
    return ranks


def hits(ranks: pd.DataFrame, k: int, *, layer: int | None = None) -> pd.DataFrame:
    """One row per (condition, lens, eval, item, intermediate) with a 0/1 hit.

    With ``layer`` given the hit is that layer's rank alone; otherwise it is the
    minimum rank over all readout layers.
    """
    frame = ranks if layer is None else ranks[ranks["layer"] == layer]
    keys = ["condition", "lens", "eval", "item", "intermediate"]
    best = frame.groupby(keys, observed=True)["rank"].min().reset_index()
    best["hit"] = (best["rank"] <= k).astype(float)
    return best


def recall(hit_frame: pd.DataFrame, *, weighting: str = "item") -> pd.DataFrame:
    """Recall per (condition, lens, eval) plus a macro row per (condition, lens)."""
    if weighting == "item":
        per_item = hit_frame.groupby(["condition", "lens", "eval", "item"], observed=True)[
            "hit"
        ].mean()
        per_eval = per_item.groupby(["condition", "lens", "eval"], observed=True).mean()
    elif weighting == "pair":
        per_eval = hit_frame.groupby(["condition", "lens", "eval"], observed=True)["hit"].mean()
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    per_eval = per_eval.reset_index(name="recall")
    macro = (
        per_eval[per_eval["eval"].isin(MACRO_EVALS)]
        .groupby(["condition", "lens"], observed=True)["recall"]
        .mean()
        .reset_index()
    )
    macro["eval"] = "macro"
    return pd.concat([per_eval, macro], ignore_index=True)


def recall_summary(ranks: pd.DataFrame, ks: tuple[int, ...] = (1, 10)) -> pd.DataFrame:
    rows = []
    for k in ks:
        hit_frame = hits(ranks, k)
        for weighting in ("item", "pair"):
            table = recall(hit_frame, weighting=weighting)
            table["k"] = k
            table["weighting"] = weighting
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


def per_layer_recall(ranks: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Recall at each readout layer separately, which is what H2 is judged on."""
    rows = []
    for layer in sorted(ranks["layer"].unique()):
        table = recall(hits(ranks, k, layer=int(layer)), weighting="item")
        table["layer"] = int(layer)
        table["k"] = k
        table["weighting"] = "item"  # named so the CSV describes itself
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def bootstrap_gaps(
    ranks: pd.DataFrame,
    *,
    k: int = 10,
    reference: str = REFERENCE_CONDITION,
    resamples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """``recall(reference) - recall(condition)`` per lens, with a 95% interval.

    Items are resampled with replacement within each eval, and the same resampled
    item set is used for every condition: the conditions are scored on identical
    items, so pairing removes the item-difficulty variance that dominates an
    unpaired interval. The macro average is rebuilt from the resampled evals.
    """
    hit_frame = hits(ranks, k)
    per_item = (
        hit_frame.groupby(["condition", "lens", "eval", "item"], observed=True)["hit"]
        .mean()
        .reset_index()
    )
    conditions = sorted(per_item["condition"].unique())
    if reference not in conditions:
        raise SystemExit(f"reference condition {reference!r} not among {conditions}")

    rng = np.random.default_rng(seed)
    rows = []
    for lens in sorted(per_item["lens"].unique()):
        lens_frame = per_item[per_item["lens"] == lens]
        # scores[condition][eval] -> item-indexed vector, all conditions sharing an
        # item order so one index draw applies to every condition at once.
        scores: dict[str, dict[str, np.ndarray]] = {}
        items_per_eval: dict[str, list] = {}
        for eval_slug in MACRO_EVALS:
            eval_frame = lens_frame[lens_frame["eval"] == eval_slug]
            if eval_frame.empty:
                continue
            items = sorted(eval_frame["item"].unique())
            items_per_eval[eval_slug] = items
            for condition in conditions:
                condition_frame = eval_frame[eval_frame["condition"] == condition]
                if condition_frame.empty:
                    # This lens was not fitted for this condition (the logit lens and
                    # the released lens exist outside the conditions entirely). Not
                    # comparable, as distinct from scored on different items.
                    continue
                table = condition_frame.set_index("item")["hit"].reindex(items)
                if table.isna().any():
                    raise SystemExit(
                        f"{lens} {condition} {eval_slug}: scored on "
                        f"{int(table.notna().sum())} of {len(items)} items; conditions "
                        "must be scored on the same items for a paired bootstrap"
                    )
                scores.setdefault(condition, {})[eval_slug] = table.to_numpy()

        present = [c for c in conditions if c in scores]
        if reference not in present:
            # Nothing to measure a gap against for this lens.
            continue
        draws = {condition: np.empty(resamples) for condition in present}
        for draw in range(resamples):
            macro = dict.fromkeys(present, 0.0)
            for eval_slug, items in items_per_eval.items():
                index = rng.integers(0, len(items), len(items))
                for condition in present:
                    macro[condition] += float(scores[condition][eval_slug][index].mean())
            for condition in present:
                draws[condition][draw] = macro[condition] / len(items_per_eval)

        point = {
            condition: float(np.mean([scores[condition][e].mean() for e in items_per_eval]))
            for condition in present
        }
        for condition in present:
            if condition == reference:
                continue
            gap_draws = draws[reference] - draws[condition]
            rows.append(
                {
                    "lens": lens,
                    "condition": condition,
                    "reference": reference,
                    "k": k,
                    "recall_reference": point[reference],
                    "recall_condition": point[condition],
                    "gap": point[reference] - point[condition],
                    "ci_low": float(np.percentile(gap_draws, 2.5)),
                    "ci_high": float(np.percentile(gap_draws, 97.5)),
                    "resamples": resamples,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_recall(
    ranks: pd.DataFrame,
    *,
    k: int = 10,
    resamples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Macro recall@k per (condition, lens) with a 95% interval.

    The same within-eval item resampling as the gap bootstrap, but reported per
    lens rather than as a difference, which is what the summary table needs.
    These intervals are wider than the gap intervals and must not be used to
    judge a difference between two conditions: a difference has its own interval,
    and the pairing makes that one much tighter.
    """
    hit_frame = hits(ranks, k)
    per_item = (
        hit_frame.groupby(["condition", "lens", "eval", "item"], observed=True)["hit"]
        .mean()
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    rows = []
    for (condition, lens), group in per_item.groupby(["condition", "lens"], observed=True):
        vectors = {
            eval_slug: frame["hit"].to_numpy()
            for eval_slug, frame in group.groupby("eval", observed=True)
            if eval_slug in MACRO_EVALS
        }
        if not vectors:
            continue
        draws = np.empty(resamples)
        for draw in range(resamples):
            draws[draw] = np.mean(
                [
                    vector[rng.integers(0, len(vector), len(vector))].mean()
                    for vector in vectors.values()
                ]
            )
        rows.append(
            {
                "condition": condition,
                "lens": lens,
                "k": k,
                "recall": float(np.mean([v.mean() for v in vectors.values()])),
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
                "resamples": resamples,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def check_against_cli(ranks: pd.DataFrame, path: Path) -> int:
    """Compare our recall table against the ``_pass_at_k.csv`` the CLI wrote."""
    reference_path = path.with_name(f"{path.stem}_pass_at_k.csv")
    if not reference_path.exists():
        raise SystemExit(f"no CLI table to check against at {reference_path}")
    theirs = pd.read_csv(reference_path).rename(columns={"pass_at_k": "recall_cli"})
    ours = recall_summary(ranks)
    merged = ours.merge(theirs, on=["eval", "lens", "k", "weighting"], how="outer", indicator=True)
    unmatched = merged[merged["_merge"] != "both"]
    merged["delta"] = (merged["recall"] - merged["recall_cli"]).abs()
    worst = merged["delta"].max()
    print(f"compared {len(merged)} cells against {reference_path.name}")
    print(f"  rows only on one side: {len(unmatched)}")
    print(f"  largest absolute difference: {worst:.2e}")
    if len(unmatched) or not (worst < 1e-9):
        print("\nMISMATCH - these definitions do not reproduce the pipeline's own")
        print(
            merged.nlargest(5, "delta")[["eval", "lens", "k", "weighting", "recall", "recall_cli"]]
        )
        return 1
    print("  match")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranks", nargs="+", type=Path, help="ranks_<cond>.csv files")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--k", type=int, default=10, help="k for per-layer recall and gaps")
    parser.add_argument("--reference", default=REFERENCE_CONDITION)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify these definitions against the CLI's own recall table.",
    )
    args = parser.parse_args(argv)

    ranks = load_ranks(args.ranks)
    if args.check:
        return check_against_cli(ranks, args.ranks[0])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = recall_summary(ranks)
    summary.to_csv(args.out_dir / "recall_summary.csv", index=False)
    per_layer = per_layer_recall(ranks, k=args.k)
    per_layer.to_csv(args.out_dir / "per_layer_recall.csv", index=False)
    for k in (1, args.k):
        intervals = bootstrap_recall(ranks, k=k, resamples=args.resamples, seed=args.seed)
        intervals.to_csv(args.out_dir / f"recall_ci_k{k}.csv", index=False)
        print(f"wrote recall_ci_k{k}.csv ({len(intervals)} rows)")
    print(
        f"wrote recall_summary.csv ({len(summary)} rows), per_layer_recall.csv ({len(per_layer)})"
    )

    if ranks["condition"].nunique() > 1:
        gaps = bootstrap_gaps(
            ranks,
            k=args.k,
            reference=args.reference,
            resamples=args.resamples,
            seed=args.seed,
        )
        gaps.to_csv(args.out_dir / "gap_to_reference.csv", index=False)
        print(f"wrote gap_to_reference.csv ({len(gaps)} rows)")
        headline = gaps[gaps["lens"].isin(("jpp_rlens", "pooled_rlens", "pooled_none"))]
        for row in headline.itertuples():
            print(
                f"  {row.lens:<14} {row.condition:<20} gap {row.gap * 100:6.2f}pp "
                f"[{row.ci_low * 100:6.2f}, {row.ci_high * 100:6.2f}]"
            )
    else:
        print("one condition only: no gaps to compute")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
