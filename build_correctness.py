"""Grade Qwen3.5-9B on every readout-eval item and write the correctness CSV that
the fitting and scoring stages filter items with.

The shipped ``data/jlens/evaluations/model_correctness.csv`` carries a column for
``Qwen/Qwen3.6-27B`` only, and ``load_recipe_items`` raises without a column for
the model in use. This grades all items once (one greedy step each) and writes a
copy of the shipped CSV with a ``Qwen/Qwen3.5-9B`` column appended.

    uv run python gibberish/build_correctness.py              # grade, write, report
    uv run python gibberish/build_correctness.py --report-only  # re-report an existing CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import jpp_cli  # noqa: E402  (needs scripts/ on the path)

from lens_evals.readout_evals.readout_eval_items import (  # noqa: E402
    MACRO_EVALS,
    MODEL_CORRECTNESS_CSV,
    READOUT_EVALS_DATA_DIR,
    filter_items_by_correctness,
    load_readout_eval_items,
)
from workspace_lens import get_hf_model  # noqa: E402

MODEL = "Qwen/Qwen3.5-9B"


def grade(args: argparse.Namespace, items: list) -> pd.DataFrame:
    """The shipped CSV with an ``args.hf_model_name`` column appended."""
    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    # Grading reads out no lens, so the layer list is irrelevant; max_seq_len must
    # match the scoring stages' truncation.
    runner = jpp_cli.readout_runner(
        model, layers=[0], max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN
    )
    graded = runner.grade_model_correctness(items, hf_model_name=args.hf_model_name)

    shipped = pd.read_csv(args.correctness_csv)
    if args.hf_model_name in shipped.columns:
        raise SystemExit(f"{args.correctness_csv} already has a {args.hf_model_name} column")
    merged = shipped.merge(graded, on=["eval", "item"], how="outer", indicator=True)
    ungraded = int((merged["_merge"] == "left_only").sum())
    unshipped = int((merged["_merge"] == "right_only").sum())
    if ungraded or unshipped:
        print(
            f"WARNING: {ungraded} shipped rows ungraded, "
            f"{unshipped} graded rows absent from the shipped CSV",
            flush=True,
        )
    return merged.drop(columns="_merge")


def report(args: argparse.Namespace) -> None:
    """Per-eval grading outcome and, from the library's own filter, how many items
    each eval contributes to a scored run. ``NA`` cells (ungradeable evals) are
    always kept, so survivors are the correct items plus the ungradeable ones."""
    graded_df = pd.read_csv(args.out)
    kept = filter_items_by_correctness(
        load_readout_eval_items(args.eval_data_dir, slugs=MACRO_EVALS),
        hf_model_name=args.hf_model_name,
        correctness_csv=args.out,
    )
    kept_per_eval: dict[str, int] = {}
    for item in kept:
        kept_per_eval[item.eval_slug] = kept_per_eval.get(item.eval_slug, 0) + 1

    print(f"\n{'eval':<16}{'items':>7}{'correct':>9}{'NA':>5}{'kept':>7}")
    for slug in (*MACRO_EVALS, "order-ops"):
        rows = graded_df[graded_df["eval"] == slug]
        if rows.empty:
            continue
        col = rows[args.hf_model_name]
        correct = int(col.fillna(False).astype(bool).sum())
        in_recipe = slug in MACRO_EVALS
        kept_str = str(kept_per_eval.get(slug, 0)) if in_recipe else "n/a"
        print(f"{slug:<16}{len(rows):>7}{correct:>9}{int(col.isna().sum()):>5}{kept_str:>7}")
    print(f"\n{len(kept)} items over the five recipe evals survive the filter")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default=MODEL)
    parser.add_argument("--eval-data-dir", default=READOUT_EVALS_DATA_DIR)
    parser.add_argument("--correctness-csv", default=MODEL_CORRECTNESS_CSV)
    parser.add_argument("--out", default="data/model_correctness_9b.csv")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip grading and re-report the survival counts of an existing --out CSV.",
    )
    args = parser.parse_args(argv)

    if not args.report_only:
        # Every eval, order-ops included: grading is per item, and the five-eval
        # recipe filter happens later in load_recipe_items.
        items = load_readout_eval_items(args.eval_data_dir)
        print(f"{len(items)} items over {len({i.eval_slug for i in items})} evals", flush=True)
        merged = grade(args, items)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(merged)} rows)")

    report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
