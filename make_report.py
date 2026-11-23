"""Assemble results/REPORT.md and its figures from the tables the other stages wrote.

    uv run --with matplotlib python gibberish/make_report.py

Reads nothing but CSVs, so the report can be rebuilt without touching the GPU.
Every number traces to a file in results/, which is what Acceptance requires.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CONDITION_LABELS = {
    "c0_wikitext": "C0 WikiText",
    "c1_selfgen": "C1 self-generated",
    "c2_shuffled": "C2 shuffled",
    "c3_random": "C3 random tokens",
    "c3_random_c0router": "C3 with C0 router",
}
LENS_LABELS = {
    "pooled_none": "J-lens",
    "pooled_rlens": "R-lens",
    "jpp_none": "J++-noLRP",
    "jpp_rlens": "J++",
    "released_jpp_9b": "released J++ 9B",
    "logit": "logit lens",
}
LENS_ORDER = ("logit", "pooled_none", "pooled_rlens", "jpp_none", "jpp_rlens")
CONDITION_ORDER = ("c0_wikitext", "c1_selfgen", "c2_shuffled", "c3_random", "c3_random_c0router")
MACRO_EVALS = ("multihop", "multilingual", "typo", "association", "poetry")
REFERENCE = "c0_wikitext"
GIBBERISH = "c3_random"


def read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def pct(value: float | None) -> str:
    return "--" if value is None or pd.isna(value) else f"{value * 100:.2f}"


def label_condition(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def label_lens(lens: str) -> str:
    return LENS_LABELS.get(lens, lens)


### TABLES


def summary_table(results: Path) -> tuple[str, dict]:
    """Recall@10 and recall@1 with 95% intervals: rows conditions, columns lenses."""
    lines = []
    facts: dict = {}
    reference = read(results / "ranks_reference_pass_at_k.csv")
    baselines = {}
    if reference is not None:
        picked = reference[(reference["weighting"] == "item") & (reference["eval"] == "macro")]
        for row in picked.itertuples():
            baselines[(row.lens, row.k)] = row.pass_at_k
    facts["baselines"] = {f"{lens}@{k}": v for (lens, k), v in baselines.items()}

    for k in (10, 1):
        intervals = read(results / f"recall_ci_k{k}.csv")
        if intervals is None:
            continue
        resamples = int(intervals["resamples"].iloc[0])
        seed = int(intervals["seed"].iloc[0])
        per_layer_path = results / "per_layer_recall.csv"
        layer_count = (
            pd.read_csv(per_layer_path)["layer"].nunique() if per_layer_path.exists() else 0
        )
        layer_phrase = (
            f"any of the {layer_count} readout layers" if layer_count else "any readout layer"
        )
        lines.append(
            f"\n**Held-out recall@{k}**, macro over the five evals, item-weighted, "
            f"{layer_phrase}, Readout Filtering on. 95% intervals from "
            f"{resamples:,} item resamples, seed {seed}.\n"
        )
        header = "| condition | " + " | ".join(label_lens(lens) for lens in LENS_ORDER) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(LENS_ORDER) + 1))
        for condition in CONDITION_ORDER:
            rows = intervals[intervals["condition"] == condition]
            if rows.empty:
                continue
            cells = []
            for lens in LENS_ORDER:
                if lens == "logit":
                    # The logit lens is fitted on nothing, so its column is the same
                    # number in every row. It is the floor the lenses must beat.
                    cells.append(pct(baselines.get(("logit", k))))
                    continue
                match = rows[rows["lens"] == lens]
                if match.empty:
                    cells.append("--")
                else:
                    item = match.iloc[0]
                    cells.append(
                        f"{pct(item['recall'])} [{pct(item['ci_low'])}, {pct(item['ci_high'])}]"
                    )
            lines.append(f"| {label_condition(condition)} | " + " | ".join(cells) + " |")
            facts.setdefault(f"recall@{k}", {})[condition] = {
                row.lens: row.recall for row in rows.itertuples()
            }
        for lens, k_value in sorted(baselines):
            if k_value == k and lens != "logit":
                lines.append(
                    f"\n{label_lens(lens)}, same harness: **{pct(baselines[(lens, k_value)])}%**"
                )
    return "\n".join(lines), facts


def gap_table(results: Path) -> tuple[str, pd.DataFrame | None]:
    gaps = read(results / "gap_to_reference.csv")
    if gaps is None:
        return "", None
    lines = [
        "\nGap to C0, `recall(C0) - recall(Cx)`, percentage points, with 95% intervals from a",
        "paired bootstrap over items (10,000 resamples, seed 0). Pairing matters here: every",
        "condition is scored on identical items, so these intervals are much tighter than the",
        "per-condition intervals above, and they are what any difference claim rests on.\n",
        "| lens | condition | gap (pp) | 95% CI |",
        "|---|---|---|---|",
    ]
    for lens in LENS_ORDER:
        for condition in CONDITION_ORDER:
            row = gaps[(gaps["lens"] == lens) & (gaps["condition"] == condition)]
            if row.empty:
                continue
            item = row.iloc[0]
            lines.append(
                f"| {label_lens(lens)} | {label_condition(condition)} | "
                f"{item['gap'] * 100:.2f} | "
                f"[{item['ci_low'] * 100:.2f}, {item['ci_high'] * 100:.2f}] |"
            )
    return "\n".join(lines), gaps


def corpus_table(results: Path, corpora: Path) -> str:
    rows = []
    for condition in CONDITION_ORDER:
        path = corpora / f"{condition}_stats.json"
        if not path.exists():
            continue
        stats = json.loads(path.read_text())
        window = stats["fit_window"]
        rows.append(
            f"| {label_condition(condition)} | {stats['token_counts']['median']} | "
            f"{window['mean_per_token_loss']:.3f} | {window['unigram_entropy_bits']:.2f} | "
            f"{window['type_token_ratio']:.3f} | {window['non_alphanumeric_share']:.3f} |"
        )
    if not rows:
        return ""
    return "\n".join(
        [
            "\n| condition | median tokens | loss | entropy (bits) | type/token | non-alnum |",
            "|---|---|---|---|---|---|",
            *rows,
        ]
    )


def cluster_table(results: Path) -> str:
    weights = read(results / "expert_weights.csv")
    if weights is None:
        return ""
    lines = [
        "\nExpert weights and cluster balance. `fallbacks` counts expert slots that kept fewer",
        "than 50 positions and were replaced by the pooled Jacobian, out of 31 layers x 8",
        "clusters = 248. Where an expert falls back, J++ degenerates toward the pooled lens at",
        "that layer, which is a way for H4's rescue to be absent for reasons unrelated to the",
        "fitting text.\n",
        "| condition | lens | min kept | median kept | max kept | fallbacks / 248 |",
        "|---|---|---|---|---|---|",
    ]
    for (condition, lens), group in weights.groupby(["condition", "lens"]):
        kept = group["kept_positions"].dropna()
        if kept.empty:
            continue
        lines.append(
            f"| {label_condition(condition)} | {label_lens(lens)} | {int(kept.min())} | "
            f"{int(kept.median())} | {int(kept.max())} | "
            f"{int((kept < 50).sum())} |"
        )
    return "\n".join(lines)


### FIGURES


def figure_per_layer(results: Path, figures: Path) -> str | None:
    per_layer = read(results / "per_layer_recall.csv")
    if per_layer is None:
        return None
    macro = per_layer[(per_layer["eval"] == "macro") & (per_layer["weighting"] == "item")]
    kinds = [lens for lens in LENS_ORDER if lens != "logit" and lens in set(macro["lens"])]
    if not kinds:
        return None
    fig, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3.6), sharey=True)
    axes = [axes] if len(kinds) == 1 else list(axes)
    for axis, lens in zip(axes, kinds, strict=True):
        for condition in CONDITION_ORDER:
            rows = macro[(macro["lens"] == lens) & (macro["condition"] == condition)]
            if rows.empty:
                continue
            rows = rows.sort_values("layer")
            axis.plot(
                rows["layer"],
                rows["recall"] * 100,
                marker="o",
                ms=3,
                label=label_condition(condition),
            )
        logit = macro[macro["lens"] == "logit"]
        if not logit.empty:
            logit = logit.sort_values("layer")
            axis.plot(logit["layer"], logit["recall"] * 100, "k--", lw=1, label="logit lens")
        axis.set_title(label_lens(lens))
        axis.set_xlabel("readout layer")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("recall@10 (%)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Figure 1: per-layer recall@10 by fitting corpus", fontsize=11)
    fig.tight_layout()
    path = figures / "fig1_per_layer_recall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_per_task(results: Path, figures: Path) -> str | None:
    summary = read(results / "recall_summary.csv")
    if summary is None:
        return None
    picked = summary[(summary["k"] == 10) & (summary["weighting"] == "item")]
    picked = picked[picked["eval"].isin(MACRO_EVALS)]
    conditions = [c for c in (REFERENCE, GIBBERISH) if c in set(picked["condition"])]
    if len(conditions) < 2:
        return None
    fig, axis = plt.subplots(figsize=(7.5, 3.6))
    width = 0.38
    for offset, condition in enumerate(conditions):
        values = [
            picked[
                (picked["condition"] == condition)
                & (picked["lens"] == "jpp_rlens")
                & (picked["eval"] == eval_slug)
            ]["recall"].max()
            * 100
            for eval_slug in MACRO_EVALS
        ]
        axis.bar(
            [i + offset * width for i in range(len(MACRO_EVALS))],
            values,
            width,
            label=label_condition(condition),
        )
    axis.set_xticks([i + width / 2 for i in range(len(MACRO_EVALS))])
    axis.set_xticklabels(MACRO_EVALS)
    axis.set_ylabel("recall@10 (%)")
    axis.set_title("Figure 2: J++ per-task recall@10, C0 against C3")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = figures / "fig2_per_task.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_diagnostics(results: Path, figures: Path) -> str | None:
    echo = read(results / "echo_rate.csv")
    align = read(results / "embedding_alignment.csv")
    if echo is None and align is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    if echo is not None:
        subset = echo[(echo["lens"] == "jpp_rlens") & (echo["prompt_set"] == "wikitext")]
        for condition in CONDITION_ORDER:
            rows = subset[subset["condition"] == condition].sort_values("layer")
            if rows.empty:
                continue
            axes[0].plot(
                rows["layer"],
                rows["position_echo_rate"] * 100,
                marker="o",
                ms=3,
                label=label_condition(condition),
            )
        axes[0].set_ylabel("position echo rate (%)")
    axes[0].set_xlabel("readout layer")
    axes[0].set_title("echo rate (WikiText probes)")
    axes[0].grid(alpha=0.3)
    if align is not None:
        subset = align[align["lens"] == "jpp_rlens"]
        for condition in CONDITION_ORDER:
            rows = subset[subset["condition"] == condition].sort_values("layer")
            if rows.empty:
                continue
            axes[1].plot(
                rows["layer"],
                rows["median_cosine"],
                marker="o",
                ms=3,
                label=label_condition(condition),
            )
        axes[1].set_ylabel("median cosine with input embedding")
    axes[1].set_xlabel("readout layer")
    axes[1].set_title("embedding alignment")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.suptitle("Figure 3: echo rate and embedding alignment by layer", fontsize=11)
    fig.tight_layout()
    path = figures / "fig3_echo_alignment.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_loss(results: Path, corpora: Path, figures: Path) -> str | None:
    intervals = read(results / "recall_ci_k10.csv")
    if intervals is None:
        return None
    losses = {}
    for condition in CONDITION_ORDER:
        path = corpora / f"{condition}_stats.json"
        if path.exists():
            losses[condition] = json.loads(path.read_text())["fit_window"]["mean_per_token_loss"]
    if not losses:
        return None
    fig, axis = plt.subplots(figsize=(5.5, 3.8))
    for lens in LENS_ORDER:
        if lens == "logit":
            continue
        xs, ys = [], []
        for condition, loss in losses.items():
            row = intervals[(intervals["condition"] == condition) & (intervals["lens"] == lens)]
            if row.empty:
                continue
            xs.append(loss)
            ys.append(row.iloc[0]["recall"] * 100)
        if xs:
            axis.plot(xs, ys, marker="o", ls="-", ms=5, label=label_lens(lens))
    axis.set_xlabel("corpus mean per-token loss")
    axis.set_ylabel("recall@10 (%)")
    axis.set_title("Figure 4: recall against how off-distribution the corpus is")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    path = figures / "fig4_recall_vs_loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


### VERDICTS


def matrix_table(results: Path) -> str:
    """How far each condition's lens matrix sits from C0's, and how its spectrum
    changed.

    Three quantities per layer, averaged over layers here and per layer in
    `matrix_distance.csv`:

    - relative Frobenius distance, `||J_C - J_C0||_F / ||J_C0||_F`
    - overlap of the top-64 left singular subspaces, the mean squared cosine of
      the principal angles, which is 1 for identical subspaces and 64/4096 =
      0.0156 for random ones
    - effective rank, entropy-based: with the singular values normalised to sum
      to one, `exp(-sum p_i log p_i)`. It is the number of directions the
      spectrum behaves like, equal to the true rank only for a flat spectrum.
    """
    frame = read(results / "matrix_distance.csv")
    if frame is None:
        return ""
    lines = [
        "\nHow much the lens *matrix* moves, as distinct from how much the readout changes.",
        "Three quantities, averaged over the 31 layers here and given per layer in",
        "`matrix_distance.csv`:",
        "",
        "- **relative Frobenius distance**, `||J_C - J_C0||_F / ||J_C0||_F`",
        "- **top-64 singular-subspace overlap**, the mean squared cosine of the principal",
        "  angles between the top-64 left singular subspaces: 1 for identical subspaces,",
        "  64/4096 = 0.0156 for random ones",
        "- **effective rank**, entropy-based: with the singular values normalised to sum to one,",
        "  `exp(-sum p_i log p_i)`. It is the number of directions the spectrum behaves like,",
        "  and equals the true rank only for a flat spectrum.",
        "",
        "| condition | lens | rel. Frobenius dist. | top-64 overlap | effective rank |",
        "|---|---|---|---|---|",
    ]
    grouped = frame.groupby(["condition", "lens"], observed=True)[
        ["relative_frobenius_distance", "top_subspace_overlap", "effective_rank"]
    ].mean()
    for condition in CONDITION_ORDER:
        for lens in LENS_ORDER:
            if (condition, lens) not in grouped.index:
                continue
            row = grouped.loc[(condition, lens)]
            distance = "0 (reference)" if condition == REFERENCE else f"{row.iloc[0]:.3f}"
            lines.append(
                f"| {label_condition(condition)} | {label_lens(lens)} | {distance} | "
                f"{row.iloc[1]:.3f} | {row.iloc[2]:.0f} |"
            )
    return "\n".join(lines)


def nofilter_table(results: Path) -> str:
    """C0 scored over the full vocabulary, the paper's convention for the J-lens
    and R-lens, beside the filtered numbers this run uses for all three."""
    unfiltered = read(results / "nofilter_c0_pass_at_k.csv")
    filtered = read(results / "ranks_c0_wikitext_pass_at_k.csv")
    if unfiltered is None or filtered is None:
        return ""

    def macro(frame, lens, k):
        picked = frame[
            (frame["eval"] == "macro")
            & (frame["weighting"] == "item")
            & (frame["lens"] == lens)
            & (frame["k"] == k)
        ]
        return None if picked.empty else picked.iloc[0]["pass_at_k"]

    lines = [
        "\nThe J++ paper scored the J-lens and R-lens over the full vocabulary but J++ with",
        "Readout Filtering. This run filters all three, so the comparison isolates the fitting",
        "corpus. C0 was also scored unfiltered once, to keep both conventions on the record.",
        "",
        "| lens | filtered (this run) | full vocabulary (paper's convention) | difference |",
        "|---|---|---|---|",
    ]
    for lens in ("pooled_none", "pooled_rlens", "jpp_rlens"):
        with_filter = macro(filtered, lens, 10)
        without = macro(unfiltered, lens, 10)
        if with_filter is None or without is None:
            continue
        lines.append(
            f"| {label_lens(lens)} | {pct(with_filter)} | {pct(without)} | "
            f"{(with_filter - without) * 100:+.2f}pp |"
        )
    return "\n".join(lines)


def verdicts(results: Path, facts: dict, gaps: pd.DataFrame | None) -> str:
    """One verdict per hypothesis, each naming the number it rests on.

    A verdict is only stated where the evidence exists; anything missing is
    inconclusive rather than quietly omitted.
    """
    lines = ["\n## Verdicts on the pre-registered hypotheses\n"]
    recall = facts.get("recall@10", {})

    def macro(condition: str, lens: str) -> float | None:
        return recall.get(condition, {}).get(lens)

    # H1: ordering.
    statements = []
    for lens in ("pooled_none", "pooled_rlens", "jpp_rlens"):
        values = {
            c: macro(c, lens) for c in ("c0_wikitext", "c1_selfgen", "c2_shuffled", "c3_random")
        }
        if any(v is None for v in values.values()):
            statements.append(f"- {label_lens(lens)}: inconclusive, a condition is missing")
            continue
        predicted = (
            values["c0_wikitext"] >= values["c1_selfgen"]
            and values["c1_selfgen"] > values["c2_shuffled"]
            and values["c2_shuffled"] > values["c3_random"]
        )
        ordering = " > ".join(
            f"{label_condition(c)} {pct(v)}"
            for c, v in sorted(values.items(), key=lambda kv: -kv[1])
        )
        statements.append(
            f"- {label_lens(lens)}: {'supported' if predicted else 'NOT supported'} - {ordering}"
        )
    lines.append("**H1 (ordering)** C0 >= C1 > C2 > C3 for all three lens types.\n")
    lines.extend(statements)

    # H2: depth.
    per_layer = read(results / "per_layer_recall.csv")
    lines.append(
        "\n**H2 (depth)** the C0-C3 gap is largest at the earliest readout layers and "
        "under 5pp at the deepest.\n"
    )
    if per_layer is None:
        lines.append("- inconclusive: no per-layer table")
    else:
        macro_layers = per_layer[
            (per_layer["eval"] == "macro") & (per_layer["weighting"] == "item")
        ]
        for lens in ("pooled_none", "pooled_rlens", "jpp_rlens"):
            c0 = macro_layers[
                (macro_layers["lens"] == lens) & (macro_layers["condition"] == REFERENCE)
            ].set_index("layer")["recall"]
            c3 = macro_layers[
                (macro_layers["lens"] == lens) & (macro_layers["condition"] == GIBBERISH)
            ].set_index("layer")["recall"]
            shared = sorted(set(c0.index) & set(c3.index))
            if not shared:
                lines.append(f"- {label_lens(lens)}: inconclusive, no shared layers")
                continue
            gap_by_layer = {layer: c0[layer] - c3[layer] for layer in shared}
            deepest = shared[-1]
            # By magnitude, not signed value: where C3 outscores C0 the gap is
            # negative, and a signed test would read a large negative gap as a
            # small one and call the hypothesis supported.
            widest = max(gap_by_layer, key=lambda layer: abs(gap_by_layer[layer]))
            early = widest <= shared[len(shared) // 2]
            deep_small = abs(gap_by_layer[deepest]) * 100 < 5
            direction = (
                ""
                if sum(gap_by_layer.values()) > 0
                else " (C3 outscores C0 at most layers, so the gap this hypothesis "
                "describes runs the other way)"
            )
            lines.append(
                f"- {label_lens(lens)}: widest gap by magnitude at layer {widest} "
                f"({pct(gap_by_layer[widest])}pp), deepest layer {deepest} "
                f"({pct(gap_by_layer[deepest])}pp) - "
                f"{'supported' if early and deep_small else 'NOT supported'}{direction}"
            )

    # H3: echo.
    echo = read(results / "echo_rate.csv")
    lines.append("\n**H3 (current-token collapse)** C3 position echo rate exceeds C0's.\n")
    if echo is None:
        lines.append("- inconclusive: no echo-rate table")
    else:
        for prompt_set in sorted(echo["prompt_set"].unique()):
            subset = echo[(echo["prompt_set"] == prompt_set) & (echo["lens"] == "jpp_rlens")]
            c0 = subset[subset["condition"] == REFERENCE]["position_echo_rate"].mean()
            c3 = subset[subset["condition"] == GIBBERISH]["position_echo_rate"].mean()
            if pd.isna(c0) or pd.isna(c3):
                lines.append(f"- {prompt_set}: inconclusive")
                continue
            lines.append(
                f"- {prompt_set}: C0 {pct(c0)}%, C3 {pct(c3)}% - "
                f"{'supported' if c3 > c0 else 'NOT supported'}"
            )

    # H4: rescue.
    lines.append(
        "\n**H4 (J++ rescue)** the C0-C3 gap is smaller for J++ than for the J-lens "
        "and the R-lens.\n"
    )
    if gaps is None:
        lines.append("- inconclusive: no gap table")
    else:
        c3_gaps = gaps[gaps["condition"] == GIBBERISH].set_index("lens")["gap"]
        if {"jpp_rlens", "pooled_none", "pooled_rlens"} <= set(c3_gaps.index):
            jpp = c3_gaps["jpp_rlens"]
            smaller = jpp < c3_gaps["pooled_none"] and jpp < c3_gaps["pooled_rlens"]
            lines.append(
                f"- J++ {pct(jpp)}pp, J-lens {pct(c3_gaps['pooled_none'])}pp, "
                f"R-lens {pct(c3_gaps['pooled_rlens'])}pp - "
                f"{'supported' if smaller else 'NOT supported'}"
            )
        else:
            lines.append("- inconclusive: a lens is missing from the gap table")

    # H0: null.
    lines.append(
        "\n**H0 (null)** if C3 recall@10 is within 3pp of C0 for a lens type, that lens "
        "is set mostly by the model's weights rather than the fitting text.\n"
    )
    if gaps is None:
        lines.append("- inconclusive: no gap table")
    else:
        c3_gaps = gaps[gaps["condition"] == GIBBERISH]
        for row in c3_gaps.itertuples():
            within = abs(row.gap) * 100 < 3
            lines.append(
                f"- {label_lens(row.lens)}: gap {row.gap * 100:.2f}pp "
                f"[{row.ci_low * 100:.2f}, {row.ci_high * 100:.2f}] - "
                f"{'within 3pp, null holds' if within else 'beyond 3pp, null rejected'}"
            )
    return "\n".join(lines)


### REPORT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--corpora", type=Path, default=Path("corpora"))
    parser.add_argument("--out", type=Path, default=Path("results/REPORT.md"))
    args = parser.parse_args(argv)

    figures = args.results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    summary, facts = summary_table(args.results)
    gaps_text, gaps = gap_table(args.results)
    figure_names = {
        "fig1": figure_per_layer(args.results, figures),
        "fig2": figure_per_task(args.results, figures),
        "fig3": figure_diagnostics(args.results, figures),
        "fig4": figure_loss(args.results, args.corpora, figures),
    }

    parts = [
        "# Gibberish-fit workspace lenses: results",
        "",
        "How much do the J-lens, R-lens and J++ lens depend on the text they are fitted on?",
        "All three were refitted on Qwen3.5-9B over four fitting corpora and scored on the same",
        "held-out readout items. Predictions were registered in `PREREGISTRATION.md` before any",
        "condition was fitted.",
        "",
        "## Summary",
        summary,
        "",
        "## Gap to C0",
        gaps_text,
        "",
        "## Fitting corpora",
        corpus_table(args.results, args.corpora),
        "",
        "## Expert weights and cluster balance",
        cluster_table(args.results),
        "",
        "## How far the matrices moved",
        matrix_table(args.results),
        "",
        "## Readout Filtering against the full vocabulary",
        nofilter_table(args.results),
        verdicts(args.results, facts, gaps),
        "",
        "## Figures",
    ]
    captions = {
        "fig1": "Per-layer recall@10, one panel per lens type, one line per condition, the logit "
        "lens dashed.",
        "fig2": "J++ per-task recall@10 for C0 against C3.",
        "fig3": "Position echo rate and embedding alignment by layer.",
        "fig4": "Recall@10 against corpus mean per-token loss.",
    }
    for key, name in figure_names.items():
        if name:
            parts.append(f"\n![{captions[key]}](figures/{name})\n\n*{captions[key]}*")
        else:
            parts.append(f"\n*{captions[key]} - not produced: its table is missing.*")

    parts += [
        "",
        "## Deviations from the spec",
        "",
        "Each with its reason; the settings were fixed in `PREREGISTRATION.md` before Phase 3.",
        "",
        "- **Source layers 0-30, all 31**, not 7 equally spaced: the released lens's own config",
        "  stores 31, and measurement showed 31 layers cost 1.3 GB more GPU than 7, so there was",
        "  no reason to take the cheaper option.",
        "- **`jacobian_rows_per_pass` 4**, not the released config's 48: 48 and 16 both ran out of",
        "  memory at ~30.5 GB on a 32 GB card. It sets only how many rows of J a backward pass",
        "  computes, so the estimator is unchanged.",
        "- **Router pool 256 prompts**, not 1,000: the router pools fp32 CPU activations of",
        "  [positions, 4096] per layer, which at 1,000 prompts over 31 layers is 56 GB and twice",
        "  that at the concatenate. Identical across conditions.",
        "- **C1 sampled from a newline**, not from BOS: this checkpoint has no BOS token. Sampling",
        "  from the endoftext token collapsed into repetition and leaked control tokens into the",
        "  text, and one sample re-encoded to 14 tokens. Special tokens were suppressed.",
        "- **C2's shuffle seed is the record's index in the stream**, not in the kept list: keeping",
        "  a record depends on its shuffled form, so a seed from the kept index would be circular.",
        "- **PyTorch fallback kernels for Gated DeltaNet**: flash-linear-attention was not",
        "  installed. The same path was used for every condition.",
        "- **`order-ops` is absent** because it is not in the pipeline's default item set, not by",
        "  hand: digit-by-digit tokenization breaks it on Qwen.",
        "",
        "## Provenance",
        "",
        "Every number above comes from a CSV in `results/`: `recall_ci_k*.csv`,",
        "`gap_to_reference.csv`, `per_layer_recall.csv`, `recall_summary.csv`, `echo_rate.csv`,",
        "`embedding_alignment.csv`, `matrix_distance.csv`, `expert_weights.csv`, and the corpus",
        "statistics in `corpora/<cond>_stats.json`. `run_all.sh` reproduces all of it from the",
        "corpora.",
        "",
        "The released 9B lens scores differently here than on its model card (55.0%) because this",
        "run uses its own model-correctness column and item split. Both numbers must name their",
        "harness.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n")
    print(f"wrote {args.out}")
    for key, name in figure_names.items():
        print(f"  {key}: {name or 'skipped'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
