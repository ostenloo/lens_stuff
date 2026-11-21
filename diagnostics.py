"""The four diagnostics the spec asks for, one subcommand each.

    uv run python gibberish/diagnostics.py weights   # expert weights + cluster balance
    uv run python gibberish/diagnostics.py matrix    # distance from the C0 lens
    uv run python gibberish/diagnostics.py echo      # position and window echo rate (H3)
    uv run python gibberish/diagnostics.py align     # embedding alignment

``weights`` and ``matrix`` read the saved lenses only. ``echo`` and ``align`` need
the model, so they want the GPU to themselves; run them after the fits.

A lens file holds one fp32 ``[F, N]`` matrix per source layer: rows are
target-layer dimensions, columns source-layer dimensions. The readout for an
activation ``h`` at layer l is ``unembed(J_l @ h)``, the lens transporting h to
the final layer and the model's own unembedding decoding it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch as t

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import jpp_cli  # noqa: E402  (needs scripts/ on the path)

CONDITIONS = ("c0_wikitext", "c1_selfgen", "c2_shuffled", "c3_random", "c3_random_c0router")
LENS_KINDS = ("pooled_none", "pooled_rlens", "jpp_none", "jpp_rlens")
REFERENCE = "c0_wikitext"
MODEL = "Qwen/Qwen3.5-9B"


### LENS FILES


def lens_path(artifacts: Path, condition: str, kind: str) -> Path:
    return artifacts / condition / f"{kind}.pt"


def load_jacobians(path: Path) -> dict[int, t.Tensor]:
    """``{layer: [F, N]}`` fp32 on the CPU, from a saved lens."""
    state = t.load(path, map_location="cpu", weights_only=False)
    return {int(layer): m.float() for layer, m in state["parameters"]["jacobians"].items()}


def available_lenses(artifacts: Path, conditions, kinds) -> list[tuple[str, str]]:
    found = []
    for condition in conditions:
        for kind in kinds:
            if lens_path(artifacts, condition, kind).exists():
                found.append((condition, kind))
    return found


### 1. EXPERT WEIGHTS AND CLUSTER BALANCE


def expert_weights(args: argparse.Namespace) -> int:
    """Learned weight per (condition, lens, layer, expert), with the per-expert
    kept-position counts that say whether a cluster was populated enough to be
    fitted at all. Gibberish may cluster very unevenly, which is itself a result.
    """
    rows = []
    for condition in args.conditions:
        for mode in ("none", "rlens"):
            weights_file = args.artifacts / condition / f"jpp_{mode}_expert_weights.json"
            if not weights_file.exists():
                continue
            payload = json.loads(weights_file.read_text())
            counts = position_counts(args.artifacts / condition / f"experts_{mode}.pt")
            for layer, layer_payload in payload["by_layer"].items():
                weights = layer_payload["weights_unit_l2"]
                shares = layer_payload.get("signed_shares", [None] * len(weights))
                layer_counts = counts.get(int(layer), [None] * len(weights))
                for expert, weight in enumerate(weights):
                    rows.append(
                        {
                            "condition": condition,
                            "lens": f"jpp_{mode}",
                            "layer": int(layer),
                            "expert": expert,
                            "weight_unit_l2": weight,
                            "signed_share": shares[expert],
                            "kept_positions": layer_counts[expert]
                            if expert < len(layer_counts)
                            else None,
                            "train_macro_pass": layer_payload.get("train_macro_pass"),
                            "init": layer_payload.get("init"),
                            "step": layer_payload.get("step"),
                        }
                    )
    if not rows:
        raise SystemExit(f"no expert-weight files under {args.artifacts}")
    frame = pd.DataFrame(rows)
    out = args.out_dir / "expert_weights.csv"
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} rows)")

    print(f"\n{'condition':<20}{'lens':<12}{'min kept':>10}{'max kept':>10}{'fallbacks':>11}")
    for (condition, lens), group in frame.groupby(["condition", "lens"]):
        kept = group["kept_positions"].dropna()
        if kept.empty:
            print(f"{condition:<20}{lens:<12}{'n/a':>10}{'n/a':>10}{'n/a':>11}")
            continue
        fallbacks = int((kept < args.min_kept_positions).sum())
        print(f"{condition:<20}{lens:<12}{int(kept.min()):>10}{int(kept.max()):>10}{fallbacks:>11}")
    return 0


def position_counts(experts_file: Path) -> dict[int, list[int]]:
    """``{layer: [count per expert]}`` from an experts file, memory-mapped so the
    9 GB of Jacobians is never read."""
    if not experts_file.exists():
        return {}
    try:
        state = t.load(experts_file, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        state = t.load(experts_file, map_location="cpu", weights_only=True)
    counts = state.get("position_counts", {})
    return {int(layer): [int(c) for c in value] for layer, value in counts.items()}


### 2. MATRIX DISTANCE, SUBSPACE OVERLAP, EFFECTIVE RANK


def effective_rank(singular_values: t.Tensor) -> float:
    """The entropy-based effective rank: with p_i the singular values normalised
    to sum to one, ``exp(-sum p_i log p_i)``. It is the number of directions the
    spectrum behaves like, and equals the true rank only for a flat spectrum."""
    p = singular_values / singular_values.sum()
    p = p[p > 0]
    return float(t.exp(-(p * p.log()).sum()))


def subspace_overlap(left: t.Tensor, right: t.Tensor, rank: int) -> float:
    """Mean squared cosine of the principal angles between the top-``rank`` left
    singular subspaces: ``||U_a^T U_b||_F^2 / rank``, which is 1 for identical
    subspaces and ``rank/N`` for random ones."""
    return float((left[:, :rank].T @ right[:, :rank]).pow(2).sum() / rank)


def matrix_distance(args: argparse.Namespace) -> int:
    """Per layer, how far each condition's lens is from the C0 lens of the same
    kind: relative Frobenius distance, top-64 singular-subspace overlap, and the
    effective rank of each. Computed in fp32 whatever the file dtype."""
    device = t.device(args.device)
    rows = []
    for kind in args.kinds:
        reference_file = lens_path(args.artifacts, args.reference, kind)
        if not reference_file.exists():
            print(f"skipping {kind}: no {args.reference} lens to compare against")
            continue
        reference = load_jacobians(reference_file)
        reference_svd: dict[int, tuple[t.Tensor, t.Tensor]] = {}
        for layer, matrix in reference.items():
            u, s, _ = t.linalg.svd(matrix.to(device), full_matrices=False)
            reference_svd[layer] = (u, s)

        for condition in args.conditions:
            path = lens_path(args.artifacts, condition, kind)
            if not path.exists():
                continue
            jacobians = load_jacobians(path)
            for layer in sorted(jacobians):
                if layer not in reference:
                    continue
                matrix = jacobians[layer].to(device)
                reference_matrix = reference[layer].to(device)
                u_reference, s_reference = reference_svd[layer]
                u, s, _ = t.linalg.svd(matrix, full_matrices=False)
                reference_norm = float(reference_matrix.norm())
                rows.append(
                    {
                        "condition": condition,
                        "lens": kind,
                        "layer": layer,
                        "relative_frobenius_distance": float(
                            (matrix - reference_matrix).norm() / reference_norm
                        ),
                        "frobenius_norm": float(matrix.norm()),
                        "frobenius_norm_reference": reference_norm,
                        "top_subspace_overlap": subspace_overlap(u, u_reference, args.rank),
                        "subspace_rank": args.rank,
                        "effective_rank": effective_rank(s),
                        "effective_rank_reference": effective_rank(s_reference),
                    }
                )
            print(f"  {condition} {kind}: {len(jacobians)} layers", flush=True)
            del jacobians
    if not rows:
        raise SystemExit("no lens pairs to compare")
    frame = pd.DataFrame(rows)
    out = args.out_dir / "matrix_distance.csv"
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} rows)")
    summary = (
        frame[frame["condition"] != args.reference]
        .groupby(["lens", "condition"])[["relative_frobenius_distance", "top_subspace_overlap"]]
        .mean()
    )
    if not summary.empty:
        print("\nmean over layers:")
        print(summary.round(4).to_string())
    return 0


### MODEL-SIDE HELPERS


class ModelHandles:
    """The model, its unembedding, and the Readout Filtering exclusion list."""

    def __init__(self, hf_model_name: str, layers: list[int], max_seq_len: int) -> None:
        from lens_evals.readout_evals.readout_eval_items import (
            load_readout_eval_items,
            non_semantic_token_ids,
        )
        from workspace_lens import get_hf_model
        from workspace_lens.utils import vocab_size_of

        self.model = get_hf_model(hf_model_name, attn_implementation="sdpa")
        self.tokenizer = self.model.tokenizer
        self.layers = layers
        self.max_seq_len = max_seq_len
        self.device = self.model.input_device
        self.vocab_size = vocab_size_of(self.model)
        self.excluded = t.tensor(
            sorted(
                non_semantic_token_ids(
                    self.tokenizer,
                    vocab_size=self.vocab_size,
                    items=load_readout_eval_items(jpp_cli.READOUT_EVALS_DATA_DIR),
                )
            ),
            dtype=t.long,
        )

    def activations(self, prompt: str) -> tuple[t.Tensor, dict[int, t.Tensor]]:
        from workspace_lens.routing.cluster import record_activations

        input_ids, activations = record_activations(
            self.model, prompt, self.max_seq_len, self.layers
        )
        return input_ids[0], activations

    @t.no_grad()
    def top_k_readouts(self, transported: t.Tensor, k: int, *, filtered: bool) -> t.Tensor:
        """Top-k vocabulary ids for each row of ``transported`` (final-layer
        residuals), decoded with the model's unembedding. With ``filtered`` the
        Readout Filtering exclusions are forced to -inf first, as scoring does."""
        logits = self.model.unembed(transported).float()
        if filtered:
            logits[:, self.excluded.to(logits.device)] = float("-inf")
        return logits.topk(k, dim=-1).indices


def transport(jacobian: t.Tensor, activations: t.Tensor) -> t.Tensor:
    """``J @ h`` for a batch of activations ``[P, N]`` -> ``[P, F]``."""
    return activations.to(jacobian.dtype) @ jacobian.T


### 3. ECHO RATE (H3)


def wikitext_probe_prompts(count: int, exclude: set[str], seed: int) -> list[str]:
    """WikiText records not used in any corpus, for echo rates measured away from
    the eval prompts' trailing punctuation."""
    from jlens.examples import load_wikitext_prompts

    pool = load_wikitext_prompts(count * 4)
    fresh = [record for record in pool if record not in exclude]
    random.Random(seed).shuffle(fresh)
    return fresh[:count]


@t.no_grad()
def echo_rate(args: argparse.Namespace) -> int:
    """Share of the top-10 readouts that repeat the input.

    Position echo rate, the primary measure for H3, counts matches with the input
    token at the readout position itself. Window echo rate counts matches with any
    of the last three input tokens. Both are measured on the eval items at their
    final prompt position, and on fresh WikiText records at sampled positions whose
    own token has a letter or digit -- the eval prompts often end in punctuation,
    which Readout Filtering removes from the ranking, so that set alone would
    understate the effect.
    """
    from lens_evals.readout_evals.readout_eval_items import load_recipe_items

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    lenses = available_lenses(args.artifacts, args.conditions, args.kinds)
    if not lenses:
        raise SystemExit("no lenses found")
    first = load_jacobians(lens_path(args.artifacts, *lenses[0]))
    layers = layers or sorted(first)
    del first

    handles = ModelHandles(args.hf_model_name, layers, args.readout_max_seq_len)
    items = load_recipe_items(
        data_dir=jpp_cli.READOUT_EVALS_DATA_DIR,
        correctness_csv=args.correctness_csv,
        hf_model_name=args.hf_model_name,
    )[: args.items]
    print(f"{len(items)} eval items, {len(layers)} layers", flush=True)

    corpus_texts = set()
    for condition in ("c0_wikitext", "c2_shuffled"):
        path = Path("corpora") / f"{condition}.json"
        if path.exists():
            corpus_texts.update(json.loads(path.read_text()))
    probes = wikitext_probe_prompts(args.wikitext_records, corpus_texts, args.seed)
    print(f"{len(probes)} fresh WikiText probe records", flush=True)

    # Collect activations once; every lens reads the same residuals.
    def collect(prompts, positions_for):
        store = {layer: [] for layer in layers}
        targets, windows = [], []
        for prompt in prompts:
            input_ids, activations = handles.activations(prompt)
            chosen = positions_for(input_ids)
            if not chosen:
                continue
            for layer in layers:
                store[layer].append(activations[layer][chosen])
            for position in chosen:
                targets.append(int(input_ids[position]))
                windows.append([int(i) for i in input_ids[max(0, position - 2) : position + 1]])
        return (
            {layer: t.cat(rows) for layer, rows in store.items() if rows},
            targets,
            windows,
        )

    def eval_positions(input_ids):
        return [int(input_ids.shape[0] - 1)]

    def probe_positions(input_ids):
        candidates = [
            p
            for p in range(args.skip_first_n_positions, int(input_ids.shape[0]))
            if any(c.isalnum() for c in handles.tokenizer.decode([int(input_ids[p])]))
        ]
        rng = random.Random(args.seed + len(candidates))
        rng.shuffle(candidates)
        return sorted(candidates[: args.positions_per_record])

    sets = {
        "eval_items": collect([item.prompt for item in items], eval_positions),
        "wikitext": collect(probes, probe_positions),
    }
    for name, (_store, targets, _) in sets.items():
        print(f"  {name}: {len(targets)} positions", flush=True)

    rows = []
    for condition, kind in lenses:
        jacobians = load_jacobians(lens_path(args.artifacts, condition, kind))
        for name, (store, targets, windows) in sets.items():
            target_tensor = t.tensor(targets)
            for layer in layers:
                if layer not in store or layer not in jacobians:
                    continue
                jacobian = jacobians[layer].to(handles.device)
                position_hits, window_hits, total = 0, 0, 0
                activations = store[layer]
                for start in range(0, activations.shape[0], args.chunk):
                    chunk = activations[start : start + args.chunk].to(handles.device)
                    top = handles.top_k_readouts(
                        transport(jacobian, chunk), args.k, filtered=not args.no_readout_filter
                    ).cpu()
                    chunk_targets = target_tensor[start : start + args.chunk]
                    position_hits += int((top == chunk_targets[:, None]).sum())
                    for offset, row in enumerate(top):
                        window = t.tensor(windows[start + offset])
                        window_hits += int(t.isin(row, window).sum())
                    total += top.shape[0] * args.k
                del jacobian
                rows.append(
                    {
                        "condition": condition,
                        "lens": kind,
                        "layer": layer,
                        "prompt_set": name,
                        "position_echo_rate": position_hits / total,
                        "window_echo_rate": window_hits / total,
                        "positions": activations.shape[0],
                        "k": args.k,
                    }
                )
            print(f"  {condition} {kind} {name} done", flush=True)
        del jacobians
    frame = pd.DataFrame(rows)
    out = args.out_dir / "echo_rate.csv"
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} rows)")
    print("\nmean over layers, position echo rate:")
    print(
        frame.pivot_table(
            index=["lens", "condition"], columns="prompt_set", values="position_echo_rate"
        )
        .round(4)
        .to_string()
    )
    return 0


### 4. EMBEDDING ALIGNMENT


@t.no_grad()
def embedding_alignment(args: argparse.Namespace) -> int:
    """Median cosine between a lens's vector for a token and that token's input
    embedding.

    The lens vector for token v at layer l is row v of ``W_U @ J_l``, a direction
    in layer-l residual space: the thing an activation must look like for the lens
    to read out v. If a gibberish-fitted lens has collapsed onto reading the
    current token, those directions should align with the input embeddings, so C3
    should sit highest.
    """
    from workspace_lens import get_hf_model

    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    tokenizer = model.tokenizer
    # The unembedding and embedding matrices are not part of the lens protocol.
    lm_head = model._lm_head.weight  # noqa: SLF001
    embeddings = model._embed_tokens.weight  # noqa: SLF001
    print(f"W_U {tuple(lm_head.shape)}, W_E {tuple(embeddings.shape)}", flush=True)

    corpus = json.loads((Path("corpora") / f"{args.frequency_corpus}.json").read_text())
    counts: dict[int, int] = {}
    for text in corpus[: args.frequency_prompts]:
        for token_id in tokenizer(text, truncation=True, max_length=args.max_seq_len).input_ids:
            counts[token_id] = counts.get(token_id, 0) + 1
    frequent = [
        token_id for token_id, _ in sorted(counts.items(), key=lambda kv: -kv[1])[: args.num_tokens]
    ]
    print(f"{len(frequent)} most frequent tokens of {args.frequency_corpus}", flush=True)
    token_index = t.tensor(frequent, device=lm_head.device)
    unembed_rows = lm_head[token_index].float()
    embed_rows = t.nn.functional.normalize(embeddings[token_index].float(), dim=-1)

    rows = []
    for condition, kind in available_lenses(args.artifacts, args.conditions, args.kinds):
        jacobians = load_jacobians(lens_path(args.artifacts, condition, kind))
        for layer in sorted(jacobians):
            jacobian = jacobians[layer].to(lm_head.device)
            lens_vectors = t.nn.functional.normalize(unembed_rows @ jacobian, dim=-1)
            cosine = (lens_vectors * embed_rows).sum(-1)
            rows.append(
                {
                    "condition": condition,
                    "lens": kind,
                    "layer": layer,
                    "median_cosine": float(cosine.median()),
                    "mean_cosine": float(cosine.mean()),
                    "p90_cosine": float(cosine.quantile(0.9)),
                    "tokens": len(frequent),
                }
            )
            del jacobian, lens_vectors
        print(f"  {condition} {kind} done", flush=True)
        del jacobians
    frame = pd.DataFrame(rows)
    out = args.out_dir / "embedding_alignment.csv"
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} rows)")
    print("\nmedian cosine, mean over layers:")
    print(frame.groupby(["lens", "condition"])["median_cosine"].mean().round(4).to_string())
    return 0


### CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--kinds", default=",".join(LENS_KINDS))
    parser.add_argument("--hf-model-name", default=MODEL)
    parser.add_argument("--reference", default=REFERENCE)
    subparsers = parser.add_subparsers(dest="diagnostic", required=True)

    weights_parser = subparsers.add_parser("weights", help="expert weights and cluster balance")
    weights_parser.add_argument("--min-kept-positions", type=int, default=50)
    weights_parser.set_defaults(func=expert_weights)

    matrix_parser = subparsers.add_parser("matrix", help="distance from the reference lens")
    matrix_parser.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    matrix_parser.add_argument("--rank", type=int, default=64)
    matrix_parser.set_defaults(func=matrix_distance)

    echo_parser = subparsers.add_parser("echo", help="position and window echo rate (H3)")
    echo_parser.add_argument("--k", type=int, default=10)
    echo_parser.add_argument("--layers", default=None, help="default: the lens's own layers")
    echo_parser.add_argument("--items", type=int, default=10_000)
    echo_parser.add_argument("--wikitext-records", type=int, default=200)
    echo_parser.add_argument("--positions-per-record", type=int, default=8)
    echo_parser.add_argument("--skip-first-n-positions", type=int, default=16)
    echo_parser.add_argument("--readout-max-seq-len", type=int, default=512)
    echo_parser.add_argument("--correctness-csv", default="data/model_correctness_9b.csv")
    echo_parser.add_argument("--chunk", type=int, default=256)
    echo_parser.add_argument("--no-readout-filter", action="store_true")
    echo_parser.add_argument("--seed", type=int, default=0)
    echo_parser.set_defaults(func=echo_rate)

    align_parser = subparsers.add_parser("align", help="embedding alignment")
    align_parser.add_argument("--num-tokens", type=int, default=5000)
    align_parser.add_argument("--frequency-corpus", default="c0_wikitext")
    align_parser.add_argument("--frequency-prompts", type=int, default=1000)
    align_parser.add_argument("--max-seq-len", type=int, default=128)
    align_parser.set_defaults(func=embedding_alignment)

    args = parser.parse_args(argv)
    args.conditions = [c for c in args.conditions.split(",") if c]
    args.kinds = [k for k in args.kinds.split(",") if k]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
