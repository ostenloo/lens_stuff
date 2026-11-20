"""Build the four fitting corpora and their statistics.

Each condition is written as ``corpora/<cond>.json``, a JSON list of
``--num-prompts`` strings. The pipeline takes the first ``--fit-prompts`` for the
Jacobian fit and the whole list for the router, so every prompt must tokenize to
at least ``--min-tokens`` tokens.

    uv run python gibberish/build_corpora.py                      # all four + stats
    uv run python gibberish/build_corpora.py --conditions c1_selfgen --limit 8
    uv run python gibberish/build_corpora.py --stats-only

Conditions:

C0 wikitext   the first records of wikitext-103-raw-v1 train with >= 600
              characters, streamed until --num-prompts of them tokenize long
              enough both raw and shuffled, so C0 and C2 are the same records.
C1 selfgen    unconditional samples from the model itself.
C2 shuffled   C0's records with their whitespace-separated words shuffled.
C3 random     uniform token ids, decoded to text.

Qwen3.5-9B has no BOS token (``bos_token`` is null and ``add_bos_token`` false),
so ``from_hf``'s ``force_bos`` is a no-op for it and the token counts here are
what the fitting stages will see. Two further consequences are recorded in each
stats file: C1 is seeded with ``--seed-token`` rather than BOS, and position 0
carries no attention sink (the recipe skips the first 16 positions anyway).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch as t

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import jpp_cli  # noqa: E402  (needs scripts/ on the path)

from jlens.examples import load_wikitext_prompts  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import vocab_size_of  # noqa: E402

MODEL = "Qwen/Qwen3.5-9B"
CONDITIONS = ("c0_wikitext", "c1_selfgen", "c2_shuffled", "c3_random")
WIKITEXT_STREAM_MULTIPLIER = 1.5  # stream this many times --num-prompts before filtering
SELFGEN_NEW_TOKENS = 224
RANDOM_TOKEN_IDS = 320


### MODEL HANDLES


class Handles:
    """The tokenizer the fitting stages will use, and the raw HF model.

    ``get_hf_model`` is the one loader the pipeline uses, so going through it
    keeps the tokenizer (and any BOS forcing) identical to what ``fit-shard``
    sees. Generation and teacher-forced loss are not part of the lens protocol,
    so they need the wrapped HF model, which is reachable only as a private
    attribute.
    """

    def __init__(self, hf_model_name: str) -> None:
        lens_model = get_hf_model(hf_model_name, attn_implementation="sdpa")
        self.lens_model = lens_model
        self.tokenizer = lens_model.tokenizer
        self.hf_model = lens_model._hf_model  # noqa: SLF001
        self._decoded: dict[int, str] = {}

    def token_count(self, text: str) -> int:
        """Tokens the fitting stages would see, untruncated."""
        return len(self.tokenizer(text).input_ids)

    def encode(self, text: str, max_length: int) -> list[int]:
        return self.tokenizer(text, truncation=True, max_length=max_length).input_ids

    def decoded(self, token_id: int) -> str:
        if token_id not in self._decoded:
            self._decoded[token_id] = self.tokenizer.decode([token_id])
        return self._decoded[token_id]

    def is_non_alphanumeric(self, token_id: int) -> bool:
        """No letter and no digit anywhere in the token's surface — the property
        Readout Filtering drops a token for."""
        return not any(char.isalnum() for char in self.decoded(token_id))

    def sampleable_token_ids(self) -> list[int]:
        """The vocabulary minus every special and added token (the image, video
        and chat-control ids among them)."""
        excluded = set(self.tokenizer.all_special_ids)
        excluded |= set(self.tokenizer.get_added_vocab().values())
        width = vocab_size_of(self.lens_model)
        return [token_id for token_id in range(width) if token_id not in excluded]


### CONDITIONS


def build_wikitext_and_shuffled(
    handles: Handles, *, num_prompts: int, min_tokens: int
) -> tuple[list[str], list[str], dict]:
    """``(c0, c2, diagnostics)``: the records kept, and the same records with
    their words shuffled.

    A record is kept only when both forms are long enough, so the two conditions
    hold the same multiset of words per record (Gate 1). The shuffle seed is the
    record's index in the stream, not its index in the kept list: the keep
    decision depends on the shuffled text, so a seed taken from the kept index
    would be circular.
    """
    streamed = load_wikitext_prompts(int(num_prompts * WIKITEXT_STREAM_MULTIPLIER))
    raw: list[str] = []
    shuffled: list[str] = []
    stream_indices: list[int] = []
    dropped_raw = dropped_shuffled = 0
    for stream_index, record in enumerate(streamed):
        words = record.split()
        random.Random(stream_index).shuffle(words)
        candidate = " ".join(words)
        if handles.token_count(record) < min_tokens:
            dropped_raw += 1
            continue
        if handles.token_count(candidate) < min_tokens:
            dropped_shuffled += 1
            continue
        raw.append(record)
        shuffled.append(candidate)
        stream_indices.append(stream_index)
        if len(raw) == num_prompts:
            break
    diagnostics = {
        "streamed": len(streamed),
        "kept": len(raw),
        "dropped_raw_too_short": dropped_raw,
        "dropped_shuffled_too_short": dropped_shuffled,
        "shuffle_seed": "the record's index in the stream",
        "last_stream_index_used": stream_indices[-1] if stream_indices else None,
    }
    if len(raw) < num_prompts:
        diagnostics["error"] = (
            f"only {len(raw)} of {num_prompts} records survived "
            f"{len(streamed)} streamed; raise WIKITEXT_STREAM_MULTIPLIER"
        )
    return raw, shuffled, diagnostics


@t.no_grad()
def build_selfgen(
    handles: Handles,
    *,
    num_prompts: int,
    new_tokens: int,
    seed_token: str,
    allow_special_tokens: bool = False,
) -> tuple[list[str], dict]:
    """Unconditional samples, one per seed: temperature 1.0, top-p 1.0, no chat
    template, ``min_new_tokens == max_new_tokens`` so an EOS cannot cut a sample
    short.

    The spec samples "from BOS alone", but this checkpoint has no BOS token, so
    generation starts from ``seed_token`` instead and the prompt token is dropped
    from the decoded text. One sequence per call, seeded with the prompt's index:
    batching would make the samples depend on how they were grouped.

    Two deviations the checkpoint forces, both recorded in the stats file.
    Sampling from ``<|endoftext|>`` collapses into repetition (``_____``,
    ``2. 2. 2.``) on this post-trained model and leaks ``<|im_end|>`` and
    ``</think>`` into the text, so the default seed is a newline, which samples
    ordinary text from the same model. Special tokens are suppressed: a control
    token in a fitting corpus is an artifact of unconditional sampling, not text
    the model would be fitted on.
    """
    seed_ids = handles.tokenizer(seed_token, add_special_tokens=False).input_ids
    if len(seed_ids) != 1:
        raise SystemExit(f"--seed-token {seed_token!r} is not a single token: {seed_ids}")
    input_ids = t.tensor([seed_ids], device=handles.lens_model.input_device)

    prompts: list[str] = []
    for index in range(num_prompts):
        t.manual_seed(index)
        output = handles.hf_model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_new_tokens=new_tokens,
            max_new_tokens=new_tokens,
            suppress_tokens=None if allow_special_tokens else handles.tokenizer.all_special_ids,
        )
        prompts.append(handles.tokenizer.decode(output[0][len(seed_ids) :]))
        if (index + 1) % 50 == 0:
            print(f"  generated {index + 1}/{num_prompts}", flush=True)
    return prompts, {
        "seed_token": seed_token,
        "seed_token_id": seed_ids[0],
        "new_tokens_per_prompt": new_tokens,
        "seed": "torch.manual_seed(prompt index), one sequence per call",
        "special_tokens_suppressed": not allow_special_tokens,
        "note": (
            "this checkpoint has no BOS token, so the spec's 'from BOS alone' is "
            "sampled from --seed-token instead; <|endoftext|> was tried first and "
            "produced degenerate repetition with control tokens in the text"
        ),
    }


def build_random(
    handles: Handles, *, num_prompts: int, num_token_ids: int
) -> tuple[list[str], dict]:
    """Uniform ids from the sampleable vocabulary, decoded to text.

    Decoding and re-encoding does not round-trip: the re-encoded sequence is the
    real input and is accepted as such, since it is still gibberish. Sampling
    more ids than the minimum makes a short prompt unlikely rather than
    resampling one away.
    """
    sampleable = handles.sampleable_token_ids()
    prompts: list[str] = []
    non_alphanumeric = 0
    for index in range(num_prompts):
        ids = random.Random(index).choices(sampleable, k=num_token_ids)
        non_alphanumeric += sum(handles.is_non_alphanumeric(token_id) for token_id in ids)
        prompts.append(handles.tokenizer.decode(ids))
    return prompts, {
        "sampleable_vocabulary": len(sampleable),
        "lm_head_width": vocab_size_of(handles.lens_model),
        "sampled_ids_per_prompt": num_token_ids,
        "sampled_non_alphanumeric_share": round(
            non_alphanumeric / (num_prompts * num_token_ids), 4
        ),
        "seed": "random.Random(prompt index)",
    }


### STATISTICS


@t.no_grad()
def mean_per_token_loss(
    handles: Handles, prompts: list[str], *, max_seq_len: int, batch_size: int = 8
) -> float:
    """Teacher-forced mean per-token loss over the prompts as the Jacobian fit
    sees them: truncated to ``max_seq_len``, in the model's own dtype. The main
    measure of how off-distribution a condition is."""
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = [handles.encode(text, max_seq_len) for text in batch]
        width = min(len(ids) for ids in encoded)
        input_ids = t.tensor(
            [ids[:width] for ids in encoded], device=handles.lens_model.input_device
        )
        logits = handles.hf_model(input_ids=input_ids, use_cache=False).logits.float()
        loss = t.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss)
        total_tokens += input_ids[:, 1:].numel()
    return total_loss / total_tokens


def token_statistics(handles: Handles, token_ids: list[list[int]]) -> dict:
    """Unigram entropy, type/token ratio and non-alphanumeric share over a set of
    tokenized prompts."""
    counts = Counter(token_id for ids in token_ids for token_id in ids)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    non_alphanumeric = sum(
        count for token_id, count in counts.items() if handles.is_non_alphanumeric(token_id)
    )
    return {
        "tokens": total,
        "types": len(counts),
        "unigram_entropy_bits": round(entropy, 4),
        "type_token_ratio": round(len(counts) / total, 4),
        "non_alphanumeric_share": round(non_alphanumeric / total, 4),
    }


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "min": ordered[0],
        "p05": at(0.05),
        "median": at(0.5),
        "p95": at(0.95),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 1),
    }


def write_stats(
    handles: Handles,
    condition: str,
    prompts: list[str],
    *,
    out_dir: Path,
    fit_prompts: int,
    max_seq_len: int,
    min_tokens: int,
    build_diagnostics: dict,
    examples: int,
) -> dict:
    """``corpora/<cond>_stats.json``, and the Gate 1 facts for the caller."""
    full = [handles.tokenizer(text).input_ids for text in prompts]
    counts = [len(ids) for ids in full]
    window = [ids[:max_seq_len] for ids in full[:fit_prompts]]
    stats = {
        "condition": condition,
        "prompts": len(prompts),
        "token_counts": distribution(counts),
        "prompts_under_min_tokens": sum(1 for count in counts if count < min_tokens),
        "min_tokens": min_tokens,
        "whole_pool": token_statistics(handles, full),
        "fit_window": {
            "prompts": len(window),
            "max_seq_len": max_seq_len,
            **token_statistics(handles, window),
            "mean_per_token_loss": round(
                mean_per_token_loss(handles, prompts[:fit_prompts], max_seq_len=max_seq_len), 4
            ),
        },
        "build": build_diagnostics,
        "examples": [text[:400] for text in prompts[:examples]],
    }
    (out_dir / f"{condition}_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


### GATE 1


def check_gate_1(
    stats_by_condition: dict[str, dict],
    *,
    num_prompts: int,
    min_tokens: int,
    wikitext: list[str] | None,
    shuffled: list[str] | None,
) -> bool:
    """Every condition has ``num_prompts`` prompts of at least ``min_tokens``
    tokens, and C0 and C2 hold the same multiset of words per record."""
    passed = True
    print("\nGate 1")
    for condition, stats in stats_by_condition.items():
        short = stats["prompts_under_min_tokens"]
        ok = stats["prompts"] == num_prompts and short == 0
        passed &= ok
        print(
            f"  {'PASS' if ok else 'FAIL'}  {condition}: {stats['prompts']} prompts, "
            f"{short} under {min_tokens} tokens, "
            f"min {stats['token_counts']['min']}, median {stats['token_counts']['median']}"
        )
    if wikitext is not None and shuffled is not None:
        mismatched = sum(
            1
            for raw, shuf in zip(wikitext, shuffled, strict=True)
            if Counter(raw.split()) != Counter(shuf.split())
        )
        passed &= mismatched == 0
        print(
            f"  {'PASS' if mismatched == 0 else 'FAIL'}  C0/C2 word multiset: "
            f"{len(wikitext) - mismatched}/{len(wikitext)} records match"
        )
    return bool(passed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default=MODEL)
    parser.add_argument("--out-dir", default="corpora")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--fit-prompts", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=jpp_cli.RECIPE_MAX_SEQ_LEN)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument(
        "--seed-token",
        default="\n",
        help="C1's unconditional start, in place of the BOS this checkpoint lacks. "
        "Default: a newline. <|endoftext|> degenerates on this post-trained checkpoint.",
    )
    parser.add_argument(
        "--allow-special-tokens",
        action="store_true",
        help="Let C1 sample control tokens (<|im_end|>, </think>) into the corpus text.",
    )
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build this many prompts per condition instead of --num-prompts (a smoke test; "
        "Gate 1 is skipped).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Recompute the stats files from the corpora already written.",
    )
    args = parser.parse_args(argv)

    conditions = [c for c in args.conditions.split(",") if c]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    num_prompts = args.limit or args.num_prompts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    handles = Handles(args.hf_model_name)
    print(
        f"tokenizer: bos_token_id={handles.tokenizer.bos_token_id}, "
        f"add_bos_token={getattr(handles.tokenizer, 'add_bos_token', 'n/a')}",
        flush=True,
    )

    corpora: dict[str, list[str]] = {}
    build_diagnostics: dict[str, dict] = {}
    if args.stats_only:
        for condition in conditions:
            corpora[condition] = json.loads((out_dir / f"{condition}.json").read_text())
            build_diagnostics[condition] = {"note": "--stats-only; build diagnostics not rebuilt"}
    else:
        if {"c0_wikitext", "c2_shuffled"} & set(conditions):
            print("building c0_wikitext and c2_shuffled...", flush=True)
            raw, shuffled, diagnostics = build_wikitext_and_shuffled(
                handles, num_prompts=num_prompts, min_tokens=args.min_tokens
            )
            if "error" in diagnostics:
                print(f"  WARNING: {diagnostics['error']}", flush=True)
            corpora["c0_wikitext"] = raw
            corpora["c2_shuffled"] = shuffled
            build_diagnostics["c0_wikitext"] = diagnostics
            build_diagnostics["c2_shuffled"] = diagnostics
        if "c1_selfgen" in conditions:
            print(f"building c1_selfgen ({num_prompts} samples)...", flush=True)
            corpora["c1_selfgen"], build_diagnostics["c1_selfgen"] = build_selfgen(
                handles,
                num_prompts=num_prompts,
                new_tokens=SELFGEN_NEW_TOKENS,
                seed_token=args.seed_token,
                allow_special_tokens=args.allow_special_tokens,
            )
        if "c3_random" in conditions:
            print("building c3_random...", flush=True)
            corpora["c3_random"], build_diagnostics["c3_random"] = build_random(
                handles, num_prompts=num_prompts, num_token_ids=RANDOM_TOKEN_IDS
            )
        for condition in conditions:
            path = out_dir / f"{condition}.json"
            path.write_text(json.dumps(corpora[condition], indent=1) + "\n")
            print(f"  wrote {path} ({len(corpora[condition])} prompts)")

    stats_by_condition: dict[str, dict] = {}
    for condition in conditions:
        print(f"stats for {condition}...", flush=True)
        stats_by_condition[condition] = write_stats(
            handles,
            condition,
            corpora[condition],
            out_dir=out_dir,
            fit_prompts=min(args.fit_prompts, len(corpora[condition])),
            max_seq_len=args.max_seq_len,
            min_tokens=args.min_tokens,
            build_diagnostics=build_diagnostics[condition],
            examples=args.examples,
        )

    print(
        f"\n{'condition':<14}{'tokens/prompt':>16}{'loss':>8}{'entropy':>9}{'TTR':>7}{'non-alnum':>11}"
    )
    for condition, stats in stats_by_condition.items():
        window = stats["fit_window"]
        print(
            f"{condition:<14}{stats['token_counts']['median']:>16}"
            f"{window['mean_per_token_loss']:>8.3f}{window['unigram_entropy_bits']:>9.2f}"
            f"{window['type_token_ratio']:>7.3f}{window['non_alphanumeric_share']:>11.3f}"
        )

    if args.limit:
        print("\n--limit set: Gate 1 skipped")
        return 0
    passed = check_gate_1(
        stats_by_condition,
        num_prompts=args.num_prompts,
        min_tokens=args.min_tokens,
        wikitext=corpora.get("c0_wikitext"),
        shuffled=corpora.get("c2_shuffled"),
    )
    print("\nGate 1 PASSED" if passed else "\nGate 1 FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
