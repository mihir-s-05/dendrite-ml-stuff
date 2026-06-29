"""Resumable seed-sweep driver shared by the streaming state-tracking probes.

All three streaming experiments (parity, mod-k, S_3) have the same shape: train
one model per ``(model, seed)`` and report per-position / final-position accuracy
at several eval lengths. This module owns that loop and its reporting so each
experiment script only defines its task, loss, and ``train_one``.

Per-(model, seed) accuracies are flushed to an append-only CSV via ``ResultStore``
(keyed ``section in {pp, fin}``, ``d = eval length``), so a crashed run resumes by
skipping seeds already recorded. ``--seed-list`` runs an explicit seed subset
(e.g. ``--seed-list 3 4 5 6 7`` to finish a run that died at seed 3).
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np

from src.results_store import ResultStore

# train_one(name, cfg, args, target_params, seed) -> ({L: (pp, fin)}, mixer_params, model)
TrainOne = Callable[[str, object, argparse.Namespace, int, int],
                    "tuple[dict[int, tuple[float, float]], int, object]"]


def add_sweep_args(ap: argparse.ArgumentParser) -> None:
    """Register the resumability flags shared by every streaming runner."""
    ap.add_argument("--out", type=str, default=None,
                    help="append-only CSV for resumable per-seed results; rerun the "
                         "same command to skip seeds already recorded")
    ap.add_argument("--seed-list", type=int, nargs="+", default=None,
                    help="explicit seeds to run, overriding --seeds (e.g. resume the "
                         "tail of a crashed run with --seed-list 3 4 5 6 7)")
    ap.add_argument("--train-lens", type=int, nargs="+", default=None,
                    help="length curriculum: sample each batch's length uniformly from "
                         "this set instead of the fixed --train-len (stabilizes "
                         "length generalization). Single value == fixed length.")


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    return list(args.seed_list) if args.seed_list else list(range(args.seeds))


def train_lengths(args: argparse.Namespace) -> list[int]:
    """The length curriculum (``--train-lens`` if given, else the single train_len)."""
    return list(args.train_lens) if args.train_lens else [args.train_len]


def sample_train_len(lens: list[int], rng) -> int:
    """Pick a batch length. A single-length curriculum draws NO rng value, so the
    default (fixed-length) training path is bit-identical to before this flag."""
    return lens[0] if len(lens) == 1 else int(rng.choice(lens))


def _has_seed(store: ResultStore, model: str, seed: int, eval_lens: list[int]) -> bool:
    return all(store.get("pp", L, model, seed) is not None
               and store.get("fin", L, model, seed) is not None for L in eval_lens)


def _print_means(store: ResultStore, section: str, models: list[str],
                 eval_lens: list[int], seeds: list[int]) -> None:
    print("  model           " + "".join(f"{'L='+str(L):>10s}" for L in eval_lens))
    for m in models:
        line = f"  {m:14s}  "
        for L in eval_lens:
            vals = store.values(section, L, m, seeds)
            line += f"{(np.mean(vals) * 100 if vals else float('nan')):>10.1f}"
        print(line)


def run_sweep(args: argparse.Namespace, cfg, target_params: int, train_one: TrainOne,
              *, task_tag: str, stream_title: str, hard_desc: str, chance: float) -> None:
    """Run (or resume) the (model x seed) sweep and print the three report blocks.

    ``task_tag`` labels the training block (e.g. ``"S_3"``); ``stream_title`` is the
    per-position summary header (e.g. ``"STREAMING S_3"``); ``hard_desc`` describes
    the final-position quantity; ``chance`` is the chance accuracy in percent.
    """
    store = ResultStore(args.out)
    seeds = resolve_seeds(args)
    eval_lens = args.eval_lens
    models = args.models

    print(f"\n=== TRAINING (per-position acc / final-position acc, by eval length; "
          f"{task_tag}) ===")
    for m in models:
        for s in seeds:
            cached = _has_seed(store, m, s, eval_lens)
            if not cached:
                t0 = time.time()
                accs, mp, _model = train_one(m, cfg, args, target_params, s)
                secs = time.time() - t0
                for L, (pp, fin) in accs.items():
                    store.record("pp", L, m, s, pp, mp, secs)
                    store.record("fin", L, m, s, fin, mp, secs)
            shown = "  ".join(
                f"L{L}:{store.get('pp', L, m, s) * 100:4.0f}/"
                f"{store.get('fin', L, m, s) * 100:4.0f}" for L in eval_lens)
            mark = " (cached)" if cached else ""
            print(f"  [{m:14s} seed={s}] {shown}{mark}", flush=True)

    print(f"\n=== {stream_title} (per-position acc %, mean over seeds; "
          f"chance={chance:.1f}%) ===")
    _print_means(store, "pp", models, eval_lens, seeds)
    print(f"\n=== final-position acc % (the hard bit: {hard_desc}) ===")
    _print_means(store, "fin", models, eval_lens, seeds)
    print()
