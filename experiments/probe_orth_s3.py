"""Probe what a good ``dendritic_orth`` seed learns on the S_3 word problem.

The 8-seed S_3 run showed the free O(2) block can *fully* length-generalize S_3
(seed 4: ~97% final-position at 8x the train length), while the hand-constrained
per-symbol variant fails. This script asks the mechanistic question: does the good
seed discover the actual 2D representation of S_3 ~= D_3 -- i.e. does each input
symbol map to a consistent O(2) element with angle near {0, +/-2*pi/3} and a
definite determinant (rotation vs reflection)?

It trains one ``dendritic_orth`` model at a chosen seed (identically to the sweep,
so the seed reproduces), captures the per-step snapped angle/sign that each O(2)
layer applies, groups them by the CURRENT input symbol, and reports how peaked the
(angle, det) is per symbol -- a clean peak per symbol = the model found the irrep.

Usage:
    uv run --no-sync python -u experiments/probe_orth_s3.py --preset gpu3080 --device cuda --seed 4
    uv run --no-sync python -u experiments/probe_orth_s3.py --preset cpu --seed 4 --steps 400
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import experiments.run_streaming_s3 as s3
from src.ssm import OrthogonalRecurrentSSM
from src.tasks import S3_SIZE, make_streaming_s3


def build_args(preset: str, device: str, threads: int, steps: int | None,
               d_state: int | None, train_lens: list[int] | None):
    """A run_streaming_s3 args Namespace for a single dendritic_orth model, with
    preset defaults filled in by the experiment's own ``configure`` (so training
    matches the sweep exactly and the seed reproduces)."""
    args = argparse.Namespace(
        preset=preset, models=["dendritic_orth"], device=device, threads=threads,
        rot_bins=12, snap_warmup_frac=0.5, out=None, seed_list=None,
        train_lens=train_lens, eval_lens=None, steps=steps, d_state=d_state,
        d_model=None, n_layers=None, n_heads=None, train_len=None, batch_size=None,
        lr=None, ffn_mult=None, n_branches=None, conv_k=None, chunk=None, seeds=None,
    )
    cfg = s3.configure(args)
    return args, cfg


def capture_orientations(model, x):
    """Run the model once, returning per-layer (theta, det, ids) on stream ``x``.

    Monkeypatches each O(2) core's ``_orient`` per-instance (keeps the core class
    clean) to stash the snapped (cos, sin, sign) it applies; ``theta`` is recovered
    as atan2(sin, cos) and ``det`` is sign(sign) in {-1,+1} (reflection vs rotation).
    """
    caps: dict[int, list] = {}
    layers = []
    for li, blk in enumerate(model.blocks):
        ssm = getattr(blk.mix, "ssm", None)
        if isinstance(ssm, OrthogonalRecurrentSSM):
            layers.append(li)
            orig = ssm._orient

            def wrapped(xx, ids=None, _orig=orig, _li=li):
                cos, sin, s = _orig(xx, ids)
                caps[_li] = (cos.detach().cpu(), sin.detach().cpu(), s.detach().cpu())
                return cos, sin, s

            ssm._orient = wrapped

    model.eval()
    with torch.no_grad():
        model(x)

    out = {}
    ids = x.detach().cpu().numpy()                          # (B, L)
    for li in layers:
        cos, sin, s = caps[li]
        theta = torch.atan2(sin, cos).numpy()              # (B, L, d) in (-pi, pi]
        det = np.sign(s.numpy())                           # (B, L, d) in {-1,+1}
        out[li] = (theta, det, ids)
    return out


def summarize_layer(theta, det, ids, n_bins: int):
    """For each input symbol, report the dominant snapped angle bin and the
    reflection fraction across all channels/positions where that symbol occurs."""
    delta = 2 * math.pi / n_bins
    bins = (np.round(theta / delta).astype(int)) % n_bins   # (B, L, d)
    refl = det < 0                                          # (B, L, d)
    B, L, d = theta.shape
    sym_col = np.broadcast_to(ids[:, :, None], (B, L, d))   # symbol at each entry

    rows = []
    for g in range(S3_SIZE):
        mask = sym_col == g
        if not mask.any():
            continue
        gb = bins[mask]
        counts = np.bincount(gb, minlength=n_bins)
        top = counts.argmax()
        top_frac = counts[top] / counts.sum()
        top_deg = (top * 360.0 / n_bins + 180) % 360 - 180  # signed degrees
        refl_frac = refl[mask].mean()
        rows.append((g, top_deg, top_frac, refl_frac))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=list(s3.PRESETS), default="gpu3080")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=4, help="seed to train + probe")
    ap.add_argument("--steps", type=int, default=None, help="override training steps")
    ap.add_argument("--d-state", type=int, default=None, help="override d_state")
    ap.add_argument("--train-lens", type=int, nargs="+", default=None,
                    help="length curriculum (default: fixed preset train_len)")
    ap.add_argument("--probe-len", type=int, default=64, help="stream length to probe on")
    ap.add_argument("--probe-rows", type=int, default=64, help="streams to probe on")
    ap.add_argument("--save", type=str, default=None, help="optional .npz of raw captures")
    pa = ap.parse_args()

    args, cfg = build_args(pa.preset, pa.device, pa.threads, pa.steps, pa.d_state,
                           pa.train_lens)
    target_params = s3.count_params(s3.CausalAttention(args.d_model, args.n_heads))

    print(f"\nProbing dendritic_orth seed={pa.seed} on S_3 (preset={pa.preset}, "
          f"device={args.device}, steps={args.steps}, d_state={args.d_state}, "
          f"train_lens={s3.train_lengths(args)}). Training...", flush=True)
    accs, mp, model = s3.train_one("dendritic_orth", cfg, args, target_params, pa.seed)
    shown = "  ".join(f"L{L}: pp{accs[L][0]*100:4.0f}/fin{accs[L][1]*100:4.0f}"
                      for L in args.eval_lens)
    print(f"  trained (mixer_params={mp}). eval: {shown}\n")

    x, _ = make_streaming_s3(pa.probe_rows, pa.probe_len, seed=12345)
    x = torch.from_numpy(x).to(args.device)
    caps = capture_orientations(model, x)

    print(f"Per-symbol O(2) transform by layer (angle = dominant snapped bin across "
          f"all channels; refl = fraction with det<0). rot_bins={cfg.rot_bins}, "
          f"2*pi/3 = 120 deg.\n")
    for li in sorted(caps):
        theta, det, ids = caps[li]
        print(f"  layer {li}:")
        print(f"    {'symbol':>6}  {'dom angle(deg)':>14}  {'peak frac':>9}  {'refl frac':>9}")
        for g, deg, frac, rf in summarize_layer(theta, det, ids, cfg.rot_bins):
            print(f"    {g:>6}  {deg:>14.0f}  {frac:>9.2f}  {rf:>9.2f}")
        print()

    if pa.save:
        np.savez(pa.save, **{f"theta_{li}": caps[li][0] for li in caps},
                 **{f"det_{li}": caps[li][1] for li in caps},
                 ids=next(iter(caps.values()))[2])
        print(f"Saved raw captures to {pa.save}")


if __name__ == "__main__":
    main()
