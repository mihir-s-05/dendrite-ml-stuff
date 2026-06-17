"""Quick timing diagnostic: per-step train time for each FFN block, on the
chosen device, with proper CUDA synchronization and flushed output.

Usage:
    uv run --no-sync python -u experiments/bench.py --device cuda --steps 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from experiments.run_lm import GPT, make_block_factory
from src.train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--target-params", type=int, default=200000)
    ap.add_argument("--vocab", type=int, default=32)
    args = ap.parse_args()
    dev = pick_device(args.device)
    print(f"device={dev} torch={torch.__version__}", flush=True)

    for name in ["mlp", "swiglu", "dendritic"]:
        make_ffn, _ = make_block_factory(name, args.d_model, args.target_params)
        model = GPT(args.vocab, args.d_model, args.n_layers, args.n_heads,
                    args.block_size, make_ffn).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        x = torch.randint(0, args.vocab, (args.batch_size, args.block_size), device=dev)
        y = torch.randint(0, args.vocab, (args.batch_size, args.block_size), device=dev)

        # warmup
        for _ in range(5):
            opt.zero_grad()
            loss = F.cross_entropy(model(x).view(-1, args.vocab), y.view(-1))
            loss.backward(); opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(args.steps):
            opt.zero_grad()
            loss = F.cross_entropy(model(x).view(-1, args.vocab), y.view(-1))
            loss.backward(); opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / args.steps * 1000
        print(f"  {name:10s}  {dt:7.2f} ms/step", flush=True)


if __name__ == "__main__":
    main()
