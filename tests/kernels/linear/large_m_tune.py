# SPDX-License-Identifier: Apache-2.0
"""Checks the mma-family heuristic against an exhaustive sweep of plans
(every config x split-K factor x data-parallel/tail split) per (shape, M).
Reports heuristic and best-plan times for NF12 with L2 flushed.
Run under a GPU lock:
  uv run --group bench python tests/kernels/linear/large_m_tune.py [--shapes KxN,...] [--ms ...] [--json out.json]
--json also writes every plan's time for offline heuristic fitting."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import large_m_harness as H  # noqa: E402
from large_m_bench import SHAPES, Timer, mma_label, warm_clocks  # noqa: E402


def tile_dims(name):
    bn, bm = name.split("_")[1].split("x")
    return int(bn), int(bm)


def plans(shim, m, n, sms):
    out = []
    for c in range(shim.num_configs(H.FAMILY_MMA)):
        bn, bm = tile_dims(shim.config_name(H.FAMILY_MMA, c))
        tiles = ((n + bn - 1) // bn) * ((m + bm - 1) // bm)
        for s in (1, 2, 4, 8):
            out.append(c + 16 * (s - 1))
            if s > 1:
                for f in {tiles // sms, tiles // sms - 1}:
                    if f >= 1 and f * sms < tiles:
                        out.append(c + 16 * (s - 1) + 256 * f * sms)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--ms", default="96,128,192,256,384,512,1024,2048,4096")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--json", help="write every plan's time here")
    a = ap.parse_args()
    shapes = SHAPES if a.shapes == "all" else [tuple(map(int, s.split("x"))) for s in a.shapes.split(",")]
    shim, timer = H.Shim(), Timer()
    warm_clocks()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    results = []
    for k, n in shapes:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        nw = H.NF12Packed(w, bf16t=False)
        for m in map(int, a.ms.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            codes = []
            for c in plans(shim, m, n, sms):
                try:
                    shim.linear(x, nw, y=y, family=H.FAMILY_MMA, config=c)
                    codes.append(c)
                except H.Unsupported:
                    pass
            ts = timer.many([lambda c=c: shim.linear(x, nw, y=y, family=H.FAMILY_MMA, config=c) for c in codes],
                            a.reps)
            hc = shim.mma_heuristic(m, n, k)
            th = timer(lambda: shim.linear(x, nw, y=y, family=H.FAMILY_MMA, config=-1), a.reps)
            best = min(zip(ts, codes))
            print(f"{k:>6} {n:>6} {m:>5} | heur {th:8.1f} {mma_label(shim, hc):<34} | best {best[0]:8.1f} "
                  f"{mma_label(shim, best[1]):<34} | {th / best[0]:5.2f}", flush=True)
            results.append(dict(k=k, n=n, m=m, heur=hc, heur_us=th,
                                plans={str(c): t for c, t in zip(codes, ts)}))
            if a.json:
                with open(a.json, "w") as f:
                    json.dump(dict(device=torch.cuda.get_device_name(), sms=sms, results=results), f, indent=1)


if __name__ == "__main__":
    main()
