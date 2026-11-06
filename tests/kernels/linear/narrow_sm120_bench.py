# SPDX-License-Identifier: Apache-2.0
"""Narrow-weight kernel benchmark (router gates, N <= 128) vs cuBLAS.
Graph timing rotates weight copies (bench/common.py). Every timed config
must produce NF12 output matching its bf16t twin without tolerance and close to an
FP32 reference.

    flock /tmp/nf12-sm120.lock uv run --group bench python tests/kernels/linear/narrow_sm120_bench.py --sweep
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "bench"))
import common as BC  # noqa: E402
import large_m_harness as H  # noqa: E402
import large_m_sm120_harness as S  # noqa: E402
from large_m_bench import warm_clocks  # noqa: E402
from large_m_sm120_bench import Copies  # noqa: E402

SHAPES = "2048x128,2048x64,4096x128"
MS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384"


def candidates(m, k):
    tfs = [tf for tf in range(4) if tf == 0 or 8 << (tf - 1) < m]  # token tiles up to the first covering M
    # SM90 also takes non-portable clusters of up to 16 split CTAs (K up to 8192).
    splits = (1, 2, 4, 8) + ((12, 14, 16) if torch.cuda.get_device_capability() == (9, 0) else ())
    return [tf | (wi << 2) | (s << 4) for tf in tfs for wi in range(4) for s in splits if s <= k // 64]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=SHAPES)
    ap.add_argument("--ms", default=MS)
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    shim = S.Sm120Shim()
    assert shim.narrow_supported(), "narrow kernel unavailable"
    warm_clocks()
    print(f"{'shape':>12} {'M':>6} | {'nf12':>7} {'bf16t':>7} {'cublas':>7} | {'x cub':>6}  config (heuristic us)")
    results = []
    for key in a.shapes.split(","):
        shape = BC.shape_by_key(key)
        k, n = shape.k, shape.n
        w, origin = BC.make_weight(shape, a.weights, torch.device("cuda"))
        nw = H.NF12Packed(w)
        ncopy = BC.cold_copies(nw.packed.numel())
        nws = Copies(nw, ncopy)
        wts = [w] + [w.clone() for _ in range(BC.cold_copies(2 * n * k) - 1)]
        print(f"# {key} ({origin}), {ncopy} copies", flush=True)
        for m in map(int, a.ms.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
            y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            ref = x.float() @ w.float().T + bias.float()
            heur = shim.narrow_heuristic(m, n, k)
            cands = candidates(m, k) if a.sweep else [heur]
            timed = {}
            for c in cands:
                try:
                    yn = shim.narrow(x, nw, bias, config=c).clone()
                except H.Unsupported:
                    continue
                yt = shim.narrow(x, nw, bias, source="bf16t", config=c)
                assert torch.equal(yn.view(torch.int16), yt.view(torch.int16)), f"{key} m={m} {c}: nf12 != bf16t"
                torch.testing.assert_close(yn.float(), ref, rtol=2e-2, atol=2e-2)
                timed[c] = BC.time_graph(lambda i, c=c: shim.narrow(x, nws[i], bias, y=y, config=c), ncopy,
                                         budget_s=0.5)["median_us"]
                if a.verbose:
                    print(f"    {shim.narrow_config_name(c):>14}: {timed[c]:8.2f}", flush=True)
            best = min(timed, key=timed.get)
            t_nf = BC.time_graph(lambda i: shim.narrow(x, nws[i], bias, y=y, config=best), ncopy, budget_s=1.0)
            t_bt = BC.time_graph(lambda i: shim.narrow(x, nws[i], bias, y=y, source="bf16t", config=best), ncopy,
                                 budget_s=1.0)
            t_cb = BC.time_graph(lambda i: torch.addmm(bias, x, wts[i % len(wts)].T, out=y), len(wts), budget_s=1.0)
            nf, bt, cb = t_nf["median_us"], t_bt["median_us"], t_cb["median_us"]
            print(f"{key:>12} {m:>6} | {nf:7.2f} {bt:7.2f} {cb:7.2f} | {cb / nf:6.2f}  {shim.narrow_config_name(best)}"
                  f" ({shim.narrow_config_name(heur)} {timed.get(heur, float('nan')):.2f})", flush=True)
            results.append(dict(shape=key, m=m, nf12_us=nf, bf16t_us=bt, cublas_us=cb, best=best, heuristic=heur,
                                sweep={str(c): t for c, t in timed.items()}))
        del w, nw, nws, wts
        torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
