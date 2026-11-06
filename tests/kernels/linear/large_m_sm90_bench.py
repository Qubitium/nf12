# SPDX-License-Identifier: Apache-2.0
"""SM90 large-M benchmark: the SM90 family (NF12 and its bf16t twin) vs
torch.matmul (cuBLAS) vs the portable mma.sync family (NF12, its heuristic).
Run with `flock /tmp/nf12-gpu2.lock uv run --group bench python tests/kernels/linear/large_m_sm90_bench.py`.

`--timing graph` (default) follows bench/common.py with back-to-back graph
launches and enough rotating weight copies to read weights from DRAM.
`events` times single launches round-robin with an L2 flush before each.

Shapes are bench/common.py keys (KxN or tags such as 4096x4096-mistral-l0q).
--weights real loads the checkpoint layer behind a key where one exists
(HF_HOME must hold it), else random normal(0, 0.02) weights.

--sweep times every SM90 config x schedule per point and reports the best.
--escapes F forces a fraction F of escaped groups. --stress also times NF12
on a copy with 10% escaped groups.
"""

import argparse
import json
import os
import sys

import torch

os.environ.setdefault("NF12_SHIM_DECODE", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "bench"))
import common as BC  # noqa: E402
import large_m_harness as H  # noqa: E402
import large_m_sm90_harness as S  # noqa: E402
from large_m_bench import MS, Timer, warm_clocks  # noqa: E402

DEFAULT_SHAPES = [s.key for s in BC.SHAPES + BC.EXTRA_SHAPES]


def candidates(shim):
    """Every config x schedule, splitting k 2-8 ways for the split schedules."""
    return [c + 16 * sch for c in range(shim.sm90_num_configs()) for sch in (S.HYBRID, S.DATA_PARALLEL, S.STREAM_K)] + \
        [c + 16 * S.SPLIT + 256 * s for c in range(shim.sm90_num_configs()) for s in (2, 3, 4, 6, 8)] + \
        [c + 16 * S.CLUSTER_SPLIT + 256 * s for c in range(shim.sm90_num_configs()) for s in (2, 3, 4, 6, 8)
         if shim.sm90_cluster_split_ok(c)]


class Copies:
    """Independent packed weights and escape tables for cold-L2 rotation.
    The bf16t twin is cloned when present."""

    def __init__(self, nw, n):
        self.items = [nw]
        for _ in range(n - 1):
            c = object.__new__(H.NF12Packed)
            c.__dict__.update(nw.__dict__)
            c.packed = nw.packed.clone()
            c.escapes = nw.escapes.clone()
            c.bf16t = nw.bf16t.clone() if nw.bf16t is not None else None
            self.items.append(c)

    def __getitem__(self, i):
        return self.items[i % len(self.items)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(DEFAULT_SHAPES), help="comma list of bench/common.py shape keys")
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--timing", choices=["graph", "events"], default="graph")
    ap.add_argument("--configs", default=None, help="comma list of SM90 configs to sweep (default: all)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="with --sweep: print every candidate")
    ap.add_argument("--reps", type=int, default=15, help="events timing: repetitions")
    ap.add_argument("--escapes", type=float, default=0.0, help="force this fraction of escaped groups")
    ap.add_argument("--stress", action="store_true", help="also time NF12 with 10%% escaped groups")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    shapes = [BC.shape_by_key(s) for s in a.shapes.split(",")]
    ms = list(map(int, a.ms.split(",")))
    shim = S.Sm90Shim()
    assert shim.sm90_supported(), "SM90 family unavailable"
    cands = list(map(int, a.configs.split(","))) if a.configs else candidates(shim)
    timer = Timer() if a.timing == "events" else None

    def time_many(fns, copies):
        """Medians (us) of fns[i](copy index), in order."""
        if timer is not None:
            return timer.many([lambda f=f: f(0) for f in fns], a.reps)
        return [BC.time_graph(f, copies, budget_s=1.0)["median_us"] for f in fns]

    warm_clocks()
    name = torch.cuda.get_device_name()
    print(f"# {name}  {a.timing} timing, medians (us); TFLOP/s and %cuBLAS for SM90 NF12; "
          f"%mma = mma time / SM90 time")
    print(f"{'shape':>22} {'M':>6} | {'nf12':>8} {'bf16t':>8} {'cublas':>8} {'mma':>8}{' stress':>8} | "
          f"{'TF/s':>6} {'%cub':>5} {'nf/t':>5} {'%mma':>5}  config")
    results = []
    for shape in shapes:
        k, n = shape.k, shape.n
        w, origin = BC.make_weight(shape, a.weights, torch.device("cuda"))
        if a.escapes:
            w = H.force_escapes(w, a.escapes)
        nw = H.NF12Packed(w)
        ncopy = BC.cold_copies(nw.packed.numel()) if a.timing == "graph" else 1
        nws = Copies(nw, ncopy)
        bcopy = BC.cold_copies(2 * n * k) if a.timing == "graph" else 1
        wts = [w] + [w.clone() for _ in range(bcopy - 1)]
        stress = Copies(H.NF12Packed(H.force_escapes(w, 0.10), bf16t=False), ncopy) if a.stress else None
        print(f"# {shape.key} ({origin}): escaped groups {100.0 * nw.num_escapes / (n * k / 8):.3f}%"
              + (f", stress {100.0 * stress[0].num_escapes / (n * k / 8):.2f}%" if stress else ""), flush=True)
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            flops = 2.0 * m * n * k
            cfg = -1
            if a.sweep:
                best = None
                for c in cands:
                    fn = lambda i, c=c: shim.sm90(x, nws[i], y=y, config=c)  # noqa: E731
                    try:
                        fn(0)
                    except H.Unsupported:
                        continue
                    t = time_many([fn], ncopy)[0]
                    if a.verbose:
                        print(f"    {shim.sm90_config_name(c):>28}: {t:9.1f}", flush=True)
                    if best is None or t < best[0]:
                        best = (t, c)
                cfg = best[1]
            ts = time_many([lambda i: shim.sm90(x, nws[i], y=y, config=cfg),
                            lambda i: shim.sm90(x, nws[i], y=y, source="bf16t", config=cfg),
                            lambda i: torch.matmul(x, wts[i % len(wts)].T, out=y),
                            lambda i: shim.linear(x, nws[i], y=y, family=H.FAMILY_MMA, config=-1)]
                           + ([lambda i: shim.sm90(x, stress[i], y=y, config=cfg)] if stress else []),
                           max(ncopy, bcopy))
            ts += [float("nan")] * (5 - len(ts))
            t_nf12, t_t, t_cub, t_mma, t_st = ts
            tf = flops / t_nf12 / 1e6
            label = shim.sm90_config_name(cfg if cfg >= 0 else shim.sm90_heuristic(m, n, k))
            print(f"{shape.key:>22} {m:>6} | {t_nf12:8.1f} {t_t:8.1f} {t_cub:8.1f} {t_mma:8.1f} {t_st:8.1f}"
                  f" | {tf:6.1f} {100 * t_cub / t_nf12:4.0f}% {t_nf12 / t_t:5.2f} {100 * t_mma / t_nf12:4.0f}%  {label}",
                  flush=True)
            results.append(dict(shape=shape.key, k=k, n=n, m=m, nf12_us=t_nf12, bf16t_us=t_t, cublas_us=t_cub,
                                mma_us=t_mma, stress_us=t_st, tflops=tf, pct_cublas=100 * t_cub / t_nf12,
                                config=cfg, label=label, escape_rate=nw.num_escapes / (n * k / 8)))
            del x, y
        del w, nw, nws, wts, stress
        torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(device=name, timing=a.timing, weights=a.weights, results=results), f, indent=1)


if __name__ == "__main__":
    main()
