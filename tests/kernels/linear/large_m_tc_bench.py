# SPDX-License-Identifier: Apache-2.0
"""Thor/SM100 benchmark of the tcgen05 large-M kernels vs cuBLAS.

For each weight shape (K x N) and token count M: NF12 and bf16t through
linear_large_m_tc (heuristic config unless --configs is given) and
torch.matmul(x, w.T) (cuBLAS). CUDA events report median time over reps.
Before each timed launch, a 128 MB buffer is written to flush L2 and a
64 MB buffer is read to drain the flush's dirty lines. Variants are
interleaved after a >= 300 ms warm-up to share DVFS/thermal drift.
Their order rotates each rep because Thor launches immediately after cuBLAS
can measure ~5% faster than later launches. Run under `flock /tmp/nf12-thor.lock`.

--graph follows bench/common.py with captured launches and rotating cold
weight copies, each owning its escape table. B200's ~2 us event-timer steps
are too coarse for microsecond-scale launches at M <= 512.

Usage: uv run --group bench python tests/kernels/linear/large_m_tc_bench.py [--shapes 4096x4096,...] [--ms 64,128,...]
                                  [--configs 0,3,6] [--reps 10] [--graph]
                                  [--compare-lib lib.so [--compare-config N]]
                                  [--escapes 0.1 | --real layer.safetensors]
"""

import argparse
import copy
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "bench"))
import common as C  # noqa: E402
import large_m_harness as H  # noqa: E402

DEFAULT_SHAPES = "4096x4096,4096x14336,14336x4096,4096x28672,5120x25600,25600x5120,8192x8192,8192x57344"
DEFAULT_MS = "16,32,64,96,128,192,256,512,1024,2048,4096,8192,16384"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=DEFAULT_SHAPES, help="comma list of KxN")
    ap.add_argument("--ms", default=DEFAULT_MS)
    ap.add_argument("--configs", default="-1", help="tc config ids (-1 = heuristic)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--max-seconds", type=float, default=20.0, help="per (shape, M) timing budget")
    ap.add_argument("--no-bf16t", action="store_true")
    ap.add_argument("--graph", action="store_true", help="CUDA-graph timing over rotated weight copies (cold)")
    ap.add_argument("--compare-lib", help="another build of the shim (e.g. an older revision): its "
                    "NF12 timing (--compare-config) is interleaved as 'before'")
    ap.add_argument("--compare-config", type=int, default=-1, help="config of the --compare-lib build (-1: heuristic)")
    ap.add_argument("--escapes", type=float, default=0.0, help="fraction of groups with a forced escape")
    ap.add_argument("--real", help="safetensors file holding one real weight matrix [N, K] (replaces --shapes)")
    args = ap.parse_args()

    shim = H.Shim()
    before = H.Shim(args.compare_lib) if args.compare_lib else None
    fam = H.FAMILY_TC
    if not shim.supported(fam):
        print("tcgen05 family not supported here")
        return 1
    cfgs = [int(c) for c in args.configs.split(",")]
    flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    clean = torch.empty(16 << 20, dtype=torch.float32, device="cuda")
    print("| K x N | M | cuBLAS us | " + " | ".join(
        f"{'auto' if c < 0 else shim.config_name(fam, c)} {src} us (TF/s, % cuBLAS)"
        for c in cfgs for src in (("nf12",) if args.no_bf16t else ("nf12", "bf16t")))
        + (" | before nf12 us (TF/s, % cuBLAS)" if before else "") + " |", flush=True)

    if args.real:
        from safetensors import safe_open

        with safe_open(args.real, "pt", device="cuda") as f:
            real = f.get_tensor(next(iter(f.keys()))).to(torch.bfloat16).contiguous()
        label = " " + os.path.basename(args.real).removesuffix(".safetensors")
        shapes = [(real.shape[1], real.shape[0])]
    else:
        label = f" esc={args.escapes:g}" if args.escapes else ""
        shapes = [tuple(int(v) for v in shape.split("x")) for shape in args.shapes.split(",")]
    for k, n in shapes:
        if args.real:
            w = real
        else:
            torch.manual_seed(0)
            w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
            if args.escapes:
                w = H.force_escapes(w, args.escapes)
        P = H.NF12Packed(w)
        # Graph mode: rotated copies, each owning its tensors (escapes included).
        copies = C.cold_copies(P.bytes_nf12()) if args.graph else 1
        ps, wts = [P], [w]
        for _ in range(copies - 1):
            c = copy.copy(P)
            c.packed, c.escapes = P.packed.clone(), P.escapes.clone()
            c.bf16t = None if args.no_bf16t else P.bf16t.clone()
            ps.append(c)
            wts.append(w.clone())
        for m in (int(v) for v in args.ms.split(",")):
            x = torch.randn(m, k, device="cuda").bfloat16()
            y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            variants = [("cublas", lambda i: torch.matmul(x, wts[i].T))]
            for c in cfgs:
                ws, nbytes = shim.workspace(fam, c, m, n, k)
                for src in ("nf12",) if args.no_bf16t else ("nf12", "bf16t"):
                    variants.append((f"{c}/{src}", lambda i, c=c, src=src, ws=ws: shim.linear(
                        x, ps[i], None, y, source=src, family=fam, config=c, ws=ws)))
            if before:
                bc = args.compare_config
                bws, _ = before.workspace(fam, bc, m, n, k)
                variants.append(("before", lambda i: before.linear(x, ps[i], None, y, family=fam, config=bc, ws=bws)))
            if args.graph:
                # Two rounds in alternating order against drift.
                ts = {name: [] for name, _ in variants}
                for order in (variants, variants[::-1]):
                    for name, fn in order:
                        ts[name].append(C.time_graph(fn, copies, budget_s=0.3)["median_us"])
            else:
                t0 = time.time()
                while time.time() - t0 < 0.3:
                    for _, fn in variants:
                        fn(0)
                torch.cuda.synchronize()
                reps = args.reps
                ts = {name: [] for name, _ in variants}
                start = time.time()
                for rep in range(reps):
                    shift = rep % len(variants)
                    for name, fn in variants[shift:] + variants[:shift]:
                        flush.fill_(rep & 0xFF)
                        clean.sum()
                        a = torch.cuda.Event(enable_timing=True)
                        b = torch.cuda.Event(enable_timing=True)
                        a.record()
                        fn(0)
                        b.record()
                        torch.cuda.synchronize()
                        ts[name].append(a.elapsed_time(b) * 1e3)
                    if rep >= len(variants) - 1 and time.time() - start > args.max_seconds:
                        break
            med = {name: statistics.median(v) for name, v in ts.items()}
            fl = 2.0 * m * n * k
            cb = med["cublas"]
            cells = [f"{k}x{n}{label}", str(m), f"{cb:.1f} ({fl / cb / 1e6:.1f})"]
            for name, _ in variants[1:]:
                t = med[name]
                cells.append(f"{t:.1f} ({fl / t / 1e6:.1f}, {cb / t * 100:.0f}%)")
            print("| " + " | ".join(cells) + " |", flush=True)
            del x, y
        del w, P, ps, wts
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
