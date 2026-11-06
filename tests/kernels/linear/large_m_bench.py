# SPDX-License-Identifier: Apache-2.0
"""Large-M linear benchmark: NF12 vs bf16t twin vs torch.matmul (cuBLAS) vs
Codec decode + torch.matmul. CUDA-event medians with L2 flushed before every
call. Run with `flock /tmp/nf12-thor.lock uv run --group bench python tests/kernels/linear/large_m_bench.py`.

--sweep times every mma config x split (or every tc config) per point and
reports the best, which is how the heuristic tables were tuned.
"""

import argparse
import json
import os
import sys

import torch

os.environ.setdefault("NF12_SHIM_DECODE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import large_m_harness as H  # noqa: E402

SHAPES = [(4096, 4096), (4096, 14336), (14336, 4096), (4096, 28672), (5120, 25600), (25600, 5120), (8192, 8192),
          (8192, 57344)]  # (K, N)
MS = [64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192, 16384]


class Timer:
    """L2-flushed CUDA-event medians. Several functions are timed round-robin
    (one call of each per repetition) so clock and power drift on a busy
    machine hits all of them alike."""

    def __init__(self):
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        # A read-only flush: a write flush leaves L2 full of dirty lines whose
        # write-back then competes with the timed kernel for DRAM bandwidth.
        self.flush_buf = torch.ones(max(2 * l2, 64 << 20) // 4, dtype=torch.float32, device="cuda")
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")

    def many(self, fns, reps):
        for fn in fns:
            fn()  # warm-up (lazy init, smem attributes)
        ts = [[] for _ in fns]
        for _ in range(reps):
            for i, fn in enumerate(fns):
                torch.sum(self.flush_buf, dim=0, out=self.sink)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                fn()
                e1.record()
                ts[i].append((e0, e1))
        torch.cuda.synchronize()
        out = []
        for t in ts:
            v = sorted(a.elapsed_time(b) * 1e3 for a, b in t)
            out.append(v[len(v) // 2])  # us
        return out

    def __call__(self, fn, reps):
        return self.many([fn], reps)[0]


def warm_clocks(seconds=0.3):
    # Warm Thor's slowly ramping clocks before timing the first points.
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    while True:
        for _ in range(10):
            a @ a
        e1.record()
        e1.synchronize()
        if e0.elapsed_time(e1) > seconds * 1e3:
            break


def mma_label(shim, code):
    dp = code // 256
    return f"{shim.config_name(H.FAMILY_MMA, code % 16)}/s{code // 16 % 16 + 1}" + (f"+dp{dp}" if dp else "")


def mma_candidates(shim, m, n, sms):
    # Split-K only pays when the unsplit grid is a few waves at most (below
    # SM90 through the workspace, from SM90 through clusters).
    tiles = ((m + 127) // 128) * ((n + 127) // 128)
    splits = [1, 2, 4, 8] if tiles < sms else [1, 2, 4] if tiles < 6 * sms else [1]
    return [c + 16 * (s - 1) for c in range(shim.num_configs(H.FAMILY_MMA)) for s in splits]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["mma", "tc", "auto"], default="auto")
    ap.add_argument("--shapes", default="all", help="comma list of KxN, or all")
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="with --sweep: print every candidate (nf12 and bf16t)")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-escapes", action="store_true", help="weights with zero escaped groups")
    ap.add_argument("--escapes", type=float, default=0.0, help="force this fraction of escaped groups")
    a = ap.parse_args()
    fam = {"mma": H.FAMILY_MMA, "tc": H.FAMILY_TC, "auto": H.FAMILY_AUTO}[a.family]
    shapes = SHAPES if a.shapes == "all" else [tuple(map(int, s.split("x"))) for s in a.shapes.split(",")]
    ms = list(map(int, a.ms.split(",")))
    shim = H.Shim()
    timer = Timer()
    warm_clocks()
    name = torch.cuda.get_device_name()
    print(f"# {name}  family={a.family}  L2-flushed CUDA-event medians (us); TFLOP/s for NF12")
    hdr = f"{'K':>6} {'N':>6} {'M':>6} | {'nf12':>9} {'bf16t':>9} {'cublas':>9} {'dec+cub':>9} | {'TF/s':>6} {'%cub':>6}"
    if a.sweep:
        hdr += "  best-config"
    print(hdr)
    results = []
    for k, n in shapes:
        g = torch.Generator(device="cuda").manual_seed(0)
        w = torch.randn(n, k, device="cuda", generator=g) * 0.02
        if a.no_escapes:
            w = torch.where(w < 0, -1.0, 1.0) * w.abs().clamp_min(2.0**-12)  # all values inside one window
        w = w.to(torch.bfloat16)
        if a.escapes:
            w = H.force_escapes(w, a.escapes)
        nw = H.NF12Packed(w)
        print(f"# K={k} N={n}: escaped groups {100.0 * nw.num_escapes / (n * k / 8):.3f}%", flush=True)
        wdec = torch.empty_like(w)
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            flops = 2.0 * m * n * k
            reps = a.reps if flops < 2e12 else max(5, a.reps // 3)
            best_cfg, label = -1, ""
            if a.sweep:
                sms = torch.cuda.get_device_properties(0).multi_processor_count
                cands = (mma_candidates(shim, m, n, sms) if fam != H.FAMILY_TC
                         else list(range(shim.num_configs(H.FAMILY_TC))))
                sweep_fam = H.FAMILY_TC if fam == H.FAMILY_TC else H.FAMILY_MMA
                best = None
                for c in cands:
                    fn = lambda c=c: shim.linear(x, nw, y=y, family=sweep_fam, config=c)  # noqa: E731
                    try:
                        fn()
                    except RuntimeError:
                        continue
                    if a.verbose:
                        t, tt = timer.many(
                            [fn, lambda c=c: shim.linear(x, nw, y=y, source="bf16t", family=sweep_fam, config=c)],
                            max(3, reps // 2))
                        print(f"    cfg {c:4d} {mma_label(shim, c):>28}: "
                              f"nf12 {t:9.1f}  bf16t {tt:9.1f}  ratio {t / tt:5.2f}", flush=True)
                    else:
                        t = timer(fn, max(3, reps // 2))
                    if best is None or t < best[0]:
                        best = (t, c)
                best_cfg = best[1]
                fam_run = sweep_fam
                label = (mma_label(shim, best_cfg)
                         if sweep_fam == H.FAMILY_MMA else shim.config_name(sweep_fam, best_cfg))
            else:
                fam_run = fam
                if fam != H.FAMILY_TC:
                    hc = shim.mma_heuristic(m, n, k)
                    label = mma_label(shim, hc)
            fns = [lambda: shim.linear(x, nw, y=y, family=fam_run, config=best_cfg),
                   lambda: shim.linear(x, nw, y=y, source="bf16t", family=fam_run, config=best_cfg)]
            if not a.no_baselines:
                fns.append(lambda: torch.matmul(x, w.T, out=y))
                if shim.has_decode:
                    def dec_mm():
                        shim.decode(nw, wdec)
                        torch.matmul(x, wdec.T, out=y)
                    fns.append(dec_mm)
            ts = timer.many(fns, reps) + [float("nan")] * (4 - len(fns))
            t_nf12, t_t, t_cub, t_dec = ts
            tf = flops / t_nf12 / 1e6
            pct = 100.0 * t_cub / t_nf12
            line = (f"{k:>6} {n:>6} {m:>6} | {t_nf12:9.1f} {t_t:9.1f} {t_cub:9.1f} {t_dec:9.1f} | {tf:6.1f} "
                    f"{pct:5.0f}%")
            if label:
                line += f"  {label}"
            print(line, flush=True)
            results.append(dict(k=k, n=n, m=m, nf12_us=t_nf12, bf16t_us=t_t, cublas_us=t_cub, dec_cublas_us=t_dec,
                                tflops=tf, pct_cublas=pct, config=best_cfg, label=label))
            del x, y
        del w, nw, wdec
        torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(device=name, family=a.family, results=results), f, indent=1)


if __name__ == "__main__":
    main()
