# SPDX-License-Identifier: Apache-2.0
"""Thor/SM100 benchmark of the tcgen05 grouped-GEMM moe_linear family.

Per model projection (w13: gathered x -> sorted rows; w2: sorted rows ->
scattered, row-scaled) and token count T with random top-k routing: NF12 and
bf16t through moe_linear_large_tc (heuristic config unless --configs is
given) and the BF16 Triton fused-MoE kernel (bench/triton_fused_moe.py,
vLLM's default BF16 config). CUDA events report median time over reps.
Before each timed launch, a 128 MB buffer is written to flush L2. Each
format rotates >= --rotate-gib of weight-stack copies, each with its own
escape table, and --draws routing draws to avoid warm weights and TLBs.
Variants are interleaved in rotating order after a >= 300 ms warm-up.
Runs whose sampled Tegra clocks dipped are repeated. Columns report
microseconds, active-expert weight bytes (including NF12 escapes) / time in
GB/s, and useful TFLOP/s.
Run under `flock /tmp/nf12-thor.lock`.

Usage: uv run --group bench python tests/kernels/moe/large_tc_bench.py [--models qwen3-30b,...] [--projs w13,w2]
                                [--ts 32,64,...] [--configs -1,0,3] [--reps 12]
                                [--routing uniform|skew] [--rotate-gib 2]
                                [--real ~/.cache/nf12-bench/moe-real/qwen3-30b-a3b.L0.{proj}.safetensors]
"""

import argparse
import dataclasses
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "bench"))
import common  # noqa: E402  (bench/common.py: Tegra clock sampling)
import large_tc_harness as H  # noqa: E402

# name: (E, topk, hidden, moe_intermediate), w13 = [E, 2I, H], w2 = [E, H, I].
MODELS = {
    "qwen3-30b": (128, 8, 2048, 768),
    "dsv2-lite": (64, 6, 2048, 1408),
    "mixtral": (8, 2, 4096, 14336),
    "qwen3-235b": (128, 8, 4096, 1536),
}
DEFAULT_TS = "32,64,128,256,512,1024,2048,4096"


def routing(T, E, topk, kind, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "skew":  # Zipf-like popularity: a few hot experts
        rank = torch.randperm(E, generator=g, device="cuda")
        logp = -2.0 * torch.log1p(rank.float())
    else:
        logp = torch.zeros(E, device="cuda")
    u = torch.rand(T, E, generator=g, device="cuda").clamp_(1e-20, 1.0)
    ids = (logp - torch.log(-torch.log(u))).topk(topk, dim=1).indices.to(torch.int32)
    w = torch.softmax(torch.randn(T, topk, generator=g, device="cuda"), dim=1)
    return ids, w


def time_variants(fns, reps, flush):
    """Median ms for each fns[name](rep), with rotating order and L2 flushes.
    Retry up to four times if sampled Tegra GPC/EMC clocks dip below maximum."""
    for _ in range(4):
        ms, lo = _time_once(fns, reps, flush)
        if all(f >= common._mhz(common._TEGRA_CLOCKS[k] / "max_freq") for k, f in lo.items()):
            break
    return ms


def _time_once(fns, reps, flush):
    names = list(fns)
    t_end = time.time() + 0.3
    i = 0
    while time.time() < t_end:
        for f in fns.values():
            f(i)
        i += 1
    torch.cuda.synchronize()
    times = {n: [] for n in names}
    lo = {}
    for r in range(reps):
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in names]
        order = names[r % len(names):] + names[:r % len(names)]
        for i, n in enumerate(order):
            flush.fill_(r & 0xFF)
            ev[i][0].record()
            fns[n](r)
            ev[i][1].record()
        for k, f in common.wait_sampling_clocks(ev[-1][1]).items():
            lo[k] = min(lo.get(k, f), f)
        for i, n in enumerate(order):
            times[n].append(ev[i][0].elapsed_time(ev[i][1]))
    return {n: statistics.median(v) for n, v in times.items()}, lo


def copies(nbytes, gib):
    """Copies of an nbytes stack so that the rotation spans at least gib GiB."""
    return max(1, math.ceil(gib * (1 << 30) / nbytes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-30b,dsv2-lite,mixtral")
    ap.add_argument("--projs", default="w13,w2")
    ap.add_argument("--ts", default=DEFAULT_TS)
    ap.add_argument("--configs", default="-1")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--routing", default="uniform", choices=["uniform", "skew"])
    ap.add_argument("--draws", type=int, default=4, help="routing draws rotated with the weight copies")
    ap.add_argument("--rotate-gib", type=float, default=2.0,
                    help="weight copies (each with its own escape table) rotated per format, in GiB")
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--no-bf16t", action="store_true")
    ap.add_argument("--distinct", type=int, default=16, help="distinct experts encoded (bytes are tiled)")
    ap.add_argument("--real", help="safetensors path template with {proj} holding a real BF16 expert stack "
                    "[E, N, K] under key 'weight' (one model; e.g. bench_moe.py's qwen3-30b-a3b.L0.{proj})")
    ap.add_argument("--lib", help="prebuilt shim library (e.g. a variant build) instead of building")
    args = ap.parse_args()

    lib = H.Lib(args.lib)
    if not lib.supported():
        print("tcgen05 MoE family not supported here")
        return 1
    cfgs = [int(c) for c in args.configs.split(",")]
    tfm = None
    if not args.no_triton:
        try:
            import triton_fused_moe as tfm  # noqa: F811
        except Exception as e:  # noqa: BLE001
            print(f"triton baseline unavailable: {e}", file=sys.stderr)
    flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")

    srcs = ("nf12",) if args.no_bf16t else ("nf12", "bf16t")
    cols = [f"{lib.config_name(c)} {s}" for c in cfgs for s in srcs] + ([] if tfm is None else ["triton"])
    print("| model proj | T | active E | " + " | ".join(f"{c} us (GB/s, TF/s)" for c in cols) + " |", flush=True)
    print("|---|---|---|" + "---|" * len(cols), flush=True)
    for model in args.models.split(","):
        E, topk, hidden, inter = MODELS[model]
        for proj in args.projs.split(","):
            n, k = (2 * inter, hidden) if proj == "w13" else (hidden, inter)
            if args.real:
                from safetensors.torch import load_file
                w = load_file(os.path.expanduser(args.real.format(proj=proj)))["weight"].cuda()
                assert tuple(w.shape) == (E, n, k), f"{tuple(w.shape)} is not {model} {proj}"
                enc = H.encode_moe(w)
                print(f"real {proj}: {enc.num_escapes} escaped groups "
                      f"({100.0 * enc.num_escapes * 8 / w.numel():.3f}%)", file=sys.stderr)
            else:
                d = min(args.distinct, E)
                wd = H.random_experts(d, n, k, seed=1)
                enc = H.encode_moe(wd, num_experts=E)
                w = wd[torch.arange(E, device="cuda") % d].contiguous()  # the BF16 stack matching the encoded bytes
                del wd
            # Rotated copies: each launch reads weights (and escape tables)
            # no earlier launch of the rep left in L2 or the TLBs.
            nf12_bytes = enc.packed.numel() + enc.escapes.numel() * 2
            stacks = {"nf12": [enc] + [dataclasses.replace(enc, packed=enc.packed.clone(), escapes=enc.escapes.clone())
                                       for _ in range(copies(nf12_bytes, args.rotate_gib) - 1)]}
            if not args.no_bf16t:
                stacks["bf16t"] = [enc] + [dataclasses.replace(enc, bf16t=enc.bf16t.clone())
                                           for _ in range(copies(enc.bf16t.numel(), args.rotate_gib) - 1)]
            if tfm is not None:
                stacks["bf16"] = [w] + [w.clone() for _ in range(copies(w.numel() * 2, args.rotate_gib) - 1)]
            esc_rows = enc.escape_offsets.diff()
            for T in [int(t) for t in args.ts.split(",")]:
                A = T * topk
                draws = []
                for dr in range(args.draws):
                    ids, tw = routing(T, E, topk, args.routing, seed=T * 131 + dr)
                    sorted_ids, offsets = H.align(ids, E)
                    live = offsets.diff() > 0
                    draws.append(dict(ids=ids, tw=tw.view(-1), sorted=sorted_ids, offsets=offsets,
                                      active=int(live.sum()), esc=int(esc_rows[live].sum()), asg={}))
                active = statistics.mean(d["active"] for d in draws)
                flops = 2.0 * A * n * k
                x = torch.randn(T if proj == "w13" else A, k, device="cuda").bfloat16()
                y = torch.empty(A, n, dtype=torch.bfloat16, device="cuda")

                def kw(d):
                    return (dict(gather=d["sorted"], gather_div=topk) if proj == "w13"
                            else dict(scatter=d["sorted"], row_scale=d["tw"]))

                fns, bytes_of = {}, {}
                for c in cfgs:
                    for s in srcs:
                        src = H.NF12 if s == "nf12" else H.BF16T
                        name = f"{lib.config_name(c)} {s}"

                        def run(r, c=c, src=src, st=stacks[s]):
                            d = draws[r % len(draws)]
                            lib.moe_linear(c, x, st[r % len(st)], d["offsets"], A, y, source=src, **kw(d))
                        fns[name] = run
                        bytes_of[name] = statistics.mean(
                            d["active"] * enc.packed[0].numel() + 16 * d["esc"] if s == "nf12"
                            else d["active"] * n * k * 2 for d in draws)
                if tfm is not None:
                    cfg = tfm.get_default_config(T, E)
                    out = torch.empty(A, n, dtype=torch.bfloat16, device="cuda")

                    def triton(r, cfg=cfg, out=out):
                        d = draws[r % len(draws)]
                        if cfg["BLOCK_SIZE_M"] not in d["asg"]:
                            d["asg"][cfg["BLOCK_SIZE_M"]] = tfm.prepare_assignment(d["ids"], cfg["BLOCK_SIZE_M"], E)
                        asg = d["asg"][cfg["BLOCK_SIZE_M"]]
                        wb = stacks["bf16"][r % len(stacks["bf16"])]
                        if proj == "w13":
                            tfm.invoke(x, wb, out, None, asg, topk, A, cfg)
                        else:
                            tfm.invoke(x, wb, out, d["tw"], asg, 1, A, cfg)
                    fns["triton"] = triton
                    bytes_of["triton"] = active * n * k * 2
                ms = time_variants(fns, args.reps, flush)
                cells = []
                for c in cols:
                    t = ms[c]
                    cells.append(f"{t * 1e3:.0f} ({bytes_of[c] / t / 1e6:.0f}, {flops / t / 1e9:.1f})")
                print(f"| {model} {proj} [{E},{n},{k}] | {T} | {active:.0f} | " + " | ".join(cells) + " |",
                      flush=True)
            del w, enc, stacks
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
