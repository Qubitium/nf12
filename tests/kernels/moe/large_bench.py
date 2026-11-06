# SPDX-License-Identifier: Apache-2.0
"""Grouped (MoE) prefill benchmark: the SM120 / mma.sync families on NF12 and
their bf16t twin, per config, vs vLLM's Triton fused-MoE BF16 kernel.

    flock /tmp/nf12-gpu1.lock uv run --group bench python tests/kernels/moe/large_bench.py --family sm120 --models qwen3-30b-a3b --t 512 2048
    ... --sweep        # report the best config and the heuristic's pick

Shapes, routing draws and the Triton baseline come from bench/bench_moe.py.
For w13, x rows are gathered through sorted assignments (gather_div = topk).
For w2, sorted rows are scattered with their routing weights.
Triton uses the per-(model, proj, T) config tuned by
bench_moe.py when its cache (--triton-configs) has one, else vLLM's default.
Timing: bench/common.py CUDA-graph mode, rotating over enough weight copies
(each with its own routing draw) that every launch streams its experts from
DRAM. Variants are timed round-robin per point (the board power-caps).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "bench"))
import bench_moe as BM  # noqa: E402
import common  # noqa: E402
import large_harness as H  # noqa: E402
import triton_fused_moe as tfm  # noqa: E402


def clone_weights(mw):
    c = object.__new__(H.MoeWeights)
    c.__dict__.update(mw.__dict__)
    c.packed = mw.packed.clone()
    c.bf16t = mw.bf16t.clone()
    c.escapes = mw.escapes.clone() if mw.escapes is not None else None
    return c


def run_point(shim, args, family, model, proj, t, kind, stacks, nfs, configs, triton_cfg):
    n, k = model.shape(proj)
    e, topk = model.experts, model.topk
    a = t * topk
    w13 = proj == "w13"
    copies = len(nfs)
    draws = [BM.make_draw(t, model, kind, 1000 * t + i + (0 if kind == "uniform" else 17)) for i in range(copies)]
    x = torch.randn(t if w13 else a, k, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(a, n, device="cuda", dtype=torch.bfloat16)

    def ours(source, config):
        def fn(i):
            d = draws[i % copies]
            shim.run(x, nfs[i % copies], d.offsets, a, y, gather=d.sorted if w13 else None, gather_div=topk,
                     scatter=None if w13 else d.sorted, row_scale=None if w13 else d.weights.view(-1),
                     source=source, family=family, config=config)
            return y
        return fn

    def triton(i):
        d = draws[i % copies]
        asg = BM.triton_assignment(d, triton_cfg["BLOCK_SIZE_M"], e)
        tfm.invoke(x, stacks[i % copies], y, None if w13 else d.weights.view(-1), asg, topk if w13 else 1, a,
                   triton_cfg)
        return y

    def timed(fn):
        try:
            return common.time_graph(fn, copies)["median_us"]
        except H.Unsupported:
            return None

    heur = shim.heuristic(family, a, e, n, k)
    row = {"t": t, "routing": kind, "heuristic": shim.config_name(family, heur)}
    if args.sweep:
        per = {}
        for c in configs:
            per[c] = timed(ours("nf12", c))
        ok = {c: us for c, us in per.items() if us is not None}
        best = min(ok, key=ok.get)
        row["sweep"] = {shim.config_name(family, c): us for c, us in ok.items()}
        row["best"] = shim.config_name(family, best)
        row["best_us"] = ok[best]
        row["heur_us"] = ok.get(heur)
        cfg = best
    else:
        cfg = configs[0] if len(configs) == 1 else heur
    # Round-robin: our config on both sources, then the baseline.
    row["config"] = shim.config_name(family, cfg)
    row["nf12"] = timed(ours("nf12", cfg))
    if not args.no_bf16t:
        row["bf16t"] = timed(ours("bf16t", cfg))
    if not args.no_triton:
        row["triton"] = timed(triton)
        row["triton_config"] = tfm.config_key(triton_cfg)
    return row


def fmt(v, f=".1f"):
    return "-" if v is None else format(v, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(H.FAMILIES), default="sm120")
    ap.add_argument("--models", nargs="*", default=list(BM.MODELS), choices=list(BM.MODELS))
    ap.add_argument("--proj", nargs="*", default=list(BM.PROJS), choices=list(BM.PROJS))
    ap.add_argument("--t", nargs="*", type=int, default=[64, 128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--routing", nargs="*", default=["uniform", "zipf"], choices=list(BM.ROUTINGS))
    ap.add_argument("--weights", choices=["random", "real"], default="random")
    ap.add_argument("--layer", type=int, default=None,
                    help="with --weights real: this MoE layer instead of bench_moe.py's (e.g. 0 for Qwen3-30B's "
                         "heavy-escape layer)")
    ap.add_argument("--configs", default=None, help="comma list of config indices (default: heuristic)")
    ap.add_argument("--sweep", action="store_true", help="time every config per point")
    ap.add_argument("--no-bf16t", action="store_true")
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--triton-configs", default=None,
                    help="bench_moe.py tuned Triton config cache (default: results/moe_triton_configs_<machine>.json)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    shim = H.Shim()
    family = H.FAMILIES[args.family]
    if not shim.supported(family, 1):
        sys.exit(f"family {args.family} unsupported on this device")
    configs = [int(c) for c in args.configs.split(",")] if args.configs else list(range(shim.num_configs(family)))
    tpath = Path(args.triton_configs) if args.triton_configs else \
        common.RESULTS_DIR / f"moe_triton_configs_{common.machine_tag()}.json"
    tuned = json.loads(tpath.read_text()) if tpath.exists() else {}
    out = []
    print(f"# {torch.cuda.get_device_name()}  family {args.family}  graph timing, medians (us); "
          f"nf/t = NF12 / bf16t twin, nf/tr = NF12 / Triton BF16")
    for key in args.models:
        model = BM.MODELS[key]
        for proj in args.proj:
            n, k = model.shape(proj)
            if args.weights == "real" and args.layer is not None:
                w = BM.load_real_stack(model, args.layer, proj)
                if w is None:
                    sys.exit(f"{key} layer {args.layer} {proj}: no real stack")
                w, origin = w.to("cuda", torch.bfloat16).contiguous(), f"{model.repo} layer {args.layer}"
            else:
                w, origin = BM.make_stack(model, proj, args.weights)
            mw = H.MoeWeights(w)
            copies = common.cold_copies(mw.packed.nbytes)  # the smallest stack a variant streams
            nfs = [mw] + [clone_weights(mw) for _ in range(copies - 1)]
            stacks = [w] + [w.clone() for _ in range(copies - 1)]
            print(f"# {model.name} {proj} [E={model.experts}, N={n}, K={k}] topk={model.topk} ({origin}), "
                  f"escaped groups {100 * mw.escape_fraction():.3f}%, {copies} copies")
            print(f"{'T':>6} {'routing':>8} | {'nf12':>8} {'bf16t':>8} {'triton':>8} | {'nf/t':>5} {'nf/tr':>5} | config")
            for t in args.t:
                for kind in args.routing:
                    tcfg = tuned.get(f"{key}/{proj}/{t}", {}).get("config") or tfm.get_default_config(t, model.experts)
                    r = run_point(shim, args, family, model, proj, t, kind, stacks, nfs, configs, tcfg)
                    r.update(model=key, proj=proj)
                    out.append(r)
                    nt = r["nf12"] / r["bf16t"] if r.get("bf16t") and r["nf12"] else None
                    ntr = r["nf12"] / r["triton"] if r.get("triton") and r["nf12"] else None
                    extra = ""
                    if args.sweep:
                        extra = f"  (best {r['best']} {fmt(r['best_us'])}, heuristic {r['heuristic']} {fmt(r['heur_us'])})"
                    print(f"{t:>6} {kind:>8} | {fmt(r['nf12']):>8} {fmt(r.get('bf16t')):>8} {fmt(r.get('triton')):>8} | "
                          f"{fmt(nt, '.2f'):>5} {fmt(ntr, '.2f'):>5} | {r['config']}{extra}", flush=True)
            del nfs, stacks, w, mw
            torch.cuda.empty_cache()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
