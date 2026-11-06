# SPDX-License-Identifier: Apache-2.0
"""MoE decode benchmark: NF12 vs its bf16t twin (same kernel and config).
The BF16 Triton fused-MoE baseline lives in bench/bench_moe.py.

Timing: bench/common.py time_graph (launches captured in one CUDA graph,
median per launch). Launch i uses weight copy i % C and routing (i / C) % R
(R = --routings, drawn up front). The C copies of all experts, each with its
own escape table, rotate >= 2 GiB of NF12 weights and reuse a copy only after
>= 4x L2 of other active-expert traffic, capped by --max-gib. Bandwidth = bytes of the
active experts' weights (distinct experts with >= 1 assignment) / time,
against the streaming-read rate (--stream-gbs, default per arch).

  uv run --group bench python tests/kernels/moe/small_bench.py
  uv run --group bench python tests/kernels/moe/small_bench.py --models qwen3-30b --ts 1,8 --sweep
  uv run --group bench python tests/kernels/moe/small_bench.py --models qwen3-30b --real <dir>
"""

import argparse
import json
import math
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "bench"))
import small_harness as h  # noqa: E402
from common import gpu_lock, l2_bytes, time_graph  # noqa: E402

MODELS = {  # name: (E, topk, hidden, moe_intermediate)
    "qwen3-30b": (128, 8, 2048, 768),
    "qwen3-235b": (128, 8, 4096, 1536),
    "dsv2-lite": (64, 6, 2048, 1408),
    "mixtral": (8, 2, 4096, 14336),
}
TS = [1, 2, 4, 8, 16, 32, 64]
STREAM_GBS = {120: 1537.0, 110: 268.0}
# --real: MoEBench's expert stacks <dir>/<prefix>.<layer>.<w13|w2>.safetensors (key "weight", BF16 [E, N, K]).
REAL_PREFIX = {"qwen3-30b": "qwen3-30b-a3b", "dsv2-lite": "dsv2-lite", "mixtral": "mixtral-8x7b"}
REAL_CACHE = {}


def routings(T, topk, E, count, skew, seed):
    out = []
    g = torch.Generator(device="cuda").manual_seed(seed)
    for r in range(count):
        if skew:  # every token picks the same topk experts (a different set per routing)
            base = torch.randperm(E, generator=g, device="cuda")[:topk].to(torch.int32)
            ids = base.repeat(T, 1)
        else:
            ids = torch.rand(T, E, generator=g, device="cuda").argsort(dim=1)[:, :topk].to(torch.int32)
        sorted_ids, offsets = h.align(ids, E)
        active = int((offsets[1:] > offsets[:-1]).sum())
        out.append((sorted_ids, offsets, active))
    return out


class Problem:
    """One GEMM of one model: weights (copies), routings, x, y, row_scale."""

    def __init__(self, model, half, T, skew, args):
        E, topk, H, I = MODELS[model]
        self.n, self.k = (2 * I, H) if half == "w13" else (H, I)
        self.E, self.topk, self.T, self.half = E, topk, T, half
        if args.real:
            enc = REAL_CACHE.get((model, half))
            if enc is None:
                from safetensors.torch import load_file

                path = os.path.join(args.real, f"{REAL_PREFIX[model]}.{args.real_layer}.{half}.safetensors")
                w = load_file(path, device="cuda")["weight"]
                assert tuple(w.shape) == (E, self.n, self.k), (path, w.shape)
                enc = REAL_CACHE[(model, half)] = h.encode_moe(w)
        else:
            w = h.random_experts(min(E, args.distinct), self.n, self.k, seed=1, escapes=args.escapes)
            enc = h.encode_moe(w, num_experts=E)
        self.esc_groups = enc.num_escapes / (E * self.n * self.k / 8)
        expert_nf12 = self.n * self.k * 3 // 2
        self.expert_bytes = {h.NF12: expert_nf12, h.BF16T: self.n * self.k * 2}
        self.routes = routings(T, topk, E, args.routings, skew, seed=T * 7 + (1 if skew else 0))
        # Shared escape tables would stay L2-resident across weight copies.
        per_copy = E * (expert_nf12 + self.n * self.k * 2)
        want = max(math.ceil(4 * l2_bytes() / self.active_bytes(h.NF12)) + 1,
                   math.ceil(2 * 2**30 / (E * expert_nf12)))
        copies = min(want, max(1, int(args.max_gib * 2**30 // per_copy)))
        self.encs = [enc] + [h.MoeEncoded(enc.w, enc.packed.clone(), enc.escapes.clone(), enc.escape_offsets,
                                          enc.exp_base, enc.bf16t.clone(), E, self.n, self.k)
                             for _ in range(copies - 1)]
        self.A = T * topk
        rows = T if half == "w13" else self.A
        self.x = (torch.randn(rows, self.k, device="cuda") * 0.5).to(torch.bfloat16)
        self.y = torch.empty(self.A, self.n, dtype=torch.bfloat16, device="cuda")
        self.scale = torch.rand(self.A, device="cuda")

    def launches(self):
        return len(self.routes) * len(self.encs)

    def run(self, lib, cfg, source, i):
        sorted_ids, offsets, _ = self.routes[(i // len(self.encs)) % len(self.routes)]
        enc = self.encs[i % len(self.encs)]
        if self.half == "w13":
            lib.moe_linear(cfg, self.x, enc, offsets, self.A, self.y, gather=sorted_ids, gather_div=self.topk,
                           source=source)
        else:
            lib.moe_linear(cfg, self.x, enc, offsets, self.A, self.y, scatter=sorted_ids, row_scale=self.scale,
                           source=source)

    def active_bytes(self, source):
        return statistics.mean(a for _, _, a in self.routes) * self.expert_bytes[source]


def sweep_configs(lib, heur, E, n, k):
    """Sweep neighbouring mt, all warps/kc/PDL and whole-range or 2/4-SA slices.
    Below SM90, use K parts 1-8 instead of kc/PDL and allow 8/16-SA slices."""
    sm8x = torch.cuda.get_device_capability()[0] < 9
    out = []
    for mt in sorted({max(1, heur.mt // 2), heur.mt, min(8, 2 * heur.mt)}):
        for warps in (4, 8):
            for kc in range(1, 9):
                for sl in (0, 2, 4, 8, 16) if sm8x else (0, 2, 4):
                    for pdl in (0, 1):
                        for kp in (1, 2, 4, 8) if sm8x else (1,):
                            cfg = h.Config(mt, warps, kc, sl, pdl, kp)
                            if lib.supported(cfg, E, n, k):
                                out.append(cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--halves", default="w13,w2")
    ap.add_argument("--ts", default=",".join(map(str, TS)))
    ap.add_argument("--routing", default="uniform,skew")
    ap.add_argument("--routings", type=int, default=8, help="distinct routings rotated per graph")
    ap.add_argument("--distinct", type=int, default=4, help="distinct expert matrices (tiled to E)")
    ap.add_argument("--escapes", type=float, default=0.0, help="fraction of out-of-window values (0: natural)")
    ap.add_argument("--real", metavar="DIR", help="real expert stacks instead of random weights")
    ap.add_argument("--real-layer", default="L0", help="layer tag of the --real stacks (L0: heavy escapes)")
    ap.add_argument("--max-gib", type=float, default=24.0, help="memory cap for the rotated weight copies")
    ap.add_argument("--sweep", action="store_true", help="time every supported config (NF12 only)")
    ap.add_argument("--stream-gbs", type=float, default=None)
    ap.add_argument("--lock", help="flock this file around each measurement (shared GPUs, e.g. /tmp/nf12-thor.lock)")
    ap.add_argument("--json", help="write results here (rewritten after every row)")
    ap.add_argument("--small", action="store_true",
                    help="also time the decode path's heuristic config where dispatch picks a grouped GEMM "
                         "(build with NF12_MOE_WITH_LARGE=1 to time that GEMM too)")
    args = ap.parse_args()

    def timed(fn, launches, **kw):
        with gpu_lock(args.lock):
            return time_graph(fn, launches, **kw)["median_us"]

    lib = h.Lib()
    cc = torch.cuda.get_device_capability()
    stream = args.stream_gbs or STREAM_GBS.get(cc[0] * 10 + cc[1], 1000.0)
    print(f"# {torch.cuda.get_device_name()} stream {stream:.0f} GB/s, cold L2, graph timing")
    results = []
    for model in args.models.split(","):
        for half in args.halves.split(","):
            for skew in [r == "skew" for r in args.routing.split(",")]:
                for T in map(int, args.ts.split(",")):
                    prob = Problem(model, half, T, skew, args)
                    heur = lib.heuristic(prob.A, prob.E, prob.n, prob.k)
                    row = dict(model=model, half=half, T=T, routing="skew" if skew else "uniform", n=prob.n,
                               k=prob.k, E=prob.E, topk=prob.topk, cfg=str(heur), esc_groups=prob.esc_groups,
                               active=prob.active_bytes(h.NF12) / prob.expert_bytes[h.NF12])
                    try:
                        for src, name in ((h.NF12, "nf12"), (h.BF16T, "bf16t")):
                            t = timed(lambda i: prob.run(lib, h.AUTO, src, i), prob.launches())
                            row[name + "_us"] = t
                            row[name + "_pct"] = prob.active_bytes(src) / (t * 1e-6) / 1e9 / stream * 100
                    except h.LargeStub:
                        if not (args.small or args.sweep):
                            print(f"{model} {half} T={T}: dispatches to a grouped-GEMM stand-in, skipped")
                            continue
                        row["nf12_us"] = row["bf16t_us"] = row["nf12_pct"] = row["bf16t_pct"] = float("nan")
                    if args.small:
                        row["small_us"] = timed(lambda i: prob.run(lib, heur, h.NF12, i), prob.launches())
                    if args.sweep:
                        row["sweep"] = {}
                        for cfg in sweep_configs(lib, heur, prob.E, prob.n, prob.k):
                            row["sweep"][str(cfg)] = timed(lambda i: prob.run(lib, cfg, h.NF12, i), prob.launches(),
                                                           budget_s=0.5)
                        row["best_cfg"], row["best_us"] = min(row["sweep"].items(), key=lambda kv: kv[1])
                    line = (f"{model:11s} {half:3s} {row['routing']:7s} T={T:3d} n={prob.n:5d} k={prob.k:5d}  "
                            f"nf12 {row['nf12_us']:8.1f} us ({row['nf12_pct']:5.1f}%)  "
                            f"bf16t {row['bf16t_us']:8.1f} us ({row['bf16t_pct']:5.1f}%)  "
                            f"x{row['bf16t_us'] / row['nf12_us']:.2f} vs bf16t")
                    if args.sweep:
                        line += f"  best {row['best_us']:8.1f} ({row['best_cfg']}, x{row['nf12_us'] / row['best_us']:.3f})"
                    if args.small:
                        line += f"  small {row['small_us']:8.1f} (x{row['nf12_us'] / row['small_us']:.3f})"
                    print(line + f"  {heur}", flush=True)
                    results.append(row)
                    if args.json:
                        with open(args.json, "w") as f:
                            json.dump(results, f, indent=1)
                    del prob
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
