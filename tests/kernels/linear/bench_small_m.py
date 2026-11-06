# SPDX-License-Identifier: Apache-2.0
"""Small-M linear benchmark: NF12 vs bf16t twin vs torch.matmul (cuBLAS).

Timing follows bench/common.py. Each CUDA graph replay contains `n` launches
bracketed by CUDA events. Samples are replay_time / n, reported as a median
over >= 100 launches. Cold mode rotates weight copies with >= 3x L2 of other
traffic between reads. Hot mode reuses one copy, keeping fitting weights in
L2. Effective bandwidth is (weight + x + y bytes) / time, compared with the
streaming-read rate measured using the kernel's load instruction.

  uv run --group bench python tests/kernels/linear/bench_small_m.py
  uv run --group bench python tests/kernels/linear/bench_small_m.py --hot
  uv run --group bench python tests/kernels/linear/bench_small_m.py --sweep
  uv run --group bench python tests/kernels/linear/bench_small_m.py --shapes 4096x4096 --ms 1,16
"""

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "bench"))
import small_m_harness as h  # noqa: E402
from common import gpu_lock  # noqa: E402

SHAPES = ["4096x4096", "4096x14336", "14336x4096", "4096x28672", "5120x25600", "25600x5120", "8192x8192",
          "8192x57344", "2048x8192", "896x4864"]
MS = [1, 2, 4, 8, 16, 32, 64]

# Real layers (label -> HF repo, tensors concatenated along N), incl. the most
# escape-heavy matrices of the escape study (Mistral layer-0 q_proj, early Qwen3 gate).
REAL = {
    "mistral7b.L0.q_proj": ("unsloth/mistral-7b-instruct-v0.3", ["model.layers.0.self_attn.q_proj.weight"]),
    "qwen3-8b.L0.gate_up": ("Qwen/Qwen3-8B", ["model.layers.0.mlp.gate_proj.weight",
                                              "model.layers.0.mlp.up_proj.weight"]),
    "qwen3-8b.L18.down": ("Qwen/Qwen3-8B", ["model.layers.18.mlp.down_proj.weight"]),
    "llama3.1-8b.L10.gate_up": ("unsloth/Meta-Llama-3.1-8B-Instruct", ["model.layers.10.mlp.gate_proj.weight",
                                                                       "model.layers.10.mlp.up_proj.weight"]),
    "qwen3-32b.L30.gate": ("Qwen/Qwen3-32B", ["model.layers.30.mlp.gate_proj.weight"]),
    "qwen3-32b.L30.down": ("Qwen/Qwen3-32B", ["model.layers.30.mlp.down_proj.weight"]),
}


def load_real(label):
    import json as _json

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    local = os.environ.get("NF12_REAL_DIR")  # <dir>/<label>.safetensors with tensor "w" (see --export-real)
    if local:
        with safe_open(os.path.join(local, f"{label}.safetensors"), "pt", device="cuda") as f:
            return f.get_tensor("w").to(torch.bfloat16).contiguous()
    repo, names = REAL[label]
    root = snapshot_download(repo, allow_patterns=["*.json"], local_files_only=True)
    index = os.path.join(root, "model.safetensors.index.json")
    files = _json.load(open(index))["weight_map"] if os.path.exists(index) else {}
    parts = []
    for name in names:
        with safe_open(os.path.join(root, files.get(name, "model.safetensors")), "pt", device="cuda") as f:
            parts.append(f.get_tensor(name).to(torch.bfloat16))
    return torch.cat(parts, 0).contiguous()


def l2_bytes():
    return torch.cuda.get_device_properties(0).L2_cache_size


def copies_for(nbytes, cold):
    return math.ceil(3 * l2_bytes() / nbytes) + 1 if cold else 1


def time_graph(fn, copies, min_launches=100, replay_target_ms=1.0, max_replays=60, warm_ms=200):
    """Median per-launch microseconds of fn(i % copies) launched back to back in a CUDA graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(copies):
            fn(i)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for i in range(copies):
            fn(i)
        b.record()
    b.synchronize()
    est_ms = max(a.elapsed_time(b) / copies, 1e-4)
    n = copies * max(1, math.ceil(replay_target_ms / (copies * est_ms)))
    replays = max(10, math.ceil(min_launches / n), min(max_replays, int(2e3 / (n * est_ms))))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for i in range(n):
            fn(i % copies)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(replays)]
    with torch.cuda.stream(stream):
        # Encoding weights between points lets clocks (notably Thor's EMC)
        # ramp down. Warm them with graph replays before timing.
        for _ in range(max(1, math.ceil(warm_ms / (n * est_ms)))):
            graph.replay()
        for s, e in events:
            s.record()
            graph.replay()
            e.record()
    stream.synchronize()
    del graph
    return statistics.median(s.elapsed_time(e) * 1e3 / n for s, e in events)


def calibrate(lib):
    """Best streaming-read bandwidth (GB/s) over launch shapes on a buffer of 8x L2 (>= 512 MiB)."""
    props = torch.cuda.get_device_properties(0)
    nbytes = max(1 << 29, 8 * l2_bytes())
    buf = torch.ones(nbytes // 4, dtype=torch.int32, device="cuda").view(torch.uint8)
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    best = 0.0
    for per_sm in (2, 4, 8, 16):
        for depth in (2, 4, 8):
            t = time_graph(lambda i: lib.stream_read(buf, props.multi_processor_count * per_sm, depth, out), 1,
                           min_launches=20, max_replays=10)
            best = max(best, nbytes / t / 1e3)
    del buf
    return best


class Point:
    """Weights (rotating copies when cold) and activations for one (shape, M)."""

    def __init__(self, enc, m, cold, sources):
        self.enc, self.m = enc, m
        self.x = torch.randn(m, enc.k, device="cuda").to(torch.bfloat16)
        self.y = torch.empty(m, enc.n, dtype=torch.bfloat16, device="cuda")
        self.encs, self.ws = {}, None
        for src in sources:
            if src == "torch":
                c = copies_for(enc.n * enc.k * 2, cold)
                self.ws = [enc.w] + [enc.w.clone() for _ in range(c - 1)]
            else:
                nbytes = enc.n * enc.k * (3 if src == "nf12" else 4) // 2
                c = copies_for(nbytes, cold)
                field = "packed" if src == "nf12" else "bf16t"
                self.encs[src] = [enc] + [replace(enc, **{field: getattr(enc, field).clone()}) for _ in range(c - 1)]

    def time(self, src, cfg, warm_ms=200):
        if src == "large":
            # linear_large_m() (production large-M family) on the same NF12 copies, for the crossover.
            import types

            import large_m_harness

            if not hasattr(Point, "large"):
                Point.large = large_m_harness.Shim()
            views = [types.SimpleNamespace(n=e.n, k=e.k, packed=e.packed, bf16t=None, escapes=e.escapes_u4,
                                           num_escapes=e.num_escapes, exp_base=e.exp_base) for e in self.encs["nf12"]]
            ws, _ = Point.large.workspace(large_m_harness.FAMILY_AUTO, -1, self.m, self.enc.n, self.enc.k)
            return time_graph(lambda i: Point.large.linear(self.x, views[i], y=self.y, family=large_m_harness.FAMILY_AUTO,
                                                           ws=ws), len(views))
        if src == "torch":
            ws, x = self.ws, self.x
            return time_graph(lambda i: torch.matmul(x, ws[i].T), len(ws))
        encs, s = self.encs[src], (h.NF12 if src == "nf12" else h.BF16T)
        return time_graph(lambda i: self.lib.linear(cfg, self.x, encs[i], y=self.y, source=s), len(encs),
                          warm_ms=warm_ms)


def time_chain(lib, k, n, m, cold, sources):
    """Per-layer microseconds of 4 dependent linears (K->N, N->K, K->N, N->K)
    captured back to back in a graph, each consuming the previous output, as
    in a layer sequence (exercises programmatic dependent launch)."""
    a, b = h.encode(h.random_weight(n, k, seed=1)), h.encode(h.random_weight(k, n, seed=2))
    xs = [torch.randn(m, k, device="cuda").to(torch.bfloat16), torch.empty(m, n, dtype=torch.bfloat16, device="cuda"),
          torch.empty(m, k, dtype=torch.bfloat16, device="cuda"), torch.empty(m, n, dtype=torch.bfloat16, device="cuda"),
          torch.empty(m, k, dtype=torch.bfloat16, device="cuda")]
    out = {}
    for src in sources:
        per_chain = 2 * n * k * (3 if src == "nf12" else 4) // 2
        copies = copies_for(per_chain, cold)
        if src == "torch":
            ws = [[e.w.clone() for e in (a, b, a, b)] for _ in range(copies)]

            def fn(i):
                for j, w in enumerate(ws[i]):
                    torch.matmul(xs[j], w.T, out=xs[j + 1])
        else:
            field = "packed" if src == "nf12" else "bf16t"
            s = h.NF12 if src == "nf12" else h.BF16T
            layers = [[replace(e, **{field: getattr(e, field).clone()}) for e in (a, b, a, b)] for _ in range(copies)]
            cfgs = [lib.heuristic(m, e.n, e.k) for e in (a, b, a, b)]

            def fn(i):
                for j, e in enumerate(layers[i]):
                    lib.linear(cfgs[j], xs[j], e, y=xs[j + 1], source=s)
        out[src] = time_graph(fn, copies) / 4
    return out


def sweep_configs(lib, m, n, k, model_slices=False):
    """Every warps/kc/kp/slice config. model_slices restricts each (warps, kc, kp)
    to its heuristic slice for tune_small_m.py."""
    mt = next(mt for mt in (1, 2, 4, 8) if 8 * mt >= m)
    kb = k // 64
    out = []
    for warps in (4, 8, 16):
        for kc in range(1, 9):
            for kp in (1, 2, 4, 8):
                cnt = -(-kb // kc)
                slices = sorted({0} | {s * kp for s in (2, 4, 8, 16) if s * kp < cnt})
                if model_slices:
                    slices = [lib.heuristic_slice(mt, warps, kc, kp, m, n, k)]
                for sl in slices:
                    cfg = h.Config(mt, warps, kc, sl, kp)
                    if sl >= 0 and lib.supported(cfg, m, n, k):
                        out.append(cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--hot", action="store_true", help="reuse one weight copy (weights may stay in L2)")
    ap.add_argument("--sweep", action="store_true", help="time every supported config (NF12 only)")
    ap.add_argument("--outliers", type=float, default=0.0,
                    help="fraction of synthetic weights replaced by out-of-window values (group escape rate ~8x)")
    ap.add_argument("--real", action="store_true", help="benchmark the real layers in REAL instead of --shapes")
    ap.add_argument("--vs-large", action="store_true",
                    help="small-M vs linear_large_m() (NF12, cold) per point, to place the crossover")
    ap.add_argument("--export-real", metavar="DIR",
                    help="write the REAL layers to DIR/<label>.safetensors for machines without the checkpoints")
    ap.add_argument("--chain", action="store_true",
                    help="per-layer time inside a graph of 4 dependent linears (K->N->K->N->K)")
    ap.add_argument("--json", help="write results here")
    ap.add_argument("--model-slices", action="store_true",
                    help="with --sweep: only the heuristic's slice per (warps, kc), for tune_small_m.py")
    ap.add_argument("--lock", help="flock this file around each (shape, M) point, each 8 configs with --sweep "
                    "(shared GPUs)")
    args = ap.parse_args()

    if args.export_real:
        from safetensors.torch import save_file

        os.makedirs(args.export_real, exist_ok=True)
        for label in REAL:
            save_file({"w": load_real(label).cpu()}, os.path.join(args.export_real, f"{label}.safetensors"))
        return
    lib = h.Lib()
    Point.lib = lib
    cold = not args.hot
    sol = calibrate(lib)
    props = torch.cuda.get_device_properties(0)
    print(f"# {props.name} ({props.multi_processor_count} SMs), streaming read {sol:.0f} GB/s, "
          f"{'cold' if cold else 'hot'} L2, graph timing", flush=True)
    results = []
    sources = ("nf12",) if args.sweep or args.vs_large else ("nf12", "bf16t", "torch")
    if args.chain:
        for shape in args.shapes.split(","):
            k, n = map(int, shape.split("x"))
            for m in map(int, args.ms.split(",")):
                r = time_chain(lib, k, n, m, cold, ("nf12", "bf16t", "torch"))
                print(f"chain K={k:6d} N={n:6d} M={m:3d}  nf12 {r['nf12']:7.1f} us/layer  bf16t {r['bf16t']:7.1f}  "
                      f"torch {r['torch']:7.1f}  x{r['torch'] / r['nf12']:.2f} vs torch", flush=True)
                results.append(dict(k=k, n=n, m=m, chain_us=r))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(dict(device=props.name, sol_gbps=sol, cold=cold, chain=True, results=results), f, indent=1)
        return
    for shape in (REAL if args.real else args.shapes.split(",")):
        if args.real:
            try:
                w = load_real(shape)
            except Exception as e:  # checkpoint not in the local HF cache
                print(f"# {shape}: skipped ({type(e).__name__})", flush=True)
                continue
            n, k = w.shape
            print(f"# {shape}: K={k} N={n}", flush=True)
        else:
            k, n = map(int, shape.split("x"))
            w = h.random_weight(n, k, seed=0, escapes=args.outliers)
        enc = h.encode(w, with_bf16t=not args.sweep)
        del w
        esc = enc.num_escapes / (n * k / 8)
        for m in map(int, args.ms.split(",")):
            # Sweeps lock per few configs: a point's hundreds of configs would starve other users.
            with contextlib.nullcontext() if args.sweep else gpu_lock(args.lock):
                io = 2 * m * (n + k)
                pt = Point(enc, m, cold, sources)
                heur = lib.heuristic(m, n, k)
                if args.vs_large:
                    t_small, t_large = pt.time("nf12", heur), pt.time("large", None)
                    print(f"K={k:6d} N={n:6d} M={m:3d}  small {t_small:8.1f} us  large {t_large:8.1f} us  "
                          f"large/small {t_large / t_small:5.2f}  {heur}", flush=True)
                    results.append(dict(k=k, n=n, m=m, small_us=t_small, large_us=t_large, cfg=str(heur)))
                elif args.sweep:
                    cfgs = sweep_configs(lib, m, n, k, args.model_slices)
                    timed = []
                    for i in range(0, len(cfgs), 8):  # the lock per 8 configs (a few seconds)
                        with gpu_lock(args.lock):
                            # Back to back under one lock, the clocks stay up after the first.
                            timed += [(pt.time("nf12", cfg, 200 if j == 0 else 20), str(cfg))
                                      for j, cfg in enumerate(cfgs[i:i + 8])]
                    timed.sort()
                    with gpu_lock(args.lock):
                        t_heur = pt.time("nf12", heur)
                    bw = (n * k * 1.5 + io) / timed[0][0] / 1e3
                    top = "  ".join(f"{c} {t:.1f}" for t, c in timed[1:4])
                    print(f"K={k:6d} N={n:6d} M={m:3d}  best {timed[0][0]:7.1f} us {bw / sol * 100:5.1f}% "
                          f"{timed[0][1]}  heur {t_heur:7.1f} {heur}  | {top}", flush=True)
                    results.append(dict(k=k, n=n, m=m, best_cfg=timed[0][1], best_us=timed[0][0], heur_cfg=str(heur),
                                        heur_us=t_heur, all=timed))
                else:
                    r = {src: pt.time(src, heur) for src in sources}
                    bw = {"nf12": (n * k * 1.5 + io) / r["nf12"] / 1e3, "bf16t": (n * k * 2 + io) / r["bf16t"] / 1e3,
                          "torch": (n * k * 2 + io) / r["torch"] / 1e3}
                    print(f"K={k:6d} N={n:6d} M={m:3d}  nf12 {r['nf12']:7.1f} us ({bw['nf12'] / sol * 100:5.1f}%)  "
                          f"bf16t {r['bf16t']:7.1f} us ({bw['bf16t'] / sol * 100:5.1f}%)  torch {r['torch']:7.1f} us "
                          f"({bw['torch'] / sol * 100:5.1f}%)  x{r['torch'] / r['nf12']:.2f} vs torch  "
                          f"x{r['bf16t'] / r['nf12']:.2f} vs bf16t  esc {esc * 100:.3f}%  {heur}", flush=True)
                    results.append(dict(shape=shape, k=k, n=n, m=m, cfg=str(heur), escape_rate=esc, us=r,
                                        pct_sol={s: bw[s] / sol * 100 for s in bw}))
            del pt
            if args.json:  # after every point, so an interrupted run keeps what it measured
                with open(args.json, "w") as f:
                    json.dump(dict(device=props.name, sol_gbps=sol, cold=cold, results=results), f, indent=1)
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
