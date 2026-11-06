# SPDX-License-Identifier: Apache-2.0
"""Benchmark NF12 MoE projections and full layers against BF16 baselines.

    uv run --group bench python bench/bench_moe.py --lock /tmp/nf12-thor.lock
    uv run --group bench python bench/bench_moe.py --layer --models qwen3-30b-a3b
    uv run --group bench python bench/bench_moe.py --table bench/results/moe_<machine>.json

Variants:
- nf12 / bf16t: NF12's packed and BF16-transposed kernels.
- triton / triton_default: standalone vLLM GEMM with searched / built-in tiles.
  Search results are cached per (model, projection, T) in
  bench/results/moe_triton_configs_<machine>.json. The first routing at each T
  supplies the search draws (uniform by default).
- cublas_oracle: one GEMM per active expert plus gather or scale/scatter.
  Row counts are baked into the graph. Device-only routing cannot realize
  this reference.
- grouped_mm: torch._grouped_mm, when graph-capturable. Torch 2.14 has native
  BF16 support on SM90/SM100 and a host-synchronizing fallback elsewhere.
- flashinfer: CUTLASS fused MoE, autotuned per T (layers only).

For A = T * topk, w13 [E, 2I, H] gathers x [T, H] into [A, 2I].
w2 [E, H, I] reads [A, I] and writes routing-weight-scaled [A, H].
Projection timing excludes alignment. --layer includes alignment, activation
and combine. Routing uses distinct experts per token, sampled uniformly or
with Zipf popularity (~1/rank). Hot routing always picks experts 0..topk-1.

Cold timing rotates routing draws sized by active-expert bytes, with full
weight copies capped by the memory budget. Each NF12 copy owns its escape
table. Results record both counts. --check-flush cross-checks NF12/Triton
rotation with event timing and a read-only L2 flush before every launch.

Real weights come from NF12_MOE_REAL_DIR exports or the local HF cache.
Missing weights fall back to normal(0, 0.02).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import torch  # noqa: E402
import triton_fused_moe as tfm  # noqa: E402
from common import import_nf12  # noqa: E402

PROJS = ("w13", "w2")
VARIANTS = ("nf12", "bf16t", "triton", "triton_default", "cublas_oracle", "grouped_mm", "flashinfer")
ROUTINGS = ("uniform", "zipf", "hot")
T_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
REAL_DIRS = [Path(p) for p in (os.environ.get("NF12_MOE_REAL_DIR"), "/workspace/hf/nf12-moe-real",
                               str(common.TENSOR_CACHE / "moe-real")) if p]


@dataclasses.dataclass(frozen=True)
class MoeModel:
    key: str
    name: str
    repo: str | None
    experts: int
    topk: int
    hidden: int
    inter: int  # per-expert intermediate size
    layout: str  # checkpoint naming: "qwen", "deepseek" or "mixtral"
    bench_layer: int  # MoE layer used for real weights in this benchmark
    # Model geometry for e2e_moe.py.
    layers: int
    moe_layers: tuple[int, ...]
    attn: tuple[tuple[str, int, int], ...]  # (name, K, N) dense attention projections
    dense_mlp: tuple[tuple[str, int, int], ...] = ()  # per MoE layer: shared experts (K, N)
    dense_layers: tuple[tuple[str, int, int], ...] = ()  # per non-MoE layer: its MLP
    vocab: int = 0

    def shape(self, proj: str) -> tuple[int, int]:
        """(N, K) of one expert of projection ``proj``."""
        return (2 * self.inter, self.hidden) if proj == "w13" else (self.hidden, self.inter)

    def expert_tensors(self, layer: int, expert: int | str, proj: str) -> tuple[str, ...]:
        """Checkpoint tensors of one expert's ``proj``, concatenated along N (gate before up)."""
        if self.layout == "mixtral":
            p = f"model.layers.{layer}.block_sparse_moe.experts.{expert}."
            return (p + "w1.weight", p + "w3.weight") if proj == "w13" else (p + "w2.weight",)
        p = f"model.layers.{layer}.mlp.experts.{expert}."
        return (p + "gate_proj.weight", p + "up_proj.weight") if proj == "w13" else (p + "down_proj.weight",)


def _qkv(heads: int, kv_heads: int, head_dim: int, hidden: int) -> tuple[tuple[str, int, int], ...]:
    return (("qkv", hidden, (heads + 2 * kv_heads) * head_dim), ("o", heads * head_dim, hidden))


MODELS = {
    "qwen3-30b-a3b": MoeModel("qwen3-30b-a3b", "Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B", 128, 8, 2048, 768, "qwen", 24,
                              48, tuple(range(48)), _qkv(32, 4, 128, 2048), vocab=151936),
    "qwen3-235b-a22b": MoeModel("qwen3-235b-a22b", "Qwen3-235B-A22B", None, 128, 8, 4096, 1536, "qwen", 47,
                                94, tuple(range(94)), _qkv(64, 4, 128, 4096), vocab=151936),
    # MLA without q LoRA: q_proj, kv_a_proj_with_mqa (512 + 64 rope), kv_b_proj, o_proj.
    "dsv2-lite": MoeModel("dsv2-lite", "DeepSeek-V2-Lite", "deepseek-ai/DeepSeek-V2-Lite", 64, 6, 2048, 1408,
                          "deepseek", 13, 27, tuple(range(1, 27)),
                          (("q", 2048, 16 * 192), ("kv_a", 2048, 576), ("kv_b", 512, 16 * 256), ("o", 2048, 2048)),
                          dense_mlp=(("shared_gate_up", 2048, 2 * 2816), ("shared_down", 2816, 2048)),
                          dense_layers=(("gate_up", 2048, 2 * 10944), ("down", 10944, 2048)), vocab=102400),
    "mixtral-8x7b": MoeModel("mixtral-8x7b", "Mixtral-8x7B", "mistralai/Mixtral-8x7B-v0.1", 8, 2, 4096, 14336,
                             "mixtral", 24, 32, tuple(range(32)), _qkv(32, 8, 128, 4096), vocab=32000),
}


def real_stack_path(model: MoeModel, layer: int, proj: str) -> Path | None:
    for d in REAL_DIRS:
        p = d / f"{model.key}.L{layer}.{proj}.safetensors"
        if p.is_file():
            return p
    return None


def load_real_stack(model: MoeModel, layer: int, proj: str, fetch: bool = False) -> torch.Tensor | None:
    """Load [E, N, K] CPU BF16 from exports or the local HF cache.
    With ``fetch``, range-download missing tensors into common.TENSOR_CACHE. Return None if unavailable."""
    from safetensors import safe_open

    p = real_stack_path(model, layer, proj)
    if p is not None:
        with safe_open(str(p), framework="pt") as f:
            return f.get_tensor("weight")
    if model.repo is None:
        return None
    experts = []
    for e in range(model.experts):
        w = common.load_hf_tensors(model.repo, model.expert_tensors(layer, e, proj), fetch=fetch)
        if w is None:
            return None
        experts.append(w)
    return torch.stack(experts)


def make_stack(model: MoeModel, proj: str, mode: str, seed: int = 0) -> tuple[torch.Tensor, str]:
    n, k = model.shape(proj)
    if mode == "real":
        w = load_real_stack(model, model.bench_layer, proj)
        if w is not None:
            if tuple(w.shape) != (model.experts, n, k):
                raise ValueError(f"{model.key} L{model.bench_layer} {proj}: shape {tuple(w.shape)}")
            return w.to("cuda", torch.bfloat16).contiguous(), f"{model.repo} layer {model.bench_layer}"
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(model.experts, n, k, generator=g, device="cuda", dtype=torch.float32).mul_(0.02)
    return w.to(torch.bfloat16), "random normal(0, 0.02)"


def make_routing(t: int, model: MoeModel, kind: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(topk_ids int32 [T, topk], topk_weights float32 [T, topk]) with distinct experts per token."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    e, k = model.experts, model.topk
    if kind == "hot":
        ids = torch.arange(k, device="cuda", dtype=torch.int32).expand(t, k).contiguous()
    else:
        # Gumbel top-k samples k distinct experts with probability ~ popularity.
        logp = torch.zeros(e, device="cuda")
        if kind == "zipf":
            rank = torch.randperm(e, generator=g, device="cuda")
            logp = -torch.log1p(rank.float())
        u = torch.rand(t, e, generator=g, device="cuda").clamp_(1e-20, 1.0)
        ids = (logp - torch.log(-torch.log(u))).topk(k, dim=1).indices.to(torch.int32)
    w = torch.softmax(torch.randn(t, k, generator=g, device="cuda"), dim=1)
    return ids, w


@dataclasses.dataclass
class Draw:
    """Routing and cached assignments for each implementation."""

    ids: torch.Tensor
    weights: torch.Tensor
    sorted: torch.Tensor  # assignment ids stably sorted by expert (torch reference of moe_align)
    offsets: torch.Tensor  # int64 [E + 1]
    counts: list[int]  # host copy, for cublas_oracle only
    triton: dict  # BLOCK_M -> tfm.Assignment
    nf12: tuple | None = None  # nf12.moe_align outputs


def make_draw(t: int, model: MoeModel, kind: str, seed: int) -> Draw:
    ids, w = make_routing(t, model, kind, seed)
    flat = ids.view(-1).long()
    order = torch.sort(flat, stable=True).indices.to(torch.int32)
    counts = torch.bincount(flat, minlength=model.experts)
    offsets = torch.zeros(model.experts + 1, dtype=torch.int64, device="cuda")
    offsets[1:] = counts.cumsum(0)
    return Draw(ids, w, order, offsets, counts.tolist(), {})


def triton_assignment(d: Draw, block_m: int, num_experts: int) -> tfm.Assignment:
    if block_m not in d.triton:
        d.triton[block_m] = tfm.prepare_assignment(d.ids, block_m, num_experts)
    return d.triton[block_m]


class Nf12Moe:
    def __init__(self, nf12) -> None:
        self.nf12 = nf12
        self.available = hasattr(nf12, "moe_linear") and self._kernels_work()

    def _kernels_work(self) -> bool:
        """Probe whether the loaded build supports the MoE kernels."""
        w = torch.randn(2, 16, 64, device="cuda").to(torch.bfloat16)
        ids = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
        try:
            sorted_, offsets = self.nf12.moe_align(ids, 2)
            self.nf12.moe_linear(torch.randn(2, 64, device="cuda").to(torch.bfloat16), self.nf12.encode_moe(w),
                                 offsets, gather=sorted_)
            return True
        except (RuntimeError, ImportError) as e:
            print(f"nf12 MoE kernels unavailable: {str(e).splitlines()[-1][:160]}", file=sys.stderr)
            return False

    def encode(self, w: torch.Tensor):
        return self.nf12.encode_moe(w)

    def pack_bf16t(self, w: torch.Tensor) -> torch.Tensor:
        return self.nf12.pack_bf16t_moe(w)

    def clone(self, nw):
        if isinstance(nw, torch.Tensor):
            return nw.clone()
        sd = {k: v.clone() for k, v in nw.state_dict("w.").items()}
        return self.nf12.NF12MoEWeight.from_state_dict(sd, "w.")

    def nbytes_per_expert(self, nw) -> list[int]:
        """Packed bytes + escape-table bytes of every expert."""
        per = nw.packed[0].numel()
        return [per + 16 * c for c in nw.escape_offsets.diff().tolist()]

    def align(self, ids: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.nf12.moe_align(ids, num_experts)

    def linear(self, x, w, sorted_, offsets, *, gather_div: int | None, scatter: bool, row_scale):
        """``w``: NF12MoEWeight or a bf16t stack (nf12.moe_linear dispatches on the type)."""
        return self.nf12.moe_linear(x, w, offsets, gather=sorted_ if gather_div else None,
                                    gather_div=gather_div or 1, scatter=sorted_ if scatter else None,
                                    row_scale=row_scale)

    def fused_moe(self, x, w13, w2, topk_weights, topk_ids):
        """``w13``/``w2``: NF12MoEWeight or bf16t stacks."""
        return self.nf12.fused_moe(x, w13, w2, topk_weights, topk_ids)


def flashinfer_moe_available() -> bool:
    try:
        from flashinfer.fused_moe import cutlass_fused_moe  # noqa: F401
    except ImportError:
        return False
    return True


def grouped_mm_capturable() -> bool:
    if not hasattr(torch, "_grouped_mm"):
        return False
    a = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 64, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([16, 32], device="cuda", dtype=torch.int32)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(s):
            torch._grouped_mm(a, b.transpose(-2, -1), offs=offs)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                torch._grouped_mm(a, b.transpose(-2, -1), offs=offs)
        return True
    except RuntimeError:
        return False
    finally:
        torch.cuda.synchronize()


@dataclasses.dataclass
class ProjCase:
    """Projection inputs and routing draws at one T."""

    model: MoeModel
    proj: str
    t: int
    kind: str  # routing
    x: torch.Tensor  # w13: [T, H]; w2: [A, I] (sorted order for nf12, assignment order for triton)
    draws: list[Draw]


def proj_fns(case: ProjCase, stacks: dict, nfm: Nf12Moe | None, cfg: dict | None, cfg_default: dict,
             variants: list[str]) -> dict[str, Callable[[int], object]]:
    m, proj, x = case.model, case.proj, case.x
    topk, e = m.topk, m.experts
    a = case.t * topk
    n, _ = m.shape(proj)
    w13 = proj == "w13"
    fns: dict[str, Callable[[int], object]] = {}

    def nf12_fn(fmt: str) -> Callable[[int], object]:
        def run(i: int):
            d = case.draws[i % len(case.draws)]
            sorted_, offsets = d.nf12
            return nfm.linear(x, stacks[fmt][i % len(stacks[fmt])], sorted_, offsets,
                              gather_div=topk if w13 else None,
                              scatter=not w13, row_scale=None if w13 else d.weights.view(-1))
        return run

    def triton_fn(c: dict) -> Callable[[int], object]:
        def run(i: int):
            d = case.draws[i % len(case.draws)]
            asg = triton_assignment(d, c["BLOCK_SIZE_M"], e)
            out = torch.empty(a, n, dtype=torch.bfloat16, device="cuda")
            tfm.invoke(x, stacks["bf16"][i % len(stacks["bf16"])], out, None if w13 else d.weights.view(-1), asg,
                       topk if w13 else 1, a, c)
            return out
        return run

    def bf16_fn(grouped: bool) -> Callable[[int], object]:
        def run(i: int):
            d = case.draws[i % len(case.draws)]
            w = stacks["bf16"][i % len(stacks["bf16"])]
            xs = x.index_select(0, d.sorted // topk) if w13 else x
            if grouped:
                y = torch._grouped_mm(xs, w.transpose(-2, -1), offs=d.offsets[1:].to(torch.int32))
            else:
                y = torch.empty(a, n, dtype=torch.bfloat16, device="cuda")
                s = 0
                for ex, c in enumerate(d.counts):
                    if c:
                        torch.mm(xs[s:s + c], w[ex].t(), out=y[s:s + c])
                        s += c
            if w13:
                return y
            out = torch.empty_like(y)
            order = d.sorted.long()
            out.index_copy_(0, order, (y.float() * d.weights.view(-1)[order, None]).to(y.dtype))
            return out
        return run

    for v in variants:
        if v in ("nf12", "bf16t"):
            fns[v] = nf12_fn(v)
        elif v == "triton" and cfg is not None:
            fns[v] = triton_fn(cfg)
        elif v == "triton_default":
            fns[v] = triton_fn(cfg_default)
        elif v in ("cublas_oracle", "grouped_mm"):
            fns[v] = bf16_fn(v == "grouped_mm")
    return fns


class TritonTuner:
    """Best tfm config per (model, proj, T) on uniform routing, cached on disk."""

    def __init__(self, path: Path, args: argparse.Namespace) -> None:
        self.path = path
        self.args = args
        self.cache = json.loads(path.read_text()) if path.exists() else {}

    def best(self, case: ProjCase, stacks: dict) -> dict:
        key = f"{case.model.key}/{case.proj}/{case.t}"
        if key in self.cache:
            return self.cache[key]["config"]
        m = case.model
        a = case.t * m.topk

        def screen(cands: list[dict], budget: float) -> list[tuple[float, dict]]:
            # Short bursts leave Tegra's memory and GPU clocks low.
            res = []
            for c in cands:
                fn = proj_fns(case, stacks, None, c, c, ["triton"])["triton"]
                try:
                    with common.gpu_lock(self.args.lock):
                        us = common.time_graph(fn, len(case.draws), min_launches=20,
                                               budget_s=budget)["median_us"]
                except Exception as e:  # noqa: BLE001 - out of shared memory / registers: not a candidate
                    print(f"    skip {tfm.config_key(c)}: {type(e).__name__}", file=sys.stderr)
                    continue
                res.append((us, c))
            return sorted(res, key=lambda r: r[0])

        # Search tiles, then warps/stages. Keep the default in the final round.
        default = tfm.get_default_config(case.t, m.experts)
        ranked = screen(tfm.tuning_tiles(a // m.experts) + [default], 0.15)
        stage2 = [{**c, "num_warps": w, "num_stages": s} for _, c in ranked[:4] for w in (4, 8) for s in (2, 3, 4, 5)]
        ranked = screen(list({tfm.config_key(c): c for c in stage2 + [r[1] for r in ranked[:4]]}.values()), 0.15)
        finalists = {tfm.config_key(c): c for c in [c for _, c in ranked[:3]] + [default]}
        best_us, best = screen(list(finalists.values()), 0.6)[0]
        print(f"    triton tuned {key}: {tfm.config_key(best)} {best_us:.2f} us (A={a})", file=sys.stderr)
        self.cache[key] = {"config": best, "us": best_us}
        common.write_json(self.path, self.cache)
        return best


def stream_gbps(tag: str) -> float | None:
    p = common.RESULTS_DIR / f"sol_{tag}.json"
    if not p.exists():
        return None
    return max(r["gbps"] for r in json.loads(p.read_text())["probe"]["read_sweep"])


def mem_budget_bytes(args: argparse.Namespace) -> int:
    free, _ = torch.cuda.mem_get_info()
    return int(min(args.mem_budget_gb * 2**30, 0.6 * free))


def active_experts(d: Draw) -> list[int]:
    return [e for e, c in enumerate(d.counts) if c]


def rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    return ((y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()


def reference(case: ProjCase, w: torch.Tensor, d: Draw, assignment_input: bool = False) -> torch.Tensor:
    """FP32 projection reference. w13 writes sorted rows, with row p reading
    x[sorted[p] // topk]. For w2, input row p feeds output sorted[p]
    (SPEC 9.2 scatter) in assignment order. With ``assignment_input``, input
    row a feeds output a instead, following vLLM's w2 convention."""
    topk = case.model.topk
    s = d.sorted.long()
    if case.proj == "w13":
        xs, experts = case.x[s // topk].float(), d.ids.view(-1).long()[s]
    elif assignment_input:
        xs, experts = case.x.float(), d.ids.view(-1).long()
    else:
        xs, experts = case.x.float(), d.ids.view(-1).long()[s]
    y = torch.empty(xs.shape[0], w.shape[1], device="cuda")
    for ex in active_experts(d):
        rows = (experts == ex).nonzero().view(-1)
        y[rows] = xs[rows] @ w[ex].float().t()
    if case.proj == "w13":
        return y
    if assignment_input:
        return y * d.weights.view(-1, 1)
    out = torch.empty_like(y)
    out[s] = y * d.weights.view(-1)[s, None]
    return out


def bench_proj(nfm: Nf12Moe | None, model: MoeModel, proj: str, ts: list[int], args: argparse.Namespace,
               tuner: TritonTuner, sol: float | None) -> dict:
    n, k = model.shape(proj)
    e = model.experts
    with common.gpu_lock(args.lock):
        w, origin = make_stack(model, proj, args.weights)
        nw = nfm.encode(w) if nfm else None
        bt = nfm.pack_bf16t(w) if nfm and "bf16t" in args.variants else None
    bytes_bf16 = 2 * n * k
    per_nf12 = nfm.nbytes_per_expert(nw) if nfm else None
    entry = {"model": model.key, "name": model.name, "proj": proj, "n": n, "k": k, "experts": e,
             "topk": model.topk, "weight_source": origin, "rows": []}
    if nw is not None:
        off = nw.escape_offsets.tolist()
        rates = [(off[i + 1] - off[i]) / (n * k // 8) for i in range(e)]
        entry.update(exp_base=nw.exp_base.tolist(), escape_rate_mean=sum(rates) / e, escape_rate_max=max(rates))
        print(f"{model.key} {proj} ({origin}): escapes mean {100 * entry['escape_rate_mean']:.3f}% "
              f"max {100 * entry['escape_rate_max']:.3f}%", file=sys.stderr)

    # Size rotation by touched bytes. Cap full weight copies by memory.
    def rotation(t: int) -> int:
        touched = len(active_experts(make_draw(t, model, "uniform", 0))) * bytes_bf16
        return common.cold_copies(touched) if args.l2 == "cold" else 1

    fmts = ["bf16"] + (["nf12"] if nw is not None and "nf12" in args.variants else []) + (
        ["bf16t"] if bt is not None else [])
    stack_bytes = {"bf16": w.nbytes, "nf12": sum(per_nf12) if per_nf12 else 0,
                   "bf16t": bt.nbytes if bt is not None else 0}
    per_copy = sum(stack_bytes[f] for f in fmts)
    want = rotation(min(ts))
    copies = max(1, min(want, mem_budget_bytes(args) // per_copy + 1))
    entry.update(copies=copies, copies_wanted=want)
    base = {"bf16": w, "nf12": nw, "bf16t": bt}
    with common.gpu_lock(args.lock):
        stacks = {f: [base[f]] + [nfm.clone(base[f]) if f == "nf12" else base[f].clone() for _ in range(copies - 1)]
                  for f in fmts}
    variants = [v for v in args.variants if (v not in ("nf12", "bf16t") or nfm) and
                (v != "grouped_mm" or args.grouped_mm_ok)]

    for t in ts:
        a = t * model.topk
        for kind in args.routing:
            rot = rotation(t)
            draws = [make_draw(t, model, kind, 1000 * t + i + (0 if kind == "uniform" else 17)) for i in range(rot)]
            if nfm:
                for d in draws:
                    d.nf12 = nfm.align(d.ids, e)
            x = torch.randn(t if proj == "w13" else a, k, device="cuda", dtype=torch.bfloat16)
            case = ProjCase(model, proj, t, kind, x, draws)
            # Tune on the first routing at this T, then reuse.
            cfg = tuner.best(case, stacks) if "triton" in variants else None
            fns = proj_fns(case, stacks, nfm, cfg, tfm.get_default_config(t, e), variants)
            with common.gpu_lock(args.lock):
                row = measure_point(case, fns, w, per_nf12, bytes_bf16, sol, args)
            row.update(t=t, routing=kind, draws=rot, triton_config=tfm.config_key(cfg) if cfg else None)
            entry["rows"].append(row)
    del stacks, base, w, nw, bt
    torch.cuda.empty_cache()
    return entry


def measure_point(case: ProjCase, fns: dict, w: torch.Tensor, per_nf12: list[int] | None,
                  bytes_bf16: int, sol: float | None, args: argparse.Namespace) -> dict:
    m = case.model
    n, k = m.shape(case.proj)
    a = case.t * m.topk
    d0 = case.draws[0]
    act = [len(active_experts(d)) for d in case.draws]
    row: dict = {"active_experts_mean": sum(act) / len(act),
                 "max_rows_per_expert": max(max(d.counts) for d in case.draws)}
    ref = reference(case, w, d0)
    ref_assign = reference(case, w, d0, assignment_input=True) if case.proj == "w2" else None
    perm = d0.sorted.long()
    for v, fn in fns.items():
        y = fn(0)
        if v.startswith("triton"):
            if case.proj == "w13":
                y = y[perm]  # triton writes w13 rows in assignment order
            row[f"rel_err_{v}"] = rel_err(y, ref if ref_assign is None else ref_assign)
        else:
            row[f"rel_err_{v}"] = rel_err(y, ref)
    del ref, ref_assign
    if "nf12" in fns and "bf16t" in fns:
        row["matches_bf16t"] = torch.equal(fns["nf12"](0).view(torch.int16), fns["bf16t"](0).view(torch.int16))

    def wbytes(v: str, d: Draw) -> int:
        ex = active_experts(d)
        return sum(per_nf12[e] for e in ex) if v == "nf12" else len(ex) * bytes_bf16

    flops = 2 * a * n * k
    act_bytes = 2 * (case.t * k if case.proj == "w13" else a * k) + 2 * a * n
    res: dict = {}
    # Alternate variant order to distribute power-cap drift.
    for r in range(args.rounds):
        order = list(fns) if r % 2 == 0 else list(reversed(fns))
        for v in order:
            t = common.time_fn(fns[v], len(case.draws), timing=args.timing, cold=args.l2 == "cold",
                               min_launches=args.min_launches, budget_s=args.budget_s)
            if v not in res or t["median_us"] < res[v]["median_us"]:
                res[v] = t
    for v, t in res.items():
        us = t["median_us"]
        wb = sum(wbytes(v, d) for d in case.draws) / len(case.draws)
        t["weight_bytes"] = wb
        t["gbps"] = (wb + act_bytes) / us / 1e3
        t["tflops"] = flops / us / 1e6
        if sol:
            t["pct_sol"] = 100 * (wb + act_bytes) / sol / 1e3 / us
    if args.check_flush:
        for v in ("nf12", "triton"):
            if v in fns:
                res[v]["flush_events_us"] = common.time_events(fns[v], cold=True, iters=50)["median_us"]
    row["timing"] = res
    if "nf12" in res:
        for v in res:
            if v != "nf12":
                res["nf12"][f"speedup_vs_{v}"] = res[v]["median_us"] / res["nf12"]["median_us"]
    print(f"  {m.key} {case.proj} T={case.t:<5} {case.kind:7} " + "  ".join(
        f"{v}={res[v]['median_us']:9.2f}us" for v in res) + f"  active={row['active_experts_mean']:.1f}",
        file=sys.stderr)
    return row


def bench_layer(nfm: Nf12Moe | None, model: MoeModel, ts: list[int], args: argparse.Namespace,
                tuner: TritonTuner) -> dict:
    """Time alignment, w13, SiLU gating, weighted w2 and combine."""
    e = model.experts
    use_fi = "flashinfer" in args.variants and flashinfer_moe_available()
    if use_fi:
        from flashinfer.autotuner import autotune as fi_autotune
        from flashinfer.fused_moe import cutlass_fused_moe as fi_moe
    with common.gpu_lock(args.lock):
        w13, o13 = make_stack(model, "w13", args.weights)
        w2, o2 = make_stack(model, "w2", args.weights, seed=1)
        nw = (nfm.encode(w13), nfm.encode(w2)) if nfm else None
        bt = (nfm.pack_bf16t(w13), nfm.pack_bf16t(w2)) if nfm else None
    entry = {"model": model.key, "name": model.name, "weight_source": [o13, o2], "rows": []}
    expert_bytes = w13[0].nbytes + w2[0].nbytes

    def rotation(t: int) -> int:
        touched = len(active_experts(make_draw(t, model, "uniform", 0))) * expert_bytes
        return common.cold_copies(touched) if args.l2 == "cold" else 1

    want = rotation(min(ts))
    per_copy = (2 * (w13.nbytes + w2.nbytes) + (w13.nbytes if use_fi else 0) +
                (sum(t.nbytes for t in nw) if nw else 0))
    copies = max(1, min(want, mem_budget_bytes(args) // per_copy + 1))
    entry.update(copies=copies, copies_wanted=want)
    with common.gpu_lock(args.lock):
        bf = [(w13, w2)] + [(w13.clone(), w2.clone()) for _ in range(copies - 1)]
        nfs = [nw] + [tuple(nfm.clone(t) for t in nw) for _ in range(copies - 1)] if nw else []
        bts = [bt] + [tuple(t.clone() for t in bt) for _ in range(copies - 1)] if bt else []
        # FlashInfer's fc1 stores up rows before gate rows.
        fis = [torch.cat([a[:, a.shape[1] // 2:], a[:, :a.shape[1] // 2]], dim=1) for a, _ in bf] if use_fi else []

    for t in ts:
        for kind in args.routing:
            rot = rotation(t)
            with common.gpu_lock(args.lock):
                draws = [make_draw(t, model, kind, 7000 * t + i) for i in range(rot)]
                x = torch.randn(t, model.hidden, device="cuda", dtype=torch.bfloat16)
                key13, key2 = f"{model.key}/w13/{t}", f"{model.key}/w2/{t}"
                default = tfm.get_default_config(t, e)
                c13 = tuner.cache.get(key13, {}).get("config", default)
                c2 = tuner.cache.get(key2, {}).get("config", default)
                fns = {}
                if nfm:
                    fns["nf12"] = lambda i: nfm.fused_moe(x, *nfs[i % copies], draws[i].weights, draws[i].ids)
                    fns["bf16t"] = lambda i: nfm.fused_moe(x, *bts[i % copies], draws[i].weights, draws[i].ids)
                fns["triton"] = lambda i: tfm.fused_experts(x, *bf[i % copies], draws[i].weights, draws[i].ids,
                                                            c13, c2)
                fns["triton_default"] = lambda i: tfm.fused_experts(x, *bf[i % copies], draws[i].weights,
                                                                    draws[i].ids, default, default)
                if use_fi:
                    out = torch.empty_like(x)
                    def fi_run(i: int) -> torch.Tensor:
                        fi_moe(x, draws[i].ids, draws[i].weights, fis[i % copies], bf[i % copies][1], torch.bfloat16,
                               [], output=out)
                        return out

                    fns["flashinfer"] = fi_run
                    with fi_autotune():
                        fns["flashinfer"](0)
                ref = fns["triton"](0).float()
                row = {"t": t, "routing": kind, "draws": rot,
                       "active_experts_mean": sum(len(active_experts(d)) for d in draws) / rot,
                       "rel_err": {v: rel_err(f(0), ref) for v, f in fns.items()}}
                if nfm:
                    row["matches_bf16t"] = torch.equal(fns["nf12"](0).view(torch.int16),
                                                      fns["bf16t"](0).view(torch.int16))
                res = {}
                timed = [v for v in fns if v in args.variants]  # the others only serve the checks above
                for r in range(args.rounds):
                    for v in (timed if r % 2 == 0 else list(reversed(timed))):
                        tm = common.time_fn(fns[v], rot, timing=args.timing, cold=args.l2 == "cold",
                                            min_launches=args.min_launches, budget_s=args.budget_s)
                        if v not in res or tm["median_us"] < res[v]["median_us"]:
                            res[v] = tm
                row["timing"] = res
            entry["rows"].append(row)
            print(f"  {model.key} layer T={t:<5} {kind:7} " + "  ".join(
                f"{v}={res[v]['median_us']:9.2f}us" for v in res), file=sys.stderr)
    del bf, nfs, bts, fis
    torch.cuda.empty_cache()
    return entry


def run(args: argparse.Namespace) -> None:
    nf12 = import_nf12()
    nfm = Nf12Moe(nf12)
    if not nfm.available:
        print("nf12 has no MoE API in this build: timing baselines only", file=sys.stderr)
        nfm = None
    tag = common.machine_tag()
    args.grouped_mm_ok = "grouped_mm" in args.variants and grouped_mm_capturable()
    if "grouped_mm" in args.variants and not args.grouped_mm_ok:
        print("torch._grouped_mm is not graph-capturable here (host-synchronizing fallback): skipped",
              file=sys.stderr)
    sol = stream_gbps(tag)
    tuner = TritonTuner(common.RESULTS_DIR / f"moe_triton_configs_{tag}.json", args)
    result = {
        "machine": tag, "info": common.machine_info(), "stream_gbps": sol,
        "config": {"timing": args.timing, "l2": args.l2, "weights": args.weights, "variants": args.variants,
                   "routing": args.routing, "rounds": args.rounds, "layer": args.layer,
                   "grouped_mm_capturable": args.grouped_mm_ok, "mem_budget_gb": args.mem_budget_gb},
        "gpu_state_before": common.gpu_state(), "entries": [],
    }
    suffix = "_layer" if args.layer else ""
    out = Path(args.out) if args.out else common.RESULTS_DIR / f"moe{suffix}_{tag}.json"
    t0 = time.time()
    for key in args.models:
        model = MODELS[key]
        if args.real_layer is not None:
            model = dataclasses.replace(model, bench_layer=args.real_layer,
                                        name=f"{model.name} L{args.real_layer}")
        if args.layer:
            result["entries"].append(bench_layer(nfm, model, args.t, args, tuner))
        else:
            for proj in args.proj:
                result["entries"].append(bench_proj(nfm, model, proj, args.t, args, tuner, sol))
                common.write_json(out, result)
        common.write_json(out, result)
    result["gpu_state_after"] = common.gpu_state()
    result["throttle_after"] = common.throttled(result["gpu_state_after"])
    result["wall_s"] = time.time() - t0
    common.write_report(out, result, table)
    print(f"wrote {out} and {out.with_suffix('.md')}", file=sys.stderr)


def _cell(t: dict | None, key: str = "median_us", fmt: str = ".1f") -> str:
    return format(t[key], fmt) if t and key in t else ""


def table(path: Path) -> str:
    r = json.loads(path.read_text())
    info, cfg = r["info"], r["config"]
    lines = [
        f"## MoE {'layer' if cfg.get('layer') else 'projections'}: {info['gpu']} (SM{info['sm']}), "
        f"timing `{cfg['timing']}`, L2 `{cfg['l2']}`, weights `{cfg['weights']}`",
        "",
        f"GPU before: {common.state_summary(r.get('gpu_state_before'))}. "
        f"After: {common.state_summary(r.get('gpu_state_after'))}. "
        f"Streaming-read SOL: {r['stream_gbps'] or 'n/a'} GB/s. "
        f"torch._grouped_mm graph-capturable: {cfg['grouped_mm_capturable']}.",
        "",
    ]
    for ent in r["entries"]:
        variants = [v for v in VARIANTS if any(v in row["timing"] for row in ent["rows"])]
        title = f"{ent['name']} {ent.get('proj', 'layer')}"
        if "proj" in ent:
            title += f" [E={ent['experts']}, N={ent['n']}, K={ent['k']}], topk {ent['topk']}"
        esc = (f". Escapes mean {100 * ent['escape_rate_mean']:.3f}%, max {100 * ent['escape_rate_max']:.3f}% "
               "of groups per expert" if "escape_rate_mean" in ent else "")
        lines += [f"### {title}", "",
                  f"Weights: {ent['weight_source']}{esc}. Weight-stack copies: {ent['copies']}"
                  + (f" ({ent['copies_wanted']} wanted at the smallest T, memory budget)"
                     if ent["copies"] < ent["copies_wanted"] else "") + ".", ""]
        has_nf12 = "nf12" in variants
        cols = ["T", "routing", "active experts", "draws"] + [f"{v} us" for v in variants]
        if "proj" in ent:
            cols += ["best BF16 GB/s"] + (["NF12 GB/s", "NF12 % SOL", "NF12 vs triton", "NF12 vs bf16t", "matches bf16t"]
                                          if has_nf12 else ["triton config"])
        elif has_nf12:
            cols += ["NF12 vs triton", "NF12 vs bf16t", "matches bf16t"]
        lines += ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
        for row in ent["rows"]:
            tm = row["timing"]
            cells = [f"{row['t']}", row["routing"], f"{row['active_experts_mean']:.1f}", str(row.get("draws", ""))]
            cells += [_cell(tm.get(v)) for v in variants]
            if "proj" in ent:
                bf = [tm[v] for v in ("triton", "triton_default") if v in tm]
                best = min(bf, key=lambda t: t["median_us"]) if bf else None
                cells.append(_cell(best, "gbps", ".0f"))
                if has_nf12:
                    nf = tm.get("nf12", {})
                    cells += [_cell(nf, "gbps", ".0f"), _cell(nf, "pct_sol", ".0f"),
                              _cell(nf, "speedup_vs_triton", ".2f"), _cell(nf, "speedup_vs_bf16t", ".3f"),
                              str(row.get("matches_bf16t", ""))]
                else:
                    cells.append(f"`{row['triton_config']}`" if row.get("triton_config") else "")
            elif has_nf12:
                nf = tm["nf12"]["median_us"]
                cells += [f"{tm['triton']['median_us'] / nf:.2f}" if "triton" in tm else "",
                          f"{tm['bf16t']['median_us'] / nf:.3f}" if "bf16t" in tm else "",
                          str(row.get("matches_bf16t", ""))]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def summary(paths: list[Path], routing: str) -> str:
    """NF12 latency and speedup over the fastest Triton/FlashInfer baseline."""
    entries = [e for p in paths for e in json.loads(p.read_text())["entries"]]
    ts = sorted({row["t"] for e in entries for row in e["rows"] if row["routing"] == routing})
    lines = ["| model | proj | " + " | ".join(f"T={t}" for t in ts) + " |", "|---" * (len(ts) + 2) + "|"]
    for key in dict.fromkeys((e["name"], e.get("proj", "layer")) for e in entries):
        by = {row["t"]: row["timing"] for e in entries if (e["name"], e.get("proj", "layer")) == key
              for row in e["rows"] if row["routing"] == routing}
        cells = []
        for t in ts:
            tm = by.get(t)
            bf = [tm[v]["median_us"] for v in ("triton", "triton_default", "flashinfer") if tm and v in tm]
            if not bf:
                cells.append("")
            elif "nf12" in tm:
                cells.append(f"{tm['nf12']['median_us']:.1f} ({min(bf) / tm['nf12']['median_us']:.2f}x)")
            else:
                cells.append(f"{min(bf):.1f}")
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, help="print Markdown tables from results JSON")
    ap.add_argument("--summary", type=Path, nargs="+",
                    help="summarize results JSONs using the first --routing value")
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--proj", nargs="*", default=list(PROJS), choices=PROJS)
    ap.add_argument("--t", nargs="*", type=int, default=T_LIST, help="tokens per batch")
    ap.add_argument("--routing", nargs="*", default=["uniform", "zipf"], choices=ROUTINGS)
    ap.add_argument("--layer", action="store_true", help="time full layers")
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--real-layer", type=int,
                    help="override the real-weight layer and label results with its index")
    ap.add_argument("--timing", choices=["graph", "events"], default="graph")
    ap.add_argument("--l2", choices=["cold", "hot"], default="cold")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--rounds", type=int, default=2, help="rounds with alternating variant order")
    ap.add_argument("--check-flush", action="store_true",
                    help="cross-check NF12/Triton rotation with flushed event timing")
    ap.add_argument("--min-launches", type=int, default=100)
    ap.add_argument("--budget-s", type=float, default=1.0, help="target seconds per measurement")
    ap.add_argument("--mem-budget-gb", type=float, default=40.0, help="cap on rotation copies' memory")
    ap.add_argument("--lock", help="measurement flock path, e.g. /tmp/nf12-thor.lock")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.table:
        print(table(args.table))
    elif args.summary:
        print(summary(args.summary, args.routing[0]))
    else:
        run(args)


if __name__ == "__main__":
    main()
