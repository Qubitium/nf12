# SPDX-License-Identifier: Apache-2.0
"""Estimate model decode time from linear and MoE work, NF12 vs BF16.

Times attention projections (including DeepSeek MLA), a BF16 router GEMM,
shared experts and routed experts with uniform random routing. NF12 and
bf16t use ``nf12.fused_moe``. BF16 uses standalone Triton with bench_moe's
cached configs or vLLM defaults.

CUDA graphs rotate routing draws and distinct weight copies, including escape
tables. ``--mem-budget-gb`` limits copies. Layer times are multiplied by their
model counts, with non-MoE layers included and lm_head reported separately.
This is a partial estimate, not full serving. It excludes attention math,
norms, routing softmax/top-k, embedding, sampling and communication.

    uv run --group bench python bench/e2e_moe.py --lock /tmp/nf12-thor.lock
    uv run --group bench python bench/e2e_moe.py --table bench/results/e2e_moe_<machine>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import torch  # noqa: E402
import triton_fused_moe as tfm  # noqa: E402
from common import import_nf12  # noqa: E402
from bench_moe import MODELS, MoeModel, Nf12Moe, make_routing, make_stack  # noqa: E402

FORMATS = ("nf12", "bf16t", "bf16")


def dense_tensors(layer: int, name: str) -> tuple[str, ...]:
    """Checkpoint tensors of one dense linear, concatenated along N."""
    p = f"model.layers.{layer}."
    a = p + "self_attn."
    table = {
        "qkv": (a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"),
        "q": (a + "q_proj.weight",),
        "kv_a": (a + "kv_a_proj_with_mqa.weight",),
        "kv_b": (a + "kv_b_proj.weight",),
        "o": (a + "o_proj.weight",),
        "shared_gate_up": (p + "mlp.shared_experts.gate_proj.weight", p + "mlp.shared_experts.up_proj.weight"),
        "shared_down": (p + "mlp.shared_experts.down_proj.weight",),
        "gate_up": (p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"),
        "down": (p + "mlp.down_proj.weight",),
    }
    return table[name]


def load_dense(model: MoeModel, layer: int, name: str, k: int, n: int, args: argparse.Namespace) -> tuple:
    names = ("lm_head.weight",) if name == "lm_head" else dense_tensors(layer, name)
    shape = common.Shape(k, n, (model.repo, names) if model.repo else None, name)
    return common.make_weight(shape, args.weights, torch.device("cuda"))


class Formats:
    """Dense and MoE calls per weight format."""

    def __init__(self, nf12, nfm: Nf12Moe | None) -> None:
        self.nf12, self.nfm = nf12, nfm

    def dense(self, f: str, w: torch.Tensor):
        if f == "nf12":
            return self.nf12.encode(w)
        return self.nf12.pack_bf16t(w) if f == "bf16t" else w

    def moe(self, f: str, w: torch.Tensor):
        if f == "nf12":
            return self.nfm.encode(w)
        return self.nfm.pack_bf16t(w) if f == "bf16t" else w

    def clone(self, f: str, t):
        if f == "nf12" and not isinstance(t, torch.Tensor):
            if hasattr(t, "escape_offsets"):
                return self.nfm.clone(t)
            return self.nf12.NF12Weight(t.packed.clone(), t.escapes.clone(), t.exp_base)
        return t.clone()

    def linear(self, f: str, x: torch.Tensor, t):
        if f == "nf12":
            return self.nf12.linear(x, t)
        return self.nf12.linear_bf16t(x, t) if f == "bf16t" else torch.matmul(x, t.t())


def triton_configs(model: MoeModel, b: int) -> tuple[dict, dict]:
    path = common.RESULTS_DIR / f"moe_triton_configs_{common.machine_tag()}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    default = tfm.get_default_config(b, model.experts)
    return tuple(cache.get(f"{model.key}/{p}/{b}", {}).get("config", default) for p in ("w13", "w2"))


def bench_model(fm: Formats, model: MoeModel, batches: list[int], formats: list[str],
                args: argparse.Namespace) -> dict:
    layer = model.bench_layer
    dev = torch.device("cuda")
    dense_specs = list(model.attn) + list(model.dense_mlp)
    with common.gpu_lock(args.lock):
        dense, origins = [], []
        for name, k, n in dense_specs:
            w, o = load_dense(model, layer, name, k, n, args)
            dense.append(w)
            origins.append(f"{name}: {o}")
        router = torch.randn(model.experts, model.hidden, device=dev, dtype=torch.bfloat16)
        w13, o13 = make_stack(model, "w13", args.weights)
        w2, o2 = make_stack(model, "w2", args.weights, seed=1)
        origins += [f"experts w13: {o13}", f"experts w2: {o2}"]
        dense_layer = []
        if model.dense_layers:
            dense_layer = [load_dense(model, 0, name, k, n, args)[0]
                           for name, k, n in (*model.attn, *model.dense_layers)]
        head, ho = load_dense(model, layer, "lm_head", model.hidden, model.vocab, args)
        origins.append(f"lm_head: {ho}")
    out = {"model": model.name, "key": model.key, "weights": origins, "moe_layers": len(model.moe_layers),
           "dense_layers": model.layers - len(model.moe_layers), "batches": []}

    for b in batches:
        row: dict = {"batch": b}
        c13, c2 = triton_configs(model, b)
        xs = [torch.randn(b, k, device=dev, dtype=torch.bfloat16) for _, k, _ in dense_specs]
        xh = torch.randn(b, model.hidden, device=dev, dtype=torch.bfloat16)
        for f in formats:
            with common.gpu_lock(args.lock):
                d0 = [fm.dense(f, w) for w in dense]
                m0 = [fm.moe(f, w13), fm.moe(f, w2)]
                # Routing draws rotate through dense plus expected active-expert bytes.
                # Memory limits the number of full layer copies.
                dense_bytes = sum(t.nbytes for t in d0)
                moe_bytes = sum(t.nbytes for t in m0)
                active = model.experts * (1 - (1 - model.topk / model.experts) ** b)
                rotation = common.cold_copies(int(dense_bytes + active * (moe_bytes / model.experts)))
                per_copy = dense_bytes + moe_bytes
                free, _ = torch.cuda.mem_get_info()
                copies = max(1, min(rotation, int(min(args.mem_budget_gb * 2**30, 0.6 * free)) // per_copy + 1))
                layers = [(d0, m0)] + [([fm.clone(f, t) for t in d0], [fm.clone(f, t) for t in m0])
                                       for _ in range(copies - 1)]
                routes = [make_routing(b, model, "uniform", 31 * b + i) for i in range(rotation)]

                def layer_fn(i: int) -> None:
                    dws, (e13, e2) = layers[i % copies]
                    for x, t in zip(xs, dws):
                        fm.linear(f, x, t)
                    torch.matmul(xh, router.t())
                    ids, wts = routes[i]
                    if f == "bf16":
                        tfm.fused_experts(xh, e13, e2, wts, ids, c13, c2)
                    else:
                        fm.nfm.fused_moe(xh, e13, e2, wts, ids)

                t_layer = common.time_graph(layer_fn, rotation, min_launches=args.min_launches,
                                            budget_s=args.budget_s)
                del layers, routes, d0, m0
                res = {"layer_us": t_layer["median_us"], "layer_p10_us": t_layer["p10_us"],
                       "layer_p90_us": t_layer["p90_us"], "copies": copies, "copies_wanted": rotation}
                if dense_layer:
                    dfmt = [fm.dense(f, w) for w in dense_layer]
                    dxs = [torch.randn(b, w.shape[1], device=dev, dtype=torch.bfloat16) for w in dense_layer]
                    dcopies = [dfmt] + [[fm.clone(f, t) for t in dfmt]
                                        for _ in range(common.cold_copies(sum(t.nbytes for t in dfmt)) - 1)]

                    def dense_fn(i: int) -> None:
                        for x, t in zip(dxs, dcopies[i]):
                            fm.linear(f, x, t)

                    t_dense = common.time_graph(dense_fn, len(dcopies), min_launches=args.min_launches,
                                                budget_s=args.budget_s)
                    res["dense_layer_us"] = t_dense["median_us"]
                    del dfmt, dcopies
                hf = fm.dense(f, head)
                heads = [hf] + [fm.clone(f, hf) for _ in range(common.cold_copies(hf.nbytes) - 1)]
                res["lm_head_us"] = common.time_graph(lambda i: fm.linear(f, xh, heads[i]), len(heads),
                                                      min_launches=args.min_launches,
                                                      budget_s=args.budget_s)["median_us"]
                del heads, hf
                torch.cuda.empty_cache()
            res["stack_ms"] = (out["moe_layers"] * res["layer_us"]
                               + out["dense_layers"] * res.get("dense_layer_us", 0.0)) / 1e3
            res["stack_plus_head_ms"] = res["stack_ms"] + res["lm_head_us"] / 1e3
            row[f] = res
        for f in formats:
            if f != "nf12" and "nf12" in row:
                row[f"nf12_speedup_vs_{f}"] = row[f]["stack_ms"] / row["nf12"]["stack_ms"]
                row[f"nf12_speedup_vs_{f}_with_head"] = (row[f]["stack_plus_head_ms"]
                                                         / row["nf12"]["stack_plus_head_ms"])
        out["batches"].append(row)
        print(f"{model.name} batch={b}: " + "  ".join(
            f"{f}={row[f]['stack_ms']:.3f}ms (layer {row[f]['layer_us']:.1f}us)" for f in formats), file=sys.stderr)
    del dense, w13, w2, head
    torch.cuda.empty_cache()
    return out


def table(path: Path) -> str:
    r = json.loads(path.read_text())
    fmts = [f for f in FORMATS if any(f in b for m in r["models"] for b in m["batches"])]
    names = {"nf12": "NF12", "bf16t": "bf16t twin", "bf16": "BF16 (cuBLAS + Triton MoE)"}
    lines = [
        f"## MoE model decode estimate: {r['info']['gpu']} (SM{r['info']['sm']})",
        "",
        f"GPU before: {common.state_summary(r.get('gpu_state_before'))}. "
        f"After: {common.state_summary(r.get('gpu_state_after'))}.",
        "",
        "Partial linear and MoE estimate in ms per decode step, not full serving. Includes attention "
        "projections, a BF16 router GEMM, shared and routed experts, and any non-MoE layers' linears. "
        "Routing is uniform random. lm_head is reported separately. Excludes attention math, norms, "
        "routing softmax/top-k, embedding, sampling and communication.",
        "",
    ]
    head_fmts = [f for f in ("nf12", "bf16") if f in fmts]
    cols = (["model", "batch"] + [f"{names[f]} ms" for f in fmts]
            + [f"lm_head us ({' / '.join(names[f].split(' ')[0] for f in head_fmts)})"])
    speedups = ("nf12_speedup_vs_bf16", "nf12_speedup_vs_bf16t", "nf12_speedup_vs_bf16_with_head")
    if "nf12" in fmts:
        cols += ["NF12 vs BF16", "NF12 vs bf16t", "NF12 vs BF16, with lm_head"]
    lines += ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for m in r["models"]:
        for b in m["batches"]:
            cells = [m["model"], str(b["batch"])] + [f"{b[f]['stack_ms']:.3f}" if f in b else "" for f in fmts]
            cells.append(" / ".join(f"{b[f]['lm_head_us']:.1f}" for f in head_fmts if f in b))
            if "nf12" in fmts:
                cells += [f"{b[k]:.3f}x" if k in b else "" for k in speedups]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Weights: " + "; ".join(f"{m['model']}: " + ", ".join(m["weights"]) for m in r["models"]) + ".", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path)
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--batch", nargs="*", type=int, default=[1, 8, 32])
    ap.add_argument("--formats", nargs="*", default=list(FORMATS), choices=FORMATS)
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--min-launches", type=int, default=100)
    ap.add_argument("--budget-s", type=float, default=2.0)
    ap.add_argument("--mem-budget-gb", type=float, default=40.0, help="cap on the layer copies' memory")
    ap.add_argument("--lock", help="flock file taken per (model, batch, format), e.g. /tmp/nf12-thor.lock")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.table:
        print(table(args.table))
        return
    nf12 = import_nf12()
    nfm = Nf12Moe(nf12)
    formats = args.formats
    if not nfm.available:
        print("nf12 MoE kernels unavailable: BF16 only", file=sys.stderr)
        formats = [f for f in formats if f == "bf16"]
    fm = Formats(nf12, nfm if nfm.available else None)
    tag = common.machine_tag()
    result = {"machine": tag, "info": common.machine_info(), "gpu_state_before": common.gpu_state(), "models": []}
    out = Path(args.out) if args.out else common.RESULTS_DIR / f"e2e_moe_{tag}.json"
    for key in args.models:
        result["models"].append(bench_model(fm, MODELS[key], args.batch, formats, args))
        common.write_json(out, result)
    result["gpu_state_after"] = common.gpu_state()
    result["throttle_after"] = common.throttled(result["gpu_state_after"])
    print(common.write_report(out, result, table))


if __name__ == "__main__":
    main()
