# SPDX-License-Identifier: Apache-2.0
"""Estimate model decode time from dense linear layers, NF12 vs BF16.

Times fused qkv, o, gate_up and down in sequence over distinct cold weight
copies, including escape tables. Per-token time is layer count times measured
layer time, plus lm_head.

This is a partial estimate, not full serving. It excludes attention, norms,
activations, embedding, sampling and communication. The paper's 1.13-1.23x
batch-1 end-to-end speedups include those costs.

    uv run --group bench python bench/e2e_decode.py --lock /tmp/nf12-thor.lock
    uv run --group bench python bench/e2e_decode.py --table bench/results/e2e_<machine>.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import torch  # noqa: E402
from common import import_nf12  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Model:
    name: str
    repo: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int

    def linears(self) -> list[tuple[str, int, int, tuple[str, ...]]]:
        """(name, K, N, checkpoint tensors of layer L/2 concatenated along N)."""
        p = f"model.layers.{self.layers // 2}."
        qkv_n = (self.heads + 2 * self.kv_heads) * self.head_dim
        return [
            ("qkv", self.hidden, qkv_n, tuple(p + f"self_attn.{x}_proj.weight" for x in "qkv")),
            ("o", self.heads * self.head_dim, self.hidden, (p + "self_attn.o_proj.weight",)),
            ("gate_up", self.hidden, 2 * self.intermediate, (p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight")),
            ("down", self.intermediate, self.hidden, (p + "mlp.down_proj.weight",)),
        ]


MODELS = {
    "qwen3-8b": Model("Qwen3-8B", "Qwen/Qwen3-8B", 4096, 36, 32, 8, 128, 12288, 151936),
    "qwen3-32b": Model("Qwen3-32B", "Qwen/Qwen3-32B", 5120, 64, 64, 8, 128, 25600, 151936),
    "llama-3.1-8b": Model("Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct", 4096, 32, 32, 8, 128, 14336, 128256),
}
FORMATS = ("nf12", "bf16t", "torch")


def load(model: Model, k: int, n: int, names: tuple[str, ...], args: argparse.Namespace) -> tuple[torch.Tensor, str]:
    shape = common.Shape(k, n, (model.repo, names), names[0])
    return common.make_weight(shape, args.weights, torch.device("cuda"), fetch=args.fetch)


def bench_model(nf12, model: Model, batches: list[int], args: argparse.Namespace) -> dict:
    lin = model.linears()
    weights, origins, escapes = [], [], []
    for name, k, n, names in lin:
        w, origin = load(model, k, n, names, args)
        weights.append(w)
        origins.append(f"{name}: {origin}")
    head, head_origin = load(model, model.hidden, model.vocab, ("lm_head.weight",), args)
    origins.append(f"lm_head: {head_origin}")

    def encode_all(ws: list[torch.Tensor]) -> dict:
        nws = [nf12.encode(w) for w in ws]
        escapes.extend(nw.num_escapes / (nw.shape[0] * nw.shape[1] // 8) for nw in nws)
        return {"nf12": nws, "bf16t": [nf12.pack_bf16t(w) for w in ws], "torch": ws}

    layer_fmt = encode_all(weights)
    head_fmt = encode_all([head])
    layer_bytes = {f: sum(t.nbytes for t in v) for f, v in layer_fmt.items()}

    def clone(f: str, t):
        if f == "nf12":
            return nf12.NF12Weight(t.packed.clone(), t.escapes.clone(), t.exp_base)
        return t.clone()

    def call(f: str, x: torch.Tensor, t):
        if f == "nf12":
            return nf12.linear(x, t)
        if f == "bf16t":
            return nf12.linear_bf16t(x, t)
        return torch.matmul(x, t.t())

    out = {"model": model.name, "weights": origins, "escape_rate_max": max(escapes),
           "escape_rate_mean": sum(escapes) / len(escapes), "layer_bytes": layer_bytes, "batches": []}
    # stacks[f][i] = the four weights of distinct layer copy i.
    stacks = {f: [layer_fmt[f]] + [[clone(f, t) for t in layer_fmt[f]]
                                   for _ in range(common.cold_copies(layer_bytes[f]) - 1)] for f in args.formats}
    for b in batches:
        xs = [torch.randn(b, k, device="cuda", dtype=torch.bfloat16) for _, k, _, _ in lin]
        xh = torch.randn(b, model.hidden, device="cuda", dtype=torch.bfloat16)
        row = {"batch": b}
        for f in args.formats:
            stack = stacks[f]

            def layer(i: int) -> None:
                for x, t in zip(xs, stack[i]):
                    call(f, x, t)

            t_layer = common.time_graph(layer, len(stack), min_launches=args.min_launches, budget_s=args.budget_s)
            head_copies = [head_fmt[f][0]] + [clone(f, head_fmt[f][0]) for _ in range(
                common.cold_copies(head_fmt[f][0].nbytes) - 1)]
            t_head = common.time_graph(lambda i: call(f, xh, head_copies[i]), len(head_copies),
                                       min_launches=args.min_launches, budget_s=args.budget_s)
            del head_copies
            row[f] = {
                "layer_us": t_layer["median_us"], "layer_p10_us": t_layer["p10_us"],
                "layer_p90_us": t_layer["p90_us"], "lm_head_us": t_head["median_us"],
                "stack_ms": model.layers * t_layer["median_us"] / 1e3,
                "stack_plus_head_ms": (model.layers * t_layer["median_us"] + t_head["median_us"]) / 1e3,
            }
        for f in args.formats:
            if f != "nf12" and "nf12" in row:
                row[f"nf12_speedup_vs_{f}"] = row[f]["stack_ms"] / row["nf12"]["stack_ms"]
                row[f"nf12_speedup_vs_{f}_with_head"] = (row[f]["stack_plus_head_ms"]
                                                         / row["nf12"]["stack_plus_head_ms"])
        out["batches"].append(row)
        print(f"{model.name} batch={b}: " + "  ".join(
            f"{f}={row[f]['stack_plus_head_ms']:.3f}ms" for f in args.formats)
            + (f"  nf12 x{row.get('nf12_speedup_vs_torch_with_head', float('nan')):.3f} vs torch (with head)"),
            file=sys.stderr)
    del stacks, layer_fmt, head_fmt, weights, head
    torch.cuda.empty_cache()
    return out


def table(path: Path) -> str:
    r = json.loads(path.read_text())
    lines = [
        f"## Linear-layer decode estimate: {r['info']['gpu']} (SM{r['info']['sm']})",
        "",
        f"GPU before: {common.state_summary(r.get('gpu_state_before'))}. "
        f"After: {common.state_summary(r.get('gpu_state_after'))}.",
        "",
        "Partial linear-layer estimate in ms per decode step (one token per sequence), not full serving. "
        "Includes fused qkv, o, gate_up, down and lm_head. Excludes attention, norms, activations, "
        "embedding, sampling and communication. The paper's 1.13-1.23x batch-1 speedups include those "
        "costs. Linear-only speedup is an upper bound when excluded costs are unchanged.",
        "",
        "| model | batch | BF16 cuBLAS | BF16 bf16t | NF12 | NF12 vs cuBLAS | NF12 vs bf16t "
        "| NF12 vs cuBLAS (stack, no head) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in r["models"]:
        for b in m["batches"]:
            def ms(f: str) -> str:
                return f"{b[f]['stack_plus_head_ms']:.3f}" if f in b else ""

            def sp(key: str) -> str:
                return f"{b[key]:.3f}x" if key in b else ""

            lines.append(f"| {m['model']} | {b['batch']} | {ms('torch')} | {ms('bf16t')} | {ms('nf12')} | "
                         f"{sp('nf12_speedup_vs_torch_with_head')} | {sp('nf12_speedup_vs_bf16t_with_head')} | "
                         f"{sp('nf12_speedup_vs_torch')} |")

    def origin(m: dict) -> str:
        rand = [w.split(":")[0] for w in m["weights"] if w.endswith("random normal(0, 0.02)")]
        return ("real checkpoint layer L/2 + lm_head" if not rand
                else f"random normal(0, 0.02) for {', '.join(rand)}, real otherwise")

    lines += ["", "Weights: " + "; ".join(
        f"{m['model']}: {origin(m)} (escape rate mean {100 * m['escape_rate_mean']:.4f}%, "
        f"max {100 * m['escape_rate_max']:.4f}%)" for m in r["models"]) + ".", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path)
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--batch", nargs="*", type=int, default=[1, 8, 32])
    ap.add_argument("--formats", nargs="*", default=list(FORMATS), choices=FORMATS)
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--fetch", action="store_true", help="range-download missing real-weight tensors")
    ap.add_argument("--min-launches", type=int, default=100)
    ap.add_argument("--budget-s", type=float, default=3.0)
    ap.add_argument("--lock", help="flock file taken per model, e.g. /tmp/nf12-thor.lock")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.table:
        print(table(args.table))
        return
    nf12 = import_nf12()
    tag = common.machine_tag()
    result = {"machine": tag, "info": common.machine_info(), "gpu_state_before": common.gpu_state(), "models": []}
    out = Path(args.out) if args.out else common.RESULTS_DIR / f"e2e_{tag}.json"
    for name in args.models:
        with common.gpu_lock(args.lock):
            result["models"].append(bench_model(nf12, MODELS[name], args.batch, args))
        common.write_json(out, result)
    result["gpu_state_after"] = common.gpu_state()
    result["throttle_after"] = common.throttled(result["gpu_state_after"])
    print(common.write_report(out, result, table))


if __name__ == "__main__":
    main()
