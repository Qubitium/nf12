# SPDX-License-Identifier: Apache-2.0
"""Benchmark NF12 linear against its BF16 twin, cuBLAS and decode + cuBLAS.

    uv run --group bench python bench/bench_linear.py --lock /tmp/nf12-thor.lock
    uv run --group bench python bench/bench_linear.py --shapes 4096x4096 --m 1 16 --l2 hot
    uv run --group bench python bench/bench_linear.py --table bench/results/linear_<machine>.json

``nf12`` uses the torch dispatcher. ``nf12_c`` calls the C API directly.
Both are checked against their route references and cuBLAS error.
``sol_read`` measures packed-weight reads in the same run. See common.py for timing.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import torch  # noqa: E402
from common import import_nf12  # noqa: E402

# sol_read streams the packed bytes of the same rotated copies alongside NF12.
VARIANTS = ("nf12", "nf12_c", "sol_read", "bf16t", "torch", "decode_matmul")


class CWeight(ctypes.Structure):
    _fields_ = [("packed", ctypes.c_void_p), ("escapes", ctypes.c_void_p), ("num_escapes", ctypes.c_uint32),
                ("n", ctypes.c_int64), ("k", ctypes.c_int64), ("exp_base", ctypes.c_uint8)]


class CApi:
    """C API calls with graph-capturable torch streams and workspace allocation."""

    def __init__(self, nf12) -> None:
        lib = ctypes.CDLL(str(nf12.library_path()))
        i64, vp = ctypes.c_int64, ctypes.c_void_p
        lib.nf12_linear_workspace_bytes.restype = ctypes.c_size_t
        lib.nf12_linear_workspace_bytes.argtypes = [i64, i64, i64, ctypes.c_int]
        lib.nf12_linear_decode_gemm_rows.restype = i64
        lib.nf12_linear_decode_gemm_rows.argtypes = [i64, i64, i64, ctypes.c_int]
        lib.nf12_linear.restype = ctypes.c_int
        lib.nf12_linear.argtypes = [vp, i64, i64, ctypes.POINTER(CWeight), vp, vp, i64, vp, ctypes.c_size_t, vp]
        self.lib = lib

    def decode_gemm_rows(self, m: int, n: int, k: int) -> int:
        return self.lib.nf12_linear_decode_gemm_rows(m, n, k, torch.cuda.current_device())

    def linear(self, x: torch.Tensor, nw) -> torch.Tensor:
        m, k = x.shape
        n = nw.shape[0]
        y = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
        need = self.lib.nf12_linear_workspace_bytes(m, n, k, torch.cuda.current_device())
        ws = torch.empty(max(need, 1), dtype=torch.uint8, device=x.device)
        w = CWeight(nw.packed.data_ptr(), nw.escapes.data_ptr() if nw.num_escapes else None, nw.num_escapes, n, k,
                    nw.exp_base)
        status = self.lib.nf12_linear(x.data_ptr(), m, k, ctypes.byref(w), None, y.data_ptr(), n, ws.data_ptr(), need,
                                      torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"nf12_linear failed: status {status}")
        return y


def route_reference(nf12, x: torch.Tensor, nw, rows: int) -> torch.Tensor:
    """The decode + cuBLAS route's calls on the fully decoded weight."""
    wd = nf12.decode(nw)
    y = torch.empty(x.shape[0], wd.shape[0], dtype=torch.bfloat16, device=x.device)
    for n0 in range(0, wd.shape[0], rows):
        torch.mm(x, wd[n0:n0 + rows].t(), out=y[:, n0:n0 + rows])
    return y


def load_sol(tag: str) -> dict:
    p = common.RESULTS_DIR / f"sol_{tag}.json"
    if not p.exists():
        return {}
    return {(r["k"], r["n"], r["m"]): r for r in json.loads(p.read_text())["sol"]}


@dataclasses.dataclass
class Prepared:
    """One shape's weights in every format, with cold-L2 rotation copies."""

    w: torch.Tensor
    nw: object  # nf12.NF12Weight
    bt: torch.Tensor
    nws: list
    bts: list
    ws: list
    copies: dict
    entry: dict
    read: common.SolRead | None
    read_cfg: int


def prepare(nf12, shape: common.Shape, args: argparse.Namespace) -> Prepared:
    dev = torch.device("cuda")
    w, origin = common.make_weight(shape, args.weights, dev, fetch=args.fetch)
    nw = nf12.encode(w)
    lossless = torch.equal(nf12.decode(nw).view(torch.int16), w.view(torch.int16))
    bt = nf12.pack_bf16t(w)
    entry = {
        "key": shape.key, "k": shape.k, "n": shape.n, "note": shape.note, "weight_source": origin,
        "exp_base": nw.exp_base, "num_escapes": nw.num_escapes, "escape_rate": nw.num_escapes / (shape.n * shape.k // 8),
        "nf12_weight_bytes": nw.nbytes, "lossless": lossless, "rows": [],
    }
    print(f"{shape.key} ({origin}): exp_base={nw.exp_base} escapes={nw.num_escapes} "
          f"({100 * entry['escape_rate']:.4f}%) lossless={lossless}", file=sys.stderr)
    need_copies = args.l2 != "hot" and args.timing == "graph"
    c_nf12 = common.cold_copies(nw.nbytes) if need_copies else 1
    c_bf16 = common.cold_copies(2 * shape.n * shape.k) if need_copies else 1
    # Each copy owns its escape table so escape reads miss L2 too.
    nws = [nw] + [nf12.NF12Weight(nw.packed.clone(), nw.escapes.clone(), nw.exp_base) for _ in range(c_nf12 - 1)]
    bts = [bt] + [bt.clone() for _ in range(c_bf16 - 1)] if "bf16t" in args.variants else []
    ws = [w] + [w.clone() for _ in range(c_bf16 - 1)] if "torch" in args.variants else [w]
    copies = {"nf12": c_nf12, "nf12_c": c_nf12, "sol_read": c_nf12, "decode_matmul": c_nf12, "bf16t": c_bf16,
              "torch": c_bf16}
    entry["copies"] = copies
    read, read_cfg = None, -1
    if "sol_read" in args.variants:
        read = common.SolRead()
        read_cfg, screen_us = read.best([q.packed for q in nws])
        entry["sol_read_config"] = read.describe(read_cfg)
        print(f"  sol_read: {entry['sol_read_config']} ({screen_us:.2f} us screening)", file=sys.stderr)
    return Prepared(w, nw, bt, nws, bts, ws, copies, entry, read, read_cfg)


def rel_err(y: torch.Tensor, ref: torch.Tensor, rows: int = 1024) -> float:
    """||y - ref|| / ||ref|| in FP32, a block of rows at a time (lm_head outputs at large M are GBs)."""
    num = den = 0.0
    for r0 in range(0, ref.shape[0], rows):
        a, b = y[r0:r0 + rows].float(), ref[r0:r0 + rows].float()
        num += (a - b).square().sum().item()
        den += b.square().sum().item()
    return num ** 0.5 / max(den ** 0.5, 1e-30)


def measure(nf12, capi: CApi, shape: common.Shape, m: int, p: Prepared, args: argparse.Namespace,
            sol: dict) -> dict:
    x = torch.randn(m, shape.k, device="cuda", dtype=torch.bfloat16)
    rows = capi.decode_gemm_rows(m, shape.n, shape.k)
    yt = nf12.linear_bf16t(x, p.bt)
    yc = capi.linear(x, p.nw)
    matches_bf16t = torch.equal(yc.view(torch.int16), yt.view(torch.int16))
    del yc
    route_ref = route_reference(nf12, x, p.nw, rows) if rows else yt
    del yt
    y = nf12.linear(x, p.nw)
    matches_route = torch.equal(y.view(torch.int16), route_ref.view(torch.int16))
    del route_ref
    ref = torch.matmul(x, p.w.t())
    row = {
        "m": m,
        "route": f"decode+cublas/{rows}" if rows else "nf12_linear",
        "matches_bf16t": matches_bf16t,
        "matches_route": matches_route,
        "rel_err_vs_torch": rel_err(y, ref),
    }
    del y, ref
    fns = {
        "nf12": lambda i: nf12.linear(x, p.nws[i]),
        "nf12_c": lambda i: capi.linear(x, p.nws[i]),
        "bf16t": lambda i: nf12.linear_bf16t(x, p.bts[i]),
        "torch": lambda i: torch.matmul(x, p.ws[i].t()),
        "decode_matmul": lambda i: torch.matmul(x, nf12.decode(p.nws[i]).t()),
        "sol_read": lambda i: p.read(p.nws[i].packed, p.read_cfg),
    }
    act = 2 * m * shape.k + 2 * m * shape.n
    flops = 2 * m * shape.n * shape.k
    moved = {"nf12": p.nw.nbytes + act, "nf12_c": p.nw.nbytes + act, "sol_read": p.nw.packed.nbytes,
             "bf16t": 2 * shape.n * shape.k + act, "torch": 2 * shape.n * shape.k + act,
             "decode_matmul": p.nw.nbytes + act}
    for mode in (["cold", "hot"] if args.l2 == "both" else [args.l2]):
        res = {}
        for v in args.variants:
            t = common.time_fn(fns[v], p.copies[v], timing=args.timing, cold=mode == "cold",
                               min_launches=args.min_launches, budget_s=args.budget_s)
            us = t["median_us"]
            t["gbps"] = moved[v] / us / 1e3
            if v != "sol_read":
                t["tflops"] = flops / us / 1e6
            res[v] = t
        if "nf12" in res:
            nf = res["nf12"]["median_us"]
            for v in res:
                if v not in ("nf12", "sol_read"):
                    res["nf12"][f"speedup_vs_{v}"] = res[v]["median_us"] / nf
            if "sol_read" in res and mode == "cold":
                # SOL = max(streaming the packed weights, 2MNK at the machine's
                # measured tensor peak from results/sol_<machine>.json).
                s = sol.get((shape.k, shape.n, m))
                sol_us = max(res["sol_read"]["median_us"], s["compute_us"] if s else 0.0)
                res["nf12"]["sol_us"] = sol_us
                res["nf12"]["pct_of_sol"] = 100 * sol_us / nf
        row[mode] = res
        print(f"  M={m:<6} {mode:4} " + "  ".join(f"{v}={res[v]['median_us']:9.2f}us" for v in res)
              + (f"  x{res['nf12'].get('speedup_vs_torch', float('nan')):.3f} vs torch" if "nf12" in res else "")
              + (f"  {res['nf12']['pct_of_sol']:.0f}% SOL" if "pct_of_sol" in res.get("nf12", {}) else ""),
              file=sys.stderr)
    return row


def bench_shape(nf12, capi: CApi, shape: common.Shape, ms: list[int], args: argparse.Namespace,
                sol: dict) -> dict:
    # Lock each point so other GPU users can interleave with long sweeps.
    with common.gpu_lock(args.lock):
        p = prepare(nf12, shape, args)
    for m in ms:
        with common.gpu_lock(args.lock):
            p.entry["rows"].append(measure(nf12, capi, shape, m, p, args, sol))
    entry = p.entry
    entry["gpu_state_after"] = common.gpu_state()
    entry["throttle_after"] = common.throttled(entry["gpu_state_after"])
    del p
    torch.cuda.empty_cache()
    return entry


def run(args: argparse.Namespace) -> None:
    nf12 = import_nf12()
    capi = CApi(nf12)
    tag = common.machine_tag()
    shapes = ([common.shape_by_key(k) for k in args.shapes] if args.shapes
              else common.SHAPES + common.EXTRA_SHAPES)
    ms = args.m or common.M_LIST
    result = {
        "machine": tag, "info": common.machine_info(),
        "config": {"timing": args.timing, "l2": args.l2, "weights": args.weights, "variants": args.variants,
                   "min_launches": args.min_launches, "rotate_min_bytes": common.ROTATE_MIN_BYTES},
        "gpu_state_before": common.gpu_state(), "shapes": [],
    }
    out = Path(args.out) if args.out else common.RESULTS_DIR / f"linear_{tag}.json"
    sol = load_sol(tag)
    t0 = time.time()
    for s in shapes:
        result["shapes"].append(bench_shape(nf12, capi, s, ms, args, sol))
        common.write_json(out, result)  # partial results survive an interrupted sweep
    result["gpu_state_after"] = common.gpu_state()
    result["wall_s"] = time.time() - t0
    common.write_report(out, result, table)
    print(f"wrote {out} and {out.with_suffix('.md')}", file=sys.stderr)


def table(path: Path) -> str:
    r = json.loads(path.read_text())
    cfg = r["config"]
    info = r["info"]
    lines = [
        f"## Linear: {info['gpu']} (SM{info['sm']}), timing `{cfg['timing']}`, weights `{cfg['weights']}`",
        "",
        f"GPU before: {common.state_summary(r.get('gpu_state_before'))}. "
        f"After: {common.state_summary(r.get('gpu_state_after'))}.",
        "",
    ]
    modes = [m for m in ("cold", "hot") if any(m in row for s in r["shapes"] for row in s["rows"])]
    for mode in modes:
        ms = list(dict.fromkeys(row["m"] for s in r["shapes"] for row in s["rows"]))
        lines += [
            f"### NF12 speedup over cuBLAS `torch.matmul` ({mode} L2), median",
            "",
            "| K x N | " + " | ".join(f"M={m}" for m in ms) + " |",
            "|---" * (len(ms) + 1) + "|",
        ]
        for s in r["shapes"]:
            by = {row["m"]: row for row in s["rows"]}
            cells = []
            for m in ms:
                row = by.get(m, {})
                v = row.get(mode, {}).get("nf12", {}).get("speedup_vs_torch")
                cells.append(f"{v:.2f}" if v is not None else "")
            lines.append(f"| {common.entry_label(s)} | " + " | ".join(cells) + " |")
        lines.append("")
    for s in r["shapes"]:
        lines += [
            f"### {common.entry_label(s)}: {s['note']}",
            "",
            f"Weights: {s['weight_source']}. exp_base {s['exp_base']}, escapes {s['num_escapes']} "
            f"({100 * s['escape_rate']:.4f}% of groups). Lossless round trip: {s['lossless']}. "
            f"Throttle after: {', '.join(s['throttle_after']) or 'none'}.",
            "",
        ]
        for mode in modes:
            variants = [v for v in VARIANTS if any(v in row.get(mode, {}) for row in s["rows"])]
            lines += [
                f"{mode} L2, median us [p10, p90]:",
                "",
                "| M | " + " | ".join(variants) + " | NF12 GB/s | NF12 TFLOP/s | vs bf16t | vs torch | % SOL | "
                "matches bf16t | route (matches reference) | rel err vs torch |",
                "|---" * (len(variants) + 9) + "|",
            ]
            for row in s["rows"]:
                res = row.get(mode, {})
                cells = [f"{res[v]['median_us']:.2f} [{res[v]['p10_us']:.2f}, {res[v]['p90_us']:.2f}]"
                         if v in res else "" for v in variants]
                nf = res.get("nf12", {})

                def g(key: str, fmt: str) -> str:
                    return format(nf[key], fmt) if key in nf else ""

                lines.append(
                    f"| {row['m']} | " + " | ".join(cells)
                    + f" | {g('gbps', '.0f')} | {g('tflops', '.1f')} | {g('speedup_vs_bf16t', '.3f')} | "
                    f"{g('speedup_vs_torch', '.3f')} | {g('pct_of_sol', '.0f')} | {row['matches_bf16t']} | "
                    f"{row.get('route', 'nf12_linear')} ({row.get('matches_route', row['matches_bf16t'])}) | "
                    f"{row['rel_err_vs_torch']:.2e} |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, help="render a results JSON as Markdown")
    ap.add_argument("--shapes", nargs="*", help="KxN keys (default: full grid)")
    ap.add_argument("--m", nargs="*", type=int)
    ap.add_argument("--weights", choices=["real", "random"], default="real",
                    help="use cached checkpoint weights, falling back to random")
    ap.add_argument("--fetch", action="store_true", help="range-download missing checkpoint tensors")
    ap.add_argument("--timing", choices=["graph", "events"], default="graph")
    ap.add_argument("--l2", choices=["cold", "hot", "both"], default="cold")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--min-launches", type=int, default=100)
    ap.add_argument("--budget-s", type=float, default=3.0, help="target seconds per measurement")
    ap.add_argument("--lock", help="flock file taken per (shape, M), e.g. /tmp/nf12-thor.lock")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.table:
        print(table(args.table))
    else:
        run(args)


if __name__ == "__main__":
    main()
