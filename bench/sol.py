# SPDX-License-Identifier: Apache-2.0
"""Calibrate GPU bandwidth and compute limits for NF12 benchmarks.

    flock /tmp/nf12-thor.lock uv run --group bench python bench/sol.py
    uv run --group bench python bench/sol.py --report

Measures DRAM/L2 reads, copies, launch overhead, MMA throughput and cuBLAS.
--report writes bench/results/sol.md from the saved calibration JSONs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

# Vendor-nominal DRAM bandwidth (GB/s), for "% of nominal" only.
NOMINAL_GBPS = {"rtxpro6000": 1792.0, "thor": 273.0}
BW_M_MAX = 64  # M <= this: report bandwidth; above: TFLOP/s


def run_probe(cold_bytes: list[int], quick: bool) -> dict:
    cmd = [str(common.build_cuda(common.BENCH_DIR / "sol" / "sol_device.cu", "sol_device")), "--cold-bytes",
           ",".join(map(str, cold_bytes))]
    if quick:
        cmd.append("--quick")
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


def bench_cublas(shapes: list[common.Shape], ms: list[int], quick: bool) -> list[dict]:
    import torch

    rows = []
    for s in shapes:
        copies = common.cold_copies(2 * s.n * s.k)
        ws = [common.make_weight(s, "random", torch.device("cuda"), seed=i)[0] for i in range(copies)]
        for m in ms:
            x = torch.randn(m, s.k, device="cuda", dtype=torch.bfloat16)
            t = common.time_graph(lambda i: torch.matmul(x, ws[i].t()), copies,
                                  min_launches=20 if quick else 30, max_replays=30 if quick else 100,
                                  budget_s=1.0 if quick else 2.0)
            us = t["median_us"]
            bytes_bf16 = 2 * s.n * s.k + 2 * m * s.k + 2 * m * s.n
            rows.append({
                "k": s.k, "n": s.n, "m": m, **t,
                "gbps": bytes_bf16 / us / 1e3,
                "tflops": 2 * m * s.n * s.k / us / 1e6,
            })
            print(f"cublas {s.key:>11} M={m:<6} {us:10.2f} us  {rows[-1]['gbps']:7.1f} GB/s  "
                  f"{rows[-1]['tflops']:7.1f} TFLOP/s", file=sys.stderr)
            del x
        del ws
        torch.cuda.empty_cache()
    return rows


def derive_sol(probe: dict, cublas: list[dict], shapes: list[common.Shape], ms: list[int]) -> list[dict]:
    """Estimate latency as max(memory time, compute time).

    Cold reads include launch overhead. Count activation/output DRAM traffic
    only above half L2 capacity. Compute peak is the best measured MMA/cuBLAS rate.
    """
    bw = probe["read_best"]["gbps"]
    l2 = probe["device"]["l2_bytes"]
    cold = {c["bytes"]: c["us"] for c in probe["cold"]}
    peak = max([probe["mma_sync_bf16_f32"]["best_tflops"]] + [r["tflops"] for r in cublas])
    cub = {(r["k"], r["n"], r["m"]): r for r in cublas}
    out = []
    for s in shapes:
        w_nf12, w_bf16 = s.n * s.k * 3 // 2, 2 * s.n * s.k
        for m in ms:
            act = 2 * m * s.k + 2 * m * s.n
            act_dram = act if 2 * act > l2 else 0
            t_c = 2 * m * s.n * s.k / peak / 1e6
            nf12_mem = cold.get(-(-w_nf12 // 256) * 256, w_nf12 / bw / 1e3) + act_dram / bw / 1e3
            bf16_mem = cold.get(-(-w_bf16 // 256) * 256, w_bf16 / bw / 1e3) + act_dram / bw / 1e3
            row = {
                "k": s.k, "n": s.n, "m": m,
                "nf12_bytes": w_nf12 + act, "bf16_bytes": w_bf16 + act, "act_dram_bytes": act_dram,
                "compute_peak_tflops": peak,
                "nf12_mem_us": nf12_mem, "bf16_mem_us": bf16_mem, "compute_us": t_c,
                "nf12_sol_us": max(nf12_mem, t_c), "bf16_sol_us": max(bf16_mem, t_c),
                "nf12_stream_us": (w_nf12 + act_dram) / bw / 1e3,
            }
            row["sol_speedup_vs_bf16_sol"] = row["bf16_sol_us"] / row["nf12_sol_us"]
            c = cub.get((s.k, s.n, m))
            if c is not None:
                row["cublas_us"] = c["median_us"]
                row["cublas_pct_of_stream_bw"] = 100 * c["gbps"] / bw
                row["cublas_tflops"] = c["tflops"]
                row["cublas_pct_of_compute_peak"] = 100 * c["tflops"] / peak
                row["sol_speedup_vs_cublas"] = c["median_us"] / row["nf12_sol_us"]
            out.append(row)
    return out


def calibrate(args: argparse.Namespace) -> None:
    tag = common.machine_tag()
    shapes = [common.shape_by_key(k) for k in args.shapes] if args.shapes else common.SHAPES
    ms = args.m or common.M_LIST
    cold_bytes = sorted({-(-b // 256) * 256 for s in shapes for b in (s.n * s.k * 3 // 2, 2 * s.n * s.k)})
    before = common.gpu_state()
    t0 = time.time()
    probe = run_probe(cold_bytes, args.quick)
    print(f"probe: read {probe['read_best']['gbps']:.1f} GB/s, mma.sync {probe['mma_sync_bf16_f32']['best_tflops']:.1f}"
          f" TFLOP/s ({time.time() - t0:.0f}s)", file=sys.stderr)
    cublas = [] if args.skip_cublas else bench_cublas(shapes, ms, args.quick)
    after = common.gpu_state()
    result = {
        "machine": tag,
        "info": common.machine_info(),
        "nominal_dram_gbps": NOMINAL_GBPS.get(tag),
        "gpu_state_before": before,
        "gpu_state_after": after,
        "throttle_after": common.throttled(after),
        "probe": probe,
        "cublas": cublas,
        "sol": derive_sol(probe, cublas, shapes, ms),
    }
    out = Path(args.out) if args.out else common.RESULTS_DIR / f"sol_{tag}.json"
    common.write_json(out, result)
    print(f"wrote {out}", file=sys.stderr)


def _f(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}"


def machine_section(r: dict) -> list[str]:
    p, info = r["probe"], r["info"]
    d = p["device"]
    bw = p["read_best"]["gbps"]
    nominal = r.get("nominal_dram_gbps")
    hot = max(h["gbps"] for h in p["l2_hot"] if h["bytes"] <= d["l2_bytes"])
    rb = p["read_best"]
    best_cfg = (f"{rb['kind']}" + (f" `{rb['ld']}` {rb['pattern']} x{rb['unroll']}, {rb['block']} thr x "
                                    f"{rb['ctas_per_sm']} CTA/SM" if rb["kind"] == "ldg" else
                                    f" {rb['stages']}x{rb['chunk']} B, {rb['ctas_per_sm']} CTA/SM"))
    cub_peak = max((c["tflops"] for c in r["cublas"]), default=float("nan"))
    lines = [
        f"## {info['gpu']} (SM{info['sm']}, {d['num_sms']} SMs), `{r['machine']}`",
        "",
        f"torch {info['torch']}, CUDA {info['cuda_runtime']}, driver {info['driver']}, "
        f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}. Throttle reasons after run: "
        f"{', '.join(r['throttle_after']) or 'none'}.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| DRAM streaming read (best) | **{_f(bw)} GB/s**"
        + (f" ({_f(100 * bw / nominal)}% of nominal {_f(nominal, 0)})" if nominal else "") + f", {best_cfg} |",
        "| best by load flavour | " + ", ".join(f"`{k}` {_f(v)}" for k, v in p["read_best_by_ld"].items()) + " |",
        f"| D2D copy | cudaMemcpy {_f(p['copy']['memcpy_rw_gbps'])} GB/s (read+write), "
        f"{_f(p['copy']['memcpy_bytes_gbps'])} GB/s copied; copy kernel {_f(p['copy']['kernel_rw_gbps'])} GB/s (r+w) |",
        f"| L2 size | {d['l2_bytes'] >> 20} MiB (attribute) |",
        f"| L2-resident read | {_f(hot)} GB/s (best footprint <= L2) |",
        f"| launch floor | graph {_f(p['launch']['graph_full_grid_us'], 2)} us/kernel, stream back-to-back "
        f"{_f(p['launch']['stream_back_to_back_us'], 2)} us, single event-bracketed "
        f"{_f(p['launch']['single_event_us'], 2)} us |",
        f"| mma.sync m16n8k16 BF16->FP32 peak | {_f(p['mma_sync_bf16_f32']['best_tflops'])} TFLOP/s "
        f"({p['mma_sync_bf16_f32']['best_warps_per_sm']} warps/SM, {p['mma_sync_bf16_f32']['best_chains']} chains) |",
        f"| cuBLAS BF16 best on grid | {_f(cub_peak)} TFLOP/s |",
        "",
        "Hot read bandwidth vs footprint (MiB: GB/s): "
        + ", ".join(f"{h['bytes'] >> 20}: {_f(h['gbps'], 0)}" for h in p["l2_hot"]),
        "",
        "Cold single-kernel read (one kernel streams the buffer from DRAM, graph launch, rotating copies), "
        "MiB: us (GB/s): "
        + ", ".join(f"{h['bytes'] / 2**20:.1f}: {_f(h['us'])} ({_f(h['gbps'], 0)})" for h in p["cold"]),
        "",
    ]
    sol = r["sol"]
    shapes = list(dict.fromkeys((x["k"], x["n"]) for x in sol))
    small = [m for m in dict.fromkeys(x["m"] for x in sol) if m <= BW_M_MAX]
    large = [m for m in dict.fromkeys(x["m"] for x in sol) if m > BW_M_MAX]
    by = {(x["k"], x["n"], x["m"]): x for x in sol}
    sections = (
        (small, f"Small M (bandwidth bound): NF12 SOL us / cuBLAS us (cuBLAS % of {_f(bw, 0)} GB/s)",
         "cublas_pct_of_stream_bw", 0, "%"),
        (large, "Large M (compute bound): NF12 SOL us / cuBLAS us (cuBLAS TFLOP/s)",
         "cublas_tflops", 1, ""),
    )
    for ms, title, metric, digits, unit in sections:
        if not ms:
            continue
        lines += [
            f"### {title}", "",
            "| K x N | " + " | ".join(f"M={m}" for m in ms) + " |",
            "|---" * (len(ms) + 1) + "|",
        ]
        for k, n in shapes:
            cells = []
            for m in ms:
                x = by[(k, n, m)]
                c = f" / {_f(x['cublas_us'])} ({_f(x[metric], digits)}{unit})" if "cublas_us" in x else ""
                cells.append(f"{_f(x['nf12_sol_us'])}{c}")
            lines.append(f"| {k}x{n} | " + " | ".join(cells) + " |")
        lines.append("")
    m1 = [by[(k, n, 1)] for k, n in shapes if (k, n, 1) in by and "cublas_us" in by[(k, n, 1)]]
    if m1:
        lines += [
            "M=1 ceilings: NF12 SOL speedup over cuBLAS = "
            + ", ".join(f"{x['k']}x{x['n']} {_f(x['sol_speedup_vs_cublas'], 2)}x" for x in m1)
            + ". Ideal bytes ratio is 1.33x. Values above it mean cuBLAS itself is below the BF16 SOL.",
            "",
        ]
    return lines


def report() -> None:
    lines = [
        "# NF12 speed of light", "",
        "Generated by `python bench/sol.py --report` from `bench/results/sol_*.json`.", "",
        "## Method", "",
        "- DRAM reads sweep load policies, access patterns and occupancy on a buffer >= max(16x L2, 1 GiB).",
        "- Per-weight cold reads use rotating copies in a CUDA graph and include launch overhead.",
        "- NF12 latency bound = max(weight-read + activation traffic, 2MNK / compute peak).",
        "  Activation traffic counts only above half L2. Peak throughput is max(measured MMA, cuBLAS).",
        "- cuBLAS uses BF16 `torch.matmul` with rotating weights.",
        "- Linear benchmarks measure `sol_read` in-run. Separate calibrations can differ with GPU state.", "",
    ]
    for path in sorted(common.RESULTS_DIR.glob("sol_*.json")):
        lines += machine_section(json.loads(path.read_text()))
    destination = common.RESULTS_DIR / "sol.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
    print(f"wrote {destination}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="render saved calibrations to bench/results/sol.md")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-cublas", action="store_true")
    ap.add_argument("--shapes", nargs="*", help="KxN keys (default: full grid)")
    ap.add_argument("--m", nargs="*", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        calibrate(args)


if __name__ == "__main__":
    main()
