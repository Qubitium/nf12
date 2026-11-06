# SPDX-License-Identifier: Apache-2.0
"""Benchmark NF12 encode, decode and BF16-twin packing.

    uv run --group bench python bench/bench_codec.py --lock /tmp/nf12-thor.lock
    uv run --group bench python bench/bench_codec.py --table bench/results/codec_<machine>.json

Decode and packing use cold CUDA graphs. Encode synchronizes and uses events.
Bandwidth counts reads + writes against the calibrated D2D copy rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import torch  # noqa: E402
from common import import_nf12  # noqa: E402


def copy_gbps(tag: str) -> float | None:
    p = common.RESULTS_DIR / f"sol_{tag}.json"
    return json.loads(p.read_text())["probe"]["copy"]["memcpy_rw_gbps"] if p.exists() else None


def bench_shape(nf12, shape: common.Shape, args: argparse.Namespace, copy_bw: float | None) -> dict:
    dev = torch.device("cuda")
    w, origin = common.make_weight(shape, args.weights, dev, fetch=args.fetch)
    nw = nf12.encode(w)
    nk = shape.n * shape.k
    lossless = torch.equal(nf12.decode(nw).view(torch.int16), w.view(torch.int16))
    entry = {"key": shape.key, "k": shape.k, "n": shape.n, "note": shape.note, "weight_source": origin,
             "exp_base": nw.exp_base, "num_escapes": nw.num_escapes, "escape_rate": nw.num_escapes / (nk // 8),
             "lossless": lossless}

    t = common.time_events(lambda i: nf12.encode(w), cold=False, iters=args.encode_iters)
    entry["encode"] = {**t, "gbps": (2 * nk + nw.nbytes) / t["median_us"] / 1e3}

    copies = common.cold_copies(nw.nbytes + 2 * nk)
    nws = [nw] + [nf12.NF12Weight(nw.packed.clone(), nw.escapes.clone(), nw.exp_base) for _ in range(copies - 1)]
    t = common.time_graph(lambda i: nf12.decode(nws[i]), copies, distinct_outputs=True)
    entry["decode"] = {**t, "gbps": (nw.nbytes + 2 * nk) / t["median_us"] / 1e3}
    del nws

    copies = common.cold_copies(4 * nk)
    ws = [w] + [w.clone() for _ in range(copies - 1)]
    t = common.time_graph(lambda i: nf12.pack_bf16t(ws[i]), copies, distinct_outputs=True)
    entry["pack_bf16t"] = {**t, "gbps": 4 * nk / t["median_us"] / 1e3}
    del ws

    if copy_bw:
        for op in ("encode", "decode", "pack_bf16t"):
            entry[op]["pct_of_copy_bw"] = 100 * entry[op]["gbps"] / copy_bw
    print(f"{shape.key}: escapes {100 * entry['escape_rate']:.4f}% lossless={lossless} "
          + "  ".join(f"{op}={entry[op]['median_us']:.1f}us ({entry[op]['gbps']:.0f} GB/s)"
                      for op in ("encode", "decode", "pack_bf16t")), file=sys.stderr)
    del w, nw
    torch.cuda.empty_cache()
    return entry


def table(path: Path) -> str:
    r = json.loads(path.read_text())
    info = r["info"]
    bw = r.get("copy_rw_gbps")
    lines = [
        f"## Codec: {info['gpu']} (SM{info['sm']})",
        "",
        "Median us (GB/s read+write" + (f", % of {bw:.0f} GB/s D2D copy" if bw else "") + "). "
        f"GPU before: {common.state_summary(r.get('gpu_state_before'))}. "
        f"After: {common.state_summary(r.get('gpu_state_after'))}.",
        "",
        "| K x N | weights | escapes | lossless | encode | decode | pack_bf16t |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in r["shapes"]:
        def cell(op: str) -> str:
            e = s[op]
            pct = f", {e['pct_of_copy_bw']:.0f}%" if "pct_of_copy_bw" in e else ""
            return f"{e['median_us']:.1f} ({e['gbps']:.0f}{pct})"

        src = "real" if not s["weight_source"].startswith("random") else "random"
        lines.append(f"| {common.entry_label(s)} | {src} | {100 * s['escape_rate']:.4f}% | {s['lossless']} | "
                     f"{cell('encode')} | {cell('decode')} | {cell('pack_bf16t')} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path)
    ap.add_argument("--shapes", nargs="*")
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--fetch", action="store_true", help="range-download missing real-weight tensors")
    ap.add_argument("--encode-iters", type=int, default=100)
    ap.add_argument("--lock", help="flock file taken per shape, e.g. /tmp/nf12-thor.lock")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.table:
        print(table(args.table))
        return
    nf12 = import_nf12()
    tag = common.machine_tag()
    shapes = ([common.shape_by_key(k) for k in args.shapes] if args.shapes
              else common.SHAPES + common.EXTRA_SHAPES)
    bw = copy_gbps(tag)
    result = {"machine": tag, "info": common.machine_info(), "copy_rw_gbps": bw,
              "gpu_state_before": common.gpu_state(), "shapes": []}
    out = Path(args.out) if args.out else common.RESULTS_DIR / f"codec_{tag}.json"
    for s in shapes:
        with common.gpu_lock(args.lock):
            result["shapes"].append(bench_shape(nf12, s, args, bw))
        common.write_json(out, result)
    result["gpu_state_after"] = common.gpu_state()
    result["throttle_after"] = common.throttled(result["gpu_state_after"])
    print(common.write_report(out, result, table))


if __name__ == "__main__":
    main()
