# SPDX-License-Identifier: Apache-2.0
"""Benchmark NF12 lookup against BF16 ``F.embedding``.

    uv run --group bench python bench/bench_embedding.py --lock /tmp/nf12-thor.lock
    uv run --group bench python bench/bench_embedding.py --table bench/results/embedding_<machine>.json

Uniform int64 IDs index random, 16-row-aligned half-table windows in a CUDA
graph. The nominal touched footprint spans 4x L2, subject to the launch cap.
Bandwidth counts unique rows (NF12: 3H bytes, BF16: 2H) and 2H output bytes
per ID. NF12 reads paired rows, including the row eight positions away.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from common import import_nf12  # noqa: E402

TABLES = [
    ("Qwen3-8B", "Qwen/Qwen3-8B", 151936, 4096),
    ("Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct", 128256, 4096),
    ("Qwen3-0.6B", "Qwen/Qwen3-0.6B", 151936, 1024),
    ("Gemma-2-9B", "unsloth/gemma-2-9b-it", 256000, 3584),
]
NUM_IDS = [1, 8, 64, 512, 4096, 32768]
TENSOR = "model.embed_tokens.weight"
MAX_LAUNCHES = 40000


def load_table(repo: str, v: int, h: int, fetch: bool) -> tuple[torch.Tensor, str]:
    w = common.load_hf_tensor(repo, TENSOR, fetch)
    if w is not None:
        if tuple(w.shape) != (v, h):
            raise ValueError(f"{repo}:{TENSOR} has shape {tuple(w.shape)}, expected {(v, h)}")
        return w.to(device="cuda", dtype=torch.bfloat16).contiguous(), f"{repo}: {TENSOR}"
    g = torch.Generator(device="cuda").manual_seed(0)
    return (torch.randn(v, h, generator=g, device="cuda") * 0.02).to(torch.bfloat16), "random normal(0, 0.02)"


def bench_table(nf12, name: str, repo: str, v: int, h: int, args: argparse.Namespace) -> dict:
    w, origin = load_table(repo, v, h, args.fetch)
    nw = nf12.encode(w)
    entry = {"name": name, "v": v, "h": h, "weight_source": origin, "exp_base": nw.exp_base,
             "num_escapes": nw.num_escapes, "escape_rate": nw.num_escapes / (v * h // 8), "points": []}
    window = v // 32 * 16  # rows per view, a multiple of 16
    g = torch.Generator(device="cuda").manual_seed(1)
    for num_ids in args.num_ids:
        ids = torch.randint(0, window, (num_ids,), generator=g, device="cuda")
        unique = ids.unique().numel()
        launches = min(MAX_LAUNCHES, max(2, math.ceil(4 * common.l2_bytes() / (unique * 2 * h))))
        offsets = (torch.randint(0, (v - window) // 16 + 1, (launches,), generator=g, device="cuda") * 16).tolist()
        views = [(w[o:o + window], nf12.NF12Weight(nw.packed[o:o + window], nw.escapes, nw.exp_base))
                 for o in offsets]
        exact = all(torch.equal(nf12.embedding(ids, views[i][1]).view(torch.int16),
                                F.embedding(ids, views[i][0]).view(torch.int16)) for i in (0, launches - 1))
        written = 2 * h * num_ids
        point = {"num_ids": num_ids, "unique_rows": unique, "exact": exact}

        # Drop results so every launch reuses the same output allocation.
        def run_torch(i: int) -> None:
            F.embedding(ids, views[i][0])

        def run_nf12(i: int) -> None:
            nf12.embedding(ids, views[i][1])

        for op, fn, row_bytes in (("torch", run_torch, 2 * h), ("nf12", run_nf12, 3 * h)):
            t = common.time_graph(fn, launches, min_launches=2 * launches, budget_s=args.budget)
            point[op] = {**t, "bytes": unique * row_bytes + written,
                         "gbps": (unique * row_bytes + written) / t["median_us"] / 1e3}
        point["speedup"] = point["torch"]["median_us"] / point["nf12"]["median_us"]
        print(f"{name} ids={num_ids} ({launches} views): torch {point['torch']['median_us']:.2f}us "
              f"({point['torch']['gbps']:.0f} GB/s)  nf12 {point['nf12']['median_us']:.2f}us "
              f"({point['nf12']['gbps']:.0f} GB/s)  x{point['speedup']:.2f} exact={exact}", file=sys.stderr)
        entry["points"].append(point)
        del views
    del w, nw
    torch.cuda.empty_cache()
    return entry


def table(path: Path) -> str:
    r = json.loads(path.read_text())
    info = r["info"]
    lines = [
        f"## Embedding: {info['gpu']} (SM{info['sm']})",
        "",
        "Cold-table median us per lookup in a CUDA graph (effective GB/s of needed bytes). "
        "speed = F.embedding time / NF12 time. "
        f"GPU before: {common.state_summary(r.get('gpu_state_before'))}. "
        f"After: {common.state_summary(r.get('gpu_state_after'))}.",
        "",
        "| table | V x H | escapes | ids | F.embedding (BF16) | NF12 | speed | exact |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in r["tables"]:
        src = "random" if t["weight_source"].startswith("random") else "real"
        for p in t["points"]:
            lines.append(
                f"| {t['name']} ({src}) | {t['v']}x{t['h']} | {100 * t['escape_rate']:.4f}% | {p['num_ids']} | "
                f"{p['torch']['median_us']:.2f} ({p['torch']['gbps']:.0f}) | "
                f"{p['nf12']['median_us']:.2f} ({p['nf12']['gbps']:.0f}) | {p['speedup']:.2f}x | {p['exact']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path)
    ap.add_argument("--tables", nargs="*", help=f"subset of {[t[0] for t in TABLES]}")
    ap.add_argument("--num-ids", type=int, nargs="*", default=NUM_IDS)
    ap.add_argument("--fetch", action="store_true", help="range-download missing real tables")
    ap.add_argument("--budget", type=float, default=2.0, help="seconds of replays per measurement")
    ap.add_argument("--lock", help="flock file taken per table, e.g. /tmp/nf12-thor.lock")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.table:
        print(table(args.table))
        return
    nf12 = import_nf12()
    tag = common.machine_tag()
    tables = [t for t in TABLES if not args.tables or t[0] in args.tables]
    result = {"machine": tag, "info": common.machine_info(), "gpu_state_before": common.gpu_state(), "tables": []}
    out = Path(args.out) if args.out else common.RESULTS_DIR / f"embedding_{tag}.json"
    for t in tables:
        with common.gpu_lock(args.lock):
            result["tables"].append(bench_table(nf12, *t, args))
        common.write_json(out, result)
    result["gpu_state_after"] = common.gpu_state()
    result["throttle_after"] = common.throttled(result["gpu_state_after"])
    print(common.write_report(out, result, table))


if __name__ == "__main__":
    main()
