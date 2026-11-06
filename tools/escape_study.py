# SPDX-License-Identifier: Apache-2.0
"""NF12 escape study over real BF16 checkpoints.

    uv sync --group bench --no-install-project
    uv run --no-sync python tools/escape_study.py analyze --out results/ Qwen/Qwen3-8B ...
    uv run --no-sync python tools/escape_study.py report --out results/ > tables.md

`analyze` writes one JSON line per weight matrix to <out>/<model>.jsonl.
Grouping and base selection use nf12.reference, so the numbers describe
the encoder's output. Eligible matrices are 2-D BF16 with N % 16 == 0 and
K % 64 == 0. Stacked 3-D experts are split into eligible 2-D slices.
For each matrix, the script records:

- the chosen even base and its escaped-group count (SPEC 2, 3);
- the same count for alternative designs: a window starting at any integer
  exponent (the paper's power-of-two scale), the paper's group geometry
  (8 contiguous values of a row), and a separate base per 16-row block;
- what caused each escape at the chosen base (zeros, subnormals, small or
  large normals, Inf/NaN, sentinel collision);
- how often a warp's escape branch fires (atoms / super-atoms with an escape).

`report` aggregates the JSON lines into the markdown tables used by
docs/escape_study.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import nf12.reference as ref  # noqa: E402

NUM_EVEN = ref.MAX_EXP_BASE + 1  # even window starts 2B, B in [0, 120]
NUM_ANY = 241  # any window start s in [0, 240]
CHUNK_ELEMS = 1 << 25  # bounds per-worker memory on huge embeddings


def exp_range(groups: torch.Tensor):
    """int16 [G, 8] BF16 bit patterns -> (emin, emax, all positive), int64 [G]."""
    e = (groups.to(torch.int32) >> 7) & 0xFF
    return e.amin(1).long(), e.amax(1).long(), (groups >= 0).all(1)


def window_counts(emin, emax, pos, step, nstarts, block=None, nblocks=1):
    """Escaped groups for every window start s = step * i, i in [0, nstarts).

    A group fits [s, s+15] iff emax - 15 <= s <= emin, and collides with the
    meta sentinel iff it is all-positive with every exponent in {s, s+1},
    i.e. emax - 1 <= s <= emin. Both are intervals of i, so the counts are
    prefix sums of interval-endpoint histograms. Returns int64 [nblocks, nstarts].
    """
    width = nstarts + 1

    def interval_hist(lo_s, hi_s, mask):
        lo = (lo_s + step - 1).div(step, rounding_mode="floor").clamp(min=0)
        hi = hi_s.div(step, rounding_mode="floor").clamp(max=nstarts - 1)
        mask = mask & (lo <= hi)
        lo, hi = lo[mask], hi[mask]
        if block is not None:
            off = block[mask] * width
            lo, hi = lo + off, hi + off
        h = torch.bincount(lo, minlength=nblocks * width) - torch.bincount(hi + 1, minlength=nblocks * width)
        return h.view(nblocks, width).cumsum(1)[:, :nstarts]

    every = torch.ones_like(pos)
    fits = interval_hist(emax - 15, emin, every)
    sentinel = interval_hist(emax - 1, emin, pos)
    total = emin.numel() // nblocks
    return total - fits + sentinel


def analyze_matrix(w: torch.Tensor) -> dict:
    n, k = w.shape
    w16 = w.contiguous().view(torch.int16)
    rows_per_chunk = max(16, (CHUNK_ELEMS // k) // 16 * 16)
    chunks = [w16[r : r + rows_per_chunk] for r in range(0, n, rows_per_chunk)]
    gpb = k // 64 * 128  # groups per 16-row block

    # Pass 1: counts for every base / window start, per matrix and per block.
    even = torch.zeros(NUM_EVEN, dtype=torch.int64)
    anys = torch.zeros(NUM_ANY, dtype=torch.int64)
    contig = torch.zeros(NUM_ANY, dtype=torch.int64)
    block_even_opt = block_any_opt = 0
    block_bases = []
    for c in chunks:
        g = ref.to_groups(c)
        emin, emax, pos = exp_range(g)
        nb = c.shape[0] // 16
        block = torch.arange(emin.numel()) // gpb
        e_blk = window_counts(emin, emax, pos, 2, NUM_EVEN, block, nb)
        a_blk = window_counts(emin, emax, pos, 1, NUM_ANY, block, nb)
        even += e_blk.sum(0)
        anys += a_blk.sum(0)
        block_even_opt += int(e_blk.min(1).values.sum())
        block_any_opt += int(a_blk.min(1).values.sum())
        block_bases.append(e_blk.argmin(1))
        # Paper geometry: 8 consecutive values of a row.
        cmin, cmax, cpos = exp_range(c.reshape(-1, 8))
        contig += window_counts(cmin, cmax, cpos, 1, NUM_ANY)[0]
    base = int(even.argmin())
    block_bases = torch.cat(block_bases)

    # Pass 2: escape causes at the chosen base.
    lo_e, hi_e = 2 * base, 2 * base + 15
    cause = collections.Counter()
    atoms_hit = super_atoms_hit = max_block = 0
    zeros = subnormals = 0
    for c in chunks:
        g = ref.to_groups(c).to(torch.int32) & 0xFFFF
        e = (g >> 7) & 0xFF
        mag = g & 0x7FFF
        is_zero = mag == 0
        is_sub = (e == 0) & ~is_zero
        zeros += int(is_zero.sum())
        subnormals += int(is_sub.sum())
        below, above = e < lo_e, e > hi_e
        out = below | above
        meta_zero = (g < 0x8000).all(1) & ((e - lo_e) >> 1 == 0).all(1)
        esc = out.any(1) | meta_zero
        o, z, s = out[esc], is_zero[esc], is_sub[esc]
        cause["sentinel"] += int((~o.any(1)).sum())
        cause["with_zero"] += int((o & z).any(1).sum())
        cause["with_subnormal"] += int((o & s).any(1).sum())
        cause["with_small_normal"] += int((below[esc] & ~z & ~s).any(1).sum())
        cause["with_large"] += int((above[esc] & (e[esc] != 255)).any(1).sum())
        cause["with_nonfinite"] += int((e[esc] == 255).any(1).sum())
        cause["only_zero"] += int((o.any(1) & ~(o & ~z).any(1)).sum())
        cause["only_zero_or_subnormal"] += int((o.any(1) & ~(o & ~z & ~s).any(1)).sum())
        atoms_hit += int(esc.view(-1, 32).any(1).sum())
        super_atoms_hit += int(esc.view(-1, 128).any(1).sum())
        max_block = max(max_block, int(esc.view(-1, gpb).sum(1).max()))

    groups = n * k // 8
    rec = {
        "shape": [n, k],
        "groups": groups,
        "base": base,
        "esc": int(even[base]),
        "esc_any": int(anys.min()),
        "any_start": int(anys.argmin()),
        "esc_contig_any": int(contig.min()),
        "esc_block_even": block_even_opt,
        "esc_block_any": block_any_opt,
        "blocks_off_base": int((block_bases != base).sum()),
        "atoms_hit": atoms_hit,
        "super_atoms_hit": super_atoms_hit,
        "max_block_esc": max_block,
        "zeros": zeros,
        "subnormals": subnormals,
        "cause": dict(cause),
    }
    return rec


def self_check(w: torch.Tensor, rec: dict) -> None:
    """The fast counters must agree with the reference encoder."""
    counts = ref.escape_counts_all_bases(w)
    assert int(counts.argmin()) == rec["base"], "base mismatch vs reference"
    assert int(counts[rec["base"]]) == rec["esc"], "escape count mismatch vs reference"
    _, escapes, _ = ref.encode(w, rec["base"])
    assert escapes.shape[0] == rec["esc"], "escape count mismatch vs encode()"


def tensor_type(name: str) -> str:
    parts = name.split(".")
    leaf = parts[-2] if parts[-1] in ("weight", "bias") and len(parts) > 1 else parts[-1]
    if leaf in ("router", "gate") and "experts" not in name:
        leaf = "router"
    prefix = ""
    if re.search(r"(^|\.)(visual|vision|vision_tower|vision_model|mm_projector)\.", name):
        prefix = "vision."
    elif ".experts." in name:
        prefix = "expert."
    elif "shared_expert" in name:
        prefix = "shared_expert."
    return prefix + leaf


def resolve(model: str) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.safetensors", "*.json"], local_files_only=True))


def _work(task):
    file, name, index = task
    from safetensors import safe_open

    with safe_open(file, framework="pt") as f:
        t = f.get_tensor(name)
    w = t if index is None else t[index]
    rec = analyze_matrix(w)
    if os.environ.get("NF12_STUDY_SELFCHECK") and w.numel() <= (1 << 24):
        self_check(w, rec)
    rec["name"] = name if index is None else f"{name}[{index}]"
    rec["type"] = tensor_type(name)
    return rec


def collect_tasks(root: Path):
    from safetensors import safe_open

    tasks, skipped = [], []
    for file in sorted(root.glob("*.safetensors")):
        with safe_open(str(file), framework="pt") as f:
            for name in f.keys():
                sl = f.get_slice(name)
                shape, dtype = sl.get_shape(), sl.get_dtype()
                numel = 1
                for d in shape:
                    numel *= d
                reason = None
                if dtype != "BF16":
                    reason = f"dtype {dtype}"
                elif len(shape) < 2:
                    reason = "1-D"
                elif len(shape) > 3:
                    reason = f"{len(shape)}-D"
                elif shape[-2] % 16 or shape[-1] % 64:
                    reason = "shape not N%16, K%64"
                if reason:
                    skipped.append({"name": name, "shape": shape, "numel": numel, "reason": reason,
                                    "type": tensor_type(name)})
                elif len(shape) == 2:
                    tasks.append((str(file), name, None, numel))
                else:
                    tasks += [(str(file), name, i, numel // shape[0]) for i in range(shape[0])]
    tasks.sort(key=lambda t: -t[3])
    return [t[:3] for t in tasks], skipped


def cmd_analyze(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        root = resolve(model)
        tasks, skipped = collect_tasks(root)
        dest = out / (model.replace("/", "__") + ".jsonl")
        print(f"{model}: {len(tasks)} matrices, {len(skipped)} skipped tensors", file=sys.stderr, flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=torch.set_num_threads, initargs=(args.threads,)) as pool, \
                open(dest, "w") as fh:
            fh.write(json.dumps({"model": model, "skipped": skipped}) + "\n")
            for i, rec in enumerate(pool.imap_unordered(_work, tasks, chunksize=1)):
                fh.write(json.dumps(rec) + "\n")
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(tasks)}", file=sys.stderr, flush=True)


def load(out: Path):
    models = []
    for path in sorted(out.glob("*.jsonl")):
        with open(path) as fh:
            head = json.loads(fh.readline())
            recs = [json.loads(line) for line in fh]
        models.append((head["model"], head["skipped"], recs))
    return models


def pct(a, b, digits=3):
    return f"{100.0 * a / b:.{digits}f}%" if b else "-"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def cmd_report(args):
    models = load(Path(args.out))
    order = {m: i for i, m in enumerate(args.models)} if args.models else {}
    models.sort(key=lambda m: order.get(m[0], len(order)))
    short = {m: m.split("/")[-1] for m, _, _ in models}
    s = lambda recs, key: sum(r[key] for r in recs)  # noqa: E731
    out = []

    out.append("### Summary per model\n")
    rows = []
    for m, skipped, recs in models:
        g = s(recs, "groups")
        params = 8 * g
        skipped_1d = sum(x["numel"] for x in skipped if x["reason"] == "1-D")
        skipped_other = [x for x in skipped if x["reason"] != "1-D"]
        e = s(recs, "esc")
        rows.append([
            short[m], f"{params / 1e9:.2f}B", len(recs), pct(e, g), f"{12 + 16 * e / g:.4f}",
            pct(s(recs, "esc_any"), g), pct(s(recs, "esc_contig_any"), g),
            pct(s(recs, "esc_block_even"), g), pct(sum(r["cause"].get("sentinel", 0) for r in recs), g, 4),
            pct(s(recs, "atoms_hit"), g // 32, 2), pct(s(recs, "super_atoms_hit"), g // 128, 2),
            f"{skipped_1d / 1e6:.2f}M",
            ", ".join(sorted({f"{x['reason']}" for x in skipped_other})) or "-",
        ])
    out.append(table(
        ["model", "NF12 params", "matrices", "escape rate", "bits/weight", "any-start", "paper geom.",
         "per-block base", "sentinel", "atoms w/ esc", "super-atoms w/ esc", "1-D params", "other skipped"],
        rows))

    out.append("\n### Escape rate per tensor type\n")
    types = collections.OrderedDict()
    for _, _, recs in models:
        for r in recs:
            types.setdefault(r["type"], None)
    rows = []
    for t in types:
        row = [t]
        for m, _, recs in models:
            sel = [r for r in recs if r["type"] == t]
            row.append(pct(s(sel, "esc"), s(sel, "groups")) if sel else "")
        rows.append(row)
    out.append(table(["type"] + [short[m] for m, _, _ in models], rows))

    out.append("\n### Chosen exponent base (matrices per base)\n")
    all_bases = sorted({r["base"] for _, _, recs in models for r in recs})
    rows = []
    for m, _, recs in models:
        c = collections.Counter(r["base"] for r in recs)
        rows.append([short[m]] + [c.get(b, "") for b in all_bases])
    out.append(table(["model"] + [f"B={b}" for b in all_bases], rows))

    out.append("\n### Chosen base per tensor type (median, min-max)\n")
    rows = []
    for t in types:
        row = [t]
        for m, _, recs in models:
            b = sorted(r["base"] for r in recs if r["type"] == t)
            row.append(f"{b[len(b) // 2]} ({b[0]}-{b[-1]})" if b else "")
        rows.append(row)
    out.append(table(["type"] + [short[m] for m, _, _ in models], rows))

    out.append("\n### Design alternatives (escaped groups relative to ours)\n")
    rows = []
    for m, _, recs in models:
        e = s(recs, "esc")
        g = s(recs, "groups")
        rows.append([
            short[m], pct(e, g), pct(s(recs, "esc_any"), g), pct(s(recs, "esc_contig_any"), g),
            pct(s(recs, "esc_block_even"), g), pct(s(recs, "esc_block_any"), g),
            pct(s(recs, "blocks_off_base"), sum(r["shape"][0] // 16 for r in recs), 1),
            f"{s(recs, 'esc_any') / e:.2f}x" if e else "-",
            f"{s(recs, 'esc_block_even') / e:.2f}x" if e else "-",
        ])
    out.append(table(
        ["model", "ours: even start, per matrix", "any start, per matrix", "any start, row-contiguous groups",
         "even start, per 16-row block", "any start, per 16-row block", "blocks preferring another base",
         "any/ours", "per-block/ours"],
        rows))

    out.append("\n### Escape causes at the chosen base (share of escaped groups)\n")
    keys = ["with_zero", "only_zero", "with_subnormal", "only_zero_or_subnormal", "with_small_normal",
            "with_large", "with_nonfinite", "sentinel"]
    rows = []
    for m, _, recs in models:
        e = s(recs, "esc")
        z = s(recs, "zeros")
        sub = s(recs, "subnormals")
        g = s(recs, "groups")
        rows.append([short[m], e] + [pct(sum(r["cause"].get(k, 0) for r in recs), e, 1) for k in keys]
                    + [pct(z, 8 * g, 4), pct(sub, 8 * g, 5)])
    out.append(table(["model", "escaped groups"] + keys + ["zeros / weights", "subnormals / weights"], rows))

    out.append(f"\n### Worst {args.worst} matrices by escape rate (>= 1M weights)\n")
    allrecs = [(m, r) for m, _, recs in models for r in recs if r["groups"] * 8 >= (1 << 20)]
    allrecs.sort(key=lambda mr: -mr[1]["esc"] / mr[1]["groups"])
    rows = []
    for m, r in allrecs[: args.worst]:
        g = r["groups"]
        c = r["cause"]
        rows.append([short[m], f"`{r['name']}`", "x".join(map(str, r["shape"])), r["base"], pct(r["esc"], g),
                     pct(r["esc_any"], g), pct(r["esc_block_even"], g), r["max_block_esc"],
                     pct(c.get("only_zero_or_subnormal", 0), r["esc"], 0) if r["esc"] else "-"])
    out.append(table(["model", "tensor", "shape", "B", "escape rate", "any-start", "per-block", "max esc / block",
                      "zero/subnormal-only"], rows))

    out.append("\n### Excluded tensors\n")
    rows = []
    for m, skipped, _ in models:
        by = collections.defaultdict(lambda: [0, 0])
        for x in skipped:
            key = (x["reason"], x["type"])
            by[key][0] += 1
            by[key][1] += x["numel"]
        for (reason, t), (cnt, numel) in sorted(by.items()):
            rows.append([short[m], reason, t, cnt, f"{numel / 1e6:.2f}M"])
    out.append(table(["model", "reason", "type", "tensors", "params"], rows))
    print("\n".join(out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("models", nargs="+", help="HF repo ids (cached) or local checkpoint dirs")
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    a.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--worst", type=int, default=20)
    r.add_argument("models", nargs="*", help="optional model order")
    args = p.parse_args()
    cmd_analyze(args) if args.cmd == "analyze" else cmd_report(args)


if __name__ == "__main__":
    main()
