# SPDX-License-Identifier: Apache-2.0
"""NF12 escape study of mixture-of-experts weights, per expert (SPEC.md 9).

    uv sync --group bench --no-install-project
    uv run --no-sync python tools/moe_escape_study.py analyze --out results/moe_escape qwen3-30b-a3b dsv2-lite mixtral-8x7b
    uv run --no-sync python tools/moe_escape_study.py report --out results/moe_escape > tables.md
    uv run --no-sync python tools/moe_escape_study.py export --dest /workspace/hf/nf12-moe-real qwen3-30b-a3b:0:w13 ...

Experts are the stacked projections the grouped kernels read (bench/bench_moe.py
MODELS): ``w13`` = gate rows then up rows [2I, H], ``w2`` = down [H, I]. Each
expert is encoded standalone with its own base (SPEC 9.1). ``analyze`` reads
every MoE layer present in the local HF cache (layers whose shards are missing
are skipped) and writes one JSON line per (layer, projection) to
<out>/<model>.jsonl with, per expert:

- its own base and escaped-group count, and the count under one shared base
  per layer (the base minimizing the layer's total), to size what per-expert
  bases buy;
- atoms (32 groups, one warp's A fragments) and super-atoms (128 groups,
  16 rows x 64 k) holding at least one escape, at both bases: how often a
  kernel's escape branch fires;
- the largest escape count in one 16-row block (one CTA row tile's table walk).

Counts come from nf12.reference.escape_counts_all_bases (the encoder's own
base search) and every per-group escape mask is checked against it.

``export`` writes real expert stacks as safetensors (``weight``: BF16
[E, N, K]) named <model>.L<layer>.<proj>.safetensors, which bench_moe.py and
the kernel tests load from NF12_MOE_REAL_DIR.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "bench"))
import nf12.reference as ref  # noqa: E402
from bench_moe import MODELS, PROJS, load_real_stack  # noqa: E402


def escaped_mask(gbits: torch.Tensor, base: int) -> torch.Tensor:
    """bool [G]: groups escaped at ``base`` (SPEC 3: outside the window, or the all-zero meta sentinel)."""
    off = ((gbits >> 7) & 0xFF) - 2 * base
    in_window = ((off >= 0) & (off <= 15)).all(1)
    sentinel = (gbits < 0x8000).all(1) & ((off >> 1) == 0).all(1)
    return ~in_window | sentinel


def hits(esc: torch.Tensor, k: int) -> dict:
    gpb = k // 64 * 128  # groups per 16-row block
    return {"atoms": int(esc.view(-1, 32).any(1).sum()), "super_atoms": int(esc.view(-1, 128).any(1).sum()),
            "max_block": int(esc.view(-1, gpb).sum(1).max())}


def analyze_stack(stack: torch.Tensor) -> dict:
    """Per-expert statistics of one [E, N, K] BF16 stack."""
    e, n, k = stack.shape
    vecs = torch.stack([ref.escape_counts_all_bases(stack[i]) for i in range(e)])  # [E, 121]
    own = vecs.argmin(1)  # first minimum = smallest base, as the encoder
    shared = int(vecs.sum(0).argmin())
    experts = []
    for i in range(e):
        gbits = ref.to_groups(stack[i].contiguous().view(torch.int16)).to(torch.int32) & 0xFFFF
        rec = {"base": int(own[i]), "esc": int(vecs[i, own[i]]), "esc_shared": int(vecs[i, shared])}
        for tag, b in (("", int(own[i])), ("_shared", shared)):
            m = escaped_mask(gbits, b)
            if int(m.sum()) != int(vecs[i, b]):
                raise AssertionError(f"escape mask disagrees with the reference count at base {b}")
            rec.update({f"{key}{tag}": v for key, v in hits(m, k).items()})
        experts.append(rec)
    return {"shape": [e, n, k], "groups": n * k // 8, "super_atoms": n * k // 1024, "atoms": n * k // 256,
            "shared_base": shared, "experts": experts}


def _work(task: tuple[str, int, str]) -> dict | None:
    key, layer, proj = task
    torch.set_num_threads(4)
    stack = load_real_stack(MODELS[key], layer, proj)
    if stack is None:
        return None
    rec = analyze_stack(stack)
    rec.update(model=key, layer=layer, proj=proj)
    return rec


def analyze(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        model = MODELS[key]
        tasks = [(key, layer, proj) for layer in model.moe_layers for proj in PROJS]
        path = out / f"{key}.jsonl"
        n = 0
        with path.open("w") as f, mp.get_context("spawn").Pool(args.jobs) as pool:
            for rec in pool.imap(_work, tasks):
                if rec is None:
                    continue
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n += 1
                print(f"{key} L{rec['layer']} {rec['proj']}: shared base {rec['shared_base']}, "
                      f"max expert esc {max(x['esc'] for x in rec['experts']) / rec['groups']:.4%}", file=sys.stderr)
        print(f"{key}: {n} of {len(tasks)} (layer, projection) stacks analyzed -> {path}", file=sys.stderr)


def pct(x: float) -> str:
    return f"{100 * x:.3f}%"


def quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def load_records(out: Path) -> dict[str, list[dict]]:
    recs: dict[str, list[dict]] = {}
    for p in sorted(out.glob("*.jsonl")):
        recs[p.stem] = [json.loads(line) for line in p.read_text().splitlines() if line]
    return recs


def report(args: argparse.Namespace) -> None:
    recs = load_records(Path(args.out))
    lines = [
        "### Escape rate per expert",
        "",
        "Escaped groups / groups for each expert matrix at its own base. Quantiles are over all experts of all "
        "analyzed MoE layers. *Super-atoms hit*: fraction of 16x64 super-atoms with at least one escape.",
        "",
        "| model | proj | layers | experts | aggregate | p50 | p90 | p99 | max | experts > 1% "
        "| super-atoms hit (aggregate) | super-atoms hit (worst expert) | max escapes in one 16-row block |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, rs in recs.items():
        for proj in PROJS:
            sel = [r for r in rs if r["proj"] == proj]
            if not sel:
                continue
            rates = [x["esc"] / r["groups"] for r in sel for x in r["experts"]]
            sa = [x["super_atoms"] / r["super_atoms"] for r in sel for x in r["experts"]]
            tot_esc = sum(x["esc"] for r in sel for x in r["experts"])
            tot_groups = sum(r["groups"] * len(r["experts"]) for r in sel)
            tot_sa = sum(x["super_atoms"] for r in sel for x in r["experts"])
            tot_sa_all = sum(r["super_atoms"] * len(r["experts"]) for r in sel)
            lines.append(
                f"| {MODELS[key].name} | {proj} | {len(sel)} | {len(rates)} | {pct(tot_esc / tot_groups)} | "
                f"{pct(quantile(rates, 0.5))} | {pct(quantile(rates, 0.9))} | {pct(quantile(rates, 0.99))} | "
                f"{pct(max(rates))} | {sum(r > 0.01 for r in rates)} | {pct(tot_sa / tot_sa_all)} | {pct(max(sa))} | "
                f"{max(x['max_block'] for r in sel for x in r['experts'])} |")

    lines += [
        "",
        "### Per-expert base vs one shared base per layer",
        "",
        "Shared base: the one base per (layer, projection) minimizing the layer's escapes. *Extra escapes*: "
        "total escapes with the shared base over total with per-expert bases. *Worst expert*: the largest "
        "single-expert escape rate under the shared base (vs its own base).",
        "",
        "| model | proj | experts off the shared base | extra escapes | worst expert, shared base (own) "
        "| super-atoms hit, shared (own) |",
        "|---|---|---|---|---|---|",
    ]
    for key, rs in recs.items():
        for proj in PROJS:
            sel = [r for r in rs if r["proj"] == proj]
            if not sel:
                continue
            ex = [(r, x) for r in sel for x in r["experts"]]
            off = sum(x["base"] != r["shared_base"] for r, x in ex)
            own = sum(x["esc"] for _, x in ex)
            shared = sum(x["esc_shared"] for _, x in ex)
            r_w, x_w = max(ex, key=lambda rx: rx[1]["esc_shared"] / rx[0]["groups"])
            sa_own = sum(x["super_atoms"] for _, x in ex)
            sa_sh = sum(x["super_atoms_shared"] for _, x in ex)
            sa_all = sum(r["super_atoms"] for r, _ in ex)
            lines.append(
                f"| {MODELS[key].name} | {proj} | {off} of {len(ex)} | "
                f"{shared / own:.2f}x ({own} -> {shared}) | {pct(x_w['esc_shared'] / r_w['groups'])} "
                f"({pct(x_w['esc'] / r_w['groups'])}) | {pct(sa_sh / sa_all)} ({pct(sa_own / sa_all)}) |")

    lines += [
        "",
        "### Escape rate by layer (max over experts, own base)",
        "",
    ]
    for key, rs in recs.items():
        layers = sorted({r["layer"] for r in rs})
        lines += [f"{MODELS[key].name}:", "", "| proj | " + " | ".join(f"L{la}" for la in layers) + " |",
                  "|---" * (len(layers) + 1) + "|"]
        for proj in PROJS:
            by = {r["layer"]: r for r in rs if r["proj"] == proj}
            cells = []
            for la in layers:
                r = by.get(la)
                cells.append(f"{100 * max(x['esc'] for x in r['experts']) / r['groups']:.2f}" if r else "")
            lines.append(f"| {proj} max % | " + " | ".join(cells) + " |")
            cells = []
            for la in layers:
                r = by.get(la)
                cells.append(f"{100 * statistics.mean(x['esc'] for x in r['experts']) / r['groups']:.2f}"
                             if r else "")
            lines.append(f"| {proj} mean % | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["### Worst experts", "",
              "| model | layer | proj | expert | base (layer shared) | escape rate | super-atoms hit | "
              "max escapes in one 16-row block |", "|---|---|---|---|---|---|---|---|"]
    allx = [(key, r, i, x) for key, rs in recs.items() for r in rs for i, x in enumerate(r["experts"])]
    allx.sort(key=lambda t: -t[3]["esc"] / t[1]["groups"])
    for key, r, i, x in allx[: args.worst]:
        lines.append(f"| {MODELS[key].name} | {r['layer']} | {r['proj']} | {i} | {x['base']} ({r['shared_base']}) | "
                     f"{pct(x['esc'] / r['groups'])} | {pct(x['super_atoms'] / r['super_atoms'])} | "
                     f"{x['max_block']} |")
    print("\n".join(lines))


def export(args: argparse.Namespace) -> None:
    from safetensors.torch import save_file

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for spec in args.specs:
        key, layer, proj = spec.split(":")
        model = MODELS[key]
        stack = load_real_stack(model, int(layer), proj, fetch=args.fetch)
        if stack is None:
            raise SystemExit(f"{spec}: tensors not in the local HF cache (--fetch downloads them)")
        path = dest / f"{key}.L{layer}.{proj}.safetensors"
        save_file({"weight": stack.contiguous()}, str(path),
                  metadata={"repo": model.repo or "", "layer": layer, "proj": proj,
                            "tensors": ",".join(model.expert_tensors(int(layer), "{e}", proj))})
        print(f"{path}: {tuple(stack.shape)}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("models", nargs="+", choices=list(MODELS))
    a.add_argument("--out", required=True)
    a.add_argument("--jobs", type=int, default=16)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--worst", type=int, default=12)
    e = sub.add_parser("export")
    e.add_argument("specs", nargs="+", help="model:layer:proj, e.g. qwen3-30b-a3b:0:w13")
    e.add_argument("--dest", required=True)
    e.add_argument("--fetch", action="store_true",
                   help="range-download tensors missing from the HF cache (into bench common.TENSOR_CACHE)")
    args = ap.parse_args()
    {"analyze": analyze, "report": report, "export": export}[args.cmd](args)


if __name__ == "__main__":
    main()
