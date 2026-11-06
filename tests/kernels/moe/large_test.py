# SPDX-License-Identifier: Apache-2.0
"""Exactness suite for the grouped (MoE) prefill families (SPEC.md section 9).

For each config: NF12 output must match the bf16t twin without tolerance, match an
FP32 reference within the BF16 output rounding plus an FP32 accumulation-order
bound, and leave every row of y not named by a valid sorted position (and the
ldy padding) untouched. Covers skewed, empty and dropped routing, every
rows-per-expert count 1..300, gather/scatter/row-scale combinations, ld
padding, light and heavy escapes (incl. Inf/NaN, bitwise), CUDA-graph replay
with new routing, and (NF12_HUGE_TESTS=1) 64-bit offsets.

Run: uv run --group test python tests/kernels/moe/large_test.py [--family mma|sm120|sm90|all] [--quick] (also collected by pytest).
"""

import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import large_harness as H  # noqa: E402

_SHIM = None
SENTINEL = 0x7FC1  # a NaN payload no kernel produces


def shim():
    global _SHIM
    if _SHIM is None:
        _SHIM = H.Shim()
    return _SHIM


def make_weights(E, n, k, seed=0, escapes=0.0, specials=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(E, n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    if escapes:
        w = torch.stack([H.LH.force_escapes(w[e], escapes, seed + e, specials) for e in range(E)])
    return w, H.MoeWeights(w)


class Routing:
    """Expert-major sorted positions mapped to x/y rows in in_rows/out_rows."""

    def __init__(self, offsets, max_assignments, gather=None, gather_div=1, scatter=None):
        self.offsets = offsets
        self.A = max_assignments
        self.gather, self.gather_div, self.scatter = gather, gather_div, scatter
        valid = int(offsets[-1].item())
        p = torch.arange(valid, device="cuda", dtype=torch.int64)
        self.in_rows = (gather[:valid].to(torch.int64) // gather_div) if gather is not None else p
        self.out_rows = scatter[:valid].to(torch.int64) if scatter is not None else p
        self.expert = torch.repeat_interleave(torch.arange(len(offsets) - 1, device="cuda"),
                                              (offsets[1:] - offsets[:-1]))


def topk_routing(T, topk, E, *, seed=0, skew=None, drop=0.0, mode="w13"):
    """Random top-k routing through moe_align. w13 gathers token rows using
    sorted / topk and writes sorted rows. w2 reads sorted rows and scatters
    into assignment order. Positions past offsets[E] hold garbage."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if skew is None:
        scores = torch.rand(T, E, device="cuda", generator=g)
    else:  # expert e drawn with weight skew**e: a few hot experts, many cold or empty ones
        scores = torch.rand(T, E, device="cuda", generator=g).log() / (skew ** torch.arange(E, device="cuda"))
    ids = scores.topk(topk, dim=1).indices.to(torch.int32)
    if drop:
        ids[torch.rand(T, topk, device="cuda", generator=g) < drop] = -1
    sorted_ids, offsets = H.align(ids, E)
    A = T * topk
    valid = int(offsets[-1].item())
    sorted_ids[valid:] = torch.randint(-(2**31), 2**31 - 1, (A - valid,), device="cuda", generator=g,
                                       dtype=torch.int32)
    if mode == "w13":
        return Routing(offsets, A, gather=sorted_ids, gather_div=topk)
    return Routing(offsets, A, scatter=sorted_ids)


def count_routing(counts, *, gather=False, scatter=False, slack=0, seed=0):
    """Explicit per-expert counts with optional permuted gather/scatter maps.
    `slack` adds garbage positions past offsets[E]."""
    counts = torch.tensor(counts, dtype=torch.int64, device="cuda")
    offsets = torch.zeros(len(counts) + 1, dtype=torch.int64, device="cuda")
    offsets[1:] = torch.cumsum(counts, 0)
    valid = int(offsets[-1].item())
    A = valid + slack
    g = torch.Generator(device="cuda").manual_seed(seed)
    garbage = lambda: torch.randint(-(2**31), 2**31 - 1, (slack,), device="cuda", generator=g, dtype=torch.int32)  # noqa: E731
    gi = sc = None
    if gather:
        gi = torch.cat([torch.randperm(valid, device="cuda", generator=g).to(torch.int32) * 3 + 1, garbage()])
    if scatter:
        sc = torch.cat([torch.randperm(valid, device="cuda", generator=g).to(torch.int32), garbage()])
    return Routing(offsets, A, gather=gi, gather_div=3 if gather else 1, scatter=sc)


def x_rows(r):
    return int(r.in_rows.max().item()) + 1 if r.in_rows.numel() else 1


def y_rows(r):
    return max(r.A, int(r.out_rows.max().item()) + 1 if r.out_rows.numel() else 1)


def make_x(rows, k, ldx=None, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn(rows, ldx or k, device="cuda", generator=g).to(torch.bfloat16)
    return buf[:, :k]


def reference(x, w, r, scale):
    """FP32 reference and tolerance over the valid sorted positions (in position order)."""
    n = w.shape[1]
    ref = torch.empty(r.in_rows.numel(), n, device="cuda")
    tol = torch.empty_like(ref)
    off = r.offsets.tolist()
    for e in range(w.shape[0]):
        if off[e + 1] == off[e]:
            continue
        xs = x[r.in_rows[off[e]:off[e + 1]]].float()
        wf = w[e].float()
        v = xs @ wf.T
        slack = (xs.abs() @ wf.abs().T) * (x.shape[1] * 2.0**-23) + 1e-30
        if scale is not None:
            s = scale[r.out_rows[off[e]:off[e + 1]]].float()[:, None]
            v, slack = v * s, slack * s.abs()
        ref[off[e]:off[e + 1]] = v
        tol[off[e]:off[e + 1]] = v.abs() * 2.0**-8 + slack
    return ref, tol


def run(family, config, x, mw, r, scale=None, ldy=None):
    """Return sentinel-filled (y_nf12, y_bf16t) buffers for both sources."""
    outs = []
    rows = y_rows(r)
    for source in ("nf12", "bf16t"):
        ybuf = torch.full((rows, ldy or mw.n), SENTINEL, dtype=torch.int16, device="cuda").view(torch.bfloat16)
        shim().run(x, mw, r.offsets, r.A, ybuf[:, :mw.n], gather=r.gather, gather_div=r.gather_div,
                   scatter=r.scatter, row_scale=scale, source=source, family=family, config=config)
        outs.append(ybuf)
    return outs


def check(family, config, w, mw, r, *, scale=False, ldx=None, ldy=None, tag="", bitwise_only=False):
    x = make_x(x_rows(r), mw.k, ldx)
    sc = None
    if scale:
        sc = torch.rand(y_rows(r), device="cuda") * 2 - 0.5
    y_nf12, y_t = run(family, config, x, mw, r, sc, ldy)
    tag = f"family={family} config={config} {tag} E={mw.E} n={mw.n} k={mw.k} A={r.A} scale={scale} ldx={ldx} ldy={ldy}"
    assert torch.equal(y_nf12.view(torch.int16), y_t.view(torch.int16)), f"NF12 != bf16t bits: {tag}"
    written = torch.zeros(y_nf12.shape[0], dtype=torch.bool, device="cuda")
    written[r.out_rows] = True
    yi = y_nf12.view(torch.int16)
    assert (yi[~written] == SENTINEL).all(), f"wrote rows outside out(p): {tag}"
    assert (yi[:, mw.n:] == SENTINEL).all(), f"wrote into the ldy padding: {tag}"
    if bitwise_only or r.out_rows.numel() == 0:
        return
    ref, tol = reference(x, w, r, sc)
    got = y_nf12[r.out_rows, :mw.n].float()
    err = (got - ref).abs()
    bad = ~(err <= tol)
    assert not bad.any(), f"tolerance: {tag} max err {err.max().item():.3e} at {bad.nonzero()[0].tolist()}"


def check_graph(family, config, E=16, n=272, k=512):
    """Captured once, replayed with new routing written into the same buffers."""
    w, mw = make_weights(E, n, k, seed=5)
    T, topk = 300, 4
    r1 = topk_routing(T, topk, E, seed=1, mode="w2")
    r2 = topk_routing(T, topk, E, seed=2, skew=1.5, drop=0.2, mode="w2")
    x = make_x(r1.A, k)
    sc = torch.rand(r1.A, device="cuda")
    offsets, scatter = r1.offsets.clone(), r1.scatter.clone()
    y = torch.zeros(r1.A, n, dtype=torch.bfloat16, device="cuda")
    call = lambda: shim().run(x, mw, offsets, r1.A, y, scatter=scatter, row_scale=sc, family=family, config=config)  # noqa: E731
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()  # warm-up: sets the kernel attributes outside capture
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for r in (r2, r1):
        offsets.copy_(r.offsets)
        scatter.copy_(r.scatter)
        y.fill_(0)
        graph.replay()
        ref = torch.zeros_like(y)
        shim().run(x, mw, r.offsets, r.A, ref, scatter=r.scatter, row_scale=sc, family=family, config=config)
        torch.cuda.synchronize()
        assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), f"graph replay differs: config {config}"


def check_huge(family, config):
    """64-bit offsets: y rows beyond 2^31 elements, x rows beyond 2^31 elements,
    and expert weight offsets beyond 2^31 bytes."""
    # 12 experts x 4096 x 32768 x 1.5 B: expert 11 starts past 2^31 bytes.
    E, n, k = 12, 4096, 32768
    g = torch.Generator(device="cuda").manual_seed(9)
    w = (torch.randn(E, n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    mw = H.MoeWeights(w)
    mw.bf16t = None  # only NF12 is checked here
    offs = torch.tensor([0] * 11 + [5, 75], dtype=torch.int64, device="cuda")
    A = 75
    # x rows / y rows placed past 2^31 elements: index * ld > 2^31.
    x_full = torch.zeros(70000, k, dtype=torch.bfloat16, device="cuda")  # 4.6 GB
    gather = torch.randint(66000, 70000, (A,), dtype=torch.int32, device="cuda", generator=g)
    x_full[gather.long()] = torch.randn(A, k, device="cuda", generator=g).to(torch.bfloat16)
    y_full = torch.full((600000, n), SENTINEL, dtype=torch.int16, device="cuda").view(torch.bfloat16)  # 4.9 GB
    scatter = torch.randperm(80000, device="cuda", generator=g)[:A].to(torch.int32) + 520000
    shim().run(x_full, mw, offs, A, y_full, gather=gather, gather_div=1, scatter=scatter, source="nf12",
               family=family, config=config)
    torch.cuda.synchronize()
    for e, (p0, p1) in ((10, (0, 5)), (11, (5, 75))):
        xs = x_full[gather[p0:p1].long()].float()
        ref = xs @ w[e].float().T
        got = y_full[scatter[p0:p1].long()].float()
        tol = ref.abs() * 2.0**-8 + (xs.abs() @ w[e].float().abs().T) * (k * 2.0**-23) + 1e-30
        assert ((got - ref).abs() <= tol).all(), f"huge: expert {e} config {config}"


def check_huge_contiguous(family, config):
    """Without gather, only expert 11 positions [66000, 66075) are valid.
    Their x offsets exceed 2^31 elements, and x ends at the last valid position."""
    E, n, k = 12, 4096, 32768
    g = torch.Generator(device="cuda").manual_seed(10)
    w = (torch.randn(E, n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    mw = H.MoeWeights(w)
    mw.bf16t = None  # only NF12 is checked here
    p0, A = 66000, 66075
    x_rows = torch.zeros(A, k, dtype=torch.bfloat16, device="cuda")  # 4.3 GB
    x_rows[p0:] = torch.randn(A - p0, k, device="cuda", generator=g).to(torch.bfloat16)
    offs = torch.tensor([p0] * 12 + [A], dtype=torch.int64, device="cuda")
    y_full = torch.full((600000, n), SENTINEL, dtype=torch.int16, device="cuda").view(torch.bfloat16)  # 4.9 GB
    scatter = torch.cat([torch.zeros(p0, dtype=torch.int32, device="cuda"),
                         torch.randperm(80000, device="cuda", generator=g)[:A - p0].to(torch.int32) + 520000])
    shim().run(x_rows, mw, offs, A, y_full, scatter=scatter, source="nf12", family=family, config=config)
    torch.cuda.synchronize()
    xs = x_rows[p0:].float()
    ref = xs @ w[11].float().T
    got = y_full[scatter[p0:].long()].float()
    tol = ref.abs() * 2.0**-8 + (xs.abs() @ w[11].float().abs().T) * (k * 2.0**-23) + 1e-30
    assert ((got - ref).abs() <= tol).all(), f"huge, no gather: config {config}"


def configs_for(family, quick):
    s = shim()
    if not s.supported(family, 1):
        return []
    cs = list(range(s.num_configs(family)))
    return cs[: max(1, len(cs) // 2)] if quick else cs


def cases(quick):
    """(name, builder) pairs returning (w, mw, routing, kwargs)."""
    out = []
    W = {}

    def weights(E, n, k, seed=0, escapes=0.0, specials=False):
        key = (E, n, k, seed, escapes, specials)
        if key not in W:
            W[key] = make_weights(E, n, k, seed, escapes, specials)
        return W[key]

    def no_escape_weights():
        key = "no-escapes"
        if key not in W:
            w, _ = weights(8, 384, 512, seed=0)
            # copysign also moves exact RNG zeros off the special-value path.
            w = torch.copysign(w.abs().clamp(0.004, 0.06), w).to(torch.bfloat16)
            mw = H.MoeWeights(w)
            assert mw.num_escapes == 0
            W[key] = w, mw
        return W[key]

    # Typical layer halves, uniform routing.
    out.append(("w13-uniform", lambda: (*weights(8, 384, 512), topk_routing(300, 2, 8, seed=1), {})))
    out.append(("w2-uniform-scale", lambda: (*weights(8, 256, 384), topk_routing(300, 2, 8, seed=2, mode="w2"),
                                             {"scale": True})))
    # SM90 BN=256/BM>=128 must preserve operands while compressed planes
    # from both k-blocks are live. Configs 3 and 5 must match the twin.
    out.append(("w13-no-escapes", lambda: (*no_escape_weights(), topk_routing(300, 2, 8, seed=1), {})))
    # Skewed (hot experts, many empty), dropped (-1 ids: offsets[E] < A, garbage beyond).
    out.append(("w13-skew-drop", lambda: (*weights(32, 160, 256), topk_routing(400, 4, 32, seed=3, skew=1.6, drop=0.3), {})))
    out.append(("w2-skew-drop-scale", lambda: (*weights(32, 160, 256),
                                               topk_routing(400, 4, 32, seed=4, skew=1.6, drop=0.3, mode="w2"),
                                               {"scale": True})))
    # Every rows-per-expert count 1..300 on some expert, n not a multiple of any BN.
    halves = [list(range(1, 151)), list(range(151, 301))] if not quick else [list(range(1, 80, 3)) + [255, 256, 257]]
    for i, counts in enumerate(halves):
        out.append((f"counts-{counts[0]}..{counts[-1]}",
                    lambda c=counts, i=i: (*weights(len(c), 144, 128, seed=10 + i),
                                           count_routing(c, gather=True, scatter=True, slack=7, seed=i), {"scale": True})))
    # Row-contiguous activations (no gather), counts past two 256-row m-tiles.
    contig = list(range(1, 601, 9)) + [255, 256, 257, 511, 512, 513, 768, 1025]
    out.append(("counts-contiguous", lambda: (*weights(len(contig), 144, 128, seed=12),
                                              count_routing(contig, scatter=True, slack=5, seed=3), {"scale": True})))
    # Empty experts at the ends and between them, plus an all-empty routing.
    out.append(("empty-mix", lambda: (*weights(6, 128, 192), count_routing([0, 3, 0, 0, 200, 0], slack=5), {})))
    out.append(("all-empty", lambda: (*weights(4, 128, 192), count_routing([0, 0, 0, 0], slack=9), {})))
    # Gather/scatter/scale combinations, ld padding.
    for gth in (False, True):
        for sct in (False, True):
            for scl in (False, True):
                out.append((f"combo-g{int(gth)}s{int(sct)}r{int(scl)}",
                            lambda g_=gth, s_=sct, r_=scl: (*weights(5, 208, 320, seed=2),
                                                            count_routing([40, 1, 130, 0, 77], gather=g_, scatter=s_,
                                                                          slack=3, seed=7),
                                                            {"scale": r_, "ldx": 320 + 24, "ldy": 208 + 40})))
    # Escapes: light, heavy, and Inf/NaN specials (bitwise only).
    out.append(("escapes-5%", lambda: (*weights(4, 256, 1024, seed=3, escapes=0.05), topk_routing(200, 2, 4, seed=5), {})))
    out.append(("escapes-30%", lambda: (*weights(4, 128, 2048, seed=4, escapes=0.3),
                                        topk_routing(150, 2, 4, seed=6, mode="w2"), {"scale": True})))
    out.append(("escapes-specials", lambda: (*weights(3, 128, 512, seed=6, escapes=0.05, specials=True),
                                             topk_routing(120, 2, 3, seed=7), {"bitwise_only": True})))
    # Many experts (E > the SM120 and SM90 limit runs on mma only), single k-block.
    out.append(("E300-k64", lambda: (*weights(300, 32, 64, seed=8), topk_routing(500, 6, 300, seed=8), {})))
    return out


def families(arg):
    return [H.FAMILY_MMA, H.FAMILY_SM120, H.FAMILY_SM90] if arg == "all" else [H.FAMILIES[arg]]


def run_suite(family_arg="all", quick=False):
    n = 0
    for family in families(family_arg):
        configs = configs_for(family, quick)
        if not configs:
            print(f"family {family}: unsupported on this device, skipped")
            continue
        for name, build in cases(quick):
            w, mw, r, kw = build()
            if not shim().supported(family, mw.E):
                continue
            for c in configs:
                try:
                    check(family, c, w, mw, r, tag=name, **kw)
                    n += 1
                except H.Unsupported:  # shared memory for this E exceeds the device limit
                    print(f"  {name}: config {c} unsupported, skipped")
        for c in configs:
            try:
                check_graph(family, c)
                n += 1
            except H.Unsupported:
                print(f"  config {c} unsupported, skipped")
            if os.environ.get("NF12_HUGE_TESTS") == "1":
                for huge in (check_huge, check_huge_contiguous):
                    try:
                        huge(family, c)
                        n += 1
                    except H.Unsupported:
                        print(f"  {huge.__name__}: config {c} unsupported, skipped")
        print(f"family {family}: {len(configs)} configs passed")
    return n


def check_ceil_div():
    """Ceiling division near INT32_MAX must avoid count + bm - 1 overflow.
    Uses the kernel's moe_ceil_div helper with offsets [0, 2^31 - 1] and
    bm = 256 because the real case needs a >1 TiB y."""
    f = shim().lib.nf12_test_moe_ceil_div
    f.restype = H.ctypes.c_int
    f.argtypes = [H.ctypes.c_int] * 2
    imax = 2**31 - 1
    for count, bm in [(imax, 256), (imax, 128), (imax, 1), (imax - 255, 256), (0, 256), (1, 256), (256, 256),
                      (257, 256), (imax, imax), (imax - 1, imax)]:
        assert f(count, bm) == -(-count // bm), (count, bm)


def test_ceil_div():
    check_ceil_div()



@pytest.mark.parametrize("family", ["mma", "sm120", "sm90"])
def test_moe_large(family):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not configs_for(H.FAMILIES[family], True):
        pytest.skip(f"{family} unsupported on this device")
    run_suite(family, quick=os.environ.get("NF12_QUICK") == "1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["mma", "sm120", "sm90", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    check_ceil_div()
    print(f"{run_suite(a.family, a.quick)} checks passed")
