# SPDX-License-Identifier: Apache-2.0
"""Exactness suite for the SM120 large-M kernel family.

For every config and schedule: NF12 output matches the bf16t twin without tolerance,
both within BF16 tolerance of an FP32 reference, ldy padding untouched.
Run on SM120: uv run --group test python tests/kernels/linear/large_m_sm120_test.py [--quick]
64-bit offset cases (>2^31-byte weights) require NF12_HUGE_TESTS=1.
"""

import argparse
import itertools
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import large_m_harness as H  # noqa: E402
import large_m_sm120_harness as S  # noqa: E402
from large_m_test import make_weight, make_x, reference  # noqa: E402

_SHIM = None


def shim():
    global _SHIM
    if _SHIM is None:
        _SHIM = S.Sm120Shim()
    return _SHIM


def run(config, x, nw, bias=None, ldy=None):
    m, n = x.shape[0], nw.n
    outs = []
    for source in ("nf12", "bf16t"):
        ybuf = torch.full((m, ldy or n), float("nan"), dtype=torch.bfloat16, device="cuda")
        y = ybuf[:, :n]
        shim().sm120(x, nw, bias=bias, y=y, source=source, config=config)
        if ldy:
            assert torch.isnan(ybuf[:, n:].float()).all(), "kernel wrote past n into the ldy padding"
        outs.append(y)
    return outs


def check_out(tag, x, w, b, y_nf12, y_t, exact_only=False):
    assert torch.equal(y_nf12.view(torch.int16), y_t.view(torch.int16)), f"NF12 != bf16t bits: {tag}"
    if exact_only:
        return
    ref, tol = reference(x, w, b)
    err = (y_nf12.float() - ref).abs()
    bad = ~(err <= tol)  # NaN counts as bad
    assert not bad.any(), f"tolerance: {tag} max err {err.max().item():.3e} at {bad.nonzero()[0].tolist()}"


def check(config, m, n, k, *, bias=False, ldx=None, ldy=None, escapes=0.0, seed=0, w_cache=None):
    key = (n, k, escapes, seed)
    if w_cache is not None and key in w_cache:
        w, nw = w_cache[key]
    else:
        w, nw = make_weight(n, k, seed, escapes)
        if w_cache is not None:
            w_cache[key] = (w, nw)
    x = make_x(m, k, ldx, seed + 1)
    b = (torch.randn(n, device="cuda") * 0.5).to(torch.bfloat16) if bias else None
    y_nf12, y_t = run(config, x, nw, b, ldy)
    tag = f"config={shim().sm120_config_name(config)} m={m} n={n} k={k} bias={bias} ldx={ldx} ldy={ldy} esc={escapes}"
    check_out(tag, x, w, b, y_nf12, y_t)


def check_specials(config, m=96, n=128, k=512):
    """Inf/NaN/-0 escapes: NF12 must preserve the twin's payloads and signs."""
    w, nw = make_weight(n, k, 3, escapes=0.02, specials=True)
    assert nw.num_escapes > 0
    x = make_x(m, k)
    y_nf12, y_t = run(config, x, nw)
    check_out(f"specials config={config}", x, w, None, y_nf12, y_t, exact_only=True)


def check_no_escapes(config, m=150, n=256, k=1024):
    """Weights entirely in-window: num_escapes = 0 and a NULL escape table."""
    g = torch.Generator(device="cuda").manual_seed(7)
    mag = torch.rand(n, k, device="cuda", generator=g) * 0.9 + 0.05  # exponents 2^-5 .. 2^-1
    sign = torch.randint(0, 2, (n, k), device="cuda", generator=g) * 2 - 1
    w = (mag * sign * 0.25).to(torch.bfloat16)
    nw = H.NF12Packed(w)
    assert nw.num_escapes == 0, nw.num_escapes
    x = make_x(m, k)
    y_nf12, y_t = run(config, x, nw)
    check_out(f"no-escapes config={config}", x, w, None, y_nf12, y_t)


def check_graph(config=-1, m=200, n=512, k=2048):
    w, nw = make_weight(n, k)
    x = make_x(m, k)
    y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    ws, _ = shim().sm120_workspace(config, m, n, k)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        shim().sm120(x, nw, y=y, config=config, ws=ws)  # warm-up (sets smem attributes)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        shim().sm120(x, nw, y=y, config=config, ws=ws)
    ref = shim().sm120(x, nw, config=config)
    for _ in range(3):  # replays reuse the workspace: stream-K tags must be reset
        y.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), "graph replay differs"


def check_deterministic(config, m=300, n=1024, k=4096):
    w, nw = make_weight(n, k, 5, escapes=0.01)
    x = make_x(m, k)
    a = shim().sm120(x, nw, config=config)
    for _ in range(3):
        b = shim().sm120(x, nw, config=config)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"nondeterministic: config={config}"


def check_long(config, m=16384, n=2048, k=4096):
    """Thousands of stages per CTA let fast consumers run a full ring ahead
    of escape fetchers and expose pipeline-phase hazards absent in short runs."""
    w, nw = make_weight(n, k, 9, escapes=0.0005)
    x = make_x(m, k)
    y_nf12, y_t = run(config, x, nw)
    check_out(f"long config={config}", x, w, None, y_nf12, y_t, exact_only=True)


def check_interleaved(rounds=4):
    """Compare alternating shapes and sources bitwise against their first runs
    to expose timing-dependent pipeline races such as early barrier arrivals."""
    shapes = [(4096, 4096), (4096, 14336), (8192, 8192)]
    ws = {kn: make_weight(kn[1], kn[0], 0, escapes=0.001) for kn in shapes}
    xs = {m: make_x(m, 8192) for m in (96, 192, 512, 2048)}
    golden = {}
    for _ in range(rounds):
        for (k, n), m in itertools.product(shapes, xs):
            nw = ws[(k, n)][1]
            ys = [shim().sm120(xs[m][:, :k], nw, source=src) for src in ("nf12", "bf16t")]
            assert torch.equal(ys[0].view(torch.int16), ys[1].view(torch.int16)), f"interleaved NF12 != bf16t {k} {n} {m}"
            if (k, n, m) not in golden:
                golden[(k, n, m)] = ys[0].clone()
            assert torch.equal(ys[0].view(torch.int16), golden[(k, n, m)].view(torch.int16)), "interleaved repeat differs"


def check_prefilled_tags(config, m, n, k, escapes=0.001):
    """Prefill workspace with tags from a clean run and NaN partials.
    Tags persist after publication. Per-launch reset must make owners wait
    for new partials and match the clean-workspace result for both sources."""
    w, nw = make_weight(n, k, 11, escapes)
    x = make_x(m, k)
    ws, nbytes = shim().sm120_workspace(config, m, n, k)
    assert nbytes > 0, "plan has no stream-K work"
    ws = ws[:nbytes]
    # Workspace has grid * BN * BM FP32 partials, then grid 64-bit tags
    # rounded up to 256 bytes. Recover the grid from its size.
    name = shim().sm120_config_name(config if config >= 0 else shim().sm120_heuristic(m, n, k))
    bn, bm = (int(v) for v in re.match(r"w(\d+)_t(\d+)", name).groups())
    grids = [g for g in range(1, 4097) if g * bn * bm * 4 + (g * 8 + 255) // 256 * 256 == nbytes]
    assert len(grids) == 1, (name, nbytes, grids)
    part = grids[0] * bn * bm * 4
    for src in ("nf12", "bf16t"):
        ws.zero_()
        clean = shim().sm120(x, nw, source=src, config=config, ws=ws).clone()
        torch.cuda.synchronize()
        tags = ws[part:].clone()
        assert tags.view(torch.int64).ne(0).any(), "no contributor published a tag"
        for _ in range(2):
            ws[:part].view(torch.float32).fill_(float("nan"))
            ws[part:].copy_(tags)
            y = shim().sm120(x, nw, source=src, config=config, ws=ws)
            assert torch.equal(y.view(torch.int16), clean.view(torch.int16)), \
                f"prefilled tags config={config} m={m} n={n} k={k}: {src} differs from the clean-workspace run"


def check_huge():
    """64-bit offsets where 16 * K * 1.5 or N * K * 1.5 exceeds 2^31 bytes."""
    free = torch.cuda.mem_get_info()[0]
    for (n, k), m in [((16 * 1024, 98304), 192), ((64, 1 << 21), 96)]:
        need = n * k * 2 * 4  # w + bf16t + packed + encoder temporaries, roughly
        if need > free * 0.8:
            print(f"  skip huge n={n} k={k}: needs ~{need >> 30} GiB")
            continue
        check(-1, m, n, k, bias=True)
        print(f"  ok huge n={n} k={k} m={m}")
        torch.cuda.empty_cache()


def configs(quick):
    s = shim()
    scheds = [S.HYBRID, S.DATA_PARALLEL, S.STREAM_K]
    return [c + 16 * sch for c in range(s.sm120_num_configs()) for sch in scheds] + \
        [c + 16 * S.SPLIT + 256 * f for c in range(s.sm120_num_configs()) for f in (2, 4, 16)]


def suite(quick=False, verbose=True):
    s = shim()
    if not s.sm120_supported():
        print("SM120 family not supported on this device/image; skipping")
        narrow_suite(quick, verbose)  # also runs on SM90
        return
    cache = {}
    cfgs = configs(quick)
    for config in cfgs:
        label = s.sm120_config_name(config)
        ms = [65, 128, 131, 257] if quick else [1, 7, 65, 100, 128, 131, 255, 257, 300, 513]
        for m in ms:
            check(config, m, 208, 1024, w_cache=cache)  # n tail: 208 = 13 row blocks
        check(config, 190, 384, 768, bias=True, ldx=800, ldy=392, w_cache=cache)
        check(config, 150, 256, 512, escapes=0.01, w_cache=cache)
        check(config, 150, 256, 512, escapes=0.10, w_cache=cache)
        check_specials(config)
        check_no_escapes(config)
        check_deterministic(config)
        if not quick:
            check_long(config)
        if verbose:
            print(f"  ok {label}")
    # Heuristic: every M in 65..300 plus sparse large M, real shapes.
    for m in (range(65, 301, 7) if quick else range(65, 301)):
        check(-1, m, 256, 1024, bias=(m % 2 == 0), w_cache=cache)
    for m in ([1024, 4099] if quick else [512, 1000, 1024, 2047, 4096, 4099, 8192, 16384]):
        check(-1, m, 512, 1024, bias=True, w_cache=cache)
    shapes = [(4096, 4096), (14336, 4096), (4096, 14336)] if quick else [
        (4096, 4096), (14336, 4096), (4096, 14336), (25600, 5120), (5120, 25600), (8192, 8192)]
    for (k, n), m in itertools.product(shapes, [96, 384] if quick else [64, 96, 192, 512, 2048]):
        check(-1, m, n, k, bias=True, escapes=0.001, w_cache=cache)
    for m in [96, 384, 2048]:
        check(-1, m, 4096, 4096, escapes=0.10, w_cache=cache)
    # Narrow weights (router gates): few tiles, the split schedule's all-CTA reduction.
    for (k, n), m in itertools.product([(2048, 64), (4096, 128)], [65, 128, 300, 1024]):
        check(-1, m, n, k, bias=(m % 2 == 0), escapes=0.001, w_cache=cache)
    # Hybrid schedule with K >= 16384: one data-parallel wave, then each of the
    # 20 remaining tiles split k-wise over 9 CTAs (aligned tail).
    check(4, 832, 2048, 16384, bias=True, escapes=0.001, w_cache=cache)
    check_no_escapes(-1)
    for config, m, n, k in [(-1, 96, 4096, 4096), (-1, 192, 14336, 4096), (-1, 512, 4096, 14336),
                            (-1, 1024, 8192, 8192), (2 + 16 * S.STREAM_K, 300, 1024, 2048), (4, 832, 2048, 16384),
                            (1 + 16 * S.SPLIT + 256 * 4, 200, 1024, 2048)]:
        check_prefilled_tags(config, m, n, k)
    check_interleaved(2 if quick else 8)
    check_graph(-1)
    check_graph(2 + 16 * S.STREAM_K)
    if os.environ.get("NF12_HUGE_TESTS") == "1":
        check_huge()
    print(f"sm120: {len(cfgs)} configs passed")
    narrow_suite(quick, verbose)


def check_narrow(config, m, n, k, *, bias=False, ldx=None, ldy=None, escapes=0.0, w_cache=None):
    key = (n, k, escapes, 0)
    if key not in w_cache:
        w_cache[key] = make_weight(n, k, 0, escapes)
    w, nw = w_cache[key]
    x = make_x(m, k, ldx, 1)
    b = (torch.randn(n, device="cuda") * 0.5).to(torch.bfloat16) if bias else None
    outs = []
    for source in ("nf12", "bf16t", "nf12"):
        ybuf = torch.full((m, ldy or n), float("nan"), dtype=torch.bfloat16, device="cuda")
        shim().narrow(x, nw, bias=b, y=ybuf[:, :n], source=source, config=config)
        if ldy:
            assert torch.isnan(ybuf[:, n:].float()).all(), "narrow kernel wrote into the ldy padding"
        outs.append(ybuf[:, :n])
    tag = f"narrow {shim().narrow_config_name(config)} m={m} n={n} k={k} bias={bias} esc={escapes}"
    assert torch.equal(outs[0].view(torch.int16), outs[2].view(torch.int16)), f"nondeterministic: {tag}"
    check_out(tag, x, w, b, outs[0], outs[1])


def narrow_suite(quick=False, verbose=True):
    """Test every narrow-weight config with ragged M/N, K splits, bias,
    padded strides and dense escapes. Then test the heuristic."""
    s = shim()
    if not s.narrow_supported():
        print("narrow kernel not supported on this device/image; skipping")
        return
    cache = {}
    n_cfg = 0
    for tf, wi, splits in itertools.product(range(4), range(4), (1, 2, 4, 8)):
        config = tf | (wi << 2) | (splits << 4)
        for m, n, k in [(1, 64, 512), (7, 112, 1024), (33, 128, 512), (100, 48, 256)] + (
                [] if quick else [(300, 128, 512), (64, 80, 128)]):
            if -(-k // 64 // splits) > 8 or splits > k // 64:
                continue
            check_narrow(config, m, n, k, bias=(m % 2 == 1), w_cache=cache)
        check_narrow(config, 19, 64, 512, ldx=520, ldy=72, escapes=0.10, w_cache=cache)
        n_cfg += 1
        if verbose:
            print(f"  ok {s.narrow_config_name(config)}")
    if torch.cuda.get_device_capability() == (9, 0):
        # SM90's non-portable clusters: up to 16 split CTAs, K up to 8192.
        for tf, wi, splits in [(0, 1, 16), (1, 3, 14), (2, 2, 12), (3, 0, 16)]:
            config = tf | (wi << 2) | (splits << 4)
            for m, n, k in [(1, 256, 7168), (37, 384, 7168), (130, 128, 8192), (64, 96, 6144)] + (
                    [] if quick else [(300, 256, 7168), (9, 48, 6144)]):
                if -(-k // 64 // splits) > 8:
                    continue
                check_narrow(config, m, n, k, bias=(m % 2 == 1), w_cache=cache)
            k = 448 * splits  # 7 k-blocks per split CTA
            check_narrow(config, 19, 64, k, ldx=k + 8, ldy=72, escapes=0.10, w_cache=cache)
            n_cfg += 1
            if verbose:
                print(f"  ok {s.narrow_config_name(config)}")
    shapes = [(2048, 64), (2048, 128), (4096, 128)]
    if torch.cuda.get_device_capability() == (9, 0):
        shapes += [(7168, 256), (7168, 384)]
    for (k, n), m in itertools.product(shapes, [1, 8, 16, 37, 64, 128, 256] if quick else range(1, 257, 3)):
        check_narrow(-1, m, n, k, bias=True, escapes=0.002, w_cache=cache)
    print(f"narrow: {n_cfg} configs passed")


def test_sm120_quick():
    suite(quick=True, verbose=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    suite(quick=ap.parse_args().quick)
