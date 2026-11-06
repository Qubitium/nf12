# SPDX-License-Identifier: Apache-2.0
"""Exactness suite for the large-M linear kernels.

For each config: NF12 output must match the bf16t twin without tolerance, and both
must match an FP32 reference within the BF16 output rounding plus an FP32
accumulation-order bound. Also collected by pytest.
Run: uv run --group test python tests/kernels/linear/large_m_test.py [--family mma|tc|auto] [--quick]
"""

import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import large_m_harness as H  # noqa: E402

_SHIM = None


def shim():
    global _SHIM
    if _SHIM is None:
        _SHIM = H.Shim()
    return _SHIM


def make_weight(n, k, seed=0, escapes=0.0, specials=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    if escapes:
        w = H.force_escapes(w, escapes, seed, specials)
    return w, H.NF12Packed(w)


def make_x(m, k, ldx=None, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ldx = ldx or k
    buf = torch.randn(m, ldx, device="cuda", generator=g).to(torch.bfloat16)
    return buf[:, :k]


def reference(x, w, bias):
    xf, wf = x.float(), w.float()
    ref = xf @ wf.T
    if bias is not None:
        ref += bias.float()
    # BF16 output rounding (2^-8 relative) plus FP32 accumulation-order slack
    # (K * 2^-23 * sum |x||w|).
    slack = (xf.abs() @ wf.abs().T) * (x.shape[1] * 2.0**-23) + 1e-30
    return ref, ref.abs() * 2.0**-8 + slack


def run(family, config, x, nw, bias=None, ldy=None):
    m, n = x.shape[0], nw.n
    outs = []
    for source in ("nf12", "bf16t"):
        ybuf = torch.full((m, ldy or n), float("nan"), dtype=torch.bfloat16, device="cuda")
        y = ybuf[:, :n]
        shim().linear(x, nw, bias=bias, y=y, source=source, family=family, config=config)
        if ldy:
            pad = ybuf[:, n:]
            assert torch.isnan(pad.float()).all(), "kernel wrote past n into the ldy padding"
        outs.append(y)
    return outs


def check(family, config, m, n, k, *, bias=False, ldx=None, ldy=None, escapes=0.0, seed=0, w_cache=None):
    if w_cache is not None and (n, k, escapes, seed) in w_cache:
        w, nw = w_cache[(n, k, escapes, seed)]
    else:
        w, nw = make_weight(n, k, seed, escapes)
        if w_cache is not None:
            w_cache[(n, k, escapes, seed)] = (w, nw)
    x = make_x(m, k, ldx, seed + 1)
    b = (torch.randn(n, device="cuda") * 0.5).to(torch.bfloat16) if bias else None
    y_nf12, y_t = run(family, config, x, nw, b, ldy)
    tag = f"family={family} config={config} m={m} n={n} k={k} bias={bias} ldx={ldx} ldy={ldy} esc={escapes}"
    assert torch.equal(y_nf12.view(torch.int16), y_t.view(torch.int16)), f"NF12 != bf16t bits: {tag}"
    ref, tol = reference(x, w, b)
    err = (y_nf12.float() - ref).abs()
    bad = err > tol
    assert not bad.any(), f"tolerance: {tag} max err {err.max().item():.3e} at {bad.nonzero()[0].tolist()}"


def check_specials(family, config, m=96, n=128, k=512):
    w, nw = make_weight(n, k, 3, escapes=0.02, specials=True)
    x = make_x(m, k)
    y_nf12, y_t = run(family, config, x, nw)
    assert nw.num_escapes > 0
    assert torch.equal(y_nf12.view(torch.int16), y_t.view(torch.int16)), "specials: NF12 != bf16t"


def check_graph(family, config, m=200, n=256, k=1024):
    w, nw = make_weight(n, k)
    x = make_x(m, k)
    y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    ws, _ = shim().workspace(family, config, m, n, k)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        shim().linear(x, nw, y=y, family=family, config=config, ws=ws)  # warm-up (sets smem attributes)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        shim().linear(x, nw, y=y, family=family, config=config, ws=ws)
    y.zero_()
    graph.replay()
    torch.cuda.synchronize()
    ref = shim().linear(x, nw, family=family, config=config)
    assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), "graph replay differs"


def mma_configs(quick):
    """(config code, label) pairs: table index + 16 * (splits - 1) + 256 * dp_tiles.

    dp_tiles = 3 runs the first 3 tiles whole-K in one launch and splits the
    rest in a second (the test shapes have 4+ tiles for every config). Splits
    reduce through clusters from SM90 and through the workspace below."""
    s = shim()
    splits = [1, 2, 4] if quick else [1, 2, 3, 4, 8]
    out = []
    for c in range(s.num_configs(H.FAMILY_MMA)):
        for sp in splits:
            out.append((c + 16 * (sp - 1), f"{s.config_name(H.FAMILY_MMA, c)}/split{sp}"))
            if sp > 1:
                out.append((c + 16 * (sp - 1) + 256 * 3, f"{s.config_name(H.FAMILY_MMA, c)}/split{sp}+dp3"))
    return out


def configs_for(family, quick):
    if family == H.FAMILY_MMA:
        return mma_configs(quick)
    if family == H.FAMILY_TC:
        s = shim()
        return [(c, s.config_name(H.FAMILY_TC, c)) for c in range(s.num_configs(H.FAMILY_TC))]
    return [(-1, "auto")]


def suite(family, quick=False, verbose=True):
    s = shim()
    if family == H.FAMILY_TC and not s.supported(H.FAMILY_TC):
        print("tcgen05 family not supported on this device; skipping")
        return
    cache = {}
    cfgs = configs_for(family, quick)
    skipped = []
    for config, label in cfgs:
        ms = [65, 128, 131, 257] if quick else [1, 7, 65, 100, 128, 131, 255, 257, 300, 513]
        try:
            for m in ms:
                check(family, config, m, 208, 1024, w_cache=cache)  # n tail: 208 = 13 row blocks
        except H.Unsupported:
            skipped.append(label)  # e.g. split-K dump exceeds this device's shared memory
            continue
        check(family, config, 190, 384, 768, bias=True, ldx=800, ldy=392, w_cache=cache)
        check(family, config, 150, 256, 512, escapes=0.01, w_cache=cache)
        check_specials(family, config)
        if verbose:
            print(f"  ok {label}")
    if skipped:
        print(f"  unsupported on this device: {', '.join(skipped)}")
    # Default heuristic: every M in 65..300 plus sparse large M, real shapes.
    auto = H.FAMILY_AUTO if family == H.FAMILY_AUTO else family
    auto_cfg = -1
    for m in (range(65, 301, 7) if quick else range(65, 301)):
        check(auto, auto_cfg, m, 256, 1024, bias=(m % 2 == 0), w_cache=cache)
    for m in ([1024, 4099] if quick else [512, 1000, 1024, 2047, 4096, 4099, 8192, 16384]):
        check(auto, auto_cfg, m, 512, 1024, bias=True, w_cache=cache)
    shapes = [(4096, 4096), (14336, 4096), (4096, 14336)] if quick else [
        (4096, 4096), (14336, 4096), (4096, 14336), (25600, 5120), (5120, 25600), (8192, 8192)]
    for (k, n), m in itertools.product(shapes, [96, 384] if quick else [64, 96, 192, 512, 2048]):
        check(auto, auto_cfg, m, n, k, bias=True, escapes=0.001, w_cache=cache)
    check_graph(auto, auto_cfg)
    print(f"family {family}: {len(cfgs)} configs passed")


def check_tile_bounds():
    """A padded 192-token tile at m0 = 2^31 - 128 must reject rows past M
    without wrapping. Uses the kernel's bounds helper because the real case
    needs a >256 GiB x."""
    f = shim().lib.nf12_test_tile_offset_in_range
    f.restype = H.ctypes.c_int
    f.argtypes = [H.ctypes.c_int] * 3
    m, m0 = 2147483521, 2147483520
    assert f(m0, 0, m) and not f(m0, 1, m) and not f(m0, 128, m) and not f(m0, 191, m)
    imax = 2**31 - 1
    assert f(imax - 1, 0, imax) and not f(imax - 1, 1, imax) and f(0, 15, 16) and not f(0, 16, 16)


def test_tile_bounds():
    check_tile_bounds()


def test_mma_quick():
    suite(H.FAMILY_MMA, quick=True, verbose=False)


def test_auto_quick():
    suite(H.FAMILY_AUTO, quick=True, verbose=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["mma", "tc", "auto"], default="mma")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    fam = {"mma": H.FAMILY_MMA, "tc": H.FAMILY_TC, "auto": H.FAMILY_AUTO}[a.family]
    check_tile_bounds()
    suite(fam, quick=a.quick)
