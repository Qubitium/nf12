# SPDX-License-Identifier: Apache-2.0
"""Exactness suite for the small-M linear kernels and nf12::detail::linear().

Run: uv run --group test pytest tests/kernels/linear/test_small_m.py
"""

import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import small_m_harness as h  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_LIB = None
_ENC = {}


def lib():
    global _LIB
    if _LIB is None:
        _LIB = h.Lib()
    return _LIB


def encoded(n, k, seed=0, escapes=0.0, exp_base=None):
    key = (n, k, seed, escapes, exp_base)
    if key not in _ENC:
        _ENC[key] = h.encode(h.random_weight(n, k, seed=seed, escapes=escapes), exp_base)
    return _ENC[key]


def make_x(m, k, ldx=None, seed=1):
    ldx = k if ldx is None else ldx
    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn(m, ldx, generator=g, device="cuda").to(torch.bfloat16)
    return buf[:, :k] if ldx != k else buf


def reference(x, w, bias):
    y = x.float() @ w.float().T
    if bias is not None:
        y = y + bias.float()
    return y


def check_close(y, x, w, bias):
    """Within BF16 output rounding of an FP32 reference (accumulation order differs)."""
    ref = reference(x, w, bias)
    mag = x.float().abs() @ w.float().abs().T
    # mma.sync partial sums are not IEEE-rounded FP32. With 3e4 outliers
    # cancelling to O(1), accumulation noise is ~2e-5 of sum(|x||w|).
    tol = ref.abs() * 2.0**-7 + mag * 1e-4 + 1e-6
    err = (y.float() - ref).abs()
    bad = err > tol
    assert not bool(bad.any()), f"max err {float(err.max())}, {int(bad.sum())} elements out of tolerance"


def run_pair(cfg, m, enc, ldx=None, ldy=None, bias=None):
    """Check NF12/bf16t equality and the FP32 reference with a shared config.
    Padded output columns must remain untouched."""
    x = make_x(m, enc.k, ldx)
    ldy = enc.n if ldy is None else ldy
    outs = []
    for src in (h.NF12, h.BF16T):
        y = torch.full((m + 1, ldy), float("nan"), dtype=torch.bfloat16, device="cuda")
        try:
            lib().linear(cfg, x, enc, y=y, bias=bias, source=src, m=m)
        except h.LargeMStub as e:
            pytest.skip(str(e))
        outs.append(y)
    torch.cuda.synchronize()
    y_nf12, y_bf16t = outs
    assert torch.equal(y_nf12.view(torch.int16), y_bf16t.view(torch.int16)), f"NF12 != bf16t for {cfg} m={m}"
    assert bool(y_nf12[m].isnan().all()), "row past m written"
    if ldy > enc.n:
        assert bool(y_nf12[:m, enc.n:].isnan().all()), "columns past n written"
    check_close(y_nf12[:m, : enc.n], x, enc.w, bias)
    return y_nf12[:m, : enc.n]


def all_configs(m, n, k):
    out = []
    mts = [mt for mt in (1, 2, 4, 8) if 8 * mt >= m]
    for mt, warps, kc, sl, kp in itertools.product(mts, (4, 8, 16), (1, 2, 3, 8), (0, 2, 4, 8, 16), (1, 2, 4, 8)):
        cfg = h.Config(mt, warps, kc, sl, kp)
        if lib().supported(cfg, m, n, k):
            out.append(cfg)
    return out


@pytest.mark.parametrize("m", range(1, 65))
def test_every_m_dispatch(m):
    """Every M in [1, 64] through the production dispatch, odd strides, bias, escapes."""
    enc = encoded(400, 1280, seed=m % 3, escapes=0.002)
    assert enc.num_escapes > 0
    bias = (torch.randn(enc.n, device="cuda") * 0.1).to(torch.bfloat16)
    run_pair(h.AUTO, m, enc, ldx=enc.k + 8 * (m % 3), ldy=enc.n + 8 * (m % 2), bias=bias if m % 2 else None)


@pytest.mark.parametrize("n,k", [(144, 2112), (1024, 768)])
def test_single_token_weight_formats(n, k):
    enc = encoded(n, k, seed=7, escapes=0.003)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    run_pair(h.AUTO, 1, enc, ldx=k + 8, ldy=n + 8, bias=bias)


@pytest.mark.parametrize("m,n,k", [(3, 4096, 1280), (8, 1536, 128), (8, 2304, 1536), (8, 6144, 7168), (8, 3584, 7168)])
def test_small_batch_weight_formats(m, n, k):
    enc = encoded(n, k, seed=7, escapes=0.003)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    run_pair(h.AUTO, m, enc, ldx=k + 8, ldy=n + 8, bias=bias)


@pytest.mark.parametrize("m", [1, 3, 8, 9, 16, 17, 31, 32, 40, 64])
def test_every_config(m):
    """All template/split/slice combinations on a shape with an uneven K split and a partial row group."""
    enc = encoded(208, 1472, seed=4, escapes=0.003)
    bias = (torch.randn(enc.n, device="cuda") * 0.1).to(torch.bfloat16)
    configs = all_configs(m, enc.n, enc.k)
    assert configs
    ref = None
    for cfg in configs:
        y = run_pair(cfg, m, enc, ldx=enc.k + 8, bias=bias)
        # Different configs reduce in different orders: equal within rounding, not bitwise.
        if ref is None:
            ref = y
        else:
            assert float((y.float() - ref.float()).abs().max()) <= float(ref.float().abs().max()) * 2.0**-6


@pytest.mark.parametrize("m", [1, 3, 8])
def test_every_config_wide(m):
    """As test_every_config on a weight with N >= max(K, 8192), which SM90 runs
    with the register ring at MT = 1 (small_m_stages)."""
    enc = encoded(8208, 1472, seed=9, escapes=0.003)
    configs = all_configs(m, enc.n, enc.k)
    assert configs
    ref = None
    for cfg in configs:
        y = run_pair(cfg, m, enc, ldx=enc.k + 8)
        if ref is None:
            ref = y
        else:
            assert float((y.float() - ref.float()).abs().max()) <= float(ref.float().abs().max()) * 2.0**-6


def test_deterministic():
    enc = encoded(1024, 4096, seed=5, escapes=0.001)
    x = make_x(16, enc.k)
    cfg = h.Config(2, 8, 8, 8)  # K split 8 ways: cluster ranks from SM90, workspace partials below
    a = lib().linear(cfg, x, enc)
    for _ in range(5):
        b = lib().linear(cfg, x, enc)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_escape_heavy():
    """Many escaped groups (forced out-of-window base) still decode exactly."""
    w = h.random_weight(256, 1024, seed=6)
    enc = h.encode(w, exp_base=reference_base(w) + 4)
    assert enc.num_escapes > 1000
    for m in (1, 7, 33):
        run_pair(h.AUTO, m, enc)


def test_special_values():
    """Inf/NaN/-0/subnormal weights propagate the same through NF12 and bf16t."""
    w = h.random_weight(128, 512, seed=7)
    flat = w.view(-1)
    flat[::97] = float("inf")
    flat[5::131] = float("-inf")
    flat[11::173] = -0.0
    flat[13::89] = 1e-40
    enc = h.encode(w)
    x = make_x(4, enc.k)
    for cfg in (h.AUTO, h.Config(1, 4, 2, 0)):
        a = lib().linear(cfg, x, enc, source=h.NF12)
        b = lib().linear(cfg, x, enc, source=h.BF16T)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def reference_base(w):
    return h.reference.choose_exp_base(w)


MODEL_SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336), (28672, 4096), (25600, 5120), (5120, 25600),
                (8192, 8192), (8192, 2048), (4864, 896)]  # (n, k)


@pytest.mark.parametrize("n,k", MODEL_SHAPES)
def test_model_shapes(n, k):
    enc = h.encode(h.random_weight(n, k, seed=8, escapes=1e-4))
    bias = (torch.randn(n, device="cuda") * 0.1).to(torch.bfloat16)
    for m in (1, 2, 4, 8, 16, 32, 64):
        run_pair(h.AUTO, m, enc, bias=bias if m in (1, 16) else None)


def test_dispatch_rejects_bad_args():
    enc = encoded(64, 128)
    x = make_x(4, enc.k)
    y = torch.empty(4, enc.n, dtype=torch.bfloat16, device="cuda")
    L = lib().lib
    stream = torch.cuda.current_stream().cuda_stream
    bad = [
        dict(m=4, ldx=enc.k - 8),          # ldx < k
        dict(m=4, ldx=enc.k + 4),          # ldx not a multiple of 8
        dict(m=4, n=enc.n - 8),            # n % 16
        dict(m=4, k=enc.k - 32),           # k % 64
        dict(m=4, ldy=enc.n - 8),          # ldy < n
        dict(m=0),
    ]
    for over in bad:
        args = dict(m=4, ldx=enc.k, n=enc.n, k=enc.k, ldy=enc.n)
        args.update(over)
        err = L.nf12_sm_linear(0, 0, 0, 0, 1, x.data_ptr(), args["m"], args["ldx"], h.NF12, enc.packed.data_ptr(),
                               None, 0, enc.exp_base, args["n"], args["k"], None, y.data_ptr(), args["ldy"], None, 0,
                               stream)
        assert err == 1, f"{over} -> {err}"  # cudaErrorInvalidValue


def test_graph_capture():
    """linear() is capturable even when its first call (occupancy queries,
    shared-memory cap) happens inside the capture."""
    enc = encoded(1040, 9024, seed=9)  # a shape no other test dispatches
    x = make_x(24, enc.k)  # small-M on every arch
    y = torch.empty(24, enc.n, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        lib().linear(h.AUTO, x, enc, y=y)
    y.zero_()
    graph.replay()
    torch.cuda.synchronize()
    ref = lib().linear(h.AUTO, x, enc)
    assert torch.equal(y.view(torch.int16), ref.view(torch.int16))


@pytest.mark.skipif(os.environ.get("NF12_HUGE_TESTS") != "1", reason="set NF12_HUGE_TESTS=1 (needs ~40 GB of GPU memory)")
@pytest.mark.parametrize("n,k", [(8388608, 64),    # > 65535 row groups: the grid must not use gridDim.y
                                 (32768, 65536)])  # > 2^31 weight bytes: 64-bit offsets throughout
def test_huge(n, k):
    import large_m_harness as lm  # chunked encoding avoids ~25x matrix-sized reference temporaries

    w = h.random_weight(n, k, seed=10, escapes=1e-4)
    p = lm.NF12Packed(w)
    enc = h.Encoded(w, p.packed, p.escapes[: p.num_escapes], p.exp_base, p.bf16t, n, k)
    rows = torch.cat([torch.arange(0, 64), torch.arange(n // 2, n // 2 + 64), torch.arange(n - 64, n)]).cuda()
    for m in (1, 16):
        x = make_x(m, k)
        y = lib().linear(h.AUTO, x, enc, source=h.NF12)
        y_t = lib().linear(h.AUTO, x, enc, source=h.BF16T)
        torch.cuda.synchronize()
        assert torch.equal(y.view(torch.int16), y_t.view(torch.int16))
        check_close(y[:, rows], x, w[rows], None)
