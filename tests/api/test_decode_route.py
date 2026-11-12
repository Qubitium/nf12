# SPDX-License-Identifier: Apache-2.0
"""Large-M / large-T decode-then-GEMM routes (SPEC.md sections 6 and 9.4).

- C API: from a per-architecture M, ``nf12_linear`` decodes the weight to
  bf16t in N-slices inside the caller's workspace and runs the bf16t large-M
  kernel per slice; ``nf12_linear_bf16t`` takes the same slices straight from
  the twin, so the two produce the same outputs.
- torch op: where ``nf12_linear_decode_gemm_rows`` is non-zero, ``nf12.linear``
  decodes row slices with ``nf12_decode`` and runs one cuBLAS ``addmm``/``mm``
  per slice, matching the same calls on ``nf12.decode(w)``.
- MoE: where ``nf12_moe_linear_decode_gemm_rows`` is non-zero, a gathered
  ``nf12.moe_linear`` without scatter/row_scale/zero_fill decodes the active experts per N-slice
  (``nf12_moe_decode``) and runs ``torch._grouped_mm`` on the rows gathered by
  ``nf12_moe_grouped_operands``, matching the same calls on the decoded
  experts.
"""

from __future__ import annotations

import ctypes
import os

import nf12
import pytest
import torch
from _helpers import bits
from nf12 import reference
from test_api import OK, Weight, lib, ptr, stream  # noqa: F401  (lib is a fixture)
from test_moe import MoeWeight, experts, routing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"
M_PROBE = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
# (n, k): one slice; several unequal slices (16 * 1031 rows of 16 KiB) for slice bounds up to 256 MiB; a
# router-gate-sized weight (narrow route, small M included where the device takes it).
SHAPES = [(1024, 2048), (16 * 1031, 8192), (128, 2048)]




def rand(*shape: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=g) * scale).to(torch.bfloat16)


def escaping_weight(n: int, k: int) -> torch.Tensor:
    """Normal weights with sparse tiny values that force groups to escape."""
    w = rand(n, k, seed=0, scale=0.02)
    w[::97, ::89] = 1e-30
    return w


def routed_rows(lib, n: int, k: int) -> tuple[int, int]:
    """Smallest probed decode + cuBLAS M and its slice rows. Skips if none routes."""
    dev = torch.cuda.current_device()
    for m in M_PROBE:
        rows = lib.nf12_linear_decode_gemm_rows(m, n, k, dev)
        if rows > 0:
            return m, rows
    pytest.skip(f"no decode + cuBLAS route on this device for {n}x{k}")


def cublas_slices(x: torch.Tensor, wd: torch.Tensor, bias: torch.Tensor | None, rows: int) -> torch.Tensor:
    """The route's calls, issued from Python on the fully decoded weight."""
    y = torch.empty(x.shape[0], wd.shape[0], dtype=torch.bfloat16, device=DEV)
    for n0 in range(0, wd.shape[0], rows):
        n1 = min(n0 + rows, wd.shape[0])
        if bias is None:
            torch.mm(x, wd[n0:n1].t(), out=y[:, n0:n1])
        else:
            torch.addmm(bias[n0:n1], x, wd[n0:n1].t(), out=y[:, n0:n1])
    return y




@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_torch_route_matches_cublas_on_decoded(lib, n, k, with_bias):
    m, rows = routed_rows(lib, n, k)
    if n > 1024:
        assert rows < n and n % rows != 0, "expected several slices with a shorter last one"
    w = escaping_weight(n, k)
    nw = nf12.encode(w)
    assert nw.num_escapes > 0
    bias = rand(n, seed=3) if with_bias else None
    x = rand(m, k, seed=1)
    y = nf12.linear(x, nw, bias)
    assert torch.equal(bits(y), bits(cublas_slices(x, nf12.decode(nw), bias, rows)))
    assert torch.equal(bits(y), bits(nf12.linear(x, nw, bias)))  # deterministic
    # Leading dims flatten to the same [M, K] call.
    assert torch.equal(bits(nf12.linear(x.view(2, m // 2, k), nw, bias)), bits(y.view(2, m // 2, n)))
    ref = x.float() @ w.float().t() + (bias.float() if with_bias else 0)
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=2e-2)


def test_torch_route_graph_capture(lib):
    n, k = SHAPES[1]
    m, _ = routed_rows(lib, n, k)
    nw = nf12.encode(escaping_weight(n, k))
    bias = rand(n, seed=3)
    x = rand(m, k, seed=1)
    stream_ = torch.cuda.Stream()
    stream_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream_):
        nf12.linear(x, nw, bias)  # warm up cuBLAS and the allocator outside capture
    torch.cuda.current_stream().wait_stream(stream_)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = nf12.linear(x, nw, bias)
    for seed in (2, 3):
        x.copy_(rand(m, k, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(y), bits(nf12.linear(x, nw, bias)))




def c_weight(nw: nf12.NF12Weight) -> Weight:
    n, k = nw.shape
    esc = ptr(nw.escapes) if nw.num_escapes else None
    return Weight(ptr(nw.packed), esc, nw.num_escapes, n, k, nw.exp_base)


@pytest.mark.parametrize("n,k,m", [(n, k, m) for n, k in SHAPES for m in (2048, 8192)] +
                         [(64, 2048, 7), (128, 4096, 7), (128, 2048, 129), (64, 2048, 513)])
def test_c_api_linear_matches_bf16t(lib, n, k, m):
    """Exact-size or NULL workspace, padded x/y rows, bias: NF12 == bf16t, deterministic."""
    w = escaping_weight(n, k)
    nw = nf12.encode(w)
    wt = nf12.pack_bf16t(w)
    weight = c_weight(nw)
    x = rand(m, k + 8, seed=1)[:, :k]
    bias = rand(n, seed=3)
    ldy = n + 8
    need = lib.nf12_linear_workspace_bytes(m, n, k, torch.cuda.current_device())
    outs = []
    for fill, run in [(0xFF, "nf12"), (0x7F, "nf12"), (0xA5, "bf16t"), (0xFF, "static")]:
        ws = torch.full((need,), fill, dtype=torch.uint8, device=DEV) if need else None
        workspace = ptr(ws) if ws is not None else None
        y = torch.full((m, ldy), -7.0, dtype=torch.bfloat16, device=DEV)
        if run != "bf16t":
            linear = lib.nf12_linear_static_weights if run == "static" else lib.nf12_linear
            status = linear(ptr(x), m, x.stride(0), ctypes.byref(weight), ptr(bias), ptr(y), ldy,
                            workspace, need, stream())
        else:
            status = lib.nf12_linear_bf16t(ptr(x), m, x.stride(0), ptr(wt), n, k, ptr(bias), ptr(y), ldy,
                                           workspace, need, stream())
        assert status == OK
        assert torch.equal(y[:, n:], torch.full_like(y[:, n:], -7.0))
        outs.append(y[:, :n])
    assert torch.equal(bits(outs[0]), bits(outs[1]))
    assert torch.equal(bits(outs[0]), bits(outs[2]))
    assert torch.equal(bits(outs[0]), bits(outs[3]))
    ref = x.float() @ w.float().t() + bias.float()
    torch.testing.assert_close(outs[0].float(), ref, rtol=2e-2, atol=2e-2)




@pytest.mark.skipif(os.environ.get("NF12_HUGE_TESTS") != "1", reason="set NF12_HUGE_TESTS=1 (~10 GB of device memory)")
def test_slices_beyond_4gib(lib):
    """N x K = 49152 x 65536: 4.5 GiB packed, so later slices start beyond 2^32 bytes.

    Row block b holds the constant (b % 7 + 1) / 16 (assembled from one encoded
    112-row matrix, so one exp_base and no escapes); x is 1 on columns 0..15 and
    0 elsewhere, so y[:, n] = 16 * W[n, 0] exactly in any accumulation order."""
    n, k, blocks = 49152, 65536, 7
    small = torch.repeat_interleave(torch.arange(1, blocks + 1, device=DEV, dtype=torch.float32) / 16, 16)
    small_nw = nf12.encode(small[:, None].expand(16 * blocks, k).to(torch.bfloat16).contiguous())
    assert small_nw.num_escapes == 0
    idx = torch.arange(n // 16, device=DEV) % blocks
    packed = small_nw.packed.view(blocks, 16, -1)[idx].view(n, -1)
    nw = nf12.NF12Weight(packed, small_nw.escapes, small_nw.exp_base)
    want = (16 * (idx + 1).float() / 16).repeat_interleave(16).to(torch.bfloat16)
    dev = torch.cuda.current_device()
    for m in (2048, 4096):
        x = torch.zeros(m, k, dtype=torch.bfloat16, device=DEV)
        x[:, :16] = 1.0
        need = lib.nf12_linear_workspace_bytes(m, n, k, dev)
        ws = torch.empty(max(need, 1), dtype=torch.uint8, device=DEV)
        y = torch.empty(m, n, dtype=torch.bfloat16, device=DEV)
        weight = c_weight(nw)
        assert lib.nf12_linear(ptr(x), m, k, ctypes.byref(weight), None, ptr(y), n, ptr(ws), need, stream()) == OK
        assert torch.equal(y, want.expand(m, n))
        if lib.nf12_linear_decode_gemm_rows(m, n, k, dev) > 0:
            assert torch.equal(nf12.linear(x, nw), want.expand(m, n))
        del x, ws, y




@pytest.fixture(scope="module")
def moe_lib():
    lib = ctypes.CDLL(str(nf12.library_path()))
    p, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.nf12_moe_decode.restype = i32
    lib.nf12_moe_decode.argtypes = [ctypes.POINTER(MoeWeight), p, i64, i64, i64, p, i64, p]
    lib.nf12_moe_grouped_operands.restype = i32
    lib.nf12_moe_grouped_operands.argtypes = [p, i64, p, i64, p, i64, i64, i64, p, i64, p, p]
    lib.nf12_moe_linear_decode_gemm_rows.restype = i64
    lib.nf12_moe_linear_decode_gemm_rows.argtypes = [i64, i64, i64, i64, i32]
    return lib


def c_moe_weight(mw: nf12.NF12MoEWeight) -> MoeWeight:
    e, n, k = mw.shape
    return MoeWeight(mw.packed.data_ptr(), mw.escapes.data_ptr() if mw.num_escapes else None,
                     mw.escape_offsets.data_ptr(), mw.exp_base.data_ptr(), e, n, k)


# Malformed offsets as moe_linear reads them (clamped to [0, A = 10], running max):
# starts 2, 2, 7, 7, 10, 10 -> experts 1 and 3 own positions, 0, 2 and 4 none.
BAD_OFFSETS, BAD_A, BAD_ENDS = [2, 1, 7, 7, 20, 30], 10, [2, 7, 7, 10, 10]


def test_moe_decode_slices_and_gating(moe_lib):
    e, n, k = 5, 48, 128
    w = experts(e, n, k)
    mw = nf12.encode_moe(w)
    assert mw.num_escapes > 0
    assert torch.equal(bits(mw.decode()), bits(w))
    cw = c_moe_weight(mw)
    out = torch.empty(e, 32, k + 8, dtype=torch.bfloat16, device=DEV)  # rows [16, 48), ldo > k
    assert moe_lib.nf12_moe_decode(ctypes.byref(cw), None, 0, 16, 32, out.data_ptr(), k + 8, stream()) == OK
    assert torch.equal(bits(out[:, :, :k]), bits(w[:, 16:]))
    for offsets, a, active in [([0, 3, 5, 5, 9, 9], 9, [1, 1, 0, 1, 0]), (BAD_OFFSETS, BAD_A, [0, 1, 0, 1, 0])]:
        off = torch.tensor(offsets, dtype=torch.int64, device=DEV)
        out = torch.full((e, n, k), -3.0, dtype=torch.bfloat16, device=DEV)
        assert moe_lib.nf12_moe_decode(ctypes.byref(cw), off.data_ptr(), a, 0, n, out.data_ptr(), k, stream()) == OK
        for i in range(e):
            want = w[i] if active[i] else torch.full_like(w[i], -3.0)
            assert torch.equal(bits(out[i]), bits(want)), i
    # Past-N and stride cases run against the shared ABI in test_shared_abi.py.
    bad = [(-16, 16), (8, 16)]  # row0 < 0, unaligned rows
    for row0, rows in bad:
        assert moe_lib.nf12_moe_decode(ctypes.byref(cw), None, 0, row0, rows, out.data_ptr(), k, stream()) != OK


def test_moe_grouped_operands(moe_lib):
    e, k, a, div = 5, 64, BAD_A, 3
    x = rand(4, k, seed=4)
    gather = torch.arange(a, dtype=torch.int32, device=DEV).flip(0) % 12  # in(p) = gather[p] // 3 in [0, 3]
    off = torch.tensor(BAD_OFFSETS, dtype=torch.int64, device=DEV)
    ends = torch.full((e,), -1, dtype=torch.int32, device=DEV)
    xg = torch.full((a, k), -5.0, dtype=torch.bfloat16, device=DEV)
    assert moe_lib.nf12_moe_grouped_operands(x.data_ptr(), k, gather.data_ptr(), div, off.data_ptr(), a, e, k,
                                             xg.data_ptr(), k, ends.data_ptr(), stream()) == OK
    assert ends.tolist() == BAD_ENDS
    valid = BAD_ENDS[-1]
    assert torch.equal(bits(xg[:valid]), bits(x[gather[:valid].long() // div]))
    # The no-gather copy contract is in test_shared_abi.py; only div validation here.
    assert moe_lib.nf12_moe_grouped_operands(None, k, None, 0, off.data_ptr(), a, e, k, None, k, ends.data_ptr(),
                                             stream()) != OK
    # Rows past the valid count are left alone.
    off = torch.tensor([0, 2, 4, 4, 4, 6], dtype=torch.int64, device=DEV)
    xg.fill_(-5.0)
    assert moe_lib.nf12_moe_grouped_operands(x.data_ptr(), k, gather.data_ptr(), div, off.data_ptr(), a, e, k,
                                             xg.data_ptr(), k, ends.data_ptr(), stream()) == OK
    assert torch.equal(xg[6:], torch.full_like(xg[6:], -5.0))


def moe_routed(moe_lib, e: int, n: int, k: int, topk: int) -> tuple[int, int]:
    """Smallest probed decode + grouped GEMM T and its slice rows. Skips if none routes."""
    dev = torch.cuda.current_device()
    for t in (256, 1024, 4096, 8192, 16384):
        rows = moe_lib.nf12_moe_linear_decode_gemm_rows(t * topk, e, n, k, dev)
        if rows > 0:
            return t, rows
    pytest.skip(f"no MoE decode + grouped GEMM route on this device for E={e} {n}x{k}")


def grouped_slices(xg: torch.Tensor, wd: torch.Tensor, ends: torch.Tensor, rows: int) -> torch.Tensor:
    """The route's grouped GEMMs, issued from Python on the decoded experts."""
    outs = [torch._grouped_mm(xg, wd[:, n0:n0 + rows].contiguous().transpose(1, 2), offs=ends)
            for n0 in range(0, wd.shape[1], rows)]
    return torch.cat(outs, dim=1)


# (E, N, K, topk): one slice; two unequal slices of the 1 GiB bound (8 x 8208 x 8192 BF16).
MOE_SHAPES = [(8, 1024, 1024, 2), (8, 8208, 8192, 2)]


@pytest.mark.parametrize("e,n,k,topk", MOE_SHAPES)
def test_moe_route_matches_grouped_mm(moe_lib, e, n, k, topk):
    t, rows = moe_routed(moe_lib, e, n, k, topk)
    if n > 1024:
        assert rows < n and n % rows != 0, "expected several slices with a shorter last one"
    w = experts(e, n, k, spread=1)
    mw = nf12.encode_moe(w)
    x = rand(t, k, seed=1)
    _, ids = routing(t, e, topk)
    ids[::5, 0] = -1  # dropped assignments: positions past the valid count are unspecified
    sorted_ids, offsets = nf12.moe_align(ids, e)
    valid = int(offsets[-1])
    ends = offsets[1:].to(torch.int32)
    wd = mw.decode()
    y = nf12.moe_linear(x, mw, offsets, gather=sorted_ids, gather_div=topk)
    xg = x[sorted_ids[:valid].long() // topk]
    xg = torch.cat([xg, torch.zeros(t * topk - valid, k, dtype=xg.dtype, device=DEV)])
    want = grouped_slices(xg, wd, ends, rows)
    assert torch.equal(bits(y[:valid]), bits(want[:valid]))
    assert torch.equal(bits(y[:valid]), bits(nf12.moe_linear(x, mw, offsets, gather=sorted_ids,
                                                              gather_div=topk)[:valid]))  # deterministic
    ref = reference.moe_linear(x, w, offsets, gather=sorted_ids, gather_div=topk)
    torch.testing.assert_close(y[:valid].float(), ref[:valid].float(), rtol=2e-2, atol=2e-2)


def test_moe_route_graph_capture(moe_lib):
    e, n, k, topk = MOE_SHAPES[0]
    t, _ = moe_routed(moe_lib, e, n, k, topk)
    mw = nf12.encode_moe(experts(e, n, k, spread=1))
    x = rand(t, k, seed=1)
    _, ids = routing(t, e, topk)
    sorted_ids, offsets = nf12.moe_align(ids, e)
    run = lambda: nf12.moe_linear(x, mw, offsets, gather=sorted_ids, gather_div=topk)  # noqa: E731
    stream_ = torch.cuda.Stream()
    stream_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream_):
        run()
    torch.cuda.current_stream().wait_stream(stream_)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = run()
    for seed in (2, 3):
        x.copy_(rand(t, k, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(y), bits(run()))
