# SPDX-License-Identifier: Apache-2.0
"""Shared C-ABI contract regressions, run on whichever backend is loaded."""

import ctypes as C

import nf12
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU required")


class Weight(C.Structure):
    _fields_ = [
        ("packed", C.c_void_p), ("escapes", C.c_void_p),
        ("num_escapes", C.c_uint32), ("n", C.c_int64),
        ("k", C.c_int64), ("exp_base", C.c_uint8),
    ]


class MoeWeight(C.Structure):
    _fields_ = [("packed", C.c_void_p), ("escapes", C.c_void_p),
                ("escape_offsets", C.c_void_p), ("exp_base", C.c_void_p),
                ("experts", C.c_int64), ("n", C.c_int64), ("k", C.c_int64)]


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


@pytest.fixture(scope="module")
def abi():
    lib = C.CDLL(str(nf12.library_path()))
    p, i, sz = C.c_void_p, C.c_int64, C.c_size_t
    for name, restype, args in [
        ("nf12_moe_linear_workspace_bytes", sz, [i, i, i, i, C.c_int]),
        ("nf12_moe_linear", C.c_int, [p, i, p, i, p, p, i, C.POINTER(MoeWeight), p, p, i, p, sz, p]),
        ("nf12_moe_decode", C.c_int, [C.POINTER(MoeWeight), p, i, i, i, p, i, p]),
        ("nf12_linear_workspace_bytes", sz, [i, i, i, C.c_int]),
        ("nf12_linear_decode_gemm_rows", i, [i, i, i, C.c_int]),
        ("nf12_linear", C.c_int, [p, i, i, C.POINTER(Weight), p, p, i, p, sz, p]),
        ("nf12_linear_bf16t", C.c_int, [p, i, i, p, i, i, p, p, i, p, sz, p]),
        ("nf12_moe_align_workspace_bytes", sz, [i, i, i]),
        ("nf12_moe_combine", C.c_int, [p, i, i, i, i, p, i, p]),
        ("nf12_moe_grouped_operands", C.c_int, [p, i, p, i, p, i, i, i, p, i, p, p]),
    ]:
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, args
    return lib


def _weight(nw):
    n, k = nw.shape
    return Weight(nw.packed.data_ptr(), nw.escapes.data_ptr() if nw.num_escapes else None,
                  nw.num_escapes, n, k, nw.exp_base)


@pytest.mark.parametrize("row0,rows,ldo", [
    (16, 16, 64), (0, 32, 64), (32, 0, 64),
    (0, 16, 0), (0, 16, 56), (0, 16, 65),
    (2**63 - 16, 16, 64), (0, 2**63 - 16, 64),
])
def test_moe_decode_rejects_invalid_slice_or_stride(abi, row0, rows, ldo):
    # A trailing encoded matrix makes an out-of-range read observable safely.
    w = torch.cat([torch.ones(16, 64), torch.full((16, 64), 2.)]).bfloat16().cuda()
    nw = nf12.encode(w)
    ends = torch.tensor([0, nw.num_escapes], device="cuda", dtype=torch.int64)
    base = torch.tensor([nw.exp_base], device="cuda", dtype=torch.uint8)
    out = torch.full((34, 80), -7., device="cuda", dtype=torch.bfloat16)
    cw = MoeWeight(nw.packed.data_ptr(), nw.escapes.data_ptr(), ends.data_ptr(),
                   base.data_ptr(), 1, 16, 64)
    assert abi.nf12_moe_decode(C.byref(cw), None, 0, row0, rows,
                               out[1:].data_ptr(), ldo, _stream()) == 1
    torch.cuda.synchronize()
    assert bool((out == -7).all())


def test_moe_decode_grid_overflow_is_invalid_argument(abi):
    experts, n, k = 65536, 16, 64
    packed = torch.empty((experts, n, k * 3 // 2), device="cuda", dtype=torch.uint8)
    offsets = torch.zeros(experts + 1, device="cuda", dtype=torch.int64)
    bases = torch.zeros(experts, device="cuda", dtype=torch.uint8)
    out = torch.full((experts, n, k), -7., device="cuda", dtype=torch.bfloat16)
    cw = MoeWeight(packed.data_ptr(), None, offsets.data_ptr(), bases.data_ptr(), experts, n, k)
    assert abi.nf12_moe_decode(C.byref(cw), None, 0, 0, n, out.data_ptr(), k, _stream()) == 1
    torch.cuda.synchronize()
    assert bool((out == -7).all())


@pytest.mark.parametrize("offsets", [
    [-1, 1], [0, 3, 1, 4], [3, 1, 2, 4], [-9, -4, 2, 99],
    [2**63 - 1, -(2**63), 0, 4],
])
def test_moe_offsets_are_clamped_and_monotone(abi, offsets):
    e, n, k, count = len(offsets) - 1, 16, 64, 4
    w = torch.stack([torch.full((n, k), j + 1., device="cuda", dtype=torch.bfloat16)
                     for j in range(e)])
    nw = nf12.encode_moe(w)
    cw = MoeWeight(nw.packed.data_ptr(), nw.escapes.data_ptr(), nw.escape_offsets.data_ptr(),
                   nw.exp_base.data_ptr(), e, n, k)
    raw = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    normalized = raw.clamp(0, count).cummax(0).values
    x = torch.ones((count + 2, k), device="cuda", dtype=torch.bfloat16)
    need = abi.nf12_moe_linear_workspace_bytes(count, e, n, k, torch.cuda.current_device())
    ws = torch.empty(max(1, need), device="cuda", dtype=torch.uint8)
    results = []
    for off in (raw, normalized):
        out = torch.full((count + 2, n), -7., device="cuda", dtype=torch.bfloat16)
        assert abi.nf12_moe_linear(x[1:].data_ptr(), k, None, 1, None, off.data_ptr(), count,
                                   C.byref(cw), None, out[1:].data_ptr(), n,
                                   ws.data_ptr(), need, _stream()) == 0
        decoded = torch.full((e, n, k), -7., device="cuda", dtype=torch.bfloat16)
        assert abi.nf12_moe_decode(C.byref(cw), off.data_ptr(), count, 0, n,
                                   decoded.data_ptr(), k, _stream()) == 0
        ends = torch.empty(e, device="cuda", dtype=torch.int32)
        assert abi.nf12_moe_grouped_operands(None, k, None, 1, off.data_ptr(), count, e, k,
                                             None, k, ends.data_ptr(), _stream()) == 0
        torch.cuda.synchronize()
        assert bool((out[[0, -1]] == -7).all())
        assert torch.equal(ends, normalized[1:].int())
        results.append((out, decoded))
    for raw_result, norm_result in zip(*results):
        assert torch.equal(raw_result.view(torch.int16), norm_result.view(torch.int16))


@pytest.mark.parametrize("m", [0, 1, 16])
def test_bf16t_rejects_misaligned_bias_before_writing(abi, m):
    n, k = 16, 64
    twin = nf12.pack_bf16t(torch.ones((n, k), device="cuda", dtype=torch.bfloat16))
    x = torch.ones((max(m, 1), k), device="cuda", dtype=torch.bfloat16)
    out = torch.full((max(m, 1), n), -7., device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(n + 8, device="cuda", dtype=torch.bfloat16)[1:1 + n]
    assert abi.nf12_linear_bf16t(x.data_ptr(), m, k, twin.data_ptr(), n, k,
                                 bias.data_ptr(), out.data_ptr(), n, None, 0, _stream()) == 1
    assert bool((out == -7).all())


@pytest.mark.parametrize("m", [1, 15, 16, 17, 64, 128, 256, 257, 272])
def test_padded_linear_strides_preserve_guard_columns(abi, m):
    n, k = 256, 512
    generator = torch.Generator(device="cuda").manual_seed(m)
    w = (torch.randn(n, k, device="cuda", generator=generator) * .02).bfloat16()
    nw, twin = nf12.encode(w), nf12.pack_bf16t(w)
    cw = _weight(nw)
    xpad = (torch.randn(m, k + 8, device="cuda", generator=generator) * .05).bfloat16()
    x = xpad[:, :k]
    dense_x = x.contiguous()
    bias = (torch.randn(n, device="cuda", generator=generator) * .01).bfloat16()
    need = abi.nf12_linear_workspace_bytes(m, n, k, torch.cuda.current_device())
    ws = torch.empty(max(1, need), device="cuda", dtype=torch.uint8)
    expected = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    assert abi.nf12_linear(dense_x.data_ptr(), m, k, C.byref(cw), bias.data_ptr(),
                           expected.data_ptr(), n, ws.data_ptr(), need, _stream()) == 0
    for bf16t in (False, True):
        guarded = torch.full((m + 2, n + 8), -7., device="cuda", dtype=torch.bfloat16)
        out = guarded[1:-1, :n]
        if bf16t:
            rc = abi.nf12_linear_bf16t(x.data_ptr(), m, x.stride(0), twin.data_ptr(), n, k,
                                       bias.data_ptr(), out.data_ptr(), out.stride(0),
                                       ws.data_ptr(), need, _stream())
        else:
            rc = abi.nf12_linear(x.data_ptr(), m, x.stride(0), C.byref(cw), bias.data_ptr(),
                                 out.data_ptr(), out.stride(0), ws.data_ptr(), need, _stream())
        assert rc == 0
        torch.cuda.synchronize()
        assert bool((guarded[[0, -1]] == -7).all())
        assert bool((guarded[:, n:] == -7).all())
        assert torch.equal(out.contiguous().view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("n,k", [(4096, 4096), (14336, 4096), (65536, 4096), (65552, 4096)])
def test_decode_slice_budget(abi, n, k):
    rows = abi.nf12_linear_decode_gemm_rows(512, n, k, torch.cuda.current_device())
    if rows == 0:
        pytest.skip("this backend selects the fused route for these shapes")
    assert rows % 16 == 0 and 16 <= rows <= n
    assert rows * k * 2 <= 256 * 1024 * 1024
    if n * k * 2 <= 256 * 1024 * 1024:
        assert rows == n


@pytest.mark.parametrize("m,n,k", [(257, 4096, 4096), (512, 4096, 4096), (257, 65552, 2048)])
def test_decode_slices_match_bf16_gemm(abi, m, n, k):
    generator = torch.Generator(device="cuda").manual_seed(11)
    w = (torch.randn(n, k, device="cuda", generator=generator) * .02).bfloat16()
    x = (torch.randn(m, k, device="cuda", generator=generator) * .05).bfloat16()
    bias = (torch.randn(n, device="cuda", generator=generator) * .01).bfloat16()
    nw = nf12.encode(w)
    rows = abi.nf12_linear_decode_gemm_rows(m, n, k, torch.cuda.current_device())
    if rows == 0:
        pytest.skip("this backend selects the fused route for these shapes")
    expected = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for start in range(0, n, rows):
        end = min(start + rows, n)
        torch.addmm(bias[start:end], x, w[start:end].t(), out=expected[:, start:end])
    out = nf12.linear(x, nw, bias)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def test_moe_decode_accepts_large_grid_x(abi):
    n, k = 524288, 64  # 32768 blocks exceeds the old 15-bit cap.
    w = torch.ones((1, n, k), dtype=torch.bfloat16, device="cuda")
    nw = nf12.encode_moe(w)
    cw = MoeWeight(nw.packed.data_ptr(), nw.escapes.data_ptr(), nw.escape_offsets.data_ptr(),
                   nw.exp_base.data_ptr(), 1, n, k)
    offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    out = torch.empty_like(w)
    assert abi.nf12_moe_decode(C.byref(cw), offsets.data_ptr(), 1, 0, n,
                               out.data_ptr(), k, _stream()) == 0
    torch.cuda.synchronize()
    assert torch.equal(w.view(torch.int16), out.view(torch.int16))


def test_moe_invalid_expert_count_and_empty_topk(abi):
    assert abi.nf12_moe_align_workspace_bytes(1, 1, 2**31) == 0
    assert abi.nf12_moe_combine(None, 8, 0, 2**31, 8, None, 8, _stream()) == 1


def test_grouped_operands_without_gather_leaves_output_untouched(abi):
    """The no-gather form publishes group ends and copies nothing."""
    experts, k = 2, 8
    x = torch.ones((2, k), dtype=torch.bfloat16, device="cuda")
    out = torch.full_like(x, 7.0)
    offsets = torch.tensor([0, 2, 2], dtype=torch.int64, device="cuda")
    ends = torch.full((experts,), -1, dtype=torch.int32, device="cuda")
    assert abi.nf12_moe_grouped_operands(
        x.data_ptr(), k, None, 1, offsets.data_ptr(), 2, experts, k,
        out.data_ptr(), k, ends.data_ptr(), _stream(),
    ) == 0
    torch.cuda.synchronize()
    assert int((out != 7).sum()) == 0
    assert ends.tolist() == [2, 2]
