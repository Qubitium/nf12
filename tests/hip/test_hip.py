# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
"""C-ABI tests for the RDNA4 HIP library (``libnf12_hip.so``).

Validated on ROCm 10.1.0 / HIP 7.16.26385 (R9700); see README.md here.

These tests drive the plain C API in ``include/nf12/nf12_hip.h`` through ctypes. Accuracy uses the same
``|err| <= 2e-3 + 2e-3*|ref|`` gate as the CUDA tests.

The primary contract is exact equality with the matched BF16t kernel
(SPEC.md section 6.2). A separate dense BF16 GEMM check allows the normal
accumulation-order differences of section 6.3.
"""

from __future__ import annotations

import ctypes
import threading

import nf12
import pytest
import torch
import torch.nn.functional as F
from nf12 import reference


def _arch() -> str:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]


pytestmark = pytest.mark.skipif(
    _arch() not in ("gfx1200", "gfx1201"),
    reason="requires an RDNA4 (gfx1200/gfx1201) ROCm device",
)

_VOID = ctypes.c_void_p
_I64 = ctypes.c_int64
_SIZE = ctypes.c_size_t
_U8 = ctypes.c_uint8
_U32 = ctypes.c_uint32
_INT = ctypes.c_int


class _Nf12Weight(ctypes.Structure):
    """``nf12_weight_t`` from include/nf12/nf12.h, for the CUDA-named C ABI."""

    _fields_ = [
        ("packed", _VOID),
        ("escapes", _VOID),
        ("num_escapes", _U32),
        ("n", _I64),
        ("k", _I64),
        ("exp_base", _U8),
    ]


@pytest.fixture(scope="module")
def lib():
    path = nf12.library_path()
    assert path.name == "libnf12_hip.so", path
    lib = ctypes.CDLL(str(path))
    lib.nf12_hip_encode_workspace_bytes.restype = _SIZE
    lib.nf12_hip_encode_workspace_bytes.argtypes = [_I64, _I64]
    lib.nf12_hip_encode_analyze.restype = _INT
    lib.nf12_hip_encode_analyze.argtypes = [
        _VOID, _I64, _I64, _I64, _VOID, _SIZE,
        ctypes.POINTER(_U8), ctypes.POINTER(_U32), _VOID,
    ]
    lib.nf12_hip_encode.restype = _INT
    lib.nf12_hip_encode.argtypes = [
        _VOID, _I64, _I64, _I64, _U8, _VOID, _VOID, _U32,
        _VOID, _SIZE, _VOID,
    ]
    lib.nf12_hip_decode.restype = _INT
    lib.nf12_hip_decode.argtypes = [
        _VOID, _VOID, _U8, _I64, _I64, _VOID, _I64, _VOID,
    ]
    lib.nf12_hip_selftest.restype = _INT
    lib.nf12_hip_selftest.argtypes = [_VOID, _VOID]
    lib.nf12_hip_linear_fused_small_m_supported.restype = _INT
    lib.nf12_hip_linear_fused_small_m_supported.argtypes = [_I64]
    lib.nf12_hip_linear_fused.restype = _INT
    lib.nf12_hip_linear_fused.argtypes = [
        _VOID, _VOID, _U8, _VOID, _VOID, _I64, _I64, _I64, _VOID, _VOID,
    ]
    lib.nf12_hip_wmma_max_m.restype = _INT
    lib.nf12_hip_wmma_max_m.argtypes = []
    lib.nf12_hip_wmma_supported.restype = _INT
    lib.nf12_hip_wmma_supported.argtypes = [_I64]
    lib.nf12_hip_wmma_supported_device.restype = _INT
    lib.nf12_hip_wmma_supported_device.argtypes = [_I64, _INT]
    lib.nf12_hip_linear_wmma_workspace_bytes.restype = _SIZE
    lib.nf12_hip_linear_wmma_workspace_bytes.argtypes = [_I64, _I64, _I64]
    lib.nf12_hip_linear_wmma.restype = _INT
    lib.nf12_hip_linear_wmma.argtypes = [
        _VOID, _VOID, _U8, _VOID, _VOID, _I64, _I64, _I64, _VOID,
        _VOID, _SIZE, _VOID,
    ]
    lib.nf12_hip_linear_fused_bf16t.restype = _INT
    lib.nf12_hip_linear_fused_bf16t.argtypes = [
        _VOID, _VOID, _VOID, _I64, _I64, _I64, _VOID, _VOID,
    ]
    lib.nf12_hip_linear_wmma_bf16t.restype = _INT
    lib.nf12_hip_linear_wmma_bf16t.argtypes = [
        _VOID, _VOID, _VOID, _I64, _I64, _I64, _VOID, _VOID, _SIZE, _VOID,
    ]
    # The CUDA-named C ABI, so the ROCm status codes can be compared with the
    # CUDA build's (csrc/api.cpp) rather than only with the HIP-level ones.
    lib.nf12_linear.restype = _INT
    lib.nf12_linear.argtypes = [
        _VOID, _I64, _I64, ctypes.POINTER(_Nf12Weight), _VOID, _VOID, _I64,
        _VOID, _SIZE, _VOID,
    ]
    lib.nf12_decode.restype = _INT
    lib.nf12_decode.argtypes = [ctypes.POINTER(_Nf12Weight), _VOID, _I64, _VOID]
    lib.nf12_moe_combine.restype = _INT
    lib.nf12_moe_combine.argtypes = [_VOID, _I64, _I64, _I64, _I64, _VOID, _I64, _VOID]
    lib.nf12_moe_silu_mul.restype = _INT
    lib.nf12_moe_silu_mul.argtypes = [_VOID, _I64, _I64, _I64, _VOID, _I64, _VOID]
    lib.nf12_moe_grouped_operands.restype = _INT
    lib.nf12_moe_grouped_operands.argtypes = [
        _VOID, _I64, _VOID, _I64, _VOID, _I64, _I64, _I64, _VOID, _I64,
        _VOID, _VOID,
    ]
    lib.nf12_packed_bytes.restype = _SIZE
    lib.nf12_packed_bytes.argtypes = [_I64, _I64]
    lib.nf12_bf16t_bytes.restype = _SIZE
    lib.nf12_bf16t_bytes.argtypes = [_I64, _I64]
    lib.nf12_encode_workspace_bytes.restype = _SIZE
    lib.nf12_encode_workspace_bytes.argtypes = [_I64, _I64]
    return lib


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _encode(lib, w: torch.Tensor):
    n, k = w.shape
    ws_bytes = lib.nf12_hip_encode_workspace_bytes(n, k)
    ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device=w.device)
    eb = _U8(0)
    ne = _U32(0)
    rc = lib.nf12_hip_encode_analyze(
        w.data_ptr(), n, k, k, ws.data_ptr(), ws_bytes,
        ctypes.byref(eb), ctypes.byref(ne), _stream(),
    )
    assert rc == 0, rc
    packed = torch.empty((n, k * 3 // 2), dtype=torch.uint8, device=w.device)
    escapes = torch.empty((int(ne.value), 8), dtype=torch.int16, device=w.device)
    rc = lib.nf12_hip_encode(
        w.data_ptr(), n, k, k, eb, packed.data_ptr(), escapes.data_ptr(),
        int(ne.value), ws.data_ptr(), ws_bytes, _stream(),
    )
    assert rc == 0, rc
    return packed, escapes, int(eb.value)


def _decode(lib, packed, escapes, exp_base):
    n, row_bytes = packed.shape
    k = row_bytes * 2 // 3
    out = torch.empty((n, k), dtype=torch.bfloat16, device=packed.device)
    rc = lib.nf12_hip_decode(
        packed.data_ptr(), escapes.data_ptr(), exp_base, n, k, out.data_ptr(), k,
        _stream(),
    )
    assert rc == 0, rc
    return out


def _linear(lib, x, packed, escapes, exp_base, bias=None):
    m, k = x.shape
    n = packed.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    bias_ptr = None if bias is None else bias.data_ptr()
    if m < 16:
        rc = lib.nf12_hip_linear_fused(
            packed.data_ptr(), escapes.data_ptr(), exp_base, x.data_ptr(),
            out.data_ptr(), n, k, m, bias_ptr, _stream(),
        )
    else:
        ws_bytes = lib.nf12_hip_linear_wmma_workspace_bytes(n, k, m)
        ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device=x.device)
        rc = lib.nf12_hip_linear_wmma(
            packed.data_ptr(), escapes.data_ptr(), exp_base, x.data_ptr(),
            out.data_ptr(), n, k, m, bias_ptr, ws.data_ptr(), ws_bytes, _stream(),
        )
    assert rc == 0, rc
    return out


def _violations(got, ref) -> int:
    err = (got.double() - ref).abs()
    tol = 2e-3 + 2e-3 * ref.abs()
    # A NaN in `got` compares False against every tolerance, so a naive
    # `err > tol` would report a clean pass. Non-finite outputs are violations.
    bad = (err > tol) | ~torch.isfinite(got)
    return int(bad.sum())


def test_selftest(lib):
    out = torch.zeros(1, dtype=torch.uint32, device="cuda")
    rc = lib.nf12_hip_selftest(out.data_ptr(), _stream())
    assert rc == 0
    assert int(out.item()) == 0x4E463132




@pytest.mark.parametrize("n,k", [(16, 64), (256, 256), (1024, 4096)])
def test_native_codec_round_trip(lib, n, k):
    """Native encode/decode match the reference and reject short workspaces."""
    torch.manual_seed(0)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    need = int(lib.nf12_hip_encode_workspace_bytes(n, k))
    assert need > 0
    ws = torch.full((need,), 0xA5, dtype=torch.uint8, device="cuda")
    base, count = _U8(0), _U32(0)
    for declared in (0, need - 1):
        assert lib.nf12_hip_encode_analyze(w.data_ptr(), n, k, k, ws.data_ptr(), declared,
                                           ctypes.byref(base), ctypes.byref(count), _stream()) != 0
    assert int((ws != 0xA5).sum()) == 0
    packed, escapes, exp_base = _encode(lib, w)
    ref_packed, ref_escapes, ref_base = reference.encode(w.cpu())
    assert exp_base == ref_base
    assert packed.cpu().equal(ref_packed) and escapes.cpu().equal(ref_escapes)
    assert _decode(lib, packed, escapes, exp_base).view(torch.int16).equal(w.view(torch.int16))


@pytest.mark.parametrize("n,k", [(256, 256), (1024, 4096)])
@pytest.mark.parametrize("m", [1, 15, 16, 17, 64, 256])
def test_native_linear_matches_twin_and_dense(lib, n, k, m):
    """SPEC 6.2/6.3: the native routes match the bf16t twin bitwise."""
    if m >= 16 and lib.nf12_hip_wmma_supported(m) == 0:
        pytest.skip(f"wmma path not available for m={m}")
    torch.manual_seed(m)
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    x = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    packed, escapes, exp_base = _encode(lib, w)
    got = _linear(lib, x, packed, escapes, exp_base)
    assert bool(torch.isfinite(got).all())
    assert torch.equal(got.view(torch.int16), _twin_linear(lib, x, w).view(torch.int16))
    decoded = _decode(lib, packed, escapes, exp_base)
    assert torch.equal(decoded.view(torch.int16), w.view(torch.int16))
    assert _violations(got, F.linear(x, decoded).double()) == 0


def test_route_bounds(lib):
    assert lib.nf12_hip_wmma_max_m() >= 64
    assert lib.nf12_hip_linear_fused_small_m_supported(1) == 1
    assert lib.nf12_hip_linear_fused_small_m_supported(16) == 1


def test_wmma_supported_device_matches_the_current_device(lib):
    """The op layer routes on an explicit device without switching it.

    `nf12_linear_decode_gemm_rows` calls this for the weight's device, so it
    must answer for that ordinal and must reject an ordinal that does not
    exist instead of reading out of bounds.
    """
    current = torch.cuda.current_device()
    for m in (16, 64, 256):
        assert lib.nf12_hip_wmma_supported_device(m, current) == lib.nf12_hip_wmma_supported(m)
    assert lib.nf12_hip_wmma_supported_device(16, -1) == 0
    assert lib.nf12_hip_wmma_supported_device(16, 4096) == 0






def _twin_linear(lib, x, weight):
    m, k = x.shape
    n = weight.shape[0]
    twin = nf12.pack_bf16t(weight)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    args = (twin.data_ptr(), x.data_ptr(), out.data_ptr(), n, k, m, None)
    if m < 16:
        assert lib.nf12_hip_linear_fused_bf16t(*args, _stream()) == 0
    else:
        need = lib.nf12_hip_linear_wmma_workspace_bytes(n, k, m)
        workspace = torch.empty(max(need, 1), device=x.device, dtype=torch.uint8)
        assert lib.nf12_hip_linear_wmma_bf16t(*args, workspace.data_ptr(), need, _stream()) == 0
    return out










def test_wmma_split_k_requires_workspace(lib):
    """Split-K scratch is caller-owned and required, exactly like the CUDA build.

    Regression for the process-wide ``g_partial`` pointer: two concurrent
    streams used to share one partial plane and corrupt each other, so the
    library never allocates this itself and a null or short buffer is rejected
    rather than aliased. ``csrc/api.cpp`` (``run_linear``) has the same contract.
    """
    n, k, m = 1024, 4096, 64
    assert lib.nf12_hip_linear_wmma_workspace_bytes(n, k, m) > 0
    torch.manual_seed(3)
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    x = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    packed, escapes, exp_base = _encode(lib, w)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    need = lib.nf12_hip_linear_wmma_workspace_bytes(n, k, m)
    args = (packed.data_ptr(), escapes.data_ptr(), exp_base, x.data_ptr(),
            out.data_ptr(), n, k, m, None)
    assert lib.nf12_hip_linear_wmma(*args, None, 0, _stream()) != 0
    short = torch.empty(need - 1, dtype=torch.uint8, device="cuda")
    assert lib.nf12_hip_linear_wmma(
        *args, short.data_ptr(), short.numel(), _stream()) != 0






def test_wmma_concurrent_streams_are_independent(lib):
    if lib.nf12_hip_wmma_supported(64) == 0:
        pytest.skip("wmma path not available")
    torch.manual_seed(5)
    n, k, m = 1024, 4096, 64
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    x1 = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    x2 = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    packed, escapes, exp_base = _encode(lib, w)
    ref1 = _linear(lib, x1, packed, escapes, exp_base)
    ref2 = _linear(lib, x2, packed, escapes, exp_base)

    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    current = torch.cuda.current_stream()
    s1.wait_stream(current)
    s2.wait_stream(current)
    with torch.cuda.stream(s1):
        got1 = _linear(lib, x1, packed, escapes, exp_base)
    with torch.cuda.stream(s2):
        got2 = _linear(lib, x2, packed, escapes, exp_base)
    torch.cuda.synchronize()
    assert torch.equal(got1, ref1)
    assert torch.equal(got2, ref2)


def test_wmma_concurrent_host_threads(lib):
    """Free-threaded (GIL=0) and multi-stream: one stream and workspace each."""
    if lib.nf12_hip_wmma_supported(64) == 0:
        pytest.skip("wmma path not available")
    torch.manual_seed(9)
    n, k, m = 1024, 4096, 64
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    xs = [(torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda() for _ in range(4)]
    packed, escapes, exp_base = _encode(lib, w)
    refs = [_linear(lib, x, packed, escapes, exp_base) for x in xs]

    got: list[torch.Tensor | None] = [None] * len(xs)

    def run(i: int) -> None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            got[i] = _linear(lib, xs[i], packed, escapes, exp_base)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(xs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    for i in range(len(xs)):
        assert got[i] is not None
        assert torch.equal(got[i], refs[i])










@pytest.mark.parametrize("n,k,m", [
    (256, 256, 256), (256, 256, 257), (256, 256, 271),
    (256, 256, 272), (256, 256, 512), (4096, 4096, 288),
])
def test_wmma_bf16t_twin_accepts_large_m(lib, n, k, m):
    """Exercise the native symbol, including small tails and tail-sized scratch."""
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        wt = nf12.pack_bf16t(w)
        packed, escapes, base = _encode(lib, w)
        weight = _Nf12Weight(packed.data_ptr(), escapes.data_ptr(), escapes.shape[0], n, k, base)
        x = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
        need = lib.nf12_linear_workspace_bytes(m, n, k, torch.cuda.current_device())
        ws = torch.full((need + 32,), 0xA5, dtype=torch.uint8, device="cuda")
        guarded = torch.full((m + 2, n), -7., dtype=torch.bfloat16, device="cuda")
        got = guarded[1:-1]
        args = (wt.data_ptr(), x.data_ptr(), got.data_ptr(), n, k, m, None, ws.data_ptr())
        if need:
            assert lib.nf12_hip_linear_wmma_bf16t(*args, need - 1, _stream()) == 1
            assert bool((guarded == -7).all())
        assert lib.nf12_hip_linear_wmma_bf16t(*args, need, _stream()) == 0
        ref = torch.empty_like(got)
        assert lib.nf12_linear(x.data_ptr(), m, k, ctypes.byref(weight), None,
                               ref.data_ptr(), n, ws.data_ptr(), need, _stream()) == 0
    stream.synchronize()
    assert bool(torch.isfinite(got).all())
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    assert bool((guarded[[0, -1]] == -7).all())
    assert bool((ws[need:] == 0xA5).all())
