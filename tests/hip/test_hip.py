# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
"""C-ABI tests for the RDNA4 HIP library (``libnf12_hip.so``).

The CUDA torch extension is not built on ROCm, so these tests drive the plain
C API in ``include/nf12/nf12_hip.h`` through ctypes. Accuracy uses the same
``|err| <= 2e-3 + 2e-3*|ref|`` gate as the CUDA tests.

The primary contract test is against a BF16 twin (SPEC.md section 6.2): NF12
adds zero error relative to a BF16 GEMM on the exactly-decoded weight. The FP64
oracle is kept as a tighter secondary check.
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
    lib.nf12_hip_linear_wmma_workspace_bytes.restype = _SIZE
    lib.nf12_hip_linear_wmma_workspace_bytes.argtypes = [_I64, _I64, _I64]
    lib.nf12_hip_linear_wmma.restype = _INT
    lib.nf12_hip_linear_wmma.argtypes = [
        _VOID, _VOID, _U8, _VOID, _VOID, _I64, _I64, _I64, _VOID,
        _VOID, _SIZE, _VOID,
    ]
    # The CUDA-named C ABI, so the ROCm status codes can be compared with the
    # CUDA build's (csrc/api.cu) rather than only with the HIP-level ones.
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


def test_library_path_is_stable_across_threads(lib):
    """The JIT cache and loader locks are free-threaded safe."""
    paths: list = [None] * 8

    def run(i: int) -> None:
        paths[i] = nf12.library_path()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(paths))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(path == paths[0] for path in paths)
    assert paths[0].name == "libnf12_hip.so"


def test_route_bounds(lib):
    assert lib.nf12_hip_wmma_max_m() >= 64
    assert lib.nf12_hip_linear_fused_small_m_supported(1) == 1
    assert lib.nf12_hip_linear_fused_small_m_supported(16) == 1


@pytest.mark.parametrize("n,k", [(16, 64), (256, 256), (1024, 4096)])
def test_encode_matches_reference(lib, n, k):
    torch.manual_seed(0)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    packed, escapes, exp_base = _encode(lib, w)
    ref_packed, ref_escapes, ref_base = reference.encode(w.cpu())
    assert exp_base == ref_base
    assert torch.equal(packed.cpu(), ref_packed)
    assert torch.equal(escapes.cpu(), ref_escapes)


def test_decode_is_bit_exact(lib):
    torch.manual_seed(1)
    w = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
    specials = torch.tensor(
        [0x0000, 0x8000, 0x0001, 0x007F, 0x7F80, 0xFF80, 0x7FC0, 0x7FFF,
         0xFFC0, 0x3F80, 0xBF80, 0x0002, 0x0080, 0x7F7F, 0xFF7F, 0x3F00],
        dtype=torch.uint16,
    ).view(torch.int16)
    w.view(torch.int16).view(-1)[: specials.numel()] = specials
    packed, escapes, exp_base = _encode(lib, w)
    got = _decode(lib, packed, escapes, exp_base)
    assert torch.equal(got.view(torch.int16), w.view(torch.int16))


@pytest.mark.parametrize("n,k", [(256, 256), (1024, 4096)])
@pytest.mark.parametrize("m", [1, 4, 15])
def test_fused_linear_matches_fp64_oracle(lib, n, k, m):
    torch.manual_seed(m)
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    x = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    packed, escapes, exp_base = _encode(lib, w)
    ref = x.double() @ w.double().t()
    assert _violations(_linear(lib, x, packed, escapes, exp_base), ref) == 0


@pytest.mark.parametrize("m", [16, 17, 32, 64, 128, 256])
def test_wmma_linear_matches_fp64_oracle(lib, m):
    if lib.nf12_hip_wmma_supported(m) == 0:
        pytest.skip(f"wmma path not available for m={m}")
    torch.manual_seed(m)
    n, k = 1024, 4096
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    x = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    packed, escapes, exp_base = _encode(lib, w)
    ref = x.double() @ w.double().t()
    assert _violations(_linear(lib, x, packed, escapes, exp_base), ref) == 0


@pytest.mark.parametrize("m", [1, 4, 15])
def test_fused_linear_matches_bf16_twin(lib, m):
    """SPEC.md 6.2: the contract is BF16-in/BF16-out, not an FP64 ideal.

    The twin is a BF16 GEMM over the exactly-decoded weight, i.e. the same
    numbers the caller would get from ``decode`` + ``F.linear``.
    """
    torch.manual_seed(m)
    n, k = 1024, 4096
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    x = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    packed, escapes, exp_base = _encode(lib, w)
    decoded = _decode(lib, packed, escapes, exp_base)
    assert torch.equal(decoded.view(torch.int16), w.view(torch.int16))
    ref = F.linear(x, decoded)
    assert _violations(_linear(lib, x, packed, escapes, exp_base), ref.double()) == 0


@pytest.mark.parametrize("m", [16, 17, 32, 64, 128, 256])
def test_wmma_linear_matches_bf16_twin(lib, m):
    """SPEC.md 6.2 contract for the matrix-core route."""
    if lib.nf12_hip_wmma_supported(m) == 0:
        pytest.skip(f"wmma path not available for m={m}")
    torch.manual_seed(m)
    n, k = 1024, 4096
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    x = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    bias = torch.full((n,), 0.125, dtype=torch.bfloat16, device="cuda")
    packed, escapes, exp_base = _encode(lib, w)
    decoded = _decode(lib, packed, escapes, exp_base)
    ref = F.linear(x, decoded, bias)
    got = _linear(lib, x, packed, escapes, exp_base, bias)
    assert _violations(got, ref.double()) == 0


def test_wmma_split_k_requires_workspace(lib):
    """Split-K scratch is caller-owned and required, exactly like the CUDA build.

    Regression for the process-wide ``g_partial`` pointer: two concurrent
    streams used to share one partial plane and corrupt each other, so the
    library never allocates this itself and a null or short buffer is rejected
    rather than aliased. ``csrc/api.cu`` (``run_linear``) has the same contract.
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


def test_c_abi_workspace_status_matches_cuda(lib):
    """``nf12_linear`` must return the CUDA build's status for a missing buffer.

    ``csrc/api.cu`` (``run_linear``) returns NF12_ERROR_INSUFFICIENT_WORKSPACE
    when the buffer is too small, and NF12_ERROR_INVALID_ARGUMENT only for a
    malformed call (a non-zero byte count with no pointer, or a misaligned one).
    A caller ported from CUDA switches on these values, so the ROCm build has to
    match them rather than collapsing everything into INVALID_ARGUMENT.
    """
    nf12_ok, nf12_invalid, nf12_insufficient = 0, 1, 3
    n, k, m = 1024, 4096, 64
    need = lib.nf12_hip_linear_wmma_workspace_bytes(n, k, m)
    assert need > 0
    torch.manual_seed(31)
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    x = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    packed, escapes, exp_base = _encode(lib, w)
    view = _Nf12Weight(packed.data_ptr(), escapes.data_ptr(), escapes.shape[0],
                       n, k, exp_base)

    def call(ws_ptr, ws_bytes):
        return lib.nf12_linear(x.data_ptr(), m, k, ctypes.byref(view), None,
                               out.data_ptr(), n, ws_ptr, ws_bytes, _stream())

    assert call(None, 0) == nf12_insufficient
    assert call(None, need) == nf12_invalid
    workspace = torch.empty(need, dtype=torch.uint8, device="cuda")
    assert call(workspace.data_ptr(), need) == nf12_ok


def test_c_abi_validates_like_cuda(lib):
    """The C ABI must reject bad arguments instead of crashing or truncating.

    Regressions for four defects against csrc/api.cu: a null weight segfaulted
    in ``nf12_decode``; a null output returned NF12_OK and faulted
    asynchronously in ``nf12_decode``/``nf12_moe_combine``/``nf12_moe_silu_mul``;
    ``m == 0`` reported INVALID_ARGUMENT where CUDA returns NF12_OK; and a ``k``
    that was not a multiple of 64 silently truncated the GEMM to k/64 columns
    instead of being rejected.
    """
    nf12_ok, nf12_invalid = 0, 1
    n, k = 256, 256
    out = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
    y = torch.empty((64, n), dtype=torch.bfloat16, device="cuda")

    # Null pointers are rejected, not dereferenced.
    assert lib.nf12_decode(None, out.data_ptr(), k, _stream()) == nf12_invalid
    assert lib.nf12_moe_combine(None, n, 4, 2, n, out.data_ptr(), n, _stream()) == nf12_invalid
    assert lib.nf12_moe_combine(out.data_ptr(), n, 4, 2, n, None, n, _stream()) == nf12_invalid
    assert lib.nf12_moe_silu_mul(None, 2 * k, 4, k, out.data_ptr(), k, _stream()) == nf12_invalid

    # m == 0 is a successful no-op, exactly as on CUDA.
    torch.manual_seed(41)
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    packed, escapes, exp_base = _encode(lib, w)
    view = _Nf12Weight(packed.data_ptr(), escapes.data_ptr(), escapes.shape[0], n, k, exp_base)
    assert lib.nf12_linear(None, 0, k, ctypes.byref(view), None, y.data_ptr(), n,
                           None, 0, _stream()) == nf12_ok

    # Geometry that would truncate is rejected instead.
    assert lib.nf12_packed_bytes(-1, 64) == 0
    assert lib.nf12_bf16t_bytes(-1, 64) == 0
    assert lib.nf12_encode_workspace_bytes(24, 64) == 0
    assert lib.nf12_encode_workspace_bytes(256, 96) == 0
    x96 = torch.empty((16, 96), dtype=torch.bfloat16, device="cuda")
    assert lib.nf12_linear(x96.data_ptr(), 16, 96, ctypes.byref(view), None, y.data_ptr(),
                           n, None, 0, _stream()) == nf12_invalid


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


@pytest.mark.parametrize("m", [16, 32, 64, 128])
def test_wmma_bias_is_added_once(lib, m):
    """Regression: every split-K partial plane carried the bias, so the reduce
    kernel multiplied it by kSplits (4x at m<=64, 2x at m=128)."""

    if lib.nf12_hip_wmma_supported(m) == 0:
        pytest.skip(f"wmma path not available for m={m}")
    torch.manual_seed(m)
    # N=1024/K=1024 launches 8 CTAs, so pick_splits chooses 4 (m<=64) or 2.
    n, k = 1024, 1024
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    x = (torch.randn(m, k) * 0.05).to(torch.bfloat16).cuda()
    bias = torch.full((n,), 0.125, dtype=torch.bfloat16, device="cuda")
    packed, escapes, exp_base = _encode(lib, w)

    with_bias = _linear(lib, x, packed, escapes, exp_base, bias).double()
    without = _linear(lib, x, packed, escapes, exp_base).double()
    delta = (with_bias - without).mean().item()
    assert abs(delta - 0.125) <= 2e-3, f"bias scaled by {delta / 0.125:.3f}x"
    ref = x.double() @ w.double().t() + 0.125
    assert _violations(_linear(lib, x, packed, escapes, exp_base, bias), ref) == 0


def test_encode_rejects_undersized_workspace(lib):
    """The native encoder must not write a workspace it was not given room for."""
    torch.manual_seed(40)
    w = (torch.randn(16, 64, device="cuda") * 0.02).to(torch.bfloat16)
    need = int(lib.nf12_hip_encode_workspace_bytes(16, 64))
    assert need > 0
    ws = torch.full((need,), 0xA5, dtype=torch.uint8, device="cuda")
    exp_base, num_escapes = _U8(0), _U32(0)
    for declared in (0, need - 1):
        rc = lib.nf12_hip_encode_analyze(
            w.data_ptr(), 16, 64, 64, ws.data_ptr(), declared,
            ctypes.byref(exp_base), ctypes.byref(num_escapes), _stream(),
        )
        torch.cuda.synchronize()
        assert rc != 0, declared
    # Nothing was written: the buffer still holds its sentinel everywhere.
    assert int((ws != 0xA5).sum()) == 0


def test_grouped_operands_without_gather_leaves_output_untouched(lib):
    """The no-gather form publishes group ends and copies nothing, like CUDA."""
    n_experts, k = 2, 8
    x = torch.ones((2, k), dtype=torch.bfloat16, device="cuda")
    out = torch.full_like(x, 7.0)
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    ends = torch.full((n_experts,), -1, dtype=torch.int32, device="cuda")
    rc = lib.nf12_moe_grouped_operands(
        x.data_ptr(), k, None, 1, offsets.data_ptr(), 2, n_experts, k,
        out.data_ptr(), k, ends.data_ptr(), _stream(),
    )
    torch.cuda.synchronize()
    assert rc == 0
    assert int((out != 7).sum()) == 0
    assert ends.tolist() == [2, 2]


def test_c_abi_linear_large_m_is_supported(lib):
    """m above the matrix-core cap keeps working through the chunked route."""
    torch.manual_seed(41)
    n, k, m = 16, 64, 512
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    packed, escapes, exp_base = _encode(lib, w)
    x = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
    out = torch.full((m, n), -7.0, dtype=torch.bfloat16, device="cuda")
    weight = _Nf12Weight(packed.data_ptr(), escapes.data_ptr(), escapes.shape[0], n, k, exp_base)
    ws_bytes = int(lib.nf12_linear_workspace_bytes(m, n, k, 0))
    ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device="cuda")
    rc = lib.nf12_linear(
        x.data_ptr(), m, k, ctypes.byref(weight), None, out.data_ptr(), n,
        ws.data_ptr(), ws_bytes, _stream(),
    )
    torch.cuda.synchronize()
    assert rc == 0
    assert int((out == -7.0).sum()) == 0
    assert _violations(out, x.double() @ w.double().t()) == 0


def test_wmma_bf16t_twin_accepts_large_m(lib):
    """The twin chunks the M axis above the matrix-core cap."""
    torch.manual_seed(42)
    n, k, cap = 256, 256, 256
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    wt = nf12.pack_bf16t(w)
    x = (torch.randn(2 * cap, k, device="cuda") * 0.05).to(torch.bfloat16)
    got = nf12.linear_bf16t(x, wt)
    top = nf12.linear_bf16t(x[:cap], wt)
    bottom = nf12.linear_bf16t(x[cap:], wt)
    assert got.view(torch.int16).equal(torch.cat([top, bottom]).view(torch.int16))
