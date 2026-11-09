# SPDX-License-Identifier: Apache-2.0
"""C API (libnf12.so) tests through ctypes: status codes, validation, workspace sizing."""

from __future__ import annotations

import ctypes
import threading

import pytest
import torch

import nf12

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

OK, INVALID, UNSUPPORTED, INSUFFICIENT, CUDA_ERROR = range(5)

c_void_p, c_int64, c_size_t, c_uint8, c_uint32, c_int = (
    ctypes.c_void_p, ctypes.c_int64, ctypes.c_size_t, ctypes.c_uint8, ctypes.c_uint32, ctypes.c_int)


class Weight(ctypes.Structure):
    _fields_ = [
        ("packed", c_void_p),
        ("escapes", c_void_p),
        ("num_escapes", c_uint32),
        ("n", c_int64),
        ("k", c_int64),
        ("exp_base", c_uint8),
    ]


@pytest.fixture(scope="module")
def lib():
    lib = ctypes.CDLL(str(nf12.library_path()))
    sigs = {
        "nf12_status_string": (ctypes.c_char_p, [c_int]),
        "nf12_last_cuda_error": (c_int, []),
        "nf12_packed_bytes": (c_size_t, [c_int64, c_int64]),
        "nf12_bf16t_bytes": (c_size_t, [c_int64, c_int64]),
        "nf12_encode_workspace_bytes": (c_size_t, [c_int64, c_int64]),
        "nf12_encode_analyze": (c_int, [c_void_p, c_int64, c_int64, c_int64, c_void_p, c_size_t,
                                        ctypes.POINTER(c_uint8), ctypes.POINTER(c_uint32), c_void_p]),
        "nf12_encode": (c_int, [c_void_p, c_int64, c_int64, c_int64, c_uint8, c_void_p, c_void_p, c_uint32,
                                c_void_p, c_size_t, c_void_p]),
        "nf12_decode": (c_int, [ctypes.POINTER(Weight), c_void_p, c_int64, c_void_p]),
        "nf12_linear_workspace_bytes": (c_size_t, [c_int64, c_int64, c_int64, c_int]),
        "nf12_linear_decode_gemm_rows": (c_int64, [c_int64, c_int64, c_int64, c_int]),
        "nf12_linear": (c_int, [c_void_p, c_int64, c_int64, ctypes.POINTER(Weight), c_void_p, c_void_p, c_int64,
                                c_void_p, c_size_t, c_void_p]),
        "nf12_pack_bf16t": (c_int, [c_void_p, c_int64, c_int64, c_int64, c_void_p, c_void_p]),
        "nf12_linear_bf16t": (c_int, [c_void_p, c_int64, c_int64, c_void_p, c_int64, c_int64, c_void_p, c_void_p,
                                      c_int64, c_void_p, c_size_t, c_void_p]),
    }
    for name, (restype, argtypes) in sigs.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, argtypes
    return lib


def stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def ptr(t: torch.Tensor, byte_offset: int = 0) -> int:
    return t.data_ptr() + byte_offset


def bf16(*shape: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * 0.05).to(torch.bfloat16)


def c_encode(lib, w: torch.Tensor):
    n, k = w.shape
    ws_bytes = lib.nf12_encode_workspace_bytes(n, k)
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")
    exp_base, num_escapes = c_uint8(), c_uint32()
    assert lib.nf12_encode_analyze(ptr(w), n, k, w.stride(0), ptr(ws), ws_bytes, ctypes.byref(exp_base),
                                   ctypes.byref(num_escapes), stream()) == OK
    packed = torch.empty(lib.nf12_packed_bytes(n, k), dtype=torch.uint8, device="cuda")
    escapes = torch.empty(num_escapes.value * 16, dtype=torch.uint8, device="cuda")
    assert lib.nf12_encode(ptr(w), n, k, w.stride(0), exp_base.value, ptr(packed),
                           ptr(escapes) if num_escapes.value else None, num_escapes.value, ptr(ws), ws_bytes,
                           stream()) == OK
    weight = Weight(ptr(packed), ptr(escapes) if num_escapes.value else None, num_escapes.value, n, k,
                    exp_base.value)
    return weight, (packed, escapes)


def test_status_strings(lib):
    names = [lib.nf12_status_string(s).decode() for s in range(5)]
    assert names[0] == "NF12_OK"
    assert len(set(names)) == 5
    assert all(name.startswith("NF12_") for name in names)
    assert lib.nf12_status_string(99).decode() == "NF12: unknown status"


def test_sizes(lib):
    assert lib.nf12_packed_bytes(16, 64) == 1536
    assert lib.nf12_bf16t_bytes(16, 64) == 2048
    assert lib.nf12_packed_bytes(4096, 11008) == 4096 * 11008 * 3 // 2
    assert lib.nf12_bf16t_bytes(4096, 11008) == 4096 * 11008 * 2
    for n, k in [(0, 64), (16, 0), (8, 64), (16, 32), (-16, 64), (16, -64), (1 << 40, 1 << 40)]:
        assert lib.nf12_packed_bytes(n, k) == 0
        assert lib.nf12_bf16t_bytes(n, k) == 0
        assert lib.nf12_encode_workspace_bytes(n, k) == 0
        assert lib.nf12_linear_workspace_bytes(1, n, k, 0) == 0
    assert lib.nf12_encode_workspace_bytes(16, 64) > 0


def test_linear_workspace_bytes(lib):
    dev = torch.cuda.current_device()
    assert lib.nf12_linear_workspace_bytes(0, 4096, 4096, dev) == 0
    assert lib.nf12_linear_workspace_bytes(-1, 4096, 4096, dev) == 0
    for m in (1, 16, 272):
        assert lib.nf12_linear_workspace_bytes(m, 4096, 4096, -1) == 0
        assert lib.nf12_linear_workspace_bytes(m, 4096, 4096, torch.cuda.device_count()) == 0
    # Valid queries are pure functions of their arguments.
    for m, n, k in [(1, 4096, 4096), (16, 1024, 16384), (512, 4096, 4096)]:
        a = lib.nf12_linear_workspace_bytes(m, n, k, dev)
        assert a == lib.nf12_linear_workspace_bytes(m, n, k, dev)


def test_linear_decode_gemm_rows(lib):
    dev = torch.cuda.current_device()
    for m, n, k, d in [(0, 4096, 4096, dev), (-1, 4096, 4096, dev), (1 << 31, 4096, 4096, dev), (4096, 8, 64, dev),
                       (4096, 16, 32, dev), (4096, 4096, 4096, -1), (4096, 4096, 4096, torch.cuda.device_count())]:
        assert lib.nf12_linear_decode_gemm_rows(m, n, k, d) == 0
    for m in (1, 16, 64):  # small M never decodes the weight first
        assert lib.nf12_linear_decode_gemm_rows(m, 4096, 4096, dev) == 0
    for m, n, k in [(16384, 16, 64), (16384, 4096, 4096), (16384, 57344, 8192), (16384, 16, 1 << 24)]:
        rows = lib.nf12_linear_decode_gemm_rows(m, n, k, dev)
        assert rows == lib.nf12_linear_decode_gemm_rows(m, n, k, dev)
        assert rows == 0 or (rows % 16 == 0 and 16 <= rows <= n)
    # The slice height also respects the 256 MiB external decode budget.
    for n, k in [(4096, 4096), (14336, 4096), (65536, 4096), (65552, 4096)]:
        rows = lib.nf12_linear_decode_gemm_rows(512, n, k, dev)
        if rows == 0:
            continue  # this device selects the fused route for the shape
        assert rows % 16 == 0 and 16 <= rows <= n and rows * k * 2 <= 256 * 1024 * 1024
        if n * k * 2 <= 256 * 1024 * 1024:
            assert rows == n


def test_round_trip_and_linear(lib):
    n, k, m = 128, 512, 5
    w = bf16(n, k)
    weight, keep = c_encode(lib, w)
    out = torch.empty(n, k + 64, dtype=torch.bfloat16, device="cuda")  # ldo > k
    assert lib.nf12_decode(ctypes.byref(weight), ptr(out), out.stride(0), stream()) == OK
    assert torch.equal(out[:, :k].view(torch.int16), w.view(torch.int16))

    nw = nf12.encode(w)
    assert torch.equal(keep[0], nw.packed.view(-1))
    x = bf16(m, k, seed=1)
    bias = bf16(n, seed=2)
    y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    ws_bytes = lib.nf12_linear_workspace_bytes(m, n, k, torch.cuda.current_device())
    ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device="cuda")
    assert lib.nf12_linear(ptr(x), m, k, ctypes.byref(weight), ptr(bias), ptr(y), n, ptr(ws), ws_bytes,
                           stream()) == OK
    assert torch.equal(y.view(torch.int16), nf12.linear(x, nw, bias).view(torch.int16))

    wt = torch.empty(lib.nf12_bf16t_bytes(n, k), dtype=torch.uint8, device="cuda")
    assert lib.nf12_pack_bf16t(ptr(w), n, k, k, ptr(wt), stream()) == OK
    y2 = torch.empty_like(y)
    assert lib.nf12_linear_bf16t(ptr(x), m, k, ptr(wt), n, k, ptr(bias), ptr(y2), n, ptr(ws), ws_bytes,
                                 stream()) == OK
    assert torch.equal(y.view(torch.int16), y2.view(torch.int16))

    # m == 0 is a no-op that touches no pointer.
    assert lib.nf12_linear(None, 0, k, ctypes.byref(weight), None, None, n, None, 0, stream()) == OK
    assert lib.nf12_last_cuda_error() == 0


def test_linear_invalid_arguments(lib):
    n, k, m = 64, 256, 4
    weight, keep = c_encode(lib, bf16(n, k))
    x = bf16(m, k + 8, seed=1)
    y = torch.empty(m, n + 8, dtype=torch.bfloat16, device="cuda")
    bias = bf16(n + 8)
    big = 1 << 26
    ws = torch.empty(big, dtype=torch.uint8, device="cuda")

    def run(xp=ptr(x), mm=m, ldx=k + 8, w=weight, bp=ptr(bias), yp=ptr(y), ldy=n + 8, wsp=ptr(ws), wsb=big):
        return lib.nf12_linear(xp, mm, ldx, ctypes.byref(w) if w is not None else None, bp, yp, ldy, wsp, wsb,
                               stream())

    assert run() == OK
    assert run(bp=None) == OK
    assert run(xp=ptr(x, 2)) == INVALID  # misaligned x
    assert run(xp=None) == INVALID
    assert run(yp=ptr(y, 8)) == INVALID  # misaligned y
    assert run(ldx=k - 8) == INVALID
    assert run(ldx=k + 4) == INVALID  # ld % 8 != 0
    assert run(ldy=n - 8) == INVALID
    assert run(mm=0) == OK  # an empty row block is a no-op, as on CUDA
    assert run(mm=-1) == INVALID
    assert run(bp=ptr(bias, 2)) == INVALID
    assert run(w=None) == INVALID
    assert run(wsp=None) == INVALID  # bytes > 0 needs a pointer
    assert run(wsp=ptr(ws, 4)) == INVALID

    def bad_weight(**kw):
        fields = {name: getattr(weight, name) for name, _ in Weight._fields_}
        fields.update(kw)
        return Weight(**fields)

    assert run(w=bad_weight(n=n + 8)) == INVALID
    assert run(w=bad_weight(k=k + 32)) == INVALID
    assert run(w=bad_weight(n=0)) == INVALID
    assert run(w=bad_weight(exp_base=121)) == INVALID
    assert run(w=bad_weight(packed=None)) == INVALID
    assert run(w=bad_weight(packed=weight.packed + 8)) == INVALID
    assert run(w=bad_weight(escapes=None, num_escapes=3)) == INVALID
    assert run(w=bad_weight(escapes=ptr(keep[0], 4), num_escapes=1)) == INVALID

    wt = torch.empty(n * k * 2, dtype=torch.uint8, device="cuda")
    assert lib.nf12_linear_bf16t(ptr(x), m, k + 8, ptr(wt, 8), n, k, None, ptr(y), n + 8, ptr(ws), big,
                                 stream()) == INVALID
    assert lib.nf12_linear_bf16t(ptr(x), m, k + 8, ptr(wt), n, k + 1, None, ptr(y), n + 8, ptr(ws), big,
                                 stream()) == INVALID


def test_insufficient_workspace(lib):
    dev = torch.cuda.current_device()
    shapes = [(m, n, k) for m in (1, 4, 16, 64, 256, 1024) for n, k in
              [(16, 65536), (64, 32768), (256, 16384), (1024, 8192), (4096, 4096), (16384, 4096)]]
    sized = [(m, n, k, b) for m, n, k in shapes if (b := lib.nf12_linear_workspace_bytes(m, n, k, dev)) > 0]
    if not sized:
        pytest.skip("linear dispatch needs no workspace for any probed shape")
    m, n, k, need = sized[0]
    weight, keep = c_encode(lib, bf16(n, k))
    x = bf16(m, k, seed=1)
    y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(need, dtype=torch.uint8, device="cuda")
    status = lib.nf12_linear(ptr(x), m, k, ctypes.byref(weight), None, ptr(y), n, ptr(ws), need - 1, stream())
    assert status == INSUFFICIENT
    status = lib.nf12_linear(ptr(x), m, k, ctypes.byref(weight), None, ptr(y), n, None, 0, stream())
    assert status == INSUFFICIENT
    # A nonzero byte count with no pointer is a malformed call, not a short one.
    assert lib.nf12_linear(ptr(x), m, k, ctypes.byref(weight), None, ptr(y), n, None, need, stream()) == INVALID
    assert lib.nf12_linear(ptr(x), m, k, ctypes.byref(weight), None, ptr(y), n, ptr(ws), need, stream()) == OK


def test_codec_invalid_arguments(lib):
    n, k = 32, 128
    w = bf16(n, k + 8)
    w[:16, :16] = 0  # +0.0 groups hit the M == 0 sentinel: at least 32 escapes
    ws_bytes = lib.nf12_encode_workspace_bytes(n, k)
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")
    exp_base, num_escapes = c_uint8(), c_uint32()
    eb, ne = ctypes.byref(exp_base), ctypes.byref(num_escapes)

    def analyze(wp=ptr(w), nn=n, kk=k, ldw=k + 8, wsp=ptr(ws), wsb=ws_bytes, e=eb, c=ne):
        return lib.nf12_encode_analyze(wp, nn, kk, ldw, wsp, wsb, e, c, stream())

    assert analyze() == OK
    assert analyze(wp=ptr(w, 2)) == INVALID
    assert analyze(ldw=k - 8) == INVALID
    assert analyze(ldw=k + 4) == INVALID
    assert analyze(nn=24) == INVALID
    assert analyze(kk=96) == INVALID
    assert analyze(e=None) == INVALID
    assert analyze(c=None) == INVALID
    assert analyze(wsb=ws_bytes - 1) == INSUFFICIENT
    assert analyze(wsp=None) == INVALID

    packed = torch.empty(lib.nf12_packed_bytes(n, k) + 16, dtype=torch.uint8, device="cuda")
    groups = n * k // 8
    escapes = torch.empty(16 * groups, dtype=torch.uint8, device="cuda")

    def encode(base=exp_base.value, pp=ptr(packed), ep=ptr(escapes), cap=groups, wsb=ws_bytes):
        return lib.nf12_encode(ptr(w), n, k, k + 8, base, pp, ep, cap, ptr(ws), wsb, stream())

    assert encode() == OK
    assert encode(base=121) == INVALID
    assert encode(pp=None) == INVALID
    assert encode(pp=ptr(packed, 4)) == INVALID
    assert encode(ep=None) == INVALID
    assert encode(ep=ptr(escapes, 8)) == INVALID
    assert encode(wsb=ws_bytes - 1) == INSUFFICIENT
    assert num_escapes.value >= 32
    assert encode(cap=num_escapes.value - 1) == INVALID  # capacity below the escape count
    assert encode(cap=0, ep=None) == INVALID

    weight = Weight(ptr(packed), ptr(escapes), num_escapes.value, n, k, exp_base.value)
    out = torch.empty(n, k + 8, dtype=torch.bfloat16, device="cuda")
    assert lib.nf12_decode(ctypes.byref(weight), ptr(out), k + 8, stream()) == OK
    assert lib.nf12_decode(ctypes.byref(weight), ptr(out), k + 4, stream()) == INVALID
    assert lib.nf12_decode(ctypes.byref(weight), ptr(out, 2), k + 8, stream()) == INVALID
    assert lib.nf12_decode(ctypes.byref(weight), None, k + 8, stream()) == INVALID
    assert lib.nf12_decode(None, ptr(out), k + 8, stream()) == INVALID

    wt = torch.empty(lib.nf12_bf16t_bytes(n, k) + 16, dtype=torch.uint8, device="cuda")
    assert lib.nf12_pack_bf16t(ptr(w), n, k, k + 8, ptr(wt), stream()) == OK
    assert lib.nf12_pack_bf16t(ptr(w), n, k, k + 8, ptr(wt, 8), stream()) == INVALID
    assert lib.nf12_pack_bf16t(ptr(w), n, k, k + 8, None, stream()) == INVALID
    assert lib.nf12_pack_bf16t(ptr(w, 2), n, k, k + 8, ptr(wt), stream()) == INVALID
    assert lib.nf12_pack_bf16t(ptr(w), n, 100, k + 8, ptr(wt), stream()) == INVALID
    torch.cuda.synchronize()


def test_library_path_is_stable_across_threads():
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
    assert paths[0] == nf12.library_path()
