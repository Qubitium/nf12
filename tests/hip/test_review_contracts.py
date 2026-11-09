# SPDX-License-Identifier: Apache-2.0
"""CUDA/HIP ABI regression cases from the RDNA4 review."""

import ctypes as C
from concurrent.futures import ThreadPoolExecutor

import nf12
import pytest
import torch
from test_hip import _Nf12Weight, _stream
from test_hip import lib as lib  # noqa: PLC0414 -- pytest fixture re-export

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU required"
)


class MoeWeight(C.Structure):
    _fields_ = [("packed", C.c_void_p), ("escapes", C.c_void_p),
                ("escape_offsets", C.c_void_p), ("exp_base", C.c_void_p),
                ("experts", C.c_int64), ("n", C.c_int64), ("k", C.c_int64)]




@pytest.fixture
def abi(lib):
    p, i, sz = C.c_void_p, C.c_int64, C.c_size_t
    for name, restype, args in [
        ("nf12_moe_linear_workspace_bytes", sz, [i, i, i, i, C.c_int]),
        ("nf12_moe_linear", C.c_int, [p, i, p, i, p, p, i, C.POINTER(MoeWeight), p, p, i, p, sz, p]),
        ("nf12_moe_decode", C.c_int, [C.POINTER(MoeWeight), p, i, i, i, p, i, p]),
        ("nf12_linear_workspace_bytes", sz, [i, i, i, C.c_int]),
        ("nf12_linear_decode_gemm_rows", i, [i, i, i, C.c_int]),
        ("nf12_linear_bf16t", C.c_int, [p, i, i, p, i, i, p, p, i, p, sz, p]),
        ("nf12_hip_moe_decode", C.c_int, [p, p, p, p, C.c_int32, i, i, p, i, i, i, p, i, p]),
        ("nf12_hip_moe_silu_mul", C.c_int, [p, i, i, i, p, i, p]),
        ("nf12_hip_moe_combine", C.c_int, [p, i, i, i, i, p, i, p]),
        ("nf12_hip_moe_grouped_operands", C.c_int, [p, i, p, i, p, i, i, i, p, i, p, p]),
        ("nf12_moe_silu_mul", C.c_int, [p, i, i, i, p, i, p]),
        ("nf12_moe_combine", C.c_int, [p, i, i, i, i, p, i, p]),
    ]:
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, args
    return lib


@pytest.mark.parametrize("twin", [False, True])
# The rest of the invalid-argument matrix runs against the shared ABI in
# tests/api/test_moe.py::test_c_linear_validation; these are the native-only gaps.
@pytest.mark.parametrize("bad", ["k_zero", "k_tail", "n_zero", "large_count", "out", "empty"])
def test_native_moe_linear_matches_shared_validation(abi, twin, bad):
    p, i = C.c_void_p, C.c_int64
    fn = abi.nf12_hip_moe_linear_bf16t if twin else abi.nf12_hip_moe_linear
    fn.restype = C.c_int
    fn.argtypes = ([p] if twin else [p] * 4) + [C.c_int32, p, i, p, i, p, p, i, p, p, i, i, i, p]
    w = torch.ones((1, 16, 64), device="cuda", dtype=torch.bfloat16)
    weight = nf12.pack_bf16t_moe(w) if twin else nf12.encode_moe(w)
    packed = weight.data_ptr() if twin else weight.packed.data_ptr()
    x = torch.ones((1, 72), device="cuda", dtype=torch.bfloat16)
    out = torch.full((1, 16), -7., device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    n, k, ldx, ldy, div, count = 16, 64, 72, 16, 1, 1
    xp, yp, off = x.data_ptr(), out.data_ptr(), offsets.data_ptr()
    if bad == "k_zero": k = 0
    elif bad == "k_tail": k = 65
    elif bad == "n_zero": n = 0
    elif bad == "ldx": ldx = 0
    elif bad == "ldy": ldy = 0
    elif bad == "gather_div": div = 0
    elif bad == "negative_count": count = -1
    elif bad == "large_count": count = 2**31
    elif bad == "packed": packed = None
    elif bad == "x": xp = None
    elif bad == "out": yp = None
    elif bad == "offsets": off = None
    elif bad == "alignment": xp += 2
    else:
        count, xp, yp, off = 0, None, None, None
    prefix = [packed] if twin else [packed, weight.escapes.data_ptr(),
                                    weight.escape_offsets.data_ptr(), weight.exp_base.data_ptr()]
    expected = 0 if bad == "empty" else 1
    assert fn(*prefix, 1, xp, ldx, None, div, None, off, count, None, yp, ldy, n, k, _stream()) == expected
    torch.cuda.synchronize()
    assert bool((out == -7).all())


def test_moe_decode_grid_overflow_matches_cuda_status(abi):
    experts, n, k = 65536, 16, 64
    packed = torch.empty((experts, n, k * 3 // 2), device="cuda", dtype=torch.uint8)
    offsets = torch.zeros(experts + 1, device="cuda", dtype=torch.int64)
    bases = torch.zeros(experts, device="cuda", dtype=torch.uint8)
    out = torch.full((experts, n, k), -7., device="cuda", dtype=torch.bfloat16)
    cw = MoeWeight(packed.data_ptr(), None, offsets.data_ptr(), bases.data_ptr(), experts, n, k)
    assert abi.nf12_hip_moe_decode(cw.packed, cw.escapes, cw.escape_offsets, cw.exp_base,
                                 experts, n, k, None, 0, 0, n, out.data_ptr(), k, _stream()) == 1
    torch.cuda.synchronize()
    assert bool((out == -7).all())


@pytest.mark.parametrize("row0,rows,ldo", [(16, 16, 64), (0, 16, 56), (0, 2**63 - 16, 64)])
def test_native_moe_decode_rejects_invalid_slice_or_stride(abi, row0, rows, ldo):
    # A trailing encoded matrix makes the original out-of-range read observable
    # without an illegal GPU access. The descriptor owns only the first 16 rows.
    w = torch.cat([torch.ones(16, 64), torch.full((16, 64), 2.)]).bfloat16().cuda()
    nw = nf12.encode(w)
    ends = torch.tensor([0, nw.num_escapes], device="cuda", dtype=torch.int64)
    base = torch.tensor([nw.exp_base], device="cuda", dtype=torch.uint8)
    out = torch.full((34, 80), -7., device="cuda", dtype=torch.bfloat16)
    cw = MoeWeight(nw.packed.data_ptr(), nw.escapes.data_ptr(), ends.data_ptr(),
                   base.data_ptr(), 1, 16, 64)
    assert abi.nf12_hip_moe_decode(cw.packed, cw.escapes, cw.escape_offsets, cw.exp_base,
                                   1, 16, 64, None, 0, row0, rows,
                                   out[1:].data_ptr(), ldo, _stream()) == 1
    torch.cuda.synchronize()
    assert bool((out == -7).all())


@pytest.mark.parametrize("op", ["silu_mul", "combine"])
@pytest.mark.parametrize("bad", ["input_stride", "stride_alignment", "row_overflow"])
def test_native_moe_elementwise_rejects_invalid_buffers(abi, op, bad):
    x = torch.ones((4, 32), device="cuda", dtype=torch.bfloat16)
    out = torch.full((4, 32), -7., device="cuda", dtype=torch.bfloat16)
    ldi, ldo, rows, offset = 32, 32, 2, 0
    if bad == "input_stride":
        ldi = 0
    elif bad == "output_stride":
        ldo = 0
    elif bad == "stride_alignment":
        ldo = 9
    elif bad == "pointer_alignment":
        offset = 2
    elif bad == "row_overflow":
        rows = 2**62
    else:
        ldi = 2**62
    shape = (rows, 8) if op == "silu_mul" else (rows, 2, 8)
    args = (x.data_ptr(), ldi, *shape, out.data_ptr() + offset, ldo, _stream())
    assert getattr(abi, "nf12_moe_" + op)(*args) == 1
    assert getattr(abi, "nf12_hip_moe_" + op)(*args) == 1
    torch.cuda.synchronize()
    assert bool((out == -7).all())


@pytest.mark.parametrize("ldx,ldo,count", [(0, 8, 4), (8, 0, 4), (8, 8, -1), (2**62, 8, 4)])
def test_native_grouped_operands_rejects_invalid_buffers(abi, ldx, ldo, count):
    x = torch.ones((4, 8), device="cuda", dtype=torch.bfloat16)
    out = torch.full_like(x, -7.)
    gather = torch.arange(4, device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int64)
    ends = torch.full((2,), -7, device="cuda", dtype=torch.int32)
    args = (x.data_ptr(), ldx, gather.data_ptr(), 1, offsets.data_ptr(), count, 2, 8,
            out.data_ptr(), ldo, ends.data_ptr(), _stream())
    assert abi.nf12_moe_grouped_operands(*args) == 1
    assert abi.nf12_hip_moe_grouped_operands(*args) == 1
    torch.cuda.synchronize()
    assert bool((out == -7).all()) and bool((ends == -7).all())


@pytest.mark.parametrize("count", [524256, 524272, 1048544, 1048545, 1048560, 2097120])
def test_moe_wmma_grid_y_boundary(abi, count):
    # Routed tiles spill from grid.y into grid.z. Cross both old Y ceilings
    # and the second Z plane, including a one-row tail.
    n, k = 16, 64
    w = torch.ones((1, n, k), device="cuda", dtype=torch.bfloat16)
    nw = nf12.encode_moe(w)
    cw = MoeWeight(nw.packed.data_ptr(), nw.escapes.data_ptr(), nw.escape_offsets.data_ptr(),
                   nw.exp_base.data_ptr(), 1, n, k)
    x = torch.ones((count, k), device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, count], device="cuda", dtype=torch.int64)
    guarded = torch.full((count + 2, n), -7., device="cuda", dtype=torch.bfloat16)
    out = guarded[1:-1]
    stream = _stream()

    def launch():
        with torch.cuda.device(x.device):
            return abi.nf12_moe_linear(x.data_ptr(), k, None, 1, None, offsets.data_ptr(), count,
                                      C.byref(cw), None, out.data_ptr(), n, None, 0, stream)

    # Native errors intentionally persist in TLS; keep this deliberate failure
    # off the pytest thread, as in the shared ABI error-state tests.
    with ThreadPoolExecutor(max_workers=1) as pool:
        rc = pool.submit(launch).result()
    torch.cuda.synchronize()
    assert rc == 0
    twin = nf12.moe_linear(x, nf12.pack_bf16t_moe(w), offsets)
    assert bool((out == k).all())  # Independent exact result for all-one inputs.
    assert torch.equal(out.view(torch.int16), twin.view(torch.int16))
    assert bool((guarded[[0, -1]] == -7).all())
    if count == 1048545:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            launch_stream = stream.cuda_stream
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            assert abi.nf12_moe_linear(x.data_ptr(), k, None, 1, None, offsets.data_ptr(), count,
                                      C.byref(cw), None, out.data_ptr(), n, None, 0, launch_stream) == 0
        with torch.cuda.stream(stream):
            x.fill_(2.)
            graph.replay()
            assert bool((out == 2 * k).all())
            assert bool((guarded[[0, -1]] == -7).all())
        stream.synchronize()




@pytest.mark.parametrize("m", [257, 272, 288, 320, 528])
def test_chunk_workspace_covers_tail_plan(abi, m):
    n = k = 4096
    w = torch.ones((n, k), device="cuda", dtype=torch.bfloat16)
    nw, twin = nf12.encode(w), nf12.pack_bf16t(w)
    cw = _Nf12Weight(nw.packed.data_ptr(), nw.escapes.data_ptr(), nw.num_escapes, n, k, nw.exp_base)
    cap = abi.nf12_hip_wmma_max_m()
    need = abi.nf12_linear_workspace_bytes(m, n, k, torch.cuda.current_device())
    assert need >= abi.nf12_linear_workspace_bytes(m % cap, n, k, torch.cuda.current_device())
    ws = torch.full((need + 32,), 0xA5, device="cuda", dtype=torch.uint8)
    x = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
    out = torch.full((m, n), -7., device="cuda", dtype=torch.bfloat16)
    if need:
        assert abi.nf12_linear(x.data_ptr(), m, k, C.byref(cw), None, out.data_ptr(), n,
                               ws.data_ptr(), need - 1, _stream()) == 3
        assert bool((out == -7).all())  # fail before writing the first chunk
    assert abi.nf12_linear(x.data_ptr(), m, k, C.byref(cw), None, out.data_ptr(), n,
                           ws.data_ptr(), need, _stream()) == 0
    twin_out = nf12.linear_bf16t(x, twin)
    torch.cuda.synchronize()
    assert bool((ws[need:] == 0xA5).all())
    assert bool((out == k).all())
    assert torch.equal(out.view(torch.int16), twin_out.view(torch.int16))
