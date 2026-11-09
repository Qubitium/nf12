# SPDX-License-Identifier: Apache-2.0
"""CUDA/HIP ABI regression cases from the RDNA4 review."""

import ctypes as C

import nf12
import pytest
import torch
from test_hip import _stream
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
        ("nf12_linear_bf16t", C.c_int, [p, i, i, p, i, i, p, p, i, p, sz, p]),
    ]:
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, args
    return lib


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
        # Guard both sides of the output and leave unassigned rows untouched.
        out = torch.full((count + 2, n), -7., device="cuda", dtype=torch.bfloat16)
        rc = abi.nf12_moe_linear(x[1:].data_ptr(), k, None, 1, None, off.data_ptr(), count,
                                C.byref(cw), None, out[1:].data_ptr(), n,
                                ws.data_ptr(), need, _stream())
        assert rc == 0
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
