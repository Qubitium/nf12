# SPDX-License-Identifier: Apache-2.0
"""Native inference capture/replay on ROCm 10.1 / HIP 7.16.26385.

Encoding/packing happens before capture. Native callers retain inputs, outputs
and their own workspace for the lifetime of the graph; torch owns its scratch.
"""

import ctypes as C

import nf12
import pytest
import torch
from test_hip import _arch, _Nf12Weight, _stream
from test_hip import lib as lib  # noqa: PLC0414 -- pytest fixture re-export
from test_review_contracts import abi as abi  # noqa: PLC0414 -- pytest fixture re-export

pytestmark = pytest.mark.skipif(
    _arch() not in ("gfx1200", "gfx1201"), reason="RDNA4 WMMA device required"
)


def _bits_equal(a, b):
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("m", [1, 16, 64, 257])
def test_native_dense_graph_with_owned_workspace(abi, m):
    n, k = 256, 2048  # Narrow N exercises split-K, including a chunked tail.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        w = (torch.randn(n, k, device="cuda") * .02).bfloat16()
        nw, twin = nf12.encode(w), nf12.pack_bf16t(w)
        cw = _Nf12Weight(nw.packed.data_ptr(), nw.escapes.data_ptr(), nw.num_escapes, n, k, nw.exp_base)
        x = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(n, device="cuda").bfloat16()
        need = abi.nf12_linear_workspace_bytes(m, n, k, torch.cuda.current_device())
        if m >= 16:
            assert need > 0, "this case must capture the split-K reduction"
        workspace = torch.full((need + 32,), 0xA5, device="cuda", dtype=torch.uint8)
        guards = [torch.full((m + 2, n), -7., device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        outputs = [g[1:-1] for g in guards]

        def run():
            if m < 16:
                rc = abi.nf12_hip_linear_fused(
                    cw.packed, cw.escapes, cw.exp_base, x.data_ptr(), outputs[0].data_ptr(),
                    n, k, m, bias.data_ptr(), _stream())
                rt = abi.nf12_hip_linear_fused_bf16t(
                    twin.data_ptr(), x.data_ptr(), outputs[1].data_ptr(), n, k, m,
                    bias.data_ptr(), _stream())
            else:
                if m <= abi.nf12_hip_wmma_max_m():
                    rc = abi.nf12_hip_linear_wmma(
                        cw.packed, cw.escapes, cw.exp_base, x.data_ptr(), outputs[0].data_ptr(),
                        n, k, m, bias.data_ptr(), workspace.data_ptr(), need, _stream())
                else:
                    rc = abi.nf12_linear(
                        x.data_ptr(), m, k, C.byref(cw), bias.data_ptr(), outputs[0].data_ptr(),
                        n, workspace.data_ptr(), need, _stream())
                rt = abi.nf12_hip_linear_wmma_bf16t(
                    twin.data_ptr(), x.data_ptr(), outputs[1].data_ptr(), n, k, m,
                    bias.data_ptr(), workspace.data_ptr(), need, _stream())
            assert rc == rt == 0

        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    with torch.cuda.stream(stream):
        for _ in range(5):
            x.normal_(std=.05)
            run()
            expected = [y.clone() for y in outputs]
            for y in outputs:
                y.fill_(float("nan"))
            graph.replay()
            for y, ref in zip(outputs, expected):
                assert bool(y.isfinite().all())
                _bits_equal(y, ref)
            _bits_equal(*outputs)
            assert bool((workspace[need:] == 0xA5).all())
            for g in guards:
                assert bool((g[[0, -1]] == -7).all())
    stream.synchronize()


def test_native_decode_and_unpack_graph(abi):
    p, i = C.c_void_p, C.c_int64
    abi.nf12_hip_unpack_bf16t.argtypes = [p, i, i, p, i, p]
    abi.nf12_hip_unpack_bf16t.restype = C.c_int
    n, k = 128, 512
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Equal-sized source buffers allow changing weights at fixed addresses.
        weights = [torch.full((n, k), v, device="cuda", dtype=torch.bfloat16) for v in (.125, -.125)]
        encoded = [nf12.encode(w) for w in weights]
        twins = [nf12.pack_bf16t(w) for w in weights]
        nw = encoded[0]
        assert nw.exp_base == encoded[1].exp_base
        assert nw.num_escapes == encoded[1].num_escapes == 0
        packed, twin = nw.packed.clone(), twins[0].clone()
        outputs = [torch.empty_like(weights[0]) for _ in range(2)]
        stamp = torch.zeros(1, device="cuda", dtype=torch.int32)

        def run():
            assert abi.nf12_hip_decode(packed.data_ptr(), nw.escapes.data_ptr(), nw.exp_base,
                                      n, k, outputs[0].data_ptr(), k, _stream()) == 0
            assert abi.nf12_hip_unpack_bf16t(twin.data_ptr(), n, k, outputs[1].data_ptr(), k, _stream()) == 0
            assert abi.nf12_hip_selftest(stamp.data_ptr(), _stream()) == 0

        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    with torch.cuda.stream(stream):
        for rep in range(5):
            j = rep % 2
            packed.copy_(encoded[j].packed)
            twin.copy_(twins[j])
            stamp.zero_()
            graph.replay()
            for out in outputs:
                _bits_equal(out, weights[j])
            assert stamp.item() == 0x4E463132
    stream.synchronize()


@pytest.mark.parametrize("gather_enabled", [False, True])
def test_native_grouped_operands_graph(abi, gather_enabled):
    count, k, experts = 16, 64, 2
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.zeros(count, k, device="cuda", dtype=torch.bfloat16)
        gather = torch.arange(count - 1, -1, -1, device="cuda", dtype=torch.int32)
        offsets = torch.tensor([0, 4, count], device="cuda", dtype=torch.int64)
        out = torch.full_like(x, -7.)
        ends = torch.empty(experts, device="cuda", dtype=torch.int32)

        def run():
            assert abi.nf12_hip_moe_grouped_operands(
                x.data_ptr(), k, gather.data_ptr() if gather_enabled else None, 1,
                offsets.data_ptr(), count, experts, k, out.data_ptr(), k,
                ends.data_ptr(), _stream()) == 0

        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    with torch.cuda.stream(stream):
        for rep in range(5):
            x.normal_()
            gather.copy_(gather.roll(1))
            offsets[1] = rep + 1
            graph.replay()
            assert torch.equal(ends, offsets[1:].int())
            if gather_enabled:
                _bits_equal(out, x[gather.long()])
            else:
                assert bool((out == -7).all())
    stream.synchronize()
