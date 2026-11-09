# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
"""Encoding scheduling boundaries, raw BF16 preservation and stream ownership."""
import ctypes as C
import threading
from concurrent.futures import ThreadPoolExecutor

import nf12
import pytest
import torch
from nf12 import reference


def _arch():
    if not torch.cuda.is_available() or torch.version.hip is None:
        return ''
    return torch.cuda.get_device_properties(0).gcnArchName.split(':', 1)[0]


pytestmark = pytest.mark.skipif(_arch() not in ('gfx1200', 'gfx1201'), reason='requires RDNA4 ROCm')


def check(w):
    got = nf12.encode(w)
    packed, escapes, base = reference.encode(w.cpu())
    assert got.exp_base == base
    assert got.packed.cpu().equal(packed)
    assert got.escapes.cpu().equal(escapes)
    assert nf12.decode(got).view(torch.int16).equal(w.view(torch.int16))
    assert nf12.pack_bf16t(w).cpu().equal(reference.pack_bf16t(w.cpu()))
    return got


@pytest.mark.parametrize('super_atoms', [1, 2, 3, 7, 8, 9, 15, 16, 17, 4095, 4096, 4097])
def test_encode_unit_and_scan_boundaries(super_atoms):
    # Nonuniform escape positions cross units, blocks, and 2048-unit scan tiles.
    g = torch.Generator().manual_seed(super_atoms)
    bits = torch.randint(-32768, 32768, (16, super_atoms * 64), dtype=torch.int16, generator=g)
    bits[:, :32] = -32768
    check(bits.to('cuda').view(torch.bfloat16))


@pytest.mark.parametrize('pattern', ['all_bits', 'zero', 'negative_zero', 'sentinel', 'high'])
def test_encode_bit_patterns(pattern):
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(256, 256)
    if pattern != 'all_bits':
        bits.fill_({'zero': 0, 'negative_zero': -32768, 'sentinel': 0x3f00, 'high': 0x7fff}[pattern])
    check(bits.to('cuda').view(torch.bfloat16))


def test_encode_mutated_input_and_padded_rows():
    storage = torch.zeros((48, 328), dtype=torch.bfloat16, device='cuda')
    w = storage[:, :320]
    check(w)
    w.view(torch.int16).fill_(-32768)
    check(w)
    w.view(torch.int16)[:, ::3] = 0x7fc1
    check(w)


def test_encode_concurrent_threads_and_streams():
    nf12._library.load()
    barrier = threading.Barrier(4)

    def worker(i):
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Independent input producers, workspaces and outputs on each stream.
            bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
            w = bits.roll(i * 997).reshape(256, 256).view(torch.bfloat16)
            barrier.wait(timeout=30)
            for j in range(5):
                w.view(torch.int16)[i::4, :64] = -32768 if j % 2 else 0
                check(w)
        stream.synchronize()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))


def test_pack_graph_replay_reads_new_input():
    stream = torch.cuda.Stream()
    w = torch.zeros((48, 320), dtype=torch.bfloat16, device='cuda')
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        nf12.pack_bf16t(w)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        packed = nf12.pack_bf16t(w)
    for bits in (0, -32768, 0x7fc1):
        with torch.cuda.stream(stream):
            w.view(torch.int16).fill_(bits)
            graph.replay()
        stream.synchronize()
        assert packed.cpu().equal(reference.pack_bf16t(w.cpu()))


@pytest.fixture
def codec():
    lib = C.CDLL(str(nf12.library_path()))
    ptr, i64, size = C.c_void_p, C.c_int64, C.c_size_t
    lib.nf12_encode_workspace_bytes.argtypes = [i64, i64]
    lib.nf12_encode_workspace_bytes.restype = size
    lib.nf12_encode_analyze.argtypes = [ptr, i64, i64, i64, ptr, size,
                                       C.POINTER(C.c_uint8), C.POINTER(C.c_uint32), ptr]
    lib.nf12_encode.argtypes = [ptr, i64, i64, i64, C.c_uint8, ptr, ptr, C.c_uint32, ptr, size, ptr]
    return lib


def test_encode_workspace_and_capacity_do_not_touch_outputs(codec):
    n, k = 16, 192
    w = torch.zeros((n, k), device='cuda', dtype=torch.bfloat16)
    packed = torch.full((n, k*3//2), 0xab, dtype=torch.uint8, device='cuda')
    escapes = torch.full((n*k//8, 8), 12345, dtype=torch.int16, device='cuda')
    need = codec.nf12_encode_workspace_bytes(n, k)
    ws = torch.full((need + 16,), 0xcd, dtype=torch.uint8, device='cuda')
    stream = torch.cuda.current_stream().cuda_stream
    base, count = C.c_uint8(99), C.c_uint32(99)
    for ptr, size, status in [(None, 0, 3), (ws.data_ptr(), need-1, 3), (None, need, 1)]:
        assert codec.nf12_encode_analyze(w.data_ptr(), n, k, k, ptr, size,
                                         C.byref(base), C.byref(count), stream) == status
        assert codec.nf12_encode(w.data_ptr(), n, k, k, 0, packed.data_ptr(), escapes.data_ptr(),
                                 n*k//8, ptr, size, stream) == status
    assert base.value == 99 and count.value == 99
    assert torch.all(ws == 0xcd)
    assert codec.nf12_encode(w.data_ptr(), n, k, k, 0, packed.data_ptr(), escapes.data_ptr(),
                             n*k//8-1, ws.data_ptr(), need, stream) == 1
    assert torch.all(packed == 0xab) and torch.all(escapes == 12345)
    assert torch.all(ws[need:] == 0xcd)


@pytest.mark.parametrize('base', [0, 55, 120])
def test_encode_explicit_base_padded_stride(codec, base):
    n, k = 48, 192
    storage = torch.arange(n * (k+8), device='cuda', dtype=torch.int32).to(torch.int16)
    w = storage.reshape(n, k+8)[:, :k].view(torch.bfloat16)
    expected, expected_escapes, _ = reference.encode(w.cpu(), base)
    packed = torch.empty_like(expected, device='cuda')
    escapes = torch.empty_like(expected_escapes, device='cuda')
    need = codec.nf12_encode_workspace_bytes(n, k)
    ws = torch.empty(need, dtype=torch.uint8, device='cuda')
    assert codec.nf12_encode(w.data_ptr(), n, k, w.stride(0), base, packed.data_ptr(),
                             escapes.data_ptr(), escapes.shape[0], ws.data_ptr(), need,
                             torch.cuda.current_stream().cuda_stream) == 0
    assert packed.cpu().equal(expected) and escapes.cpu().equal(expected_escapes)


@pytest.mark.parametrize('m', [1, 4, 16, 64])
def test_encoded_inference_matches_bf16t(m):
    g = torch.Generator(device='cuda').manual_seed(m)
    w = (torch.randn((256, 512), device='cuda', generator=g)*0.02).bfloat16()
    x = (torch.randn((m, 512), device='cuda', generator=g)*0.05).bfloat16()
    y = nf12.linear(x, nf12.encode(w))
    twin = nf12.linear_bf16t(x, nf12.pack_bf16t(w))
    assert y.view(torch.int16).equal(twin.view(torch.int16))
