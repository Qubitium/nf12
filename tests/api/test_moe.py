# SPDX-License-Identifier: Apache-2.0
"""Mixture-of-experts tests (SPEC.md section 9): stacked encoding, routing,
moe_linear / fused_moe against the reference and the bf16t twin, CUDA graphs,
torch.compile, serialization, expert parallelism and C API validation."""

from __future__ import annotations

import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor

import nf12
import pytest
import torch
from _helpers import assert_bitwise, bits
from nf12 import reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU required")

DEV = "cuda"
OK, INVALID, UNSUPPORTED, INSUFFICIENT, CUDA_ERROR = range(5)






def gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(seed)


def experts(e: int, n: int, k: int, seed: int = 0, scale: float = 0.05, spread: int = 8) -> torch.Tensor:
    w = (torch.randn(e, n, k, device=DEV, generator=gen(seed)) * scale).to(torch.bfloat16)
    # Expert-dependent escape load: +0.0 blocks hit the M == 0 sentinel, and
    # odd experts are scaled by 2^(spread * [-2, 2]) so their exp_base differs.
    for i in range(e):
        w[i, : 16 * (i % 3), : 16 * (i % 4)] = 0
        if i % 2:
            w[i] *= 2.0 ** (spread * (i % 5 - 2))
    return w


def activations(*shape: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(*shape, device=DEV, generator=gen(seed)).to(torch.bfloat16)


def routing(t: int, e: int, topk: int, seed: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct experts per token, softmax weights (float32)."""
    g = gen(seed)
    ids = torch.rand(t, e, device=DEV, generator=g).argsort(dim=1)[:, :topk].to(torch.int32)
    weights = torch.rand(t, topk, device=DEV, generator=g).softmax(dim=1)
    return weights, ids


def close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a.float(), b.float(), rtol=2e-2, atol=2e-2)




@pytest.mark.parametrize("e,n,k", [(1, 16, 64), (5, 48, 128), (8, 256, 512)])
def test_encode_moe_matches_dense_encode(e, n, k):
    w = experts(e, n, k)
    mw = nf12.encode_moe(w)
    assert mw.shape == (e, n, k) and mw.num_experts == e
    offsets = mw.escape_offsets.tolist()
    assert offsets[0] == 0 and offsets[-1] == mw.num_escapes
    for i in range(e):
        dense = nf12.encode(w[i])
        assert torch.equal(mw.packed[i], dense.packed)
        assert torch.equal(mw.escapes[offsets[i]:offsets[i + 1]], dense.escapes)
        assert int(mw.exp_base[i]) == dense.exp_base
    assert len(set(mw.exp_base.tolist())) > 1 or e == 1
    assert_bitwise(mw.decode(), w)
    ref = reference.encode_moe(w)
    for got, want in zip((mw.packed, mw.escapes, mw.escape_offsets, mw.exp_base), ref):
        assert torch.equal(got, want)


def test_encode_moe_strided_input():
    big = experts(4, 64, 256)
    w = big[:, :32, 64:192]  # rows strided, base offset 128 bytes
    assert not w.is_contiguous()
    mw = nf12.encode_moe(w)
    assert_bitwise(mw.decode(), w)
    assert torch.equal(mw.packed, nf12.encode_moe(w.contiguous()).packed)


def test_pack_bf16t_moe():
    w = experts(3, 48, 128)
    packed = nf12.pack_bf16t_moe(w)
    assert packed.shape == (3, 48, 256)
    for i in range(3):
        assert torch.equal(packed[i], reference.pack_bf16t(w[i]))




ALIGN_CASES = [
    (0, 8, 128),  # no tokens
    (1, 8, 128),  # decode
    (7, 2, 16),
    (64, 8, 256),
    (300, 8, 128),  # several tiles per warp
    (1025, 6, 64),  # multi-block
    (9000, 8, 256),  # multi-block, many chunks
    (5000, 4, 512),  # fewer counting warps
    (3, 3, 20000),  # counters in the workspace
    (50, 4, 1),  # one expert
]


@pytest.mark.parametrize("t,topk,e", ALIGN_CASES)
def test_align_matches_reference(t, topk, e):
    ids = torch.randint(-3, e + 3, (t, topk), device=DEV, dtype=torch.int32, generator=gen(t + e))
    if t:
        ids.view(-1)[::7] = torch.iinfo(torch.int32).max
        ids.view(-1)[::11] = torch.iinfo(torch.int32).min
    sorted_ids, offsets = nf12.moe_align(ids, e)
    ref_sorted, ref_offsets = reference.moe_align(ids, e)
    assert sorted_ids.dtype == torch.int32 and sorted_ids.shape == (t * topk,)
    assert offsets.dtype == torch.int64 and offsets.shape == (e + 1,)
    assert torch.equal(offsets, ref_offsets)
    valid = int(offsets[-1])
    assert torch.equal(sorted_ids[:valid], ref_sorted[:valid])
    again, _ = nf12.moe_align(ids, e)
    assert torch.equal(again[:valid], sorted_ids[:valid])


def test_align_stable_and_drops():
    ids = torch.tensor([[2, 0], [2, -1], [0, 2], [5, 0]], dtype=torch.int32, device=DEV)
    sorted_ids, offsets = nf12.moe_align(ids, 4)
    # expert 0: ids 1, 4, 7; expert 1: none; expert 2: 0, 2, 5; expert 3: none; -1 and 5 dropped.
    assert offsets.tolist() == [0, 3, 3, 6, 6]
    assert sorted_ids[:6].tolist() == [1, 4, 7, 0, 2, 5]


def test_align_int64_ids_and_noncontiguous():
    ids = torch.randint(0, 32, (40, 8), device=DEV, generator=gen(5))
    s64, o64 = nf12.moe_align(ids, 32)
    s32, o32 = nf12.moe_align(ids.to(torch.int32), 32)
    assert torch.equal(s64, s32) and torch.equal(o64, o32)
    wide = torch.randint(0, 32, (40, 16), device=DEV, dtype=torch.int32, generator=gen(6))
    s, o = nf12.moe_align(wide[:, ::2], 32)
    rs, ro = reference.moe_align(wide[:, ::2], 32)
    assert torch.equal(o, ro) and torch.equal(s, rs)


def test_align_int64_out_of_range_dropped():
    # int64 ids that would wrap into valid experts if narrowed to int32 first.
    ids = torch.tensor([[2**32, 0, -(2**32)]], dtype=torch.int64, device=DEV)
    sorted_ids, offsets = nf12.moe_align(ids, 1)
    assert offsets.tolist() == [0, 1]
    assert sorted_ids[:1].tolist() == [1]
    wide = torch.tensor([[2**32 + 3, 3, 2**31, -1, 2**63 - 1, -(2**63)]], dtype=torch.int64, device=DEV)
    sorted_ids, offsets = nf12.moe_align(wide, 4)
    assert torch.equal(offsets, reference.moe_align(wide, 4)[1])
    assert offsets.tolist() == [0, 0, 0, 0, 1] and sorted_ids[:1].tolist() == [1]




@pytest.mark.parametrize("t,topk,n", [(1, 8, 2048), (5, 3, 64), (33, 1, 8), (100, 8, 7168), (0, 4, 64)])
def test_combine_matches_reference_bitwise(t, topk, n):
    y = activations(t * topk, n)
    if t:
        y[0] = -0.0  # a -0.0 first term must survive
    out = nf12.moe_combine(y, topk)
    assert out.shape == (t, n)
    assert_bitwise(out, reference.moe_combine(y, topk))
    if t:
        assert bits(out)[0, 0] == bits(reference.moe_combine(y, topk))[0, 0]


def test_combine_order():
    # 1 + 2^-9 rounds to 1 in BF16 only if summed in a different order than left to right.
    vals = torch.tensor([[1.0], [2.0**-9], [2.0**-9], [-1.0]], device=DEV).repeat(1, 8).to(torch.bfloat16)
    out = nf12.moe_combine(vals, 4)
    assert out.float()[0, 0].item() == 2.0**-8


def test_combine_strided_rows():
    big = activations(24, 256)
    y = big[:, :128]
    assert_bitwise(nf12.moe_combine(y, 4), reference.moe_combine(y.contiguous(), 4))




@pytest.mark.parametrize("rows,inter", [(1, 8), (7, 64), (64, 1408), (300, 2048), (0, 64)])
def test_silu_mul_matches_reference_bitwise(rows, inter):
    h = activations(rows, 2 * inter) * 4
    out = nf12.moe_silu_mul(h)
    assert out.shape == (rows, inter)
    assert_bitwise(out, reference.moe_silu_mul(h))


def test_silu_mul_strided_rows():
    big = activations(24, 512)
    h = big[:, :256]
    assert_bitwise(nf12.moe_silu_mul(h), reference.moe_silu_mul(h.contiguous()))


def test_silu_mul_special_values():
    inf, nan = float("inf"), float("nan")
    gate = [0.0, -0.0, inf, -inf, nan, -100.0, -1e30, 1e30, 3.0, -3.0, 0.5, 89.0, -89.0, 1.0, 2.0, -2.0]
    up = [1.0, -1.0, 2.0, 3.0, 1.0, 5.0, 1.0, -1.0, inf, -inf, nan, 0.0, -0.0, -0.0, 1.0, 1e30]
    g = torch.tensor(gate, device=DEV).repeat(len(up), 1)
    u = torch.tensor(up, device=DEV).repeat(len(gate), 1).t()
    h = torch.cat([g, u], dim=1).to(torch.bfloat16)
    out, ref = nf12.moe_silu_mul(h), reference.moe_silu_mul(h)
    # NaN payloads are not part of the contract; everything else is bitwise.
    nans = ref.isnan()
    assert torch.equal(out.isnan(), nans)
    assert torch.equal(bits(out)[~nans], bits(ref)[~nans])



LINEAR_SHAPES = [(8, 128, 256, 2), (16, 256, 512, 4), (4, 512, 2048, 2)]
TOKENS = [1, 3, 16, 64, 257]


def twin(w):
    return nf12.encode_moe(w), nf12.pack_bf16t_moe(w)


@pytest.mark.parametrize("e,n,k,topk", LINEAR_SHAPES)
@pytest.mark.parametrize("t", TOKENS)
def test_moe_linear_gather(t, e, n, k, topk):
    w = experts(e, n, k, seed=t)
    mw, tw = twin(w)
    x = activations(t, k, seed=t)
    _, ids = routing(t, e, topk, seed=t)
    sorted_ids, offsets = nf12.moe_align(ids, e)
    y = nf12.moe_linear(x, mw, offsets, gather=sorted_ids, gather_div=topk)
    assert y.shape == (t * topk, n)
    ref = reference.moe_linear(x, w, offsets, gather=sorted_ids, gather_div=topk)
    close(y, ref)
    assert_bitwise(y, nf12.moe_linear(x, tw, offsets, gather=sorted_ids, gather_div=topk))
    assert_bitwise(y, nf12.moe_linear(x, mw, offsets, gather=sorted_ids, gather_div=topk))  # deterministic


@pytest.mark.parametrize("e,n,k,topk", LINEAR_SHAPES)
@pytest.mark.parametrize("t", TOKENS)
def test_moe_linear_scatter_row_scale(t, e, n, k, topk):
    w = experts(e, n, k, seed=t + 1)
    mw, tw = twin(w)
    weights, ids = routing(t, e, topk, seed=t + 1)
    sorted_ids, offsets = nf12.moe_align(ids, e)
    h = activations(t * topk, k, seed=t + 2)  # rows in sorted order
    scale = weights.reshape(-1)
    y = nf12.moe_linear(h, mw, offsets, scatter=sorted_ids, row_scale=scale)
    close(y, reference.moe_linear(h, w, offsets, scatter=sorted_ids, row_scale=scale))
    assert_bitwise(y, nf12.moe_linear(h, tw, offsets, scatter=sorted_ids, row_scale=scale))


def test_moe_linear_plain_and_dropped_rows():
    e, n, k, t, topk = 6, 128, 256, 20, 3
    w = experts(e, n, k)
    mw, tw = twin(w)
    ids = routing(t, e, topk)[1]
    ids[::3, 0] = -1  # dropped assignments
    sorted_ids, offsets = nf12.moe_align(ids, e)
    valid = int(offsets[-1])
    assert valid < t * topk
    h = activations(t * topk, k)
    y = nf12.moe_linear(h, mw, offsets)  # in(p) = out(p) = p
    ref = reference.moe_linear(h, w, offsets)
    close(y[:valid], ref[:valid])
    assert_bitwise(y[:valid], nf12.moe_linear(h, tw, offsets)[:valid])
    # Scatter with zero_fill: unwritten (dropped) rows are zero.
    y = nf12.moe_linear(h, mw, offsets, scatter=sorted_ids, zero_fill=True)
    close(y, reference.moe_linear(h, w, offsets, scatter=sorted_ids))
    dropped = (ids.reshape(-1) < 0).nonzero().squeeze(1)
    assert torch.equal(y[dropped], torch.zeros_like(y[dropped]))


def test_moe_linear_untouched_rows():
    e, n, k = 2, 64, 128
    mw = nf12.encode_moe(experts(e, n, k))
    offsets = torch.tensor([0, 1, 2], dtype=torch.int64, device=DEV)
    scatter = torch.tensor([3, 0, 1, 2], dtype=torch.int32, device=DEV)
    h = activations(4, k)
    y = nf12.moe_linear(h, mw, offsets, scatter=scatter, zero_fill=True)
    # Positions 0 and 1 are valid (rows 3 and 0); rows 1 and 2 stay zero.
    assert torch.equal(y[1:3], torch.zeros_like(y[1:3]))
    assert bool((y[0] != 0).any()) and bool((y[3] != 0).any())


def test_moe_linear_strided_x():
    e, n, k, t, topk = 4, 128, 256, 9, 2
    w = experts(e, n, k)
    mw = nf12.encode_moe(w)
    big = activations(t, 2 * k)
    x = big[:, k:]
    ids = routing(t, e, topk)[1]
    sorted_ids, offsets = nf12.moe_align(ids, e)
    y = nf12.moe_linear(x, mw, offsets, gather=sorted_ids, gather_div=topk)
    assert_bitwise(y, nf12.moe_linear(x.contiguous(), mw, offsets, gather=sorted_ids, gather_div=topk))


def test_moe_linear_empty():
    mw = nf12.encode_moe(experts(2, 64, 128))
    offsets = torch.zeros(3, dtype=torch.int64, device=DEV)
    empty = torch.empty(0, dtype=torch.int32, device=DEV)
    y = nf12.moe_linear(activations(4, 128), mw, offsets, gather=empty, gather_div=2)
    assert y.shape == (0, 64)




def moe_layer(e: int, k: int, inter: int, seed: int = 0):
    # A spread of 1 (scales 1/4..4) still varies exp_base without the 2^+-16
    # magnitudes, which two chained projections would turn into cancellation.
    w13 = experts(e, 2 * inter, k, seed=seed, spread=1)
    w2 = experts(e, k, inter, seed=seed + 100, spread=1)
    return w13, w2


def composed_moe(x, w13, w2, weights, ids):
    """Public primitives, retaining each BF16 rounding boundary."""
    e = w13.num_experts if isinstance(w13, nf12.NF12MoEWeight) else w13.shape[0]
    sorted_ids, offsets = nf12.moe_align(ids, e)
    h = nf12.moe_linear(x, w13, offsets, gather=sorted_ids, gather_div=ids.shape[1])
    h = nf12.moe_silu_mul(h)
    y = nf12.moe_linear(h, w2, offsets, scatter=sorted_ids, row_scale=weights.reshape(-1).float(),
                        zero_fill=True)
    return nf12.moe_combine(y, ids.shape[1])


@pytest.mark.parametrize("t", [1, 4, 32, 200])
@pytest.mark.parametrize("e,k,inter,topk", [(8, 256, 128, 2), (16, 512, 256, 4)])
def test_fused_moe(t, e, k, inter, topk):
    w13, w2 = moe_layer(e, k, inter, seed=t)
    x = activations(t, k, seed=t)
    weights, ids = routing(t, e, topk, seed=t)
    m13, t13 = twin(w13)
    m2, t2 = twin(w2)
    out = nf12.fused_moe(x, m13, m2, weights, ids)
    assert out.shape == (t, k)
    close(out, reference.fused_moe(x, w13, w2, weights, ids))
    assert_bitwise(out, nf12.fused_moe(x, t13, t2, weights, ids))
    assert_bitwise(out, nf12.fused_moe(x, m13, m2, weights, ids))


@pytest.mark.parametrize("k,inter,n,topk", [(1024, 512, 1024, 1), (2048, 1024, 1024, 4),
                                         (1024, 1536, 2048, 8), (4096, 2048, 4096, 2),
                                         (1024, 2048, 1024, 1), (2048, 3584, 2048, 2),
                                         (4096, 4096, 4096, 2), (4096, 1024, 4096, 4)])
def test_fused_moe_decode_sources(k, inter, n, topk):
    w13 = experts(2, 2 * inter, k, spread=1)
    w2 = experts(2, n, inter, seed=100, spread=1)
    m13, t13 = twin(w13)
    m2, t2 = twin(w2)
    x = activations(1, k)
    ids = (torch.arange(topk, device=DEV, dtype=torch.int32) % 2).view(1, topk)
    weights = torch.linspace(-0.75, 0.875, topk, device=DEV).view(1, topk)
    out = nf12.fused_moe(x, m13, m2, weights, ids)
    assert out.shape == (1, n)
    close(out, composed_moe(x, m13, m2, weights, ids))
    for a, b in ((t13, t2), (m13, t2), (t13, m2)):
        assert_bitwise(out, nf12.fused_moe(x, a, b, weights, ids))
    assert_bitwise(out, nf12.fused_moe(x, m13, m2, weights, ids))


@pytest.mark.parametrize("scales", [[1.0, 2.0**-9, 2.0**-9, -1.0], [-0.0]])
def test_fused_moe_decode_rounding_boundaries(scales):
    k, inter = 1024, 512
    w13 = torch.zeros((1, 2 * inter, k), dtype=torch.bfloat16, device=DEV)
    w2 = torch.zeros((1, k, inter), dtype=torch.bfloat16, device=DEV)
    w13[0, 0, 0] = w13[0, inter, 0] = 1
    w2[0, :, 0] = 1
    m13, t13 = twin(w13)
    m2, t2 = twin(w2)
    x = torch.ones((1, k), dtype=torch.bfloat16, device=DEV)
    ids = torch.zeros((1, len(scales)), dtype=torch.int32, device=DEV)
    weights = torch.tensor([scales], device=DEV)
    expected = composed_moe(x, m13, m2, weights, ids)
    # Preserve per-assignment rounding, ordered FP32 top-k sum and leading -0.
    if len(scales) == 1:
        assert bool(torch.signbit(expected).all()) and bool((expected == 0).all())
    else:
        assert bool((expected != 0).all())
    for a, b in ((m13, m2), (t13, t2), (m13, t2), (t13, m2)):
        assert_bitwise(nf12.fused_moe(x, a, b, weights, ids), expected)


@pytest.mark.parametrize("t,k,inter,topk", [(0, 1024, 512, 4), (1, 256, 128, 4),
                                         (1, 1024, 512, 9), (2, 1024, 512, 4), (5, 2048, 1024, 2)])
def test_fused_moe_compositional_fallback(t, k, inter, topk):
    w13, w2 = moe_layer(2, k, inter)
    m13, t2 = nf12.encode_moe(w13), nf12.pack_bf16t_moe(w2)
    x = activations(t, k)
    ids = torch.zeros((t, topk), dtype=torch.int32, device=DEV)
    weights = torch.full((t, topk), 1 / topk, device=DEV)
    assert_bitwise(nf12.fused_moe(x, m13, t2, weights, ids), composed_moe(x, m13, t2, weights, ids))


@pytest.mark.parametrize("layout", ["inner_stride", "unaligned", "row_stride"])
@pytest.mark.parametrize("t", [1, 2])
def test_fused_moe_strided_inputs(t, layout):
    k, inter = 1024, 512
    w13, w2 = moe_layer(2, k, inter)
    m13, m2 = nf12.encode_moe(w13), nf12.encode_moe(w2)
    if layout == "inner_stride":
        x = activations(t, 2 * k)[:, ::2]
    elif layout == "unaligned":
        x = activations(t * k + 1)[1:].view(t, k)
    else:
        x = activations(1, k).expand(t, k)
    ids = torch.tensor([[0, 7, 1, 7, 0, 7, 1, 7]], dtype=torch.int64, device=DEV).expand(t, -1)[:, ::2]
    weights = torch.tensor([[0.125, 9, 0.25, 9, -0.5, 9, 0.75, 9]], dtype=torch.float64,
                           device=DEV).expand(t, -1)[:, ::2]
    expected = nf12.fused_moe(x.contiguous().clone(), m13, m2, weights.float().contiguous(), ids.int().contiguous())
    assert_bitwise(nf12.fused_moe(x, m13, m2, weights, ids), expected)


@pytest.mark.parametrize("all_dropped", [False, True])
def test_fused_moe_decode_expert_map(all_dropped):
    w13, w2 = moe_layer(2, 1024, 512)
    m13, m2 = nf12.encode_moe(w13), nf12.encode_moe(w2)
    x = activations(1, 1024)
    ids = torch.tensor([[0, 0, 2, 3, -1, 2**32, -(2**32), 2**63 - 1]], dtype=torch.int64, device=DEV)
    weights = torch.tensor([[0.25, -0.125, 0.5, 0.75, 1, 1, 1, 1]], device=DEV)
    expert_map = torch.tensor([-1, -1, -1, -1] if all_dropped else [1, -1, 0, -1],
                              dtype=torch.int64, device=DEV)
    mapped = torch.full_like(ids, -1)
    if not all_dropped:
        mapped[0, :3] = torch.tensor([1, 1, 0], device=DEV)
    out = nf12.fused_moe(x, m13, m2, weights, ids, expert_map=expert_map)
    if all_dropped:
        assert_bitwise(out, torch.zeros_like(out))
    else:
        close(out, composed_moe(x, m13, m2, weights, mapped))
    assert_bitwise(out, nf12.fused_moe(x, nf12.pack_bf16t_moe(w13), nf12.pack_bf16t_moe(w2),
                                      weights, ids, expert_map=expert_map))


def test_fused_moe_expert_parallel():
    e, k, inter, topk, t, ranks = 8, 256, 128, 2, 37, 2
    w13, w2 = moe_layer(e, k, inter)
    x = activations(t, k)
    weights, ids = routing(t, e, topk)
    m13, m2 = nf12.encode_moe(w13), nf12.encode_moe(w2)
    full = nf12.fused_moe(x, m13, m2, weights, ids)
    local = e // ranks
    partial = []
    for r in range(ranks):
        expert_map = torch.full((e,), -1, dtype=torch.int32, device=DEV)
        expert_map[r * local:(r + 1) * local] = torch.arange(local, dtype=torch.int32, device=DEV)
        l13, l2 = m13.narrow(r * local, local), m2.narrow(r * local, local)
        assert l13.escape_offsets[0].item() == 0
        assert_bitwise(l13.decode(), w13[r * local:(r + 1) * local])
        out = nf12.fused_moe(x, l13, l2, weights, ids, expert_map=expert_map)
        ref = reference.fused_moe(x, w13[r * local:(r + 1) * local], w2[r * local:(r + 1) * local], weights, ids,
                                  expert_map=expert_map)
        close(out, ref)
        lt13, lt2 = nf12.pack_bf16t_moe(w13[r * local:(r + 1) * local]), nf12.pack_bf16t_moe(
            w2[r * local:(r + 1) * local])
        assert_bitwise(out, nf12.fused_moe(x, lt13, lt2, weights, ids, expert_map=expert_map))
        partial.append(out.float())
    close(sum(partial), full)


def test_fused_moe_expert_map_drops_out_of_range_global_ids():
    e, k, inter, topk, t = 8, 256, 128, 2, 12
    w13, w2 = moe_layer(e, k, inter)
    x = activations(t, k)
    weights, ids = routing(t, e, topk)
    l13, l2 = nf12.encode_moe(w13[:4]), nf12.encode_moe(w2[:4])
    expert_map = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1], dtype=torch.int64, device=DEV)
    remote = ids.to(torch.int64)
    remote[::3, 0] = 7  # an expert on another rank
    bad = remote.clone()
    # Global ids outside [0, 8): negative ones must not wrap to expert_map[-1],
    # large ones must not fault or narrow into range.
    bad[0::3, 0] = torch.tensor([-1, 8, 2**32, -(2**32)], device=DEV)[: bad[0::3, 0].numel()]
    want = nf12.fused_moe(x, l13, l2, weights, remote, expert_map=expert_map)
    assert_bitwise(nf12.fused_moe(x, l13, l2, weights, bad, expert_map=expert_map), want)
    close(want, reference.fused_moe(x, w13[:4], w2[:4], weights, bad, expert_map=expert_map))


@pytest.mark.parametrize("t,k,inter", [(8, 512, 256), (1, 1024, 512)])
def test_fused_moe_cuda_graph(t, k, inter):
    e, topk = 4, 4
    w13, w2 = moe_layer(e, k, inter)
    m13, m2 = nf12.encode_moe(w13), nf12.encode_moe(w2)
    x = activations(t, k)
    weights, ids = routing(t, e, topk)
    expert_map = torch.arange(e, dtype=torch.int32, device=DEV)
    nf12.fused_moe(x, m13, m2, weights, ids, expert_map=expert_map)  # warm up outside capture

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        nf12.fused_moe(x, m13, m2, weights, ids, expert_map=expert_map)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = nf12.fused_moe(x, m13, m2, weights, ids, expert_map=expert_map)
    for seed in (3, 4, 5):
        x.copy_(activations(t, k, seed=seed))
        new_weights, new_ids = routing(t, e, topk, seed=seed)
        weights.copy_(new_weights)
        ids.copy_(new_ids)
        ids[:, 1] = ids[:, 0]  # repeated assignments must remain independent
        expert_map.fill_(-1) if seed == 4 else expert_map.copy_(torch.arange(e, device=DEV))
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(out, nf12.fused_moe(x, m13, m2, weights, ids, expert_map=expert_map))


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("k,inter", [(256, 128), (1024, 512)])
def test_fused_moe_torch_compile(dynamic, k, inter):
    torch._dynamo.reset()
    e, topk = 2, 2
    w13, w2 = moe_layer(e, k, inter)
    m13, m2 = nf12.encode_moe(w13), nf12.encode_moe(w2)

    def layer(x, weights, ids):
        return x + nf12.fused_moe(x, m13, m2, weights, ids)

    compiled = torch.compile(layer, fullgraph=True, dynamic=dynamic)
    for t in (0, 1, 8, 33):
        x = activations(t, k, seed=t)
        weights, ids = routing(t, e, topk, seed=t)
        assert_bitwise(compiled(x, weights, ids), layer(x, weights, ids))
        # The internal fused operator and the individual primitives are opaque
        # to inductor; both interfaces retain their BF16 boundaries.
        def chain(x, w, i):
            sorted_ids, offsets = nf12.moe_align(i, e)
            h = nf12.moe_linear(x, m13, offsets, gather=sorted_ids, gather_div=topk)
            y = nf12.moe_linear(h[:, inter:], m2, offsets, scatter=sorted_ids, row_scale=w.reshape(-1))
            return nf12.moe_combine(y, topk)

        assert_bitwise(torch.compile(chain, fullgraph=True, dynamic=dynamic)(x, weights, ids), chain(x, weights, ids))


def test_fused_moe_compile_mixed_expert_map():
    torch._dynamo.reset()
    w13, w2 = moe_layer(2, 1024, 512)
    m13, t2 = nf12.encode_moe(w13), nf12.pack_bf16t_moe(w2)
    x = activations(1, 1024)
    ids = torch.tensor([[0, 0, 2**32, 2]], dtype=torch.int64, device=DEV)
    weights = torch.tensor([[0.25, 0.125, 1.0, 0.5]], device=DEV)
    expert_map = torch.tensor([1, -1, 0], dtype=torch.int64, device=DEV)

    def layer(x, weights, ids, expert_map):
        return nf12.fused_moe(x, m13, t2, weights, ids, expert_map=expert_map)

    compiled = torch.compile(layer, fullgraph=True)
    assert_bitwise(compiled(x, weights, ids, expert_map), layer(x, weights, ids, expert_map))
    expert_map.fill_(-1)
    assert_bitwise(compiled(x, weights, ids, expert_map), torch.zeros_like(x))


def test_opcheck():
    e, n, k, t, topk = 4, 64, 128, 5, 2
    w = experts(e, n, k)
    mw, tw = twin(w)
    x = activations(t, k)
    weights, ids = routing(t, e, topk)
    sorted_ids, offsets = nf12.moe_align(ids, e)
    ops = torch.ops.nf12
    torch.library.opcheck(ops.moe_align, (ids, e))
    torch.library.opcheck(ops.moe_linear, (x, mw.packed, mw.escapes, mw.escape_offsets, mw.exp_base, offsets,
                                           sorted_ids, topk, None, None, False))
    h = activations(t * topk, k)
    torch.library.opcheck(ops.moe_linear, (h, mw.packed, mw.escapes, mw.escape_offsets, mw.exp_base, offsets, None,
                                           1, sorted_ids, weights.reshape(-1), True))
    torch.library.opcheck(ops.moe_linear_bf16t, (x, tw, offsets, sorted_ids, topk, None, None, False))
    torch.library.opcheck(ops.moe_combine, (activations(t * topk, n), topk))
    torch.library.opcheck(ops.moe_silu_mul, (activations(t * topk, 2 * k),))
    w13, w2 = moe_layer(e, k, k)
    m13, m2 = nf12.encode_moe(w13), nf12.encode_moe(w2)
    torch.library.opcheck(ops._fused_moe,
                         (x, m13.packed, m13.escapes, m13.escape_offsets, m13.exp_base,
                          m2.packed, m2.escapes, m2.escape_offsets, m2.exp_base, weights, ids, False))
    torch.library.opcheck(ops._fused_moe,
                         (x, nf12.pack_bf16t_moe(w13), None, None, None,
                          m2.packed, m2.escapes, m2.escape_offsets, m2.exp_base, weights, ids, True))




def test_state_dict_round_trip(tmp_path):
    w = experts(4, 64, 128)
    mw = nf12.encode_moe(w)
    sd = mw.state_dict("w13.")
    assert set(sd) == {"w13.nf12_packed", "w13.nf12_escapes", "w13.nf12_escape_offsets", "w13.nf12_exp_base"}
    back = nf12.NF12MoEWeight.from_state_dict(sd, "w13.")
    assert_bitwise(back.decode(), w)
    cpu = mw.to("cpu")
    assert cpu.device.type == "cpu" and cpu.nbytes == mw.nbytes
    assert_bitwise(cpu.to(DEV).decode(), w)

    safetensors_torch = pytest.importorskip("safetensors.torch")
    path = tmp_path / "moe.safetensors"
    safetensors_torch.save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    loaded = nf12.NF12MoEWeight.from_state_dict(safetensors_torch.load_file(str(path), device=DEV), "w13.")
    assert_bitwise(loaded.decode(), w)




@pytest.mark.parametrize("t", [0, 1, 2])
def test_fused_moe_validates_before_dispatch(t):
    w13, w2 = moe_layer(2, 1024, 512)
    m13, m2 = nf12.encode_moe(w13), nf12.encode_moe(w2)
    x = activations(t, 1024)
    weights, ids = routing(t, 2, 2)
    args = [x, m13.packed, m13.escapes, m13.escape_offsets, m13.exp_base,
            m2.packed, m2.escapes, m2.escape_offsets, m2.exp_base, weights, ids, False]
    strided_offsets = torch.empty(2 * m13.escape_offsets.numel(), dtype=torch.int64, device=DEV)[::2]
    strided_offsets.copy_(m13.escape_offsets)
    strided_packed = torch.empty((*m13.packed.shape[:2], 2 * m13.packed.shape[2]),
                                 dtype=torch.uint8, device=DEV)[:, :, ::2]
    strided_packed.copy_(m13.packed)
    for position, invalid, message in (
        (0, x.float(), "bfloat16"),
        (1, strided_packed, "contiguous"),
        (2, None, "all be present"),
        (2, m13.escapes.float(), "escapes"),
        (3, strided_offsets, "escape_offsets"),
        (4, m13.exp_base.to(torch.int32), "exp_base"),
        (5, m2.packed.cpu(), "CUDA"),
        (9, weights.to(torch.bfloat16), "float32"),
        (10, ids.to(torch.int64), "int32"),
    ):
        bad = args.copy()
        bad[position] = invalid
        with pytest.raises(RuntimeError, match=message):
            torch.ops.nf12._fused_moe(*bad)
    with pytest.raises(ValueError, match="int32/int64"):
        nf12.fused_moe(x, m13, m2, weights, ids.float())
    with pytest.raises(RuntimeError, match="topk"):
        nf12.fused_moe(x, m13, m2, weights[:, :0], ids[:, :0])


def test_errors():
    mw = nf12.encode_moe(experts(2, 64, 128))
    offsets = torch.zeros(3, dtype=torch.int64, device=DEV)
    with pytest.raises(ValueError, match="multiple of 16"):
        nf12.encode_moe(experts(2, 24, 64))
    with pytest.raises(ValueError, match="bfloat16"):
        nf12.encode_moe(experts(2, 16, 64)[0])
    with pytest.raises(ValueError, match="escape_offsets"):
        nf12.NF12MoEWeight(mw.packed, mw.escapes, mw.escape_offsets[:2], mw.exp_base)
    with pytest.raises(ValueError, match=r"\[rows, 128\]"):
        nf12.moe_linear(activations(2, 64), mw, offsets)
    with pytest.raises(RuntimeError, match="expert_offsets"):
        nf12.moe_linear(activations(2, 128), mw, offsets[:2])
    with pytest.raises(RuntimeError, match="gather_div"):
        nf12.moe_linear(activations(2, 128), mw, offsets, gather=offsets.int(), gather_div=0)
    with pytest.raises(RuntimeError, match="row_scale"):
        nf12.moe_linear(activations(2, 128), mw, offsets, row_scale=torch.ones(3, device=DEV))
    with pytest.raises(RuntimeError, match="multiple of topk"):
        nf12.moe_combine(activations(5, 64), 2)
    with pytest.raises(RuntimeError, match="multiple of 8"):
        nf12.moe_combine(activations(4, 12), 2)
    with pytest.raises(RuntimeError, match="multiple of 8"):
        nf12.moe_silu_mul(activations(4, 24))
    with pytest.raises(RuntimeError, match="multiple of 8"):
        nf12.moe_silu_mul(activations(4, 33))
    with pytest.raises(RuntimeError, match="bfloat16"):
        nf12.moe_silu_mul(activations(4, 32).float())
    with pytest.raises(ValueError, match="activation"):
        nf12.fused_moe(activations(1, 128), mw, mw, torch.ones(1, 1, device=DEV),
                       torch.zeros(1, 1, dtype=torch.int32, device=DEV), activation="gelu")
    with pytest.raises(ValueError, match="out of range"):
        mw.narrow(1, 2)




class MoeWeight(ctypes.Structure):
    _fields_ = [
        ("packed", ctypes.c_void_p),
        ("escapes", ctypes.c_void_p),
        ("escape_offsets", ctypes.c_void_p),
        ("exp_base", ctypes.c_void_p),
        ("num_experts", ctypes.c_int64),
        ("n", ctypes.c_int64),
        ("k", ctypes.c_int64),
    ]


@pytest.fixture(scope="module")
def lib():
    lib = ctypes.CDLL(str(nf12.library_path()))
    p, i64, sz, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_size_t, ctypes.c_int
    sigs = {
        "nf12_moe_align_workspace_bytes": (sz, [i64, i64, i64]),
        "nf12_moe_align": (i32, [p, i64, i64, i64, p, p, p, sz, p]),
        "nf12_moe_linear_workspace_bytes": (sz, [i64, i64, i64, i64, i32]),
        "nf12_moe_linear": (i32, [p, i64, p, i64, p, p, i64, ctypes.POINTER(MoeWeight), p, p, i64, p, sz, p]),
        "nf12_moe_linear_bf16t": (i32, [p, i64, p, i64, p, p, i64, p, i64, i64, i64, p, p, i64, p, sz, p]),
        "nf12_moe_combine": (i32, [p, i64, i64, i64, i64, p, i64, p]),
        "nf12_moe_silu_mul": (i32, [p, i64, i64, i64, p, i64, p]),
        "nf12_moe_decode": (i32, [ctypes.POINTER(MoeWeight), p, i64, i64, i64, p, i64, p]),
        "nf12_moe_linear_decode_gemm_rows": (i64, [i64, i64, i64, i64, i32]),
        "nf12_moe_grouped_operands": (i32, [p, i64, p, i64, p, i64, i64, i64, p, i64, p, p]),
        "nf12_last_cuda_error": (i32, []),
    }
    for name, (restype, argtypes) in sigs.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, argtypes
    return lib


def stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def test_c_align_validation(lib):
    ids = torch.zeros(4, 2, dtype=torch.int32, device=DEV)
    out = torch.empty(8, dtype=torch.int32, device=DEV)
    off = torch.full((5,), -7, dtype=torch.int64, device=DEV)
    # Workspace sizes are backend-dependent; consumers use the public query.
    small_need = lib.nf12_moe_align_workspace_bytes(4, 2, 4)
    small_ws = torch.empty(small_need, dtype=torch.uint8, device=DEV)
    a = (ids.data_ptr(), 4, 2, 4, out.data_ptr(), off.data_ptr(),
         small_ws.data_ptr() if small_need else None, small_need, stream())
    assert lib.nf12_moe_align(*a) == OK
    assert lib.nf12_moe_align(a[0], 4, 0, 4, *a[4:]) == INVALID  # topk < 1
    assert lib.nf12_moe_align(a[0], 4, 2, 0, *a[4:]) == INVALID  # E < 1
    assert lib.nf12_moe_align(a[0], -1, 2, 4, *a[4:]) == INVALID
    assert lib.nf12_moe_align(a[0], 2**31, 2, 4, *a[4:]) == INVALID  # > 2^31 - 1 assignments
    assert lib.nf12_moe_align(a[0] + 2, *a[1:]) == INVALID  # misaligned ids
    assert lib.nf12_moe_align(*a[:5], off.data_ptr() + 4, *a[6:]) == INVALID
    assert lib.nf12_moe_align(*a[:5], None, *a[6:]) == INVALID
    assert lib.nf12_moe_align(None, 4, 2, 4, *a[4:]) == INVALID
    # No tokens: ids and sorted may be null, offsets are zeroed.
    assert lib.nf12_moe_align(None, 0, 2, 4, None, off.data_ptr(), None, 0, stream()) == OK
    torch.cuda.synchronize()
    assert off.tolist() == [0] * 5
    # Multi-block sizes need workspace.
    need = lib.nf12_moe_align_workspace_bytes(4096, 8, 128)
    assert need > 0
    if torch.version.hip is None:  # CUDA has a workspace-free small routing kernel.
        assert lib.nf12_moe_align_workspace_bytes(1, 8, 128) == 0
    big = torch.zeros(4096, 8, dtype=torch.int32, device=DEV)
    big_out = torch.empty(4096 * 8, dtype=torch.int32, device=DEV)
    big_off = torch.empty(129, dtype=torch.int64, device=DEV)
    assert lib.nf12_moe_align(big.data_ptr(), 4096, 8, 128, big_out.data_ptr(), big_off.data_ptr(), None, 0,
                              stream()) == INSUFFICIENT
    ws = torch.empty(need, dtype=torch.uint8, device=DEV)
    assert lib.nf12_moe_align(big.data_ptr(), 4096, 8, 128, big_out.data_ptr(), big_off.data_ptr(), ws.data_ptr(),
                              need, stream()) == OK
    torch.cuda.synchronize()
    assert big_off.tolist() == [0] + [4096 * 8] * 128
    assert torch.equal(big_out, torch.arange(4096 * 8, dtype=torch.int32, device=DEV))
    assert lib.nf12_last_cuda_error() == 0


def test_c_linear_validation(lib):
    e, n, k = 2, 64, 128
    w = experts(e, n, k)
    mw = nf12.encode_moe(w)
    tw = nf12.pack_bf16t_moe(w)
    x = activations(4, k)
    y = torch.empty(4, n, dtype=torch.bfloat16, device=DEV)
    off = torch.tensor([0, 2, 4], dtype=torch.int64, device=DEV)
    cw = MoeWeight(mw.packed.data_ptr(), mw.escapes.data_ptr() if mw.num_escapes else None,
                   mw.escape_offsets.data_ptr(), mw.exp_base.data_ptr(), e, n, k)
    ws_bytes = lib.nf12_moe_linear_workspace_bytes(4, e, n, k, torch.cuda.current_device())
    ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device=DEV)

    x_ptr, offsets_ptr, y_ptr, workspace_ptr = x.data_ptr(), off.data_ptr(), y.data_ptr(), ws.data_ptr()

    def run(x_ptr=x_ptr, ldx=k, gather=None, div=1, scatter=None, offsets=offsets_ptr, a=4, weight=cw,
            scale=None, y_ptr=y_ptr, ldy=n, ws_ptr=workspace_ptr, wsb=ws_bytes):
        return lib.nf12_moe_linear(x_ptr, ldx, gather, div, scatter, offsets, a, ctypes.byref(weight), scale, y_ptr,
                                   ldy, ws_ptr, wsb, stream())

    assert run() == OK
    assert run(a=0, x_ptr=None, y_ptr=None, offsets=None) == OK  # no launch
    assert run(a=-1) == INVALID
    assert run(div=0) == INVALID
    assert run(a=0, div=0) == INVALID
    assert run(x_ptr=x.data_ptr() + 2) == INVALID
    assert run(ldx=k - 8) == INVALID
    assert run(ldy=n + 4) == INVALID
    assert run(offsets=None) == INVALID
    assert run(offsets=off.data_ptr() + 4) == INVALID
    assert run(gather=off.data_ptr() + 2) == INVALID
    assert run(scale=off.data_ptr() + 2) == INVALID
    bad = MoeWeight(*[getattr(cw, f) for f, _ in MoeWeight._fields_])
    bad.num_experts = 0
    assert run(weight=bad) == INVALID
    bad.num_experts, bad.n = e, 24
    assert run(weight=bad) == INVALID
    bad.n, bad.escape_offsets = n, None
    assert run(weight=bad) == INVALID
    bad.escape_offsets, bad.exp_base = cw.escape_offsets, None
    assert run(weight=bad) == INVALID
    bad.exp_base, bad.packed = cw.exp_base, cw.packed + 8
    assert run(weight=bad) == INVALID
    if ws_bytes > 0:
        assert run(wsb=ws_bytes - 1) == INSUFFICIENT
    assert lib.nf12_moe_linear_workspace_bytes(4, 0, n, k, 0) == 0
    assert lib.nf12_moe_linear_workspace_bytes(4, e, n, k, -1) == 0
    assert lib.nf12_moe_linear_bf16t(x.data_ptr(), k, None, 1, None, off.data_ptr(), 4, tw.data_ptr(), e, n, k, None,
                                     y.data_ptr(), n, ws.data_ptr(), ws_bytes, stream()) == OK
    assert lib.nf12_moe_linear_bf16t(x.data_ptr(), k, None, 1, None, off.data_ptr(), 4, tw.data_ptr() + 8, e, n, k,
                                     None, y.data_ptr(), n, ws.data_ptr(), ws_bytes, stream()) == INVALID
    assert lib.nf12_moe_linear_bf16t(x.data_ptr(), k, None, 1, None, off.data_ptr(), 4, tw.data_ptr(), e, n, 96,
                                     None, y.data_ptr(), n, ws.data_ptr(), ws_bytes, stream()) == INVALID
    torch.cuda.synchronize()
    assert lib.nf12_last_cuda_error() == 0


def test_c_combine_validation(lib):
    y = activations(8, 64)
    out = torch.empty(4, 64, dtype=torch.bfloat16, device=DEV)
    s = stream()
    assert lib.nf12_moe_combine(y.data_ptr(), 64, 4, 2, 64, out.data_ptr(), 64, s) == OK
    assert lib.nf12_moe_combine(None, 64, 0, 2, 64, None, 64, s) == OK
    assert lib.nf12_moe_combine(y.data_ptr(), 64, 4, 0, 64, out.data_ptr(), 64, s) == INVALID
    assert lib.nf12_moe_combine(y.data_ptr(), 64, 4, 2, 60, out.data_ptr(), 64, s) == INVALID
    assert lib.nf12_moe_combine(y.data_ptr(), 60, 4, 2, 56, out.data_ptr(), 64, s) == INVALID
    assert lib.nf12_moe_combine(y.data_ptr() + 2, 64, 4, 2, 64, out.data_ptr(), 64, s) == INVALID
    assert lib.nf12_moe_combine(y.data_ptr(), 64, 4, 2, 64, out.data_ptr(), 32, s) == INVALID
    assert lib.nf12_moe_combine(y.data_ptr(), 64, 4, 2, 64, out.data_ptr() + 2, 64, s) == INVALID
    assert lib.nf12_moe_combine(y.data_ptr(), 64, 2**62, 2, 64, out.data_ptr(), 64, s) == INVALID
    torch.cuda.synchronize()
    assert_bitwise(out, reference.moe_combine(y, 2))


def test_c_silu_mul_validation(lib):
    h = activations(4, 128)
    out = torch.empty(4, 64, dtype=torch.bfloat16, device=DEV)
    s = stream()
    assert lib.nf12_moe_silu_mul(h.data_ptr(), 128, 4, 64, out.data_ptr(), 64, s) == OK
    assert lib.nf12_moe_silu_mul(None, 128, 0, 64, None, 64, s) == OK
    assert lib.nf12_moe_silu_mul(h.data_ptr(), 128, -1, 64, out.data_ptr(), 64, s) == INVALID
    assert lib.nf12_moe_silu_mul(h.data_ptr(), 128, 4, 0, out.data_ptr(), 64, s) == INVALID
    assert lib.nf12_moe_silu_mul(h.data_ptr(), 128, 4, 60, out.data_ptr(), 64, s) == INVALID
    assert lib.nf12_moe_silu_mul(h.data_ptr(), 120, 4, 64, out.data_ptr(), 64, s) == INVALID
    assert lib.nf12_moe_silu_mul(h.data_ptr(), 132, 4, 64, out.data_ptr(), 64, s) == INVALID
    assert lib.nf12_moe_silu_mul(h.data_ptr() + 2, 128, 4, 64, out.data_ptr(), 64, s) == INVALID
    assert lib.nf12_moe_silu_mul(h.data_ptr(), 128, 4, 64, out.data_ptr() + 2, 64, s) == INVALID
    assert lib.nf12_moe_silu_mul(h.data_ptr(), 128, 4, 64, out.data_ptr(), 32, s) == INVALID
    assert lib.nf12_moe_silu_mul(h.data_ptr(), 128, 2**62, 64, out.data_ptr(), 64, s) == INVALID
    torch.cuda.synchronize()
    assert_bitwise(out, reference.moe_silu_mul(h))


@pytest.mark.parametrize("experts,topk", [(1, 1), (4, 2), (20000, 8)])
def test_empty_moe_align_needs_no_workspace(experts, topk):
    ids = torch.empty((0, topk), device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        sorted_ids, offsets = nf12.moe_align(ids, experts)
    stream.synchronize()
    assert sorted_ids.shape == (0,)
    assert offsets.shape == (experts + 1,)
    assert bool((offsets == 0).all())


@pytest.mark.parametrize("e,zero_fill", [(1, True), (4, False)], ids=["single-expert", "multiple-experts"])
def test_routed_gather_tail_is_not_read(moe_pack, e, zero_fill):
    """SPEC 9.2 leaves the dropped gather tail unspecified, even within a tile."""
    n, k, topk = 256, 512, 2
    w = experts(e, n, k, seed=70, scale=.02, spread=0)
    x = activations(8, k, seed=71) * .05
    if e == 1:
        offsets = torch.tensor([0, 3], device=DEV, dtype=torch.int64)
        gather = torch.tensor([0, 1, 2, 1 << 28, -(1 << 28), 1 << 30, -1, 1 << 27],
                              device=DEV, dtype=torch.int32)
    else:
        ids = torch.tensor([[0, 0], [1, -1], [-1, -1], [2, 3], [0, -1], [3, -1], [-1, -1], [1, 2]],
                           device=DEV, dtype=torch.int32)
        gather, offsets = nf12.moe_align(ids, e)
        gather[int(offsets[-1]):] = torch.iinfo(torch.int32).max
    valid = int(offsets[-1])
    assert valid < gather.numel()
    out = nf12.moe_linear(x, moe_pack(w), offsets, gather=gather, gather_div=topk, zero_fill=zero_fill)
    ref = reference.moe_linear(x, w, offsets, gather=gather, gather_div=topk)
    assert bool(out[:valid].isfinite().all())
    torch.testing.assert_close(out[:valid].float(), ref[:valid].float(), rtol=2e-3, atol=2e-3)
    if zero_fill:
        assert bool((out[valid:] == 0).all())


@pytest.mark.parametrize("ids", [[0, -1], [-1, -1]], ids=["one-active", "all-dropped"])
@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_fused_moe_dropped_assignment_ignores_router_scale(moe_pack, ids, bad_scale):
    e, k, inter = 2, 256, 128
    w13 = experts(e, 2 * inter, k, seed=72, scale=.02, spread=0)
    w2 = experts(e, k, inter, seed=73, scale=.02, spread=0)
    x = activations(1, k, seed=74) * .05
    ids = torch.tensor([ids], device=DEV, dtype=torch.int32)
    weights = torch.tensor([[1., bad_scale]], device=DEV, dtype=torch.float32)
    out = nf12.fused_moe(x, moe_pack(w13), moe_pack(w2), weights, ids)
    assert bool(out.isfinite().all())
    ref = reference.fused_moe(x, w13, w2, weights, ids)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("operation", ["dense", "routed", "fused"])
def test_inference_streams_own_their_scratch(moe_pack, operation):
    """Different inputs on simultaneous streams expose accidental scratch reuse."""
    e, k, inter, t, topk = 4, 512, 256, 32, 2
    w13 = experts(e, 2 * inter, k, seed=60, scale=.02, spread=0)
    w2 = experts(e, k, inter, seed=61, scale=.02, spread=0)
    a, b = moe_pack(w13), moe_pack(w2)
    if operation == "dense":
        dense = nf12.encode(w13[0]) if isinstance(a, nf12.NF12MoEWeight) else nf12.pack_bf16t(w13[0])
        linear = nf12.linear if isinstance(a, nf12.NF12MoEWeight) else nf12.linear_bf16t
    weights, ids = routing(t, e, topk, seed=62)
    gather, offsets = nf12.moe_align(ids, e)

    def run(x):
        if operation == "dense":
            return linear(x, dense)
        if operation == "fused":
            return nf12.fused_moe(x[:1], a, b, weights[:1], ids[:1])
        return nf12.moe_linear(x, a, offsets, gather=gather, gather_div=topk,
                               scatter=gather, row_scale=weights.reshape(-1), zero_fill=True)

    inputs = [activations(t, k, seed=63 + i) * .05 for i in range(2)]
    expected = [run(x) for x in inputs]  # also warm up before threaded calls
    producer = torch.cuda.current_stream()
    streams = [torch.cuda.Stream() for _ in inputs]
    for stream in streams:
        stream.wait_stream(producer)
    barrier = threading.Barrier(2)

    def worker(i):
        with torch.cuda.device(inputs[i].device), torch.cuda.stream(streams[i]):
            barrier.wait(timeout=30)
            outputs = [run(inputs[i]) for _ in range(16)]
        streams[i].synchronize()
        for out in outputs:
            assert_bitwise(out, expected[i])

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, range(2)))
