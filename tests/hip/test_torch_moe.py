# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
"""``torch.ops.nf12`` parity on ROCm for the bf16t twin and the MoE ops.

The CUDA torch extension is the same ``csrc/torch/ops.cpp``; on ROCm it links
``libnf12_hip.so`` through the ``nf12_*`` compatibility layer. These tests run
the Python API, not the C ABI, and check every op against ``nf12.reference``.
"""

from __future__ import annotations

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


def _violations(got: torch.Tensor, ref: torch.Tensor) -> int:
    """SPEC.md 6.2 gate: |err| <= 2e-3 + 2e-3 * |ref|, plus finite output."""
    err = (got.double() - ref.double()).abs()
    tol = 2e-3 + 2e-3 * ref.double().abs()
    return int(((err > tol) | ~torch.isfinite(got)).sum())


def _bf16(shape, scale: float, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=gen) * scale).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# bf16t twin
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 15, 16, 17, 64, 128, 256, 512])
def test_linear_bf16t_matches_bf16(m):
    torch.manual_seed(0)
    n, k = 1024, 4096
    w = _bf16((n, k), 0.02, 0)
    wt = nf12.pack_bf16t(w)
    x = _bf16((m, k), 0.05, 1)
    got = nf12.linear_bf16t(x, wt)
    ref = F.linear(x, w)
    assert got.shape == ref.shape
    assert _violations(got, ref) == 0, m


def test_linear_bf16t_matches_nf12():
    """The twin is the exactness reference for the nf12 route (SPEC.md 6.2)."""
    torch.manual_seed(1)
    n, k = 512, 2048
    w = _bf16((n, k), 0.02, 2)
    nw = nf12.encode(w)
    wt = nf12.pack_bf16t(w)
    bias = _bf16((n,), 0.5, 3)
    for m in (16, 33, 128, 300):
        x = _bf16((m, k), 0.05, 10 + m)
        a = nf12.linear(x, nw, bias)
        b = nf12.linear_bf16t(x, wt, bias)
        assert _violations(b, a.double()) == 0, m


def test_linear_bf16t_multi_dim_input():
    torch.manual_seed(2)
    n, k = 256, 512
    w = _bf16((n, k), 0.02, 4)
    wt = nf12.pack_bf16t(w)
    x = _bf16((3, 5, k), 0.05, 5)
    got = nf12.linear_bf16t(x, wt)
    ref = F.linear(x, w)
    assert got.shape == ref.shape
    assert _violations(got, ref) == 0


# ---------------------------------------------------------------------------
# moe_align
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tokens,topk,experts", [(1, 1, 1), (17, 2, 4), (1024, 8, 8), (2048, 8, 64)]
)
def test_moe_align_matches_reference(tokens, topk, experts):
    torch.manual_seed(3)
    ids = torch.randint(
        -1, experts + 1, (tokens, topk), device="cuda", dtype=torch.int32
    )
    got_sorted, got_offsets = nf12.moe_align(ids, experts)
    ref_sorted, ref_offsets = reference.moe_align(ids, experts)
    assert got_offsets.cpu().equal(ref_offsets.cpu())
    valid = int(got_offsets[-1])
    # Positions past the valid prefix are unspecified by SPEC.md 9.2.
    assert got_sorted[:valid].cpu().equal(ref_sorted[:valid].cpu())


# ---------------------------------------------------------------------------
# elementwise MoE ops
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tokens,topk,n", [(1, 1, 8), (7, 4, 128), (129, 8, 512)])
def test_moe_combine_matches_reference(tokens, topk, n):
    torch.manual_seed(4)
    y = _bf16((tokens * topk, n), 0.5, 6)
    assert (
        nf12.moe_combine(y, topk)
        .view(torch.int16)
        .equal(reference.moe_combine(y, topk).view(torch.int16))
    )


@pytest.mark.parametrize("rows,inter", [(1, 8), (5, 64), (257, 512)])
def test_moe_silu_mul_matches_reference(rows, inter):
    torch.manual_seed(5)
    h = _bf16((rows, 2 * inter), 2.0, 7)
    assert (
        nf12.moe_silu_mul(h)
        .view(torch.int16)
        .equal(reference.moe_silu_mul(h).view(torch.int16))
    )


# ---------------------------------------------------------------------------
# MoE decode / linear
# ---------------------------------------------------------------------------
def _moe_weights(experts: int, n: int, k: int, seed: int) -> torch.Tensor:
    return _bf16((experts, n, k), 0.02, seed)


def test_moe_decode_matches_per_expert_decode():
    torch.manual_seed(6)
    e, n, k = 4, 256, 512
    w = _moe_weights(e, n, k, 8)
    nw = nf12.encode_moe(w)
    got = nw.decode()
    assert got.view(torch.int16).equal(w.view(torch.int16))


def _align(x, topk_ids, experts):
    sorted_ids, offsets = nf12.moe_align(topk_ids, experts)
    return sorted_ids, offsets


@pytest.mark.parametrize("zero_fill", [False, True])
def test_moe_linear_gather_matches_reference(zero_fill):
    torch.manual_seed(7)
    e, n, k, tokens, topk = 4, 256, 512, 64, 2
    w = _moe_weights(e, n, k, 9)
    nw = nf12.encode_moe(w)
    x = _bf16((tokens, k), 0.05, 10)
    ids = torch.randint(0, e, (tokens, topk), device="cuda", dtype=torch.int32)
    sorted_ids, offsets = _align(x, ids, e)
    got = nf12.moe_linear(
        x, nw, offsets, gather=sorted_ids, gather_div=topk, zero_fill=zero_fill
    )
    ref = reference.moe_linear(x, w, offsets, gather=sorted_ids, gather_div=topk)
    assert _violations(got, ref) == 0


def test_moe_linear_scatter_and_row_scale_matches_reference():
    torch.manual_seed(8)
    e, n, k, tokens, topk = 4, 256, 512, 48, 2
    w = _moe_weights(e, n, k, 11)
    nw = nf12.encode_moe(w)
    x = _bf16((tokens, k), 0.05, 12)
    ids = torch.randint(0, e, (tokens, topk), device="cuda", dtype=torch.int32)
    sorted_ids, offsets = _align(x, ids, e)
    scale = torch.rand(tokens * topk, device="cuda", dtype=torch.float32)
    got = nf12.moe_linear(
        x,
        nw,
        offsets,
        gather=sorted_ids,
        gather_div=topk,
        scatter=sorted_ids,
        row_scale=scale,
        zero_fill=True,
    )
    ref = reference.moe_linear(
        x,
        w,
        offsets,
        gather=sorted_ids,
        gather_div=topk,
        scatter=sorted_ids,
        row_scale=scale,
    )
    assert _violations(got, ref) == 0


def test_moe_linear_bf16t_matches_nf12():
    torch.manual_seed(9)
    e, n, k, tokens, topk = 4, 256, 512, 32, 2
    w = _moe_weights(e, n, k, 13)
    nw = nf12.encode_moe(w)
    wt = nf12.pack_bf16t_moe(w)
    x = _bf16((tokens, k), 0.05, 14)
    ids = torch.randint(0, e, (tokens, topk), device="cuda", dtype=torch.int32)
    sorted_ids, offsets = _align(x, ids, e)
    scale = torch.rand(tokens * topk, device="cuda", dtype=torch.float32)
    a = nf12.moe_linear(
        x,
        nw,
        offsets,
        gather=sorted_ids,
        gather_div=topk,
        scatter=sorted_ids,
        row_scale=scale,
        zero_fill=True,
    )
    b = nf12.moe_linear(
        x,
        wt,
        offsets,
        gather=sorted_ids,
        gather_div=topk,
        scatter=sorted_ids,
        row_scale=scale,
        zero_fill=True,
    )
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(b.view(torch.int16), a.view(torch.int16))


@pytest.mark.parametrize("twin", [False, True])
def test_fused_moe_matches_reference(twin):
    torch.manual_seed(10)
    e, tokens, topk, k, inter = 4, 24, 2, 256, 128
    w13 = _bf16((e, 2 * inter, k), 0.02, 15)
    w2 = _bf16((e, k, inter), 0.02, 16)
    x = _bf16((tokens, k), 0.05, 17)
    ids = torch.randint(0, e, (tokens, topk), device="cuda", dtype=torch.int32)
    weights = torch.rand(tokens, topk, device="cuda", dtype=torch.float32)
    w13q = nf12.pack_bf16t_moe(w13) if twin else nf12.encode_moe(w13)
    w2q = nf12.pack_bf16t_moe(w2) if twin else nf12.encode_moe(w2)
    got = nf12.fused_moe(x, w13q, w2q, weights, ids)
    ref = reference.fused_moe(x, w13, w2, weights, ids)
    assert got.shape == ref.shape
    assert _violations(got, ref) == 0


def test_moe_linear_dropped_ids_zero_fill():
    """Dropped assignments must not be written, and zero_fill must clear them."""
    torch.manual_seed(11)
    e, n, k, tokens, topk = 4, 256, 512, 40, 2
    w = _moe_weights(e, n, k, 18)
    nw = nf12.encode_moe(w)
    x = _bf16((tokens, k), 0.05, 19)
    # Half the assignments route to -1 (dropped), like an expert-parallel rank.
    ids = torch.randint(-1, e, (tokens, topk), device="cuda", dtype=torch.int32)
    sorted_ids, offsets = _align(x, ids, e)
    assert int(offsets[-1]) < tokens * topk
    got = nf12.moe_linear(
        x, nw, offsets, gather=sorted_ids, gather_div=topk, zero_fill=True
    )
    ref = reference.moe_linear(x, w, offsets, gather=sorted_ids, gather_div=topk)
    assert _violations(got, ref) == 0
    # The dropped tail must be exactly zero, not just close.
    assert int(got[int(offsets[-1]) :].abs().sum()) == 0


def test_moe_linear_routed_tail_is_never_read():
    """A tail tile clamps into the expert's assignments, not the gather capacity.

    SPEC.md 9.2 leaves the gather tail unspecified. The tail tile has fewer
    than 16 valid assignments, and clamping to `max_assignments` instead of
    `row_base + m` dereferences those unspecified entries, so a poisoned value
    names an activation row outside ``x``. The dropped rows are never written,
    so this is a memory-safety guard rather than a numerical one: a large
    enough poisoned index faults the kernel instead of changing the output.
    """
    torch.manual_seed(21)
    e, n, k = 1, 256, 256
    w = _moe_weights(e, n, k, 77)
    nw = nf12.encode_moe(w)
    x = _bf16((4, k), 0.05, 78)
    # Three valid assignments: one 16-row tile with thirteen dropped rows.
    offsets = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    # Entries 3..7 are the unspecified tail; each is far outside x's 4 rows.
    gather = torch.tensor(
        [0, 1, 2, 1 << 28, -(1 << 28), 1 << 30, -1, 1 << 27],
        device="cuda",
        dtype=torch.int32,
    )
    got = nf12.moe_linear(x, nw, offsets, gather=gather, gather_div=1, zero_fill=True)
    ref = reference.moe_linear(x, w, offsets, gather=gather, gather_div=1)
    assert _violations(got, ref) == 0
    assert bool(torch.isfinite(got.float()).all())
    # Rows no valid assignment writes stay zero.
    assert int(got[3:].abs().sum()) == 0


@pytest.mark.parametrize("twin", [False, True])
def test_fused_moe_expert_map_drops_remote_experts(twin):
    """An expert_map of -1 means "another rank"; the result is this rank's partial sum."""
    torch.manual_seed(12)
    e, tokens, topk, k, inter = 4, 16, 2, 256, 128
    w13 = _bf16((e, 2 * inter, k), 0.02, 20)
    w2 = _bf16((e, k, inter), 0.02, 21)
    x = _bf16((tokens, k), 0.05, 22)
    ids = torch.randint(0, e, (tokens, topk), device="cuda", dtype=torch.int32)
    weights = torch.rand(tokens, topk, device="cuda", dtype=torch.float32)
    expert_map = torch.tensor([0, -1, 1, -1], device="cuda", dtype=torch.int32)
    w13q = nf12.pack_bf16t_moe(w13) if twin else nf12.encode_moe(w13)
    w2q = nf12.pack_bf16t_moe(w2) if twin else nf12.encode_moe(w2)
    got = nf12.fused_moe(x, w13q, w2q, weights, ids, expert_map=expert_map)
    ref = reference.fused_moe(x, w13, w2, weights, ids, expert_map=expert_map)
    assert _violations(got, ref) == 0


def test_ops_are_multi_stream_safe():
    """Two concurrent streams must not share scratch: the library keeps no
    mutable device state, so interleaved launches stay independent."""
    torch.manual_seed(13)
    e, n, k, tokens, topk = 4, 256, 512, 32, 2
    w = _moe_weights(e, n, k, 23)
    nw = nf12.encode_moe(w)
    wt = nf12.pack_bf16t_moe(w)
    x = _bf16((tokens, k), 0.05, 24)
    ids = torch.randint(0, e, (tokens, topk), device="cuda", dtype=torch.int32)
    sorted_ids, offsets = _align(x, ids, e)
    scale = torch.rand(tokens * topk, device="cuda", dtype=torch.float32)

    def run():
        a = nf12.moe_linear(
            x,
            nw,
            offsets,
            gather=sorted_ids,
            gather_div=topk,
            scatter=sorted_ids,
            row_scale=scale,
            zero_fill=True,
        )
        b = nf12.moe_linear(
            x,
            wt,
            offsets,
            gather=sorted_ids,
            gather_div=topk,
            scatter=sorted_ids,
            row_scale=scale,
            zero_fill=True,
        )
        c = nf12.linear_bf16t(x, nf12.pack_bf16t(w[0]), None)
        return a, b, c

    serial = run()
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in range(2)]
    out = []
    for s in streams:
        # The inputs were produced on the default stream, so each worker waits
        # for it before reading them. The library owns no scratch, so after that
        # the two workers need no ordering between themselves.
        s.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(s):
            out.append(run())
    torch.cuda.synchronize()
    for got in out:
        for g, r in zip(got, serial):
            assert g.view(torch.int16).equal(r.view(torch.int16))
    # Two concurrent workers must also agree with each other: shared scratch
    # would show up as one overwriting the other's partial sums.
    for g0, g1 in zip(out[0], out[1]):
        assert g0.view(torch.int16).equal(g1.view(torch.int16))


# ---------------------------------------------------------------------------
# fused single-token MoE (nf12::torch_detail::fused_decode)
# ---------------------------------------------------------------------------
def _composed_moe(x, w13q, w2q, weights, ids, experts):
    """The route ``ops.cpp`` falls back to; the fused path must agree with it."""
    topk = ids.shape[1]
    sorted_ids, offsets = nf12.moe_align(ids, experts)
    h = nf12.moe_linear(x, w13q, offsets, gather=sorted_ids, gather_div=topk)
    h = nf12.moe_silu_mul(h)
    y = nf12.moe_linear(
        h, w2q, offsets, scatter=sorted_ids, row_scale=weights.reshape(-1).float()
    )
    return nf12.moe_combine(y, topk)


@pytest.mark.parametrize(
    "k,inter,n,topk",
    [
        (1024, 2048, 1024, 1),
        (2048, 3584, 2048, 2),
        (4096, 4096, 4096, 2),
        (4096, 1024, 4096, 4),
    ],
)
@pytest.mark.parametrize("twin", [False, True])
def test_fused_moe_single_token_matches_reference(twin, k, inter, n, topk):
    """T == 1 takes the two-kernel fused route on ROCm, not the composed one."""
    torch.manual_seed(14)
    e = max(2, topk)
    w13 = _bf16((e, 2 * inter, k), 0.02, 30)
    w2 = _bf16((e, n, inter), 0.02, 31)
    x = _bf16((1, k), 0.05, 32)
    ids = torch.arange(topk, device="cuda", dtype=torch.int32).reshape(1, topk)
    weights = torch.rand(1, topk, device="cuda", dtype=torch.float32)
    w13q = nf12.pack_bf16t_moe(w13) if twin else nf12.encode_moe(w13)
    w2q = nf12.pack_bf16t_moe(w2) if twin else nf12.encode_moe(w2)

    got = nf12.fused_moe(x, w13q, w2q, weights, ids)
    ref = reference.fused_moe(x, w13, w2, weights, ids)
    assert got.shape == ref.shape
    assert _violations(got, ref) == 0

    # Equal to the composed route to FP32 reassociation, not bitwise: the fused
    # kernel walks K with a different association order (the CUDA fused kernel
    # does too, which is why SPEC.md 6.2 is a tolerance and not equality).
    comp = _composed_moe(x, w13q, w2q, weights, ids, e)
    assert _violations(got, comp.double()) == 0

    again = nf12.fused_moe(x, w13q, w2q, weights, ids)
    assert got.view(torch.int16).equal(again.view(torch.int16))


@pytest.mark.parametrize("twin", [False, True])
def test_fused_moe_single_token_dropped_ids(twin):
    """A dropped assignment contributes exactly zero, including an all-dropped tile."""
    torch.manual_seed(15)
    e, k, inter, n, topk = 4, 2048, 4096, 2048, 2
    w13 = _bf16((e, 2 * inter, k), 0.02, 40)
    w2 = _bf16((e, n, inter), 0.02, 41)
    x = _bf16((1, k), 0.05, 42)
    weights = torch.rand(1, topk, device="cuda", dtype=torch.float32)
    w13q = nf12.pack_bf16t_moe(w13) if twin else nf12.encode_moe(w13)
    w2q = nf12.pack_bf16t_moe(w2) if twin else nf12.encode_moe(w2)

    ids = torch.tensor([[0, -1]], device="cuda", dtype=torch.int32)
    got = nf12.fused_moe(x, w13q, w2q, weights, ids)
    ref = reference.fused_moe(x, w13, w2, weights, ids)
    assert _violations(got, ref) == 0

    dropped = torch.full((1, topk), -1, device="cuda", dtype=torch.int32)
    zero = nf12.fused_moe(x, w13q, w2q, weights, dropped)
    assert int((zero != 0).sum()) == 0


@pytest.mark.parametrize("tokens", [2, 5])
def test_fused_moe_multi_token_matches_reference(tokens):
    """T > 1 keeps the composed route (the fused path is single-token only)."""
    torch.manual_seed(16)
    e, k, inter, n, topk = 4, 2048, 1024, 2048, 2
    w13 = _bf16((e, 2 * inter, k), 0.02, 50)
    w2 = _bf16((e, n, inter), 0.02, 51)
    x = _bf16((tokens, k), 0.05, 52)
    ids = torch.randint(0, e, (tokens, topk), device="cuda", dtype=torch.int32)
    weights = torch.rand(tokens, topk, device="cuda", dtype=torch.float32)
    w13q = nf12.encode_moe(w13)
    w2q = nf12.encode_moe(w2)
    got = nf12.fused_moe(x, w13q, w2q, weights, ids)
    ref = reference.fused_moe(x, w13, w2, weights, ids)
    assert _violations(got, ref) == 0


def test_fused_moe_single_token_multi_stream_safe():
    """Concurrent T == 1 calls must not share scratch: the fused path owns none."""
    torch.manual_seed(17)
    e, k, inter, n, topk = 4, 2048, 1024, 2048, 2
    w13 = _bf16((e, 2 * inter, k), 0.02, 60)
    w2 = _bf16((e, n, inter), 0.02, 61)
    w13q = nf12.encode_moe(w13)
    w2q = nf12.encode_moe(w2)
    x = _bf16((1, k), 0.05, 62)
    ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    weights = torch.rand(1, topk, device="cuda", dtype=torch.float32)

    serial = nf12.fused_moe(x, w13q, w2q, weights, ids)
    torch.cuda.synchronize()
    out = []
    for _ in range(2):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(stream):
            out.append(nf12.fused_moe(x, w13q, w2q, weights, ids))
    torch.cuda.synchronize()
    for got in out:
        assert got.view(torch.int16).equal(serial.view(torch.int16))


def test_moe_linear_never_reads_the_dropped_gather_tail():
    """SPEC.md 9.2 leaves gather[off[E]:] unspecified; it must not be read.

    A routed tail tile clamps its rows to the expert's last valid assignment.
    Clamping to the allocation capacity instead would read the dropped tail,
    whose value here names a row far past the activation.
    """
    torch.manual_seed(21)
    e, n, k, tokens, topk = 4, 256, 512, 8, 2
    w = _moe_weights(e, n, k, 70)
    nw = nf12.encode_moe(w)
    x = _bf16((tokens, k), 0.05, 71)
    ids = torch.tensor(
        [[0, 0], [1, -1], [-1, -1], [2, 3], [0, -1], [3, -1], [-1, -1], [1, 2]],
        device="cuda", dtype=torch.int32,
    )
    sorted_ids, offsets = _align(x, ids, e)
    valid = int(offsets[-1])
    assert valid < sorted_ids.numel()
    sorted_ids[valid:] = torch.iinfo(torch.int32).max
    got = nf12.moe_linear(x, nw, offsets, gather=sorted_ids, gather_div=topk)
    ref = reference.moe_linear(x, w, offsets, gather=sorted_ids, gather_div=topk)
    # Positions past the valid prefix are unspecified (no zero_fill), so only
    # the written rows are compared.
    assert _violations(got[:valid], ref[:valid].double()) == 0


@pytest.mark.parametrize("dropped_ids", [[0, -1], [-1, -1]])
@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf")])
def test_fused_moe_dropped_assignment_ignores_router_scale(dropped_ids, bad_scale):
    """A dropped assignment contributes +0 without reading its router weight.

    The reference zero-fills that assignment and never scales it. Reading an
    unset NaN/Inf scale multiplied the cleared plane and turned an all-dropped
    tile into NaN.
    """
    torch.manual_seed(22)
    e, k, inter, n = 2, 256, 128, 256
    w13 = _bf16((e, 2 * inter, k), 0.02, 72)
    w2 = _bf16((e, n, inter), 0.02, 73)
    x = _bf16((1, k), 0.05, 74)
    ids = torch.tensor([dropped_ids], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[1.0, bad_scale]], device="cuda", dtype=torch.float32)
    w13q = nf12.encode_moe(w13)
    w2q = nf12.encode_moe(w2)
    got = nf12.fused_moe(x, w13q, w2q, weights, ids)
    assert bool(got.isfinite().all())
    ref = reference.fused_moe(x, w13, w2, weights, ids)
    assert _violations(got, ref) == 0


@pytest.mark.parametrize("m", [16, 32, 64, 128, 256])
def test_linear_bf16t_is_bitwise_identical_to_nf12(m):
    """SPEC.md 6.2: the twin runs the selected plan, so the outputs match bits.

    The twin used to run a fixed eight-tile, one-K-pass configuration, which
    could only be compared to the tuned nf12 dispatch under a tolerance.
    """
    torch.manual_seed(23)
    n, k = 4096, 4096
    w = _bf16((n, k), 0.02, 75)
    nw = nf12.encode(w)
    wt = nf12.pack_bf16t(w)
    x = _bf16((m, k), 0.05, 76)
    a = nf12.linear(x, nw)
    b = nf12.linear_bf16t(x, wt)
    assert a.view(torch.int16).equal(b.view(torch.int16)), m
