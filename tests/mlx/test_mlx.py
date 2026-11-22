# SPDX-License-Identifier: Apache-2.0
"""MLX backend (nf12.mlx): byte-exact codec against nf12.reference, NF12 ==
bf16t twin on the fused routes, and bit parity with the PyTorch MPS backend,
which runs the same Metal kernels through the same launch code."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX Metal GPU required", allow_module_level=True)

import mlx.nn as mnn  # noqa: E402
import torch  # noqa: E402

import nf12  # noqa: E402
import nf12.mlx as nm  # noqa: E402
from nf12 import reference  # noqa: E402
from portable_random import generator, normalized, rand, randn  # noqa: E402

MPS = torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not MPS, reason="PyTorch MPS required for parity")
LINEAR_ROUTES = ("small", "mma", "gemm", "decode")
MOE_ROUTES = ("small", "mma", "gemm")


# --- conversions --------------------------------------------------------------

def to_torch(a: mx.array, device: str = "cpu") -> torch.Tensor:
    if a.dtype == mx.bfloat16:
        return torch.from_numpy(np.array(a.view(mx.int16))).view(torch.bfloat16).to(device)
    return torch.from_numpy(np.array(a)).to(device)


def to_mlx(t: torch.Tensor) -> mx.array:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.int16).numpy()).view(mx.bfloat16)
    return mx.array(t.numpy())


def assert_bitwise(a: mx.array, b) -> None:
    b = b if isinstance(b, mx.array) else to_mlx(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    if a.dtype == mx.bfloat16:
        a, b = a.view(mx.int16), b.view(mx.int16)
    assert mx.array_equal(a, b).item()


def matrix(n: int, k: int, seed: int = 0, scale: float = 0.05, escapes: bool = True) -> mx.array:
    w = randn(n, k, seed=seed, scale=scale)
    if escapes:
        w[0, 0] = 1e30
        w[n // 2, 5] = float("nan")
        w[n - 1, k - 1] = float("-inf")
        w[1, 8:16] = 0.0  # a +0.0 group hits the sentinel
    return to_mlx(w)


def activations(*shape: int, seed: int = 1) -> mx.array:
    return to_mlx(randn(*shape, seed=seed))


def close_to_reference(y: mx.array, x: mx.array, w: mx.array, bias=None) -> None:
    xt = to_torch(x).reshape(-1, x.shape[-1])
    ref = reference.linear(xt, to_torch(w), None if bias is None else to_torch(bias))
    torch.testing.assert_close(to_torch(y).float().reshape(ref.shape), ref.float(), rtol=2e-2, atol=2e-2, equal_nan=True)


# --- codec --------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(16, 64), (48, 192), (256, 512)])
def test_encode_matches_reference_bytes(n, k):
    w = matrix(n, k)
    nw = nm.encode(w)
    packed, escapes, exp_base = reference.encode(to_torch(w))
    assert nw.exp_base == exp_base
    assert np.array_equal(np.array(nw.packed), packed.numpy())
    assert np.array_equal(np.array(nw.escapes), escapes.numpy())
    assert_bitwise(nm.decode(nw), w)


def test_encode_without_escapes_and_all_escaped():
    w = matrix(32, 128, escapes=False, scale=1.0)
    nw = nm.encode(w)
    assert_bitwise(nw.decode(), w)
    zeros = mx.zeros((16, 64), dtype=mx.bfloat16)
    nz = nm.encode(zeros)
    assert nz.num_escapes == 16 * 64 // 8
    assert_bitwise(nz.decode(), zeros)


def test_encode_strided_input():
    big = matrix(64, 256)
    w = big[16:48, 64:192]  # a strided view
    assert_bitwise(nm.encode(w).decode(), w)


def test_pack_bf16t_matches_reference():
    w = matrix(48, 128)
    assert np.array_equal(np.array(nm.pack_bf16t(w)), reference.pack_bf16t(to_torch(w)).numpy())


@requires_mps
def test_encode_parity_with_mps():
    w = matrix(64, 256, seed=3)
    a = nm.encode(w)
    b = nf12.encode(to_torch(w, "mps"))
    assert a.exp_base == b.exp_base
    assert np.array_equal(np.array(a.packed), b.packed.cpu().numpy())
    assert np.array_equal(np.array(a.escapes), b.escapes.cpu().numpy())


def test_embedding():
    w = matrix(64, 128)
    nw = nm.encode(w)
    ids = mx.array([[0, 63, 7], [5, 5, 31]], dtype=mx.int32)
    assert_bitwise(nm.embedding(ids, nw), w[ids])
    out = nm.embedding(mx.array([-1, 64, 3], dtype=mx.int64), nw)
    assert_bitwise(out[:2], mx.zeros((2, 128), dtype=mx.bfloat16))
    assert_bitwise(out[2], w[3])
    assert nm.embedding(mx.zeros((0, 2), dtype=mx.int32), nw).shape == (0, 2, 128)


# --- linear ---------------------------------------------------------------------

@pytest.fixture(params=LINEAR_ROUTES)
def linear_route(request, monkeypatch):
    monkeypatch.setenv("NF12_MPS_LINEAR", request.param)
    return request.param


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 13, 16, 33])
@pytest.mark.parametrize("n,k", [(32, 128), (80, 320)])
def test_linear_all_routes(linear_route, m, n, k):
    w = matrix(n, k, seed=m)
    nw, twin = nm.encode(w), nm.pack_bf16t(w)
    x = activations(m, k, seed=m + 1)
    bias = activations(n, seed=m + 2)
    y = nm.linear(x, nw, bias)
    close_to_reference(y, x, w, bias)
    if linear_route == "decode":
        assert_bitwise(y, mx.addmm(bias, x, nm.decode(nw).T))
    else:
        assert_bitwise(y, nm.linear_bf16t(x, twin, bias))


@requires_mps
@pytest.mark.parametrize("route", ["small", "mma", "gemm"])  # decode-then-GEMM uses each framework's matmul
@pytest.mark.parametrize("m", [1, 4, 7, 16, 40])
def test_linear_parity_with_mps(monkeypatch, route, m):
    monkeypatch.setenv("NF12_MPS_LINEAR", route)
    n, k = 64, 256
    w = matrix(n, k, seed=11)
    x, bias = activations(m, k), activations(n, seed=5)
    tw = nf12.encode(to_torch(w, "mps"))
    expected = nf12.linear(to_torch(x, "mps"), tw, to_torch(bias, "mps"))
    assert_bitwise(nm.linear(x, nm.encode(w), bias), expected.cpu())
    expected_t = nf12.linear_bf16t(to_torch(x, "mps"), nf12.pack_bf16t(to_torch(w, "mps")), to_torch(bias, "mps"))
    assert_bitwise(nm.linear_bf16t(x, nm.pack_bf16t(w), bias), expected_t.cpu())


@pytest.mark.parametrize("m", [4, 7, 16])
@pytest.mark.parametrize("k", [192, 320])
def test_linear_tail_matches_the_twin(m, k):
    # 33 output blocks leave a tail with and without splitting K. Include
    # escapes and use the same accumulation configuration for the BF16 twin.
    n = 528
    w = matrix(n, k, seed=m)
    x = activations(m, k, seed=k)
    nw, twin = nm.encode(w), nm.pack_bf16t(w)
    actual = nm.linear(x, nw)
    close_to_reference(actual, x, w)
    assert_bitwise(actual, nm.linear_bf16t(x, twin))


def test_linear_leading_dims_and_empty():
    w = matrix(32, 64)
    nw = nm.encode(w)
    x = activations(2, 3, 64)
    y = nm.linear(x, nw)
    assert y.shape == (2, 3, 32)
    assert_bitwise(y.reshape(6, 32), nm.linear(x.reshape(6, 64), nw))
    assert nm.linear(mx.zeros((0, 64), dtype=mx.bfloat16), nw).shape == (0, 32)


def test_linear_strided_input():
    w = matrix(32, 128)
    nw = nm.encode(w)
    x = activations(4, 256)[:, 64:192]
    assert_bitwise(nm.linear(x, nw), nm.linear(mx.array(x), nw))


def test_errors():
    w = matrix(32, 64)
    nw = nm.encode(w)
    with pytest.raises(ValueError, match="bfloat16"):
        nm.linear(mx.zeros((2, 64)), nw)
    with pytest.raises(ValueError, match="multiple of 16"):
        nm.encode(mx.zeros((24, 64), dtype=mx.bfloat16))
    with pytest.raises(ValueError, match="multiple of 64"):
        nm.pack_bf16t(mx.zeros((16, 32), dtype=mx.bfloat16))
    with pytest.raises(ValueError, match="mlx.core.array"):
        nm.encode(np.zeros((16, 64)))


# --- MoE ------------------------------------------------------------------------

@pytest.fixture(params=MOE_ROUTES)
def moe_route(request, monkeypatch):
    monkeypatch.setenv("NF12_MPS_MOE", request.param)
    return request.param


def experts(e: int, n: int, k: int, seed: int = 0, escapes: bool = True) -> mx.array:
    w = randn(e, n, k, seed=seed, scale=0.05)
    if escapes:
        w[0, 0, 0] = 1e30
    w[e - 1, 1, 8:16] = 0.0
    return to_mlx(w)


def routing(t: int, e: int, topk: int, seed: int = 2) -> tuple[mx.array, mx.array]:
    g = generator(seed)
    ids = torch.stack([torch.randperm(e, generator=g)[:topk] for _ in range(t)]).to(torch.int32)
    weights = normalized(1 + rand(t, topk, seed=g))
    return to_mlx(weights), to_mlx(ids)


def test_encode_moe_matches_dense_encode():
    w = experts(3, 32, 128)
    mw = nm.encode_moe(w)
    assert_bitwise(mw.decode(), w)
    for i in range(3):
        d = nm.encode(w[i])
        lo, hi = int(mw.escape_offsets[i].item()), int(mw.escape_offsets[i + 1].item())
        assert int(mw.exp_base[i].item()) == d.exp_base
        assert np.array_equal(np.array(mw.packed[i]), np.array(d.packed))
        assert np.array_equal(np.array(mw.escapes[lo:hi]), np.array(d.escapes))
    assert np.array_equal(np.array(nm.pack_bf16t_moe(w)[1]), np.array(nm.pack_bf16t(w[1])))


@pytest.mark.parametrize("t,topk,e", [(1, 1, 4), (5, 2, 8), (300, 8, 64), (7, 3, 2000)])
def test_align_matches_reference(t, topk, e):
    _, ids = routing(t, min(e, 64), topk)
    sorted_ids, offsets = nm.moe_align(ids, e)
    ref_sorted, ref_offsets = reference.moe_align(to_torch(ids).long(), e)
    assert np.array_equal(np.array(offsets), ref_offsets.numpy())
    valid = int(ref_offsets[-1])
    assert np.array_equal(np.array(sorted_ids[:valid]), ref_sorted[:valid].numpy())


def test_align_drops_out_of_range():
    ids = mx.array([[0, 9], [-1, 2], [2**33, 1]], dtype=mx.int64)
    sorted_ids, offsets = nm.moe_align(ids, 4)
    assert offsets.tolist() == [0, 1, 2, 3, 3]
    assert sorted_ids[:3].tolist() == [0, 5, 3]


def test_combine_and_silu_mul_match_reference():
    y = activations(12, 64)
    assert_bitwise(nm.moe_combine(y, 3), reference.moe_combine(to_torch(y), 3))
    h = activations(5, 2 * 48)
    assert_bitwise(nm.moe_silu_mul(h), reference.moe_silu_mul(to_torch(h)))


@pytest.mark.parametrize("t,e,topk", [(1, 8, 2), (6, 4, 2), (64, 8, 4)])
def test_moe_linear_twin_and_reference(moe_route, t, e, topk):
    n, k = 48, 128
    w = experts(e, n, k)
    mw, twin = nm.encode_moe(w), nm.pack_bf16t_moe(w)
    x = activations(t, k)
    weights, ids = routing(t, e, topk)
    sorted_ids, offsets = nm.moe_align(ids, e)
    y = nm.moe_linear(x, mw, offsets, gather=sorted_ids, gather_div=topk)
    assert_bitwise(y, nm.moe_linear(x, twin, offsets, gather=sorted_ids, gather_div=topk))
    ref = reference.moe_linear(to_torch(x), to_torch(w), to_torch(offsets), to_torch(sorted_ids).long(), topk)
    torch.testing.assert_close(to_torch(y).float(), ref.float(), rtol=2e-2, atol=2e-2)
    # Scatter back to assignment order with router weights; zero_fill clears dropped rows.
    scale = weights.reshape(-1)
    ys = nm.moe_linear(x, mw, offsets, gather=sorted_ids, gather_div=topk, scatter=sorted_ids, row_scale=scale, zero_fill=True)
    assert_bitwise(ys, nm.moe_linear(x, twin, offsets, gather=sorted_ids, gather_div=topk, scatter=sorted_ids,
                                     row_scale=scale, zero_fill=True))


def test_moe_linear_zero_fill_untouched_rows():
    w = experts(2, 32, 64)
    mw = nm.encode_moe(w)
    x = activations(4, 64)
    offsets = mx.array([0, 1, 2], dtype=mx.int64)  # rows 2 and 3 belong to no expert
    y = nm.moe_linear(x, mw, offsets, zero_fill=True)
    assert_bitwise(y[2:], mx.zeros((2, 32), dtype=mx.bfloat16))
    assert_bitwise(y[:1], nm.linear(x[:1], nm.encode(w[0])))


def moe_layer(e: int, k: int, inter: int, seed: int = 0, escapes: bool = True):
    return experts(e, 2 * inter, k, seed, escapes), experts(e, k, inter, seed + 1, escapes)


@pytest.mark.parametrize("t,e,topk", [(1, 8, 2), (3, 16, 4), (40, 8, 2)])
def test_fused_moe(moe_route, t, e, topk):
    k, inter = 128, 64
    w13, w2 = moe_layer(e, k, inter)
    x = activations(t, k)
    weights, ids = routing(t, e, topk)
    y = nm.fused_moe(x, nm.encode_moe(w13), nm.encode_moe(w2), weights, ids)
    assert_bitwise(y, nm.fused_moe(x, nm.pack_bf16t_moe(w13), nm.pack_bf16t_moe(w2), weights, ids))
    ref = reference.fused_moe(to_torch(x), to_torch(w13), to_torch(w2), to_torch(weights), to_torch(ids).long())
    torch.testing.assert_close(to_torch(y).float(), ref.float(), rtol=3e-2, atol=3e-2)


@requires_mps
@pytest.mark.parametrize("t", [1, 5, 64])
def test_fused_moe_parity_with_mps(moe_route, t):
    e, k, inter, topk = 8, 128, 64, 2
    w13, w2 = moe_layer(e, k, inter, seed=4)
    x = activations(t, k, seed=9)
    weights, ids = routing(t, e, topk, seed=3)
    y = nm.fused_moe(x, nm.encode_moe(w13), nm.encode_moe(w2), weights, ids)
    expected = nf12.fused_moe(to_torch(x, "mps"), nf12.encode_moe(to_torch(w13, "mps")),
                              nf12.encode_moe(to_torch(w2, "mps")), to_torch(weights, "mps"), to_torch(ids, "mps"))
    assert_bitwise(y, expected.cpu())


def test_fused_moe_expert_map():
    e_global, e_local, k, inter, t, topk = 8, 4, 64, 64, 6, 2
    # Finite experts: the reference routes dropped assignments to expert 0 with weight 0.
    w13, w2 = moe_layer(e_local, k, inter, escapes=False)
    x = activations(t, k)
    weights, ids = routing(t, e_global, topk)
    expert_map = mx.array([0, 1, 2, 3, -1, -1, -1, -1], dtype=mx.int32)
    y = nm.fused_moe(x, nm.encode_moe(w13), nm.encode_moe(w2), weights, ids, expert_map=expert_map)
    y_ref = nm.fused_moe(x, nm.encode_moe(w13), nm.encode_moe(w2), mx.where(ids < e_local, weights, 0.0),
                         mx.where(ids < e_local, ids, 0))
    torch.testing.assert_close(to_torch(y).float(), to_torch(y_ref).float(), rtol=2e-2, atol=2e-2)
    if MPS:
        expected = nf12.fused_moe(to_torch(x, "mps"), nf12.encode_moe(to_torch(w13, "mps")),
                                  nf12.encode_moe(to_torch(w2, "mps")), to_torch(weights, "mps"),
                                  to_torch(ids, "mps"), expert_map=to_torch(expert_map, "mps"))
        assert_bitwise(y, expected.cpu())


@pytest.mark.parametrize("pack", ["nf12", "bf16t"])
@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("t", [1, 2, 40])
def test_fused_moe_dropped_assignment_contributes_zero(moe_route, pack, bad_scale, t):
    """An id of -1 drops its assignment with or without an expert_map: it adds
    zero and its router weight (even NaN/Inf) is never applied. Bit-exact checks
    compare calls with the same assignment count, so both run the same kernels."""
    e, k, inter = 4, 256, 128
    w13, w2 = moe_layer(e, k, inter, seed=72, escapes=False)
    p = nm.encode_moe if pack == "nf12" else nm.pack_bf16t_moe
    a, b = p(w13), p(w2)
    x = activations(t, k, seed=74)
    weights, ids = routing(t, e, 1, seed=75)
    dropped_ids = mx.concatenate([ids, mx.full((t, 1), -1, dtype=mx.int32)], 1)

    def scaled(s):
        return mx.concatenate([weights, mx.full((t, 1), s, dtype=mx.float32)], 1)

    out = nm.fused_moe(x, a, b, scaled(bad_scale), dropped_ids)
    assert bool(mx.isfinite(out).all())
    assert_bitwise(out, nm.fused_moe(x, a, b, scaled(1.0), dropped_ids))  # the scale is never applied
    alone = nm.fused_moe(x, a, b, weights, ids)  # fewer assignments: other kernels, so only close
    torch.testing.assert_close(to_torch(out).float(), to_torch(alone).float(), rtol=1e-2, atol=1e-2)
    all_dropped = mx.full((t, 2), -1, dtype=mx.int32)
    assert_bitwise(nm.fused_moe(x, a, b, scaled(bad_scale), all_dropped), mx.zeros((t, k), dtype=mx.bfloat16))
    identity = mx.arange(e, dtype=mx.int32)
    assert_bitwise(nm.fused_moe(x, a, b, scaled(bad_scale), dropped_ids, expert_map=identity), out)


# --- modules, top-level routing, packaging -------------------------------------

@pytest.mark.parametrize("bias", [False, True])
def test_nf12_linear_module(bias):
    mx.random.seed(0)  # module initialization draws from MLX's global generator
    lin = mnn.Linear(128, 48, bias=bias)
    lin.set_dtype(mx.bfloat16)
    layer = nm.NF12Linear.from_linear(lin)
    x = activations(3, 128)
    assert_bitwise(layer(x), nm.linear(x, nm.encode(lin.weight), lin.get("bias")))
    import mlx.utils

    assert not mlx.utils.tree_flatten(layer.trainable_parameters())
    flat = dict(mlx.utils.tree_flatten(layer.parameters()))
    assert {"weight.nf12_packed", "weight.nf12_escapes", "weight.nf12_exp_base"} <= set(flat)


def test_nf12_embedding_module_and_tied_head():
    mx.random.seed(1)
    emb = mnn.Embedding(64, 128)
    emb.set_dtype(mx.bfloat16)
    layer = nm.NF12Embedding.from_embedding(emb)
    ids = mx.array([3, 0, 63], dtype=mx.int32)
    assert_bitwise(layer(ids), emb.weight[ids])
    x = activations(2, 128)
    assert_bitwise(layer.as_linear(x), nm.linear(x, nm.encode(emb.weight)))


def test_fresh_module_is_zero():
    layer = nm.NF12Linear(64, 32, bias=False)
    assert_bitwise(layer(activations(2, 64)), mx.zeros((2, 32), dtype=mx.bfloat16))


def test_state_dict_safetensors_round_trip(tmp_path):
    nw = nm.encode(matrix(32, 128))
    path = str(tmp_path / "w.safetensors")
    mx.save_safetensors(path, nw.state_dict("w."))
    back = nm.NF12Weight.from_state_dict(mx.load(path), "w.")
    assert back.exp_base == nw.exp_base
    assert_bitwise(back.decode(), nw.decode())


def test_top_level_functions_route_mlx_arrays():
    w = matrix(32, 128)
    nw = nf12.encode(w)
    assert isinstance(nw, nm.NF12Weight)
    x = activations(2, 128)
    assert_bitwise(nf12.linear(x, nw), nm.linear(x, nw))
    assert_bitwise(nf12.decode(nw), w)
    mw = nf12.encode_moe(experts(2, 32, 64))
    assert isinstance(mw, nm.NF12MoEWeight)


def test_weights_move_between_frameworks():
    w = matrix(48, 192)
    packed, escapes, exp_base = reference.encode(to_torch(w))
    nw = nm.NF12Weight(to_mlx(packed), to_mlx(escapes), exp_base)
    assert_bitwise(nw.decode(), w)


def test_import_does_not_need_torch():
    code = "import sys, nf12.mlx; assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    subprocess.run([sys.executable, "-c", code], check=True)


# --- mx.compile, weight replacement, checkpoints --------------------------------

def test_compile_with_captured_state():
    layer = nm.NF12Linear(64, 16, bias=False)
    compiled = mx.compile(lambda x: layer(x), inputs=layer.state)
    assert_bitwise(compiled(mx.ones((1, 64), dtype=mx.bfloat16)), mx.zeros((1, 16), dtype=mx.bfloat16))


def same_shape_weights(n: int, k: int) -> tuple[nm.NF12Weight, nm.NF12Weight]:
    """Two encodings with equal array shapes (no escapes) but different exp_base."""
    w = randn(n, k, seed=21, scale=0.02)
    a, b = nm.encode(to_mlx(w)), nm.encode(to_mlx(w * 16))
    assert a.num_escapes == b.num_escapes == 0 and a.exp_base != b.exp_base
    return a, b


@pytest.mark.parametrize("route", ["small", "mma", "gemm", "decode"])
def test_compile_follows_weight_replacement(monkeypatch, route):
    """exp_base is graph data: a trace reused after update() (same shapes, so no
    retrace) must use the new base, not one baked in at trace time."""
    monkeypatch.setenv("NF12_MPS_LINEAR", route)
    a, b = same_shape_weights(48, 128)
    layer = nm.NF12Linear.from_weight(a)
    f = mx.compile(lambda x: layer(x), inputs=layer.state)
    x = activations(3, 128)
    assert_bitwise(f(x), nm.linear(x, a))
    layer.update({"weight": b.state_dict()})
    assert layer.nf12_weight.exp_base == b.exp_base
    assert_bitwise(f(x), nm.linear(x, b))

    emb = nm.NF12Embedding.from_weight(a)
    g = mx.compile(lambda i: emb(i), inputs=emb.state)
    ids = mx.array([0, 7, 47], dtype=mx.int32)
    assert_bitwise(g(ids), nm.embedding(ids, a))
    emb.update({"weight": b.state_dict()})
    assert_bitwise(g(ids), nm.embedding(ids, b))


def test_exp_base_as_array():
    nw = nm.encode(matrix(32, 128))
    base = mx.array([nw.exp_base], dtype=mx.uint8)
    x = activations(2, 128)
    ext = nm._load()
    assert_bitwise(ext.linear(x, nw.packed, nw.escapes, base), nm.linear(x, nw))
    assert_bitwise(ext.decode(nw.packed, nw.escapes, base), nw.decode())
    ids = mx.array([3, 0], dtype=mx.int32)
    assert_bitwise(ext.embedding(ids, nw.packed, nw.escapes, base), nm.embedding(ids, nw))
    with pytest.raises(ValueError, match="uint8 with one element"):
        ext.linear(x, nw.packed, nw.escapes, mx.array([1, 2], dtype=mx.uint8))


def test_load_weights_nf12_checkpoint(tmp_path):
    import mlx.utils

    mx.random.seed(2)
    lin = mnn.Linear(128, 48)
    lin.set_dtype(mx.bfloat16)
    src = nm.NF12Linear.from_linear(lin)
    path = str(tmp_path / "layer.safetensors")
    mx.save_safetensors(path, dict(mlx.utils.tree_flatten(src.parameters())))
    fresh = nm.NF12Linear(128, 48)  # different escape count: shapes differ from the checkpoint
    fresh.load_weights(path)
    x = activations(4, 128)
    assert_bitwise(fresh(x), src(x))
    assert fresh.nf12_weight.exp_base == src.nf12_weight.exp_base
    with pytest.raises(ValueError, match="not in the model"):
        nm.NF12Linear(128, 48, bias=False).load_weights(path)
    with pytest.raises(ValueError, match="this module's weight is"):  # wrong shape for this module
        nm.NF12Linear(64, 48).load_weights(path)
