# SPDX-License-Identifier: Apache-2.0
"""Torch-level tests: torch.ops.nf12, the Python API and NF12Linear."""

from __future__ import annotations

import ctypes
import functools

import nf12
import pytest
import torch
from _helpers import assert_bitwise
from nf12 import reference
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU required")

DEV = "cuda"






def weight(n: int, k: int, seed: int = 0, scale: float = 0.02) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(n, k, device=DEV, generator=g) * scale).to(torch.bfloat16)


def activations(*shape: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(*shape, device=DEV, generator=g).to(torch.bfloat16)


def special_weight(n: int, k: int) -> torch.Tensor:
    """Random bit patterns plus every class the format must preserve."""
    g = torch.Generator(device=DEV).manual_seed(7)
    w = weight(n, k, seed=3)
    raw = torch.randint(-32768, 32768, (n, k), device=DEV, generator=g, dtype=torch.int32).to(torch.int16)
    w = torch.where(torch.rand(n, k, device=DEV, generator=g) < 0.05, raw.view(torch.bfloat16), w)
    flat = w.view(-1).view(torch.int16)
    specials = torch.tensor(
        [0x0000, -0x8000, 0x0001, -0x7FFF, 0x007F, 0x7F80, -0x0080, 0x7FC0, 0x7F81, -0x0001, 0x7F7F, 0x0080],
        dtype=torch.int16,
        device=DEV,
    )
    flat[: specials.numel()] = specials
    flat[-specials.numel() :] = specials
    return w


@functools.cache
def _route_query():
    f = ctypes.CDLL(str(nf12.library_path())).nf12_linear_decode_gemm_rows
    f.restype = ctypes.c_int64
    f.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
    return f


def torch_op_routes(m: int, n: int, k: int) -> bool:
    """Whether nf12.linear takes the decode + cuBLAS route at (m, n, k) on this device. Its output matches
    the same cuBLAS calls on decode(W) (tests/api/test_decode_route.py), not the bf16t twin (SPEC.md 6.5)."""
    return _route_query()(m, n, k, torch.cuda.current_device()) > 0


def assert_close_to_reference(y: torch.Tensor, x: torch.Tensor, w: torch.Tensor, bias=None) -> None:
    ref = reference.linear(x.reshape(-1, x.shape[-1]), w, bias).reshape(y.shape)
    torch.testing.assert_close(y.float(), ref.float(), rtol=2e-2, atol=2e-2)


SHAPES = [(16, 64), (48, 128), (256, 512), (1024, 4096), (4096, 1024)]
MS = [1, 2, 7, 8, 16, 33, 64, 129, 256]




@pytest.mark.parametrize("n,k", SHAPES)
def test_round_trip_bitwise(n, k):
    w = special_weight(n, k)
    nw = nf12.encode(w)
    assert nw.shape == (n, k)
    assert nw.packed.shape == (n, k * 3 // 2) and nw.packed.dtype == torch.uint8
    assert nw.escapes.dtype == torch.int16 and nw.escapes.shape[1] == 8
    assert_bitwise(nf12.decode(nw), w)
    assert_bitwise(nw.decode(), w)


@pytest.mark.parametrize("n,k", [(16, 64), (256, 512)])
def test_encode_matches_reference_bytes(n, k):
    w = special_weight(n, k)
    nw = nf12.encode(w)
    packed, escapes, exp_base = reference.encode(w)
    assert nw.exp_base == exp_base
    assert torch.equal(nw.packed, packed)
    assert torch.equal(nw.escapes, escapes)


def test_encode_deterministic_and_strided_input():
    big = special_weight(256, 1024)
    w = big[:, 256:768]  # row stride 1024, 16-byte aligned base
    a, b = nf12.encode(w), nf12.encode(w.contiguous())
    assert a.exp_base == b.exp_base
    assert torch.equal(a.packed, b.packed) and torch.equal(a.escapes, b.escapes)
    assert_bitwise(a.decode(), w)
    # Transposed and misaligned views are copied before encoding.
    t = special_weight(512, 256).t()
    assert_bitwise(nf12.encode(t).decode(), t)
    m = big[:, 1:513]
    assert_bitwise(nf12.encode(m).decode(), m)


def test_all_escaped_matrix():
    # +0.0 groups hit the M == 0 sentinel, so an all-zero matrix escapes everything.
    w = torch.zeros(32, 128, dtype=torch.bfloat16, device=DEV)
    nw = nf12.encode(w)
    assert nw.num_escapes == 32 * 128 // 8
    assert_bitwise(nw.decode(), w)
    x = activations(5, 128)
    assert torch.equal(nf12.linear(x, nw), torch.zeros(5, 32, dtype=torch.bfloat16, device=DEV))


def test_pack_bf16t_matches_reference():
    w = special_weight(64, 256)
    assert torch.equal(nf12.pack_bf16t(w), reference.pack_bf16t(w))




@pytest.mark.parametrize("m,n,k", [
    (m, n, k) for n, k in SHAPES for m in MS
] + [
    pytest.param(m, 1024, 4096, id=f"route-boundary-m{m}") for m in (4, 15, 17, 128, 512)
] + [
    pytest.param(m, 4096, 4096, id=f"square-m{m}") for m in (16, 32, 64, 128, 256)
])
def test_linear_matches_reference_and_bf16t(m, n, k):
    w = weight(n, k)
    nw = nf12.encode(w)
    wt = nf12.pack_bf16t(w)
    x = activations(m, k)
    y = nf12.linear(x, nw)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    assert_close_to_reference(y, x, w)
    if not torch_op_routes(m, n, k):
        assert_bitwise(y, nf12.linear_bf16t(x, wt))
    assert_bitwise(y, nf12.linear(x, nw))  # deterministic


@pytest.mark.parametrize("m", [1, 9, 200])
def test_linear_bias_and_escapes(m):
    n, k = 128, 512
    w = special_weight(n, k)
    w = torch.where(w.abs() < 1e3, w, torch.zeros_like(w))  # drop NaN/Inf/huge; tiny values still escape
    nw = nf12.encode(w)
    assert nw.num_escapes > 0
    bias = weight(n, 64, seed=5)[:, 0].contiguous()
    x = activations(m, k)
    y = nf12.linear(x, nw, bias)
    if not torch_op_routes(m, n, k):
        assert_bitwise(y, nf12.linear_bf16t(x, nf12.pack_bf16t(w), bias))
    assert_close_to_reference(y, x, w, bias)


def test_linear_leading_dims_and_empty(dense_ops):
    encode, linear = dense_ops
    n, k = 256, 512
    w = weight(n, k)
    nw = encode(w)
    x = activations(2, 3, 5, k)
    y = linear(x, nw)
    assert y.shape == (2, 3, 5, n)
    assert_bitwise(y, linear(x.reshape(-1, k), nw).reshape(2, 3, 5, n))
    x1 = activations(k)
    assert linear(x1, nw).shape == (n,)
    assert_bitwise(linear(x1, nw), linear(x1[None], nw)[0])
    empty = torch.empty(0, 4, k, dtype=torch.bfloat16, device=DEV)
    assert linear(empty, nw).shape == (0, 4, n)


@pytest.mark.parametrize("m", [2, 16, 24, 64])
def test_linear_non_contiguous_inputs(dense_ops, m):
    encode, linear = dense_ops
    n, k = 256, 512
    w = weight(n, k)
    nw = encode(w)
    base = activations(m, 2 * k + 8)
    views = {
        "row_stride": base[:, :k],  # aligned rows, ld = 2k + 8
        "misaligned": base[:, 1 : k + 1],  # base not 16-byte aligned
        "odd_ld": activations(m, k + 3)[:, :k],  # ld % 8 != 0
        "transposed": activations(k, m).t(),
        "step": base[::2, :k],
        "batched_strided": base[:, :k].reshape(2, m // 2, k).transpose(0, 1),
    }
    for x in views.values():
        assert_bitwise(linear(x, nw), linear(x.contiguous(), nw))
    bias = activations(2 * n)[::2]
    x = base[:, :k]
    assert_bitwise(linear(x, nw, bias), linear(x, nw, bias.contiguous()))




def test_errors():
    with pytest.raises(ValueError, match="multiple of 16"):
        nf12.encode(weight(24, 64))
    with pytest.raises(ValueError, match="multiple of 64"):
        nf12.encode(weight(16, 96))
    with pytest.raises(ValueError, match="bfloat16"):
        nf12.encode(weight(16, 64).float())
    with pytest.raises(ValueError, match="CUDA"):
        nf12.encode(weight(16, 64).cpu())
    nw = nf12.encode(weight(16, 64))
    with pytest.raises(ValueError, match=r"\[\.\.\., 64\]"):
        nf12.linear(activations(2, 128), nw)
    with pytest.raises(ValueError, match="bfloat16"):
        nf12.linear(activations(2, 64).half(), nw)
    with pytest.raises(ValueError, match="CUDA"):
        nf12.linear(activations(2, 64).cpu(), nw)
    with pytest.raises(RuntimeError, match="bias"):
        nf12.linear(activations(2, 64), nw, torch.zeros(8, dtype=torch.bfloat16, device=DEV))
    with pytest.raises(ValueError, match="exp_base"):
        nf12.NF12Weight(nw.packed, nw.escapes, 121)
    with pytest.raises(RuntimeError, match="exp_base"):
        torch.ops.nf12.linear(activations(2, 64), nw.packed, nw.escapes, 121, None)
    with pytest.raises(RuntimeError, match="multiple of 16"):
        torch.ops.nf12.decode(nw.packed[:8], nw.escapes, nw.exp_base)
    with pytest.raises(ValueError, match="multiple of 16"):
        nf12.NF12Linear(64, 24)


def test_opcheck():
    w = weight(64, 256)
    nw = nf12.encode(w)
    x = activations(3, 256)
    bias = activations(64)
    torch.library.opcheck(torch.ops.nf12.linear, (x, nw.packed, nw.escapes, nw.exp_base, bias))
    torch.library.opcheck(torch.ops.nf12.linear, (x, nw.packed, nw.escapes, nw.exp_base, None))
    torch.library.opcheck(torch.ops.nf12.decode, (nw.packed, nw.escapes, nw.exp_base))
    torch.library.opcheck(torch.ops.nf12.pack_bf16t, (w,))
    torch.library.opcheck(torch.ops.nf12.linear_bf16t, (x, nf12.pack_bf16t(w), bias))




def make_linear(k: int, n: int, bias: bool = True, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(k, n, bias=bias, device=DEV, dtype=torch.bfloat16)


@pytest.mark.parametrize("bias", [True, False])
def test_nf12_linear_module(bias):
    lin = make_linear(512, 256, bias)
    layer = nf12.NF12Linear.from_linear(lin)
    assert_bitwise(layer.weight.decode(), lin.weight.detach())
    x = activations(4, 9, 512)
    with torch.no_grad():
        y = layer(x)
        torch.testing.assert_close(y.float(), lin(x).float(), rtol=2e-2, atol=2e-2)
    assert_bitwise(y, nf12.linear(x, layer.weight, lin.bias.detach() if bias else None))
    assert "exp_base" in repr(layer)


def test_nf12_linear_fresh_layer_is_zero():
    layer = nf12.NF12Linear(128, 32, device=DEV)
    assert layer.weight.num_escapes == 0
    decoded = layer.weight.decode()
    assert torch.equal(decoded, torch.zeros_like(decoded))
    assert torch.equal(layer(activations(3, 128)), torch.zeros(3, 32, dtype=torch.bfloat16, device=DEV))


def test_weight_state_dict_round_trip():
    nw = nf12.encode(special_weight(64, 256))
    sd = nw.state_dict("W.")
    assert set(sd) == {"W.nf12_packed", "W.nf12_escapes", "W.nf12_exp_base"}
    assert sd["W.nf12_exp_base"].dtype == torch.uint8 and sd["W.nf12_exp_base"].shape == (1,)
    back = nf12.NF12Weight.from_state_dict(sd, "W.")
    assert back.exp_base == nw.exp_base and back.shape == nw.shape
    assert_bitwise(back.decode(), nw.decode())
    cpu = nw.to("cpu")
    assert cpu.device.type == "cpu" and cpu.nbytes == nw.nbytes == 64 * 256 * 3 // 2 + nw.escapes.nbytes
    assert_bitwise(cpu.to(DEV).decode(), nw.decode())


def test_module_state_dict_safetensors(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    model = nn.Sequential(nf12.NF12Linear.from_linear(make_linear(256, 128)), nn.GELU(),
                          nf12.NF12Linear.from_linear(make_linear(128, 64, bias=False, seed=1)))
    sd = model.state_dict()
    assert set(sd) == {
        "0.weight.nf12_packed", "0.weight.nf12_escapes", "0.weight.nf12_exp_base", "0.bias",
        "2.weight.nf12_packed", "2.weight.nf12_escapes", "2.weight.nf12_exp_base",
    }
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    loaded = safetensors_torch.load_file(str(path))

    fresh = nn.Sequential(nf12.NF12Linear(256, 128, device=DEV), nn.GELU(),
                          nf12.NF12Linear(128, 64, bias=False, device=DEV))
    fresh.load_state_dict(loaded)
    assert fresh[0].nf12_packed.device.type == "cuda"
    x = activations(5, 256)
    with torch.no_grad():
        assert_bitwise(fresh(x), model(x))

    meta = nn.Sequential(nf12.NF12Linear(256, 128, device="meta"), nn.GELU(),
                         nf12.NF12Linear(128, 64, bias=False, device="meta"))
    meta.load_state_dict({k: v.to(DEV) for k, v in loaded.items()})
    with torch.no_grad():
        assert_bitwise(meta(x), model(x))

    with pytest.raises(RuntimeError, match="Missing key"):
        fresh.load_state_dict({k: v for k, v in loaded.items() if k != "0.bias"})
    with pytest.raises(RuntimeError, match="Unexpected key"):
        fresh.load_state_dict({**loaded, "0.weight": torch.zeros(1)})
    with pytest.raises(RuntimeError, match="does not match"):
        nf12.NF12Linear(512, 128, device=DEV).load_state_dict(
            {k[2:]: v for k, v in loaded.items() if k.startswith("0.")})




@pytest.mark.parametrize("m,n,k", [
    (1, 1024, 2048), (8, 1024, 2048), (96, 1024, 2048),
    (1024, 64, 2048), (1025, 64, 2048),
    (512, 128, 2048), (513, 128, 2048),
    (192, 128, 4096), (193, 128, 4096),
    # 15/17/65: the row bands either side of the 16-row matrix-core tile. The
    # 15 case guards a real overread: a 16-row A tile formed from a shorter
    # activation reads past the tensor and only faults when it crosses into
    # unmapped memory (it did, on gfx942, after an unrelated shape ran first).
    (15, 16, 64), (17, 16, 64), (65, 16, 64), (15, 4096, 4096), (65, 4096, 4096),
])
def test_cuda_graph_capture_replay(m, n, k):
    nw = nf12.encode(weight(n, k))
    bias = activations(n)
    x = activations(m, k)
    nf12.linear(x, nw, bias)  # warm up outside capture

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        nf12.linear(x, nw, bias)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = nf12.linear(x, nw, bias)
    for seed in (2, 3):
        x.copy_(activations(m, k, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(y, nf12.linear(x, nw, bias))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nf12.NF12Linear.from_linear(make_linear(256, 512))
        self.down = nf12.NF12Linear.from_linear(make_linear(512, 256, bias=False, seed=1))

    def forward(self, x):
        return x + self.down(torch.nn.functional.silu(self.up(x)))


@pytest.mark.parametrize("dynamic", [False, True])
def test_torch_compile_fullgraph(dynamic):
    torch._dynamo.reset()
    block = _Block()
    compiled = torch.compile(block, fullgraph=True, dynamic=dynamic)
    with torch.no_grad():
        for m in (1, 8, 33):
            x = activations(2, m, 256, seed=m)
            torch.testing.assert_close(compiled(x), block(x), rtol=1e-2, atol=1e-2)
            # The NF12 ops themselves are opaque to inductor, so their outputs are exact.
            assert_bitwise(torch.compile(block.up, fullgraph=True, dynamic=dynamic)(x), block.up(x))


def test_torch_compile_functional():
    torch._dynamo.reset()
    nw = nf12.encode(weight(128, 256))

    @torch.compile(fullgraph=True)
    def f(x):
        return nf12.linear(x, nw) * 2

    x = activations(4, 256)
    assert_bitwise(f(x), nf12.linear(x, nw) * 2)


@pytest.mark.parametrize("magnitude,bias_value", [(2.**25, 1.), (1., 2.**-25), (2.**127, 2.**127)])
def test_split_k_bias_follows_complete_reduction(dense_ops, magnitude, bias_value):
    m, n, k = 16, 16, 512
    w = torch.zeros((n, k), device="cuda", dtype=torch.bfloat16)
    w[:, 0], w[:, 64] = magnitude, -magnitude
    x = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
    bias = torch.full((n,), bias_value, device="cuda", dtype=torch.bfloat16)
    # The two nonzero products cancel exactly. Bias must survive unchanged,
    # including when adding it to an unreduced partial would overflow FP32.
    expected = bias.expand(m, n).contiguous()
    encode, linear = dense_ops
    out = linear(x, encode(w), bias)
    assert bool(out.isfinite().all())
    assert_bitwise(out, expected)
