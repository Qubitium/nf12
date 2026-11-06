# SPDX-License-Identifier: Apache-2.0
"""Embedding lookup (SPEC.md section 10): torch op, Python API, NF12Embedding and
tied weights, CUDA graphs, the C API, real embedding tables, and (with
NF12_HUGE_TESTS=1) 64-bit table and output offsets."""

from __future__ import annotations

import ctypes
import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import nf12
from nf12 import reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def assert_bitwise(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(bits(a), bits(b))


def table(v: int, h: int, seed: int = 0, scale: float = 0.02) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(v, h, device=DEV, generator=g) * scale).to(torch.bfloat16)


def special_table(v: int, h: int, seed: int = 0) -> torch.Tensor:
    """Normal rows, 5% random bit patterns, every special value, and zero (padding) rows."""
    g = torch.Generator(device=DEV).manual_seed(seed + 7)
    w = table(v, h, seed)
    raw = torch.randint(-32768, 32768, (v, h), device=DEV, generator=g, dtype=torch.int32).to(torch.int16)
    w = torch.where(torch.rand(v, h, device=DEV, generator=g) < 0.05, raw.view(torch.bfloat16), w)
    flat = w.view(-1).view(torch.int16)
    specials = torch.tensor(
        [0x0000, -0x8000, 0x0001, -0x7FFF, 0x007F, 0x7F80, -0x0080, 0x7FC0, 0x7F81, -0x0001, 0x7F7F, 0x0080],
        dtype=torch.int16, device=DEV)
    flat[: specials.numel()] = specials
    flat[-specials.numel():] = specials
    w[1::5] = 0  # padding rows: +0.0 groups are escaped at any base
    return w


def random_ids(n: int, v: int, dtype: torch.dtype, seed: int = 0, margin: int = 3) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randint(-margin, v + margin, (n,), device=DEV, generator=g).to(dtype)


SHAPES = [(16, 64), (48, 128), (32, 192), (160, 1024), (64, 3584), (48, 4096), (16, 8192), (32, 4160)]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("v,h", SHAPES)
def test_matches_reference(v, h, dtype):
    w = special_table(v, h)
    nw = nf12.encode(w)
    assert nw.num_escapes > 0
    for n in (1, 7, 64, 1000):
        ids = random_ids(n, v, dtype, seed=n)
        assert_bitwise(nf12.embedding(ids, nw), reference.embedding(ids, w))


def test_every_bit_pattern():
    """Each of the 65536 BF16 patterns once, in a table that is ~all escapes."""
    v, h = 64, 1024
    g = torch.Generator().manual_seed(3)
    w = torch.arange(-32768, 32768, dtype=torch.int32)[torch.randperm(65536, generator=g)].to(torch.int16)
    w = w.view(v, h).view(torch.bfloat16).to(DEV)
    nw = nf12.encode(w)
    assert nw.num_escapes > v * h // 8 // 2
    ids = torch.randperm(v, generator=g).to(DEV).repeat(3)  # every row, each three times
    assert_bitwise(nf12.embedding(ids, nw), w[ids])


@pytest.mark.parametrize("fill", ["zeros", "neg_zeros", "all_escaped_random"])
def test_heavy_escape_tables(fill):
    v, h = 256, 2048
    if fill == "zeros":
        w = torch.zeros(v, h, dtype=torch.bfloat16, device=DEV)
    elif fill == "neg_zeros":  # -0.0 everywhere encodes without escapes at base 0
        w = torch.full((v, h), -0.0, dtype=torch.bfloat16, device=DEV)
    else:
        raw = torch.randint(-32768, 32768, (v, h), device=DEV, dtype=torch.int32).to(torch.int16)
        w = raw.view(torch.bfloat16)
    nw = nf12.encode(w)
    expected = {"zeros": v * h // 8, "neg_zeros": 0}.get(fill)
    if expected is not None:
        assert nw.num_escapes == expected
    ids = random_ids(4096, v, torch.int64)
    assert_bitwise(nf12.embedding(ids, nw), reference.embedding(ids, w))


def test_out_of_range_ids():
    v, h = 64, 256
    w = special_table(v, h)
    nw = nf12.encode(w)
    i32 = torch.tensor([INT32_MIN, -1, v, v + 15, INT32_MAX, 0, v - 1, 5], dtype=torch.int32, device=DEV)
    # 2**32 + 1 would read row 1 if the id were truncated to 32 bits.
    i64 = torch.tensor([INT64_MIN, -1, v, 2**31, 2**32, 2**32 + 1, INT64_MAX, 0, v - 1, 1], dtype=torch.int64,
                       device=DEV)
    for ids in (i32, i64):
        out = nf12.embedding(ids, nw)
        assert_bitwise(out, reference.embedding(ids, w))
        valid = (ids >= 0) & (ids < v)
        assert torch.equal(bits(out[~valid]), torch.zeros_like(bits(out[~valid])))  # +0.0, not -0.0
        assert_bitwise(out[valid], w[ids[valid].long()])


def test_vocab_parallel_shards():
    """Shift ids into each rank's rows [r*V/R, (r+1)*V/R). Summing ranks gives the full lookup."""
    v, h, ranks = 256, 512, 4
    w = table(v, h)
    shards = [nf12.encode(s.contiguous()) for s in w.chunk(ranks)]
    ids = random_ids(300, v, torch.int64, margin=0)
    part = v // ranks
    total = sum(nf12.embedding(ids - r * part, s).float() for r, s in enumerate(shards))
    assert_bitwise(total.to(torch.bfloat16), w[ids])


def test_ids_shapes_and_layouts():
    v, h = 48, 128
    w = table(v, h)
    nw = nf12.encode(w)
    for shape in [(), (5,), (3, 5), (2, 3, 4), (0,), (2, 0)]:
        ids = random_ids(max(1, torch.Size(shape).numel()), v, torch.int64)[: torch.Size(shape).numel()].view(shape)
        out = nf12.embedding(ids, nw)
        assert out.shape == (*shape, h) and out.dtype == torch.bfloat16 and out.is_contiguous()
        assert_bitwise(out, reference.embedding(ids, w))
    base = random_ids(64, v, torch.int32).view(8, 8)
    for ids in (base[:, ::2], base.t(), base[2:5, 1:]):  # non-contiguous
        assert_bitwise(nf12.embedding(ids, nw), reference.embedding(ids, w))
    duplicates = torch.tensor([3, 3, 3, 40, 3, 40], device=DEV)
    assert_bitwise(nf12.embedding(duplicates, nw), w[duplicates])


def test_errors():
    nw = nf12.encode(table(16, 64))
    ids = torch.tensor([1, 2], device=DEV)
    with pytest.raises(ValueError, match="int32 or int64"):
        nf12.embedding(ids.float(), nw)
    with pytest.raises(ValueError, match="int32 or int64"):
        nf12.embedding(ids.to(torch.int16), nw)
    with pytest.raises(ValueError, match="CUDA"):
        nf12.embedding(ids.cpu(), nw)
    with pytest.raises(RuntimeError, match="int32 or int64"):
        torch.ops.nf12.embedding(ids.to(torch.uint8), nw.packed, nw.escapes, nw.exp_base)
    with pytest.raises(RuntimeError, match="CUDA"):
        torch.ops.nf12.embedding(ids.cpu(), nw.packed, nw.escapes, nw.exp_base)
    with pytest.raises(RuntimeError, match="exp_base"):
        torch.ops.nf12.embedding(ids, nw.packed, nw.escapes, 121)


def test_opcheck():
    nw = nf12.encode(special_table(32, 256))
    for ids in (random_ids(9, 32, torch.int32), random_ids(12, 32, torch.int64).view(3, 4)):
        torch.library.opcheck(torch.ops.nf12.embedding, (ids, nw.packed, nw.escapes, nw.exp_base))


@pytest.mark.parametrize("n", [1, 64, 5000])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cuda_graph_capture_replay(n, dtype):
    v, h = 512, 1024
    w = special_table(v, h)
    nw = nf12.encode(w)
    ids = random_ids(n, v, dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        nf12.embedding(ids, nw)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = nf12.embedding(ids, nw)
    for seed in (2, 3):
        ids.copy_(random_ids(n, v, dtype, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(out, reference.embedding(ids, w))


def test_torch_compile_fullgraph():
    torch._dynamo.reset()
    w = table(64, 256)
    emb = nf12.NF12Embedding.from_weight(nf12.encode(w))

    @torch.compile(fullgraph=True, dynamic=True)
    def f(ids):
        return emb(ids) * 2

    for n in (1, 5, 33):
        ids = random_ids(n, 64, torch.int64, seed=n).view(1, n)
        assert_bitwise(f(ids), reference.embedding(ids, w) * 2)




def make_embedding(v: int, h: int, seed: int = 0) -> nn.Embedding:
    torch.manual_seed(seed)
    return nn.Embedding(v, h, device=DEV, dtype=torch.bfloat16)


def test_nf12_embedding_module():
    emb = make_embedding(256, 512)
    layer = nf12.NF12Embedding.from_embedding(emb)
    assert (layer.num_embeddings, layer.embedding_dim) == (256, 512)
    assert_bitwise(layer.weight.decode(), emb.weight.detach())
    ids = random_ids(40, 256, torch.int64, margin=0).view(4, 10)
    with torch.no_grad():
        assert_bitwise(layer(ids), emb(ids))
        assert_bitwise(layer(ids.int()), emb(ids))
    assert "exp_base" in repr(layer) and "256, 512" in repr(layer)

    fresh = nf12.NF12Embedding(64, 128, device=DEV)
    assert fresh.weight.num_escapes == 0
    assert torch.equal(fresh(torch.tensor([0, 63], device=DEV)), torch.zeros(2, 128, dtype=torch.bfloat16, device=DEV))

    with pytest.raises(ValueError, match="max_norm"):
        nf12.NF12Embedding.from_embedding(nn.Embedding(16, 64, max_norm=1.0, device=DEV, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bfloat16"):
        nf12.NF12Embedding.from_embedding(nn.Embedding(16, 64, device=DEV))
    with pytest.raises(ValueError, match="multiple of 16"):
        nf12.NF12Embedding(24, 64)


def test_nf12_embedding_state_dict(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    layer = nf12.NF12Embedding.from_embedding(make_embedding(128, 256))
    sd = layer.state_dict()
    assert set(sd) == {"weight.nf12_packed", "weight.nf12_escapes", "weight.nf12_exp_base"}
    path = tmp_path / "emb.safetensors"
    safetensors_torch.save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    loaded = safetensors_torch.load_file(str(path))
    ids = random_ids(50, 128, torch.int64)
    for device in (DEV, "meta"):
        fresh = nf12.NF12Embedding(128, 256, device=device)
        fresh.load_state_dict({k: v.to(DEV) for k, v in loaded.items()} if device == "meta" else loaded)
        assert_bitwise(fresh(ids), layer(ids))
    with pytest.raises(RuntimeError, match="does not match"):
        nf12.NF12Embedding(64, 256, device=DEV).load_state_dict(loaded)


class _TiedLM(nn.Module):
    """embed -> lm_head with one shared NF12Weight."""

    def __init__(self, weight: nf12.NF12Weight) -> None:
        super().__init__()
        self.embed = nf12.NF12Embedding.from_weight(weight)
        self.lm_head = nf12.NF12Linear.from_weight(weight)

    def forward(self, ids):
        return self.lm_head(self.embed(ids))


def test_tied_embedding_and_lm_head():
    v, h = 1024, 512
    emb = make_embedding(v, h)
    w = emb.weight.detach()
    model = _TiedLM(nf12.encode(w))
    assert model.embed.nf12_packed.data_ptr() == model.lm_head.nf12_packed.data_ptr()
    assert model.embed.nf12_escapes is model.lm_head.nf12_escapes
    ids = random_ids(37, v, torch.int64, margin=0)
    x = model.embed(ids)
    assert_bitwise(x, F.embedding(ids, w))
    logits = model.lm_head(x)
    torch.testing.assert_close(logits.float(), F.linear(x, w).float(), rtol=2e-2, atol=2e-2)
    assert_bitwise(model(ids), logits)

    # Both modules save the same tensors, and loading them back keeps them shared.
    sd = model.state_dict()
    assert sd["embed.weight.nf12_packed"] is sd["lm_head.weight.nf12_packed"]
    fresh = _TiedLM(nf12.encode(table(v, h, seed=9)))
    fresh.load_state_dict(sd)
    assert fresh.embed.nf12_packed.data_ptr() == fresh.lm_head.nf12_packed.data_ptr()
    assert_bitwise(fresh(ids), model(ids))



OK, INVALID = 0, 1


class Weight(ctypes.Structure):
    _fields_ = [
        ("packed", ctypes.c_void_p),
        ("escapes", ctypes.c_void_p),
        ("num_escapes", ctypes.c_uint32),
        ("n", ctypes.c_int64),
        ("k", ctypes.c_int64),
        ("exp_base", ctypes.c_uint8),
    ]


@pytest.fixture(scope="module")
def lib():
    lib = ctypes.CDLL(str(nf12.library_path()))
    p, i64 = ctypes.c_void_p, ctypes.c_int64
    for name in ("nf12_embedding", "nf12_embedding_i64"):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = ctypes.c_int, [ctypes.POINTER(Weight), p, i64, p, i64, p]
    lib.nf12_last_cuda_error.restype, lib.nf12_last_cuda_error.argtypes = ctypes.c_int, []
    return lib


def c_weight(nw: nf12.NF12Weight) -> Weight:
    v, h = nw.shape
    return Weight(nw.packed.data_ptr(), nw.escapes.data_ptr() if nw.num_escapes else None, nw.num_escapes, v, h,
                  nw.exp_base)


def c_lookup(lib, weight: Weight, ids, num_ids: int, out, ldo: int) -> int:
    fn = lib.nf12_embedding_i64 if ids is not None and ids.dtype == torch.int64 else lib.nf12_embedding
    ids_ptr = ids if isinstance(ids, int) or ids is None else ids.data_ptr()
    out_ptr = out if isinstance(out, int) or out is None else out.data_ptr()
    return fn(ctypes.byref(weight), ids_ptr, num_ids, out_ptr, ldo, torch.cuda.current_stream().cuda_stream)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_c_api_padded_output(lib, dtype):
    """Output rows have stride ldo, with the ldo - H padding columns left untouched."""
    v, h, n, ldo = 96, 448, 300, 448 + 24
    w = special_table(v, h)
    nw = nf12.encode(w)
    ids = random_ids(n, v, dtype)
    sentinel = torch.full((n, ldo), 0x7FC1, dtype=torch.int16, device=DEV)
    out = sentinel.clone()
    assert c_lookup(lib, c_weight(nw), ids, n, out, ldo) == OK
    torch.cuda.synchronize()
    assert torch.equal(out[:, :h], bits(reference.embedding(ids, w)))
    assert torch.equal(out[:, h:], sentinel[:, h:])


def test_c_api_validation(lib):
    v, h = 32, 128
    nw = nf12.encode(special_table(v, h))
    weight = c_weight(nw)
    ids32 = random_ids(8, v, torch.int32)
    ids64 = random_ids(8, v, torch.int64)
    out = torch.zeros(16, h + 16, dtype=torch.bfloat16, device=DEV)
    for ids in (ids32, ids64):
        assert c_lookup(lib, weight, ids, 8, out, h) == OK
        assert c_lookup(lib, weight, ids, 8, out.data_ptr() + 2, h) == INVALID  # misaligned out
        assert c_lookup(lib, weight, ids, 8, out.data_ptr() + 8, h) == INVALID
        assert c_lookup(lib, weight, ids, 8, out, h - 8) == INVALID  # ldo < k
        assert c_lookup(lib, weight, ids, 8, out, h + 4) == INVALID  # ldo % 8
        assert c_lookup(lib, weight, ids, 8, None, h) == INVALID
        assert c_lookup(lib, weight, ids, -1, out, h) == INVALID
        assert c_lookup(lib, weight, ids, 0, out, h) == OK
        assert c_lookup(lib, weight, ids, 2**62, out, h) == INVALID  # i * ldo overflows int64
        fn = lib.nf12_embedding_i64 if ids.dtype == torch.int64 else lib.nf12_embedding
        stream = torch.cuda.current_stream().cuda_stream
        assert fn(ctypes.byref(weight), None, 0, None, h, stream) == OK  # no ids: nothing to do
        assert fn(ctypes.byref(weight), None, 8, out.data_ptr(), h, stream) == INVALID
        assert fn(ctypes.byref(weight), ids.data_ptr() + 2, 8, out.data_ptr(), h, stream) == INVALID
        assert fn(None, ids.data_ptr(), 8, out.data_ptr(), h, stream) == INVALID
        for bad in (Weight(weight.packed, None, weight.num_escapes, v, h, weight.exp_base),  # escapes missing
                    Weight(weight.packed, weight.escapes, weight.num_escapes, 8, h, weight.exp_base),  # N % 16
                    Weight(weight.packed, weight.escapes, weight.num_escapes, v, 96, weight.exp_base),  # K % 64
                    Weight(weight.packed + 8, weight.escapes, weight.num_escapes, v, h, weight.exp_base),
                    Weight(weight.packed, weight.escapes, weight.num_escapes, v, h, 121)):
            assert c_lookup(lib, bad, ids, 8, out, h) == INVALID
    assert lib.nf12_embedding_i64(ctypes.byref(weight), ids32.data_ptr() + 4, 1, out.data_ptr(), h,
                                  torch.cuda.current_stream().cuda_stream) == INVALID  # int64 ids need 8 bytes
    torch.cuda.synchronize()
    assert lib.nf12_last_cuda_error() == 0



REAL_TABLES = [
    ("Qwen/Qwen3-0.6B", 151936, 1024),  # tied with lm_head
    ("unsloth/gemma-2-9b-it", 256000, 3584),
    ("unsloth/Meta-Llama-3.1-8B-Instruct", 128256, 4096),
]


def hf_embedding_table(repo: str) -> torch.Tensor:
    hub = pytest.importorskip("huggingface_hub")
    safetensors = pytest.importorskip("safetensors")
    import json

    name = "model.embed_tokens.weight"
    index = hub.try_to_load_from_cache(repo, "model.safetensors.index.json")
    shard = json.load(open(index))["weight_map"][name] if isinstance(index, str) else "model.safetensors"
    path = hub.try_to_load_from_cache(repo, shard)
    if not isinstance(path, str) or not os.path.isfile(path):
        pytest.skip(f"{repo} {shard} is not in the local Hugging Face cache")
    with safetensors.safe_open(path, framework="pt") as f:
        return f.get_tensor(name).to(DEV)


@pytest.mark.parametrize("repo,v,h", REAL_TABLES)
def test_real_tables_match_bf16(repo, v, h):
    w = hf_embedding_table(repo)
    assert w.shape == (v, h) and w.dtype == torch.bfloat16
    nw = nf12.encode(w)
    print(f"{repo}: exp_base {nw.exp_base}, escaped groups {nw.num_escapes} "
          f"({100 * nw.num_escapes / (v * h // 8):.4f}%)")
    chunk = 32768
    for start in range(0, v, chunk):  # every row once, in a shuffled order
        ids = torch.randperm(min(chunk, v - start), device=DEV) + start
        assert_bitwise(nf12.embedding(ids, nw), w[ids])
    out_of_range = (torch.tensor([-1, v, 0, v - 1, 2**40], device=DEV),
                    torch.tensor([v - 1, -(2**31), v], device=DEV).int())
    for ids in out_of_range:
        assert_bitwise(nf12.embedding(ids, nw), reference.embedding(ids, w))
    if repo == "Qwen/Qwen3-0.6B":
        model = _TiedLM(nw)
        ids = torch.randint(0, v, (7,), device=DEV)
        x = model.embed(ids)
        assert_bitwise(x, w[ids])
        torch.testing.assert_close(model.lm_head(x).float(), F.linear(x, w).float(), rtol=2e-2, atol=2e-2)




@pytest.mark.skipif(os.environ.get("NF12_HUGE_TESTS") != "1", reason="set NF12_HUGE_TESTS=1 (~16 GB of device memory)")
def test_huge_offsets():
    """Packed table beyond 2^32 bytes and an output beyond 2^31 elements."""
    v, h = 360448, 8192
    assert v * h * 3 // 2 > 2**32
    w = torch.empty(v, h, dtype=torch.bfloat16, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(0)
    for r in range(0, v, 16384):
        w[r:r + 16384] = (torch.randn(min(16384, v - r), h, device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    w[-1] = 0  # escapes at the end of the table: high escape indices
    w[-9] = 0
    nw = nf12.encode(w)
    num_ids = 2**31 // h + 7
    for dtype in (torch.int64, torch.int32):
        ids = (v - 1 - torch.arange(num_ids, device=DEV)).to(dtype)  # last rows first
        ids[1000] = v  # out of range
        out = nf12.embedding(ids, nw)
        assert out.numel() > 2**31
        for start in range(0, num_ids, 32768):
            sel = ids[start:start + 32768]
            assert_bitwise(out[start:start + 32768], reference.embedding(sel, w))
        del out
