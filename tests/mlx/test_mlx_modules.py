# SPDX-License-Identifier: Apache-2.0
"""Module-state conformance for nf12.mlx: NF12 modules inside real parent models,
whose weights are replaced through every path users and frameworks take (a
parent's or a child's update/load_weights, files, partial checkpoints), checked
on every observable (forward, nf12_weight, decode, a compiled forward captured
before the change, a save/reload round trip).

The matrix exists because testing the module methods directly missed state
changes that never call them: MLX's parent update writes a child's arrays
without calling the child's update(), and a parent's strict load_weights checks
every shape before any child is involved."""

from __future__ import annotations

import itertools
import os

import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX Metal GPU required", allow_module_level=True)

import mlx.nn as mnn  # noqa: E402
import mlx.utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nf12.mlx as nm  # noqa: E402
from portable_random import randn  # noqa: E402


def to_mlx(t: torch.Tensor) -> mx.array:
    return mx.array(t.contiguous().view(torch.int16).numpy()).view(mx.bfloat16)


def mlx_to_torch(a: mx.array) -> torch.Tensor:
    if a.dtype == mx.bfloat16:
        return torch.from_numpy(np.array(a.view(mx.int16))).view(torch.bfloat16)
    return torch.from_numpy(np.array(a))


def bits_equal(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and bool(mx.array_equal(a.view(mx.int16), b.view(mx.int16)).item())


# Shapes of the parent model's NF12 weights.
SHAPES = {"proj": (48, 128), "layers.0": (64, 64), "layers.1": (32, 64), "emb": (64, 128)}


def weights(seed: int, scale: float, escapes: bool) -> dict[str, nm.NF12Weight]:
    out = {}
    for i, (name, (n, k)) in enumerate(SHAPES.items()):
        w = randn(n, k, seed=seed + i, scale=scale)
        # Keep magnitudes >= scale / 64 so every value fits one exponent window (no escapes).
        floor = torch.tensor(scale / 64, dtype=w.dtype, device=w.device)
        w = torch.where(w.abs() < floor, torch.where(w < 0, -floor, floor), w)
        if escapes:  # out-of-window values: escape rows (and their count) differ from the no-escape set
            w[0, : 8 * (i + 1)] = 1e30
        out[name] = nm.encode(to_mlx(w))
    return out


A = None  # filled lazily: encoding needs the GPU


def weight_sets():
    """A: initial weights. SAME: same array shapes as A, different exp_base
    (a replacement that does not retrace mx.compile). ESC: different exp_base
    and escape counts (array shapes change)."""
    global A
    if A is None:
        a, same, esc = weights(10, 0.02, False), weights(50, 0.02 * 16, False), weights(30, 0.05, True)
        for name in SHAPES:  # SAME must differ from A in every array but keep A's shapes
            assert {k: tuple(v.shape) for k, v in same[name].state_dict().items()} == \
                {k: tuple(v.shape) for k, v in a[name].state_dict().items()}, name
            assert same[name].exp_base != a[name].exp_base, name
            assert not np.array_equal(np.array(same[name].packed), np.array(a[name].packed)), name
            assert esc[name].num_escapes != a[name].num_escapes, name
        A = (a, same, esc)
    return A


class Net(mnn.Module):
    def __init__(self, w: dict[str, nm.NF12Weight] | None = None, bias: mx.array | None = None):
        super().__init__()
        if w is None:  # freshly built (all -0.0) weights, e.g. before loading a checkpoint
            self.proj = nm.NF12Linear(128, 48, bias=True)
            self.layers = [nm.NF12Linear(64, 64, bias=False), nm.NF12Linear(64, 32, bias=False)]
            self.emb = nm.NF12Embedding(64, 128)
        else:
            self.proj = nm.NF12Linear.from_weight(w["proj"], bias)
            self.layers = [nm.NF12Linear.from_weight(w["layers.0"]), nm.NF12Linear.from_weight(w["layers.1"])]
            self.emb = nm.NF12Embedding.from_weight(w["emb"])

    def __call__(self, x: mx.array, h: mx.array, ids: mx.array):
        return self.proj(x), self.layers[1](self.layers[0](h)), self.emb(ids), self.emb.as_linear(x)


BIAS = to_mlx(randn(48, seed=99))
INPUTS = (to_mlx(randn(3, 128, seed=1)), to_mlx(randn(2, 64, seed=2)), mx.array([0, 5, 63], dtype=mx.int32))


def expected(w: dict[str, nm.NF12Weight]):
    x, h, ids = INPUTS
    return (nm.linear(x, w["proj"], BIAS), nm.linear(nm.linear(h, w["layers.0"]), w["layers.1"]),
            nm.embedding(ids, w["emb"]), nm.linear(x, w["emb"]))


def checkpoint(w: dict[str, nm.NF12Weight]) -> list[tuple[str, mx.array]]:
    items = [("proj.bias", BIAS)]
    for name, weight in w.items():
        items += list(weight.state_dict(name + ".weight.").items())
    return items


def nf12_modules(net: Net) -> dict[str, nm.NF12Linear | nm.NF12Embedding]:
    return {"proj": net.proj, "layers.0": net.layers[0], "layers.1": net.layers[1], "emb": net.emb}


def assert_holds(net: Net, w: dict[str, nm.NF12Weight], compiled=None) -> None:
    """Every observable of `net` matches weights `w`."""
    for name, module in nf12_modules(net).items():
        assert module.nf12_weight.exp_base == w[name].exp_base, name
        assert module.nf12_weight.num_escapes == w[name].num_escapes, name
        assert bits_equal(module.nf12_weight.decode(), w[name].decode()), f"{name}: nf12_weight.decode()"
    want = expected(w)
    for i, (got, ref) in enumerate(zip(net(*INPUTS), want)):
        assert bits_equal(got, ref), f"forward output {i}"
    if compiled is not None:
        for i, (got, ref) in enumerate(zip(compiled(*INPUTS), want)):
            assert bits_equal(got, ref), f"compiled forward output {i} (trace captured before the change)"


# --- replacement paths ------------------------------------------------------------

def _prefixed(items, prefix):
    return [(k[len(prefix):], v) for k, v in items if k.startswith(prefix)]


def via_parent_update(net, items, tmp_path):
    net.update(mlx.utils.tree_unflatten(items))


def via_parent_load_weights_nonstrict(net, items, tmp_path):
    net.load_weights(items, strict=False)  # MLX's own non-strict load: no shape check


def via_nm_load_weights(net, items, tmp_path):
    nm.load_weights(net, items)


def via_nm_load_weights_file(net, items, tmp_path):
    path = str(tmp_path / "model.safetensors")
    mx.save_safetensors(path, dict(items))
    nm.load_weights(net, path)


def via_child_load_weights(net, items, tmp_path):
    net.proj.load_weights(_prefixed(items, "proj."))
    for i, layer in enumerate(net.layers):
        layer.load_weights(_prefixed(items, f"layers.{i}."))
    net.emb.load_weights(_prefixed(items, "emb."))


def via_child_update(net, items, tmp_path):
    for name, module in nf12_modules(net).items():
        module.update(mlx.utils.tree_unflatten(_prefixed(items, name + ".")))


PATHS = [via_parent_update, via_parent_load_weights_nonstrict, via_nm_load_weights, via_nm_load_weights_file,
         via_child_load_weights, via_child_update]


@pytest.mark.parametrize("target", ["SAME", "ESC"])
@pytest.mark.parametrize("path", PATHS, ids=lambda f: f.__name__)
def test_replacement_paths(path, target, tmp_path):
    a, same, esc = weight_sets()
    new = same if target == "SAME" else esc
    net = Net(a, BIAS)
    compiled = mx.compile(lambda x, h, ids: net(x, h, ids), inputs=net.state)
    assert_holds(net, a, compiled)
    path(net, checkpoint(new), tmp_path)
    assert_holds(net, new, compiled)
    # The new state survives a save and a reload into a freshly built model.
    fresh = Net()
    nm.load_weights(fresh, list(mlx.utils.tree_flatten(net.parameters())))
    assert_holds(fresh, new)


def test_checkpoint_into_fresh_model(tmp_path):
    """The usual way to load a model: build it, then load a checkpoint."""
    _, _, esc = weight_sets()
    path = str(tmp_path / "model.safetensors")
    mx.save_safetensors(path, dict(checkpoint(esc)))
    net = Net()
    with pytest.raises(ValueError, match="shape"):  # MLX's strict load requires unchanged escape rows
        net.load_weights(path)
    nm.load_weights(net, path)
    assert_holds(net, esc)


@pytest.mark.parametrize("keys", [c for r in (1, 2, 3) for c in itertools.combinations(
    (nm.PACKED_KEY, nm.ESCAPES_KEY, nm.EXP_BASE_KEY), r)], ids="+".join)
@pytest.mark.parametrize("loader", ["nm.load_weights", "child.load_weights"])
def test_partial_checkpoint_strict_false(keys, loader):
    """strict=False applies exactly the supplied entries; the rest keep their values."""
    a, same, _ = weight_sets()
    net = Net(a, BIAS)
    supplied = [(f"proj.weight.{key}", same["proj"].state_dict()[key]) for key in keys]
    if loader == "nm.load_weights":
        nm.load_weights(net, supplied, strict=False)
    else:
        net.proj.load_weights(_prefixed(supplied, "proj."), strict=False)
    want = {**a["proj"].state_dict(), **same["proj"].state_dict()}
    want = {k: (want[k] if k in keys else a["proj"].state_dict()[k]) for k in want}
    expect = nm.NF12Weight.from_state_dict(want)
    got = net.proj.nf12_weight
    assert got.exp_base == expect.exp_base
    assert np.array_equal(np.array(got.packed), np.array(expect.packed))
    assert bits_equal(net.proj(INPUTS[0]), nm.linear(INPUTS[0], expect, BIAS))
    for name in ("layers.0", "layers.1", "emb"):  # untouched modules keep their weights
        assert bits_equal(nf12_modules(net)[name].nf12_weight.decode(), a[name].decode())


def _bad_checkpoints():
    a, same, _esc = weight_sets()
    good = checkpoint(same)
    def edit(key, value):
        return [(k, value if k == key else v) for k, v in good]
    return {
        "extra key": good + [("nope", mx.zeros((1,)))],
        "missing key": [kv for kv in good if kv[0] != "emb.weight.nf12_exp_base"],
        "exp_base dtype": edit("proj.weight.nf12_exp_base", mx.array([3], dtype=mx.int32)),
        "exp_base range": edit("proj.weight.nf12_exp_base", mx.array([200], dtype=mx.uint8)),
        "packed shape": edit("layers.0.weight.nf12_packed", mx.zeros((16, 96), dtype=mx.uint8)),
        "escapes dtype": edit("emb.weight.nf12_escapes", mx.zeros((0, 8), dtype=mx.int32)),
        "bias shape": edit("proj.bias", mx.zeros((16,), dtype=mx.bfloat16)),
        # A truncated escape table: packed and escapes supplied together must agree.
        "truncated escapes": [(k, v[:-1] if k == "proj.weight.nf12_escapes" else v) for k, v in checkpoint(_esc)],
    }


@pytest.mark.parametrize("case", list(_bad_checkpoints()))
def test_rejected_checkpoint_leaves_model_unchanged(case):
    a, _, _ = weight_sets()
    net = Net(a, BIAS)
    before = mlx.utils.tree_flatten(net.parameters())
    with pytest.raises(ValueError):
        nm.load_weights(net, _bad_checkpoints()[case])
    after = mlx.utils.tree_flatten(net.parameters())
    assert [k for k, _ in before] == [k for k, _ in after]
    assert all(x is y for (_, x), (_, y) in zip(before, after)), "a rejected load modified the model"
    assert_holds(net, a)


def test_set_dtype_leaves_nf12_arrays():
    """Float casts (set_dtype's default predicate) must not touch the integer NF12 arrays."""
    a, _, _ = weight_sets()
    net = Net(a, BIAS)
    net.set_dtype(mx.bfloat16)
    assert_holds(net, a)


# --- one checkpoint format across frameworks ---------------------------------------

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="PyTorch MPS required")
@pytest.mark.parametrize("direction", ["torch->mlx", "mlx->torch"])
def test_checkpoints_cross_frameworks(direction, tmp_path):
    """A checkpoint saved by either framework's modules loads into the other's:
    the decoded weights are bit-identical, and so are the outputs when both
    frameworks run nf12's kernels. The decode route hands the GEMM to each
    framework's own matmul (MPS vs MLX), whose accumulation order depends on
    the GPU and shape, so there the outputs only agree to bf16 rounding."""
    safetensors_torch = pytest.importorskip("safetensors.torch")
    import nf12
    from torch import nn

    class TorchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nf12.NF12Linear(128, 48, bias=True, device="mps")
            self.layers = nn.ModuleList([nf12.NF12Linear(64, 64, bias=False, device="mps"),
                                         nf12.NF12Linear(64, 32, bias=False, device="mps")])
            self.emb = nf12.NF12Embedding(64, 128, device="mps")

        def forward(self, x, h, ids):
            return self.proj(x), self.layers[1](self.layers[0](h)), self.emb(ids)

    _, _, esc = weight_sets()
    path = str(tmp_path / "model.safetensors")
    if direction == "torch->mlx":
        source = TorchNet()
        source.load_state_dict({k: mlx_to_torch(v) for k, v in checkpoint(esc)})
        safetensors_torch.save_file({k: v.contiguous().cpu() for k, v in source.state_dict().items()}, path)
        net = Net()
        nm.load_weights(net, path)
        torch_net = source
    else:
        mx.save_safetensors(path, dict(mlx.utils.tree_flatten(Net(esc, BIAS).parameters())))
        torch_net = TorchNet()
        torch_net.load_state_dict(safetensors_torch.load_file(path))
        net = Net(esc, BIAS)
    assert_holds(net, esc)
    for name, mm in nf12_modules(net).items():
        decoded = torch_net.get_submodule(name).weight.decode().cpu()
        assert torch.equal(decoded.view(torch.int16), mlx_to_torch(mm.nf12_weight.decode()).view(torch.int16)), name
    exact = os.environ.get("NF12_MPS_LINEAR") != "decode"
    got = torch_net(*(mlx_to_torch(a).to("mps") for a in INPUTS))
    for i, (t, m) in enumerate(zip(got, net(*INPUTS)[:3])):
        if exact or i == 2:  # output 2 is the embedding: no GEMM
            assert torch.equal(t.cpu().view(torch.int16), mlx_to_torch(m).view(torch.int16)), f"output {i}"
        else:
            torch.testing.assert_close(t.cpu().float(), mlx_to_torch(m).float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("order", ["packed first", "escapes first"])
def test_sharded_checkpoint(order):
    """One weight's tensors split across two shards, loaded one shard at a time
    with strict=False: inconsistent after the first, correct after the second."""
    a, _, esc = weight_sets()
    net = Net(a, BIAS)
    items = checkpoint(esc)
    first = [kv for kv in items if kv[0].endswith(nm.PACKED_KEY)]
    second = [kv for kv in items if not kv[0].endswith(nm.PACKED_KEY)]
    if order == "escapes first":
        first, second = second, first
    nm.load_weights(net, first, strict=False)
    with pytest.raises(ValueError, match="escaped groups"):
        nm.verify_weights(net)  # half-loaded weights are flagged
    nm.load_weights(net, second, strict=False)
    nm.verify_weights(net)
    assert_holds(net, esc)


def test_verify_weights_detects_mixed_weights():
    a, _, esc = weight_sets()
    net = Net(a, BIAS)
    nm.verify_weights(net)
    net.update({"proj": {"weight": {nm.PACKED_KEY: esc["proj"].packed}}})  # raw update: no checks
    with pytest.raises(ValueError, match="proj.weight: nf12_packed has"):
        nm.verify_weights(net)


def test_scalar_exp_base_is_accepted():
    a, same, _ = weight_sets()
    net = Net(a, BIAS)
    items = [(k, mx.array(v.item(), dtype=mx.uint8) if k.endswith(nm.EXP_BASE_KEY) else v) for k, v in checkpoint(same)]
    nm.load_weights(net, items)
    assert_holds(net, same)


def test_strict_false_follows_mlx_for_other_arrays():
    """Outside NF12 weights, strict=False behaves like nn.Module.load_weights(strict=False)."""
    a, _, _ = weight_sets()
    net, ref = Net(a, BIAS), Net(a, BIAS)
    items = [("proj.bias", mx.zeros((16,), dtype=mx.bfloat16)), ("not.in.model", mx.zeros((2,)))]
    nm.load_weights(net, items, strict=False)
    mnn.Module.load_weights(ref, items, strict=False)
    assert net.proj.bias.shape == ref.proj.bias.shape == (16,)


def test_top_level_verify_weights_routes_mlx_models():
    """nf12.verify_weights must check an MLX model, not silently report success."""
    import nf12

    a, _, esc = weight_sets()
    net = Net(a, BIAS)
    nf12.verify_weights(net)
    net.update({"proj": {"weight": {nm.PACKED_KEY: esc["proj"].packed}}})
    with pytest.raises(ValueError, match="proj.weight: nf12_packed has"):
        nf12.verify_weights(net)
    with pytest.raises(TypeError):
        nf12.verify_weights(object())


def test_encode_is_evaluated(tmp_path):
    """encode evaluates its result: a compiled model built from unevaluated
    weights must not re-run the encoder on every call."""
    mx.random.seed(3)
    lin = mnn.Linear(128, 48, bias=False)
    lin.set_dtype(mx.bfloat16)
    layer = nm.NF12Linear.from_linear(lin)
    f = mx.compile(layer)
    out = f(INPUTS[0])
    dot = tmp_path / "graph.dot"
    mx.export_to_dot(str(dot), out)
    assert "encode" not in dot.read_text()


def test_narrowed_moe_weight():
    w = randn(8, 128, 128, seed=1, scale=0.05)
    mw = nm.encode_moe(to_mlx(w))
    for start in range(1, 7):
        assert bits_equal(mw.narrow(start, 2).decode(), to_mlx(w[start:start + 2]))


def test_bias_at_odd_offset():
    a, _, _ = weight_sets()
    full = to_mlx(randn(49, seed=7))
    bias = full[1:]  # starts at an odd bf16 element
    x = INPUTS[0]
    assert bits_equal(nm.linear(x, a["proj"], bias), nm.linear(x, a["proj"], mx.array(bias)))


def test_decode_route_slices_match(monkeypatch):
    """The large-M decode route decodes bounded row slices (1 MiB slices force
    several) and runs MLX's GEMM on each. Each slice is compared with the same
    GEMM on the same slice of the decoded weight: a different GEMM shape may
    accumulate in a different order on another GPU, so the whole-weight
    product is only close, not bit-identical."""
    monkeypatch.setenv("NF12_MPS_LINEAR", "decode")  # the route under test, whatever the matrix forces
    monkeypatch.delenv("NF12_MPS_DECODE_SLICE_MB", raising=False)
    w = randn(2048, 1024, seed=8, scale=0.02)
    w[5, :8] = 1e30
    nw = nm.encode(to_mlx(w))
    x, b = to_mlx(randn(40, 1024, seed=9)), to_mlx(randn(2048, seed=10))
    wd = nw.decode()
    assert bits_equal(nm.linear(x, nw, b), mx.addmm(b, x, wd.T))  # the default uses one slice
    monkeypatch.setenv("NF12_MPS_DECODE_SLICE_MB", "1")
    rows = max(16, (1 << 20) // (1024 * 2) // 16 * 16)  # launch.h decode_slice_rows: 512 rows per 1 MiB slice
    assert rows < 2048
    want = mx.concatenate([mx.addmm(b[r:r + rows], x, wd[r:r + rows].T) for r in range(0, 2048, rows)], axis=1)
    assert bits_equal(nm.linear(x, nw, b), want)
