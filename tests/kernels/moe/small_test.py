# SPDX-License-Identifier: Apache-2.0
"""Exactness suite for the MoE decode kernels, the device schedule and
nf12::detail::moe_linear() dispatch.

Run: uv run --group test pytest tests/kernels/moe/small_test.py
"""

import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import small_harness as h  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_LIB = None
_ENC = {}


def lib():
    global _LIB
    if _LIB is None:
        _LIB = h.Lib()
    return _LIB


def encoded(E, n, k, seed=0, escapes=0.0, escape_pad=0, exp_shift=0):
    key = (E, n, k, seed, escapes, escape_pad, exp_shift)
    if key not in _ENC:
        w = h.random_experts(E, n, k, seed=seed, escapes=escapes)
        bases = None
        if exp_shift:
            bases = [h.reference.choose_exp_base(w[e]) + exp_shift for e in range(E)]
        _ENC[key] = h.encode_moe(w, bases, escape_pad=escape_pad)
    return _ENC[key]


def forced(mt, warps, kc, slice_=0, pdl=0):
    """Below SM90, force a CTA-local power-of-two K split <= kc that divides
    warps and whose slice step (2 * kp) divides the slice. No clusters or PDL."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
        kp = 1
        while 2 * kp <= min(kc, warps) and slice_ % (4 * kp) == 0:
            kp *= 2
        return h.Config(mt, warps, 1, slice_, 0, kp)
    return h.Config(mt, warps, kc, slice_, pdl)


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def random_topk(T, topk, E, seed=0, drop=0.0):
    """Distinct experts per token, with a `drop` fraction set to -1 / E (dropped)."""
    g = gen(seed)
    ids = torch.rand(T, E, generator=g, device="cuda").argsort(dim=1)[:, :topk].to(torch.int32) if topk <= E else \
        torch.randint(0, E, (T, topk), generator=g, device="cuda", dtype=torch.int32)
    if drop > 0:
        mask = torch.rand(T, topk, generator=g, device="cuda") < drop
        other = torch.where(torch.rand(T, topk, generator=g, device="cuda") < 0.5, -1, E).to(torch.int32)
        ids = torch.where(mask, other, ids)
    return ids


def make_x(rows, k, ldx=None, seed=1):
    ldx = k if ldx is None else ldx
    buf = torch.randn(rows, ldx, generator=gen(seed), device="cuda").to(torch.bfloat16)
    return buf[:, :k] if ldx != k else buf


def reference(x, enc, offsets, gather, gather_div, scatter, row_scale, count):
    """FP32 reference rows keyed by output row, plus sum(|x||w|) for the tolerance."""
    off = offsets.cpu()
    p = torch.arange(count, device="cuda")
    in_rows = gather[:count].long() // gather_div if gather is not None else p
    out_rows = scatter[:count].long() if scatter is not None else p
    xs = x[in_rows].float()
    ref = torch.empty(count, enc.n, device="cuda")
    mag = torch.empty(count, enc.n, device="cuda")
    for e in range(enc.num_experts):
        lo, hi = int(off[e]), int(off[e + 1])
        if hi > lo:
            we = enc.expert(e).float()
            ref[lo:hi] = xs[lo:hi] @ we.T
            mag[lo:hi] = xs[lo:hi].abs() @ we.abs().T
    if row_scale is not None:
        s = row_scale[out_rows].float()[:, None]
        ref = ref * s
        mag = mag * s.abs()
    return out_rows, ref, mag


def check_close(y, out_rows, ref, mag):
    got = y[out_rows].float()
    # mma.sync partial sums are not IEEE-rounded FP32 (tests/kernels/linear/test_small_m.py).
    tol = ref.abs() * 2.0**-7 + mag * 1e-4 + 1e-6
    err = (got - ref).abs()
    bad = err > tol
    assert not bool(bad.any()), f"max err {float(err.max())}, {int(bad.sum())} elements out of tolerance"


def run_pair(cfg, enc, topk_ids, mode="w13", ldx=None, ldy=None, x_rows=None, max_assignments=None, seed=3):
    """Check NF12/bf16t equality and the FP32 reference with a shared config.
    Unwritten rows and columns must stay untouched.
    mode: "w13" (gather = sorted, gather_div = topk, outputs in sorted order),
          "w2" (scatter = sorted, row_scale, outputs in assignment order),
          "plain" (neither), "both" (gather and scatter and row_scale)."""
    T, topk = topk_ids.shape
    A = T * topk
    sorted_ids, offsets = h.align(topk_ids, enc.num_experts)
    count = int(offsets[-1])
    A_bound = A if max_assignments is None else max_assignments
    gather = sorted_ids if mode in ("w13", "both") else None
    scatter = sorted_ids if mode in ("w2", "both") else None
    gdiv = topk if gather is not None else 1
    if x_rows is None:
        x_rows = T if gather is not None else A
    x = make_x(x_rows, enc.k, ldx, seed=seed)
    row_scale = (torch.rand(A + 1, generator=gen(seed + 1), device="cuda") * 2 - 0.5) if mode in ("w2", "both") \
        else None
    ldy = enc.n if ldy is None else ldy
    outs = []
    for src in (h.NF12, h.BF16T):
        y = torch.full((A + 2, ldy), float("nan"), dtype=torch.bfloat16, device="cuda")
        try:
            lib().moe_linear(cfg, x, enc, offsets, A_bound, y, gather=gather, gather_div=gdiv, scatter=scatter,
                             row_scale=row_scale, source=src)
        except h.LargeStub as e:
            pytest.skip(str(e))
        outs.append(y)
    torch.cuda.synchronize()
    y_nf12, y_bf16t = outs
    assert torch.equal(y_nf12.view(torch.int16), y_bf16t.view(torch.int16)), f"NF12 != bf16t for {cfg}"
    out_rows, ref, mag = reference(x, enc, offsets, gather, gdiv, scatter, row_scale, count)
    written = torch.zeros(A + 2, dtype=torch.bool, device="cuda")
    written[out_rows] = True
    assert bool(y_nf12[~written].isnan().all()), "a row no valid position maps to was written"
    if ldy > enc.n:
        assert bool(y_nf12[:, enc.n:].isnan().all()), "columns past n written"
    check_close(y_nf12[:, : enc.n], out_rows, ref, mag)
    return y_nf12


def all_configs(E, n, k, mts=(1, 2, 4, 8)):
    out = []
    for mt, warps, kc, sl, pdl, kp in itertools.product(mts, (4, 8), (1, 2, 3, 8), (0, 2, 4), (0, 1), (1, 2, 4, 8)):
        cfg = h.Config(mt, warps, kc, sl, pdl, kp)
        if lib().supported(cfg, E, n, k):
            out.append(cfg)
    return out


def py_schedule(offsets, A, bm):
    """(tiles, row_end): offsets clamped to [0, A], then made monotone (running max), as schedule.cuh documents."""
    off, run = [], 0
    for v in offsets.cpu().tolist():
        run = max(run, min(max(int(v), 0), A))
        off.append(run)
    tiles = []
    for e in range(len(off) - 1):
        lo, hi = off[e], off[e + 1]
        for t in range((hi - lo + bm - 1) // bm):
            tiles.append((e, lo + t * bm, min(bm, hi - lo - t * bm), t))
    return tiles, off[1:]


def check_schedule(offsets, E, A, bm):
    tiles, row_end = lib().schedule(offsets, E, A, bm)
    want_tiles, want_end = py_schedule(offsets, A, bm)
    assert [tuple(r) for r in tiles.tolist()] == want_tiles
    assert row_end.tolist() == want_end


@pytest.mark.parametrize("E,T,topk,bm", [(1, 5, 1, 8), (8, 64, 2, 16), (128, 64, 8, 8), (256, 1, 8, 64),
                                         (1024, 300, 8, 128), (64, 33, 6, 256)])
def test_schedule_list(E, T, topk, bm):
    ids = random_topk(T, topk, E, seed=E + T, drop=0.1)
    _, offsets = h.align(ids, E)
    check_schedule(offsets, E, T * topk, bm)


def test_schedule_clamps_malformed_offsets():
    offsets = torch.tensor([3, 2, 40, 7, 1000, -5, 9], dtype=torch.int64, device="cuda")
    check_schedule(offsets, 6, 20, 4)
    # The decode kernel only touches sorted positions in [0, A): rows past A stay untouched.
    enc = encoded(6, *SMALL_SHAPE, seed=1)
    x = make_x(20, SMALL_SHAPE[1])
    y = torch.full((24, SMALL_SHAPE[0]), float("nan"), dtype=torch.bfloat16, device="cuda")
    for cfg in (h.AUTO, forced(1, 4, 2, 0, 0)):
        lib().moe_linear(cfg, x, enc, offsets, 20, y)
    torch.cuda.synchronize()
    assert bool(y[20:].isnan().all()) and not bool(y[3:20].isnan().any())


SMALL_SHAPE = (208, 1472)  # partial row group at 8 warps, uneven K split


@pytest.mark.parametrize("E,topk", [(1, 1), (8, 2), (64, 6), (128, 8), (256, 8), (5, 8)])
@pytest.mark.parametrize("T", [1, 3, 16, 64])
def test_routing_dispatch(E, topk, T):
    """Production dispatch over expert counts, topk and token counts, both layer halves."""
    n, k = (96, 640) if E >= 64 else SMALL_SHAPE
    enc = encoded(E, n, k, seed=E % 5, escapes=0.002)
    ids = random_topk(T, topk, E, seed=T * 31 + E)
    for mode in ("w13", "w2"):
        run_pair(h.AUTO, enc, ids, mode=mode)


@pytest.mark.parametrize("mode", ["plain", "w13", "w2", "both"])
def test_gather_scatter_modes(mode):
    enc = encoded(8, *SMALL_SHAPE, seed=1, escapes=0.002)
    ids = random_topk(13, 2, 8, seed=7, drop=0.2)
    run_pair(h.AUTO, enc, ids, mode=mode)
    run_pair(forced(2, 4, 2, 4, 0), enc, ids, mode=mode)


def test_every_config():
    """All template/split/slice/PDL combinations, with experts above one m-tile (skew)."""
    enc = encoded(6, *SMALL_SHAPE, seed=4, escapes=0.003)
    ids = random_topk(40, 2, 6, seed=9, drop=0.1)
    ids[:24, 0] = 2  # expert 2 gets >= 24 rows: several m-tiles at mt 1 and 2
    configs = all_configs(6, *SMALL_SHAPE)
    assert configs
    ref = None
    for cfg in configs:
        y = run_pair(cfg, enc, ids, mode="both", ldx=SMALL_SHAPE[1] + 8)
        if ref is None:
            ref = y
        else:  # different configs reduce in different orders
            d = (y.float() - ref.float()).nan_to_num(0).abs().max()
            assert float(d) <= float(ref.float().nan_to_num(0).abs().max()) * 2.0**-6


@pytest.mark.parametrize("cfg", [h.AUTO, forced(1, 8, 1, 0, 0), forced(8, 4, 3, 4, 1)])
def test_skewed(cfg):
    """Repeated expert sets, one expert holding all A rows, and duplicate assignments."""
    E = 16
    enc = encoded(E, *SMALL_SHAPE, seed=2, escapes=0.002)
    T = 64
    same = torch.arange(8, dtype=torch.int32, device="cuda").repeat(T, 1)  # experts 0..7 get T rows each
    run_pair(cfg, enc, same, mode="w13")
    one = torch.full((T, 8), 11, dtype=torch.int32, device="cuda")  # 512 rows in expert 11
    run_pair(cfg, enc, one, mode="w2")
    single = torch.full((1, 1), E - 1, dtype=torch.int32, device="cuda")  # one row, last expert
    run_pair(cfg, enc, single, mode="both")


def test_empty_and_dropped():
    E = 32
    enc = encoded(E, *SMALL_SHAPE, seed=3)
    ids = random_topk(16, 4, E, seed=11, drop=0.5)
    ids[ids >= 16] = -1  # experts 16.. are empty (another expert-parallel rank)
    run_pair(h.AUTO, enc, ids, mode="w2")
    run_pair(h.AUTO, enc, ids, mode="w13")
    gone = torch.full((4, 2), -1, dtype=torch.int32, device="cuda")  # nothing valid: no row written
    run_pair(h.AUTO, enc, gone, mode="w2")


def test_padding_and_bounds():
    """ldx/ldy padding, x with extra rows, max_assignments above the real count."""
    enc = encoded(8, *SMALL_SHAPE, seed=5, escapes=0.001)
    ids = random_topk(9, 2, 8, seed=13)
    run_pair(h.AUTO, enc, ids, mode="w13", ldx=SMALL_SHAPE[1] + 24, ldy=SMALL_SHAPE[0] + 8, x_rows=20)
    run_pair(h.AUTO, enc, ids, mode="w2", ldx=SMALL_SHAPE[1] + 8, ldy=SMALL_SHAPE[0] + 16, max_assignments=18 * 3)


def test_escapes_heavy_and_offset_table():
    """Many escaped groups per expert (shifted bases), and escape_offsets[0] != 0."""
    enc = encoded(8, 128, 1024, seed=6, exp_shift=4, escape_pad=37)
    assert enc.num_escapes > 8 * 1000 and int(enc.escape_offsets[0]) == 37
    ids = random_topk(16, 2, 8, seed=17)
    for cfg in (h.AUTO, forced(1, 8, 2, 0, 1), forced(4, 4, 1, 4, 0)):
        run_pair(cfg, enc, ids, mode="both")


def test_special_values():
    """Inf/NaN/-0/subnormal weights propagate the same through NF12 and bf16t."""
    w = h.random_experts(4, 128, 512, seed=7)
    flat = w.view(-1)
    flat[::97] = float("inf")
    flat[5::131] = float("-inf")
    flat[11::173] = -0.0
    flat[13::89] = 1e-40
    enc = h.encode_moe(w)
    ids = random_topk(6, 2, 4, seed=19)
    sorted_ids, offsets = h.align(ids, 4)
    x = make_x(6, 512)
    outs = []
    for src in (h.NF12, h.BF16T):
        y = torch.zeros(12, 128, dtype=torch.bfloat16, device="cuda")
        lib().moe_linear(h.AUTO, x, enc, offsets, 12, y, gather=sorted_ids, gather_div=2, source=src)
        outs.append(y)
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


def test_deterministic():
    enc = encoded(16, 512, 2048, seed=8, escapes=0.001)
    ids = random_topk(8, 4, 16, seed=23)
    sorted_ids, offsets = h.align(ids, 16)
    x = make_x(8, 2048)
    cfg = forced(1, 8, 8, 4, 1)
    ys = []
    for _ in range(4):
        y = torch.empty(32, 512, dtype=torch.bfloat16, device="cuda")
        lib().moe_linear(cfg, x, enc, offsets, 32, y, gather=sorted_ids, gather_div=4)
        ys.append(y)
    for y in ys[1:]:
        assert torch.equal(ys[0].view(torch.int16), y.view(torch.int16))


def test_graph_capture():
    """Capture moe_linear's first call, including occupancy/smem queries.
    Replays must follow new routing written into the same device buffers."""
    E, n, k = 24, 1040, 1984  # a shape no other test dispatches
    enc = encoded(E, n, k, seed=9)
    T, topk = 5, 3
    ids = random_topk(T, topk, E, seed=29)
    sorted_ids, offsets = h.align(ids, E)
    x = make_x(T, k)
    y = torch.zeros(T * topk, n, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            lib().moe_linear(h.AUTO, x, enc, offsets, T * topk, y, gather=sorted_ids, gather_div=topk)
    for seed in (29, 31, 37):
        ids = random_topk(T, topk, E, seed=seed)
        new_sorted, new_offsets = h.align(ids, E)
        sorted_ids.copy_(new_sorted)
        offsets.copy_(new_offsets)
        g.replay()
        torch.cuda.synchronize()
        expect = torch.zeros_like(y)
        lib().moe_linear(h.AUTO, x, enc, offsets, T * topk, expect, gather=sorted_ids, gather_div=topk)
        torch.cuda.synchronize()
        assert torch.equal(y.view(torch.int16), expect.view(torch.int16))
        out_rows, ref, mag = reference(x, enc, offsets, sorted_ids, topk, None, None, T * topk)
        check_close(y, out_rows, ref, mag)


MODEL_SHAPES = {  # name: (E, topk, n, k)
    "qwen3-30b-a3b.w13": (128, 8, 1536, 2048),
    "qwen3-30b-a3b.w2": (128, 8, 2048, 768),
    "deepseek-v2-lite.w13": (64, 6, 2816, 2048),
    "deepseek-v2-lite.w2": (64, 6, 2048, 1408),
    "mixtral.w2": (8, 2, 4096, 14336),
}


@pytest.mark.parametrize("name", list(MODEL_SHAPES))
def test_model_shapes(name):
    E, topk, n, k = MODEL_SHAPES[name]
    w = h.random_experts(min(E, 4), n, k, seed=10, escapes=1e-4)
    enc = h.encode_moe(w, num_experts=E)
    for T in (1, 4, 16, 64):
        run_pair(h.AUTO, enc, random_topk(T, topk, E, seed=T), mode="w2" if name.endswith("w2") else "w13")


def test_dispatch_rejects_bad_args():
    E, n, k = 4, 64, 128
    enc = encoded(E, n, k)
    ids = random_topk(4, 2, E, seed=1)
    sorted_ids, offsets = h.align(ids, E)
    x = make_x(4, k)
    y = torch.empty(8, n, dtype=torch.bfloat16, device="cuda")
    L = lib().lib
    stream = torch.cuda.current_stream().cuda_stream
    bad = [
        dict(ldx=k - 8), dict(ldx=k + 4), dict(n=n - 8), dict(k=k - 32), dict(ldy=n - 8), dict(ldy=n + 4),
        dict(gather_div=0), dict(E=0), dict(A=-1), dict(A=1 << 31), dict(offsets=None), dict(exp_base=None),
        dict(escape_offsets=None),
    ]
    for over in bad:
        a = dict(ldx=k, n=n, k=k, ldy=n, gather_div=2, E=E, A=8, offsets=offsets.data_ptr(),
                 exp_base=enc.exp_base.data_ptr(), escape_offsets=enc.escape_offsets.data_ptr())
        a.update(over)
        err = L.nf12_moe_linear(0, 0, 0, 0, 0, 0, x.data_ptr(), a["ldx"], sorted_ids.data_ptr(), a["gather_div"], None,
                                a["offsets"], a["A"], h.NF12, enc.packed.data_ptr(), enc.escapes.data_ptr(),
                                a["escape_offsets"], a["exp_base"], a["E"], a["n"], a["k"], None, y.data_ptr(),
                                a["ldy"], None, 0, stream)
        assert err == 1, f"{over} -> {err}"  # cudaErrorInvalidValue
    err = L.nf12_moe_linear(0, 0, 0, 0, 0, 0, x.data_ptr(), k, None, 1, None, offsets.data_ptr(), 8, h.NF12,
                            enc.packed.data_ptr(), None, enc.escape_offsets.data_ptr(), enc.exp_base.data_ptr(),
                            2048, n, k, None, y.data_ptr(), n, None, 0, stream)
    assert err == 801, err  # more experts than the shared-memory schedule supports: cudaErrorNotSupported


def test_zero_assignments_is_noop():
    enc = encoded(4, 64, 128)
    offsets = torch.zeros(5, dtype=torch.int64, device="cuda")
    x = make_x(1, 128)
    y = torch.full((2, 64), float("nan"), dtype=torch.bfloat16, device="cuda")
    lib().moe_linear(h.AUTO, x, enc, offsets, 0, y)
    torch.cuda.synchronize()
    assert bool(y.isnan().all())


REAL_DIR = os.environ.get("NF12_MOE_REAL_DIR")  # MoEBench's expert stacks (see small_bench.py --real)


@pytest.mark.skipif(REAL_DIR is None, reason="NF12_MOE_REAL_DIR not set")
@pytest.mark.parametrize("half", ["w13", "w2"])
def test_real_heavy_escape_layer(half):
    """Qwen3-30B-A3B layer 0: experts with 2-5.5% escaped groups."""
    from safetensors.torch import load_file

    path = os.path.join(REAL_DIR, f"qwen3-30b-a3b.L0.{half}.safetensors")
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    enc = h.encode_moe(load_file(path, device="cuda")["weight"])
    ids = random_topk(64, 8, enc.num_experts, seed=41)
    ids[:, 0] = 76  # the heaviest-escape expert
    for T in (1, 8, 64):
        run_pair(h.AUTO, enc, ids[:T], mode="w13" if half == "w13" else "w2")
