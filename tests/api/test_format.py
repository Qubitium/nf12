# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure-PyTorch NF12 reference (python/nf12/reference.py).

The reference is checked against independent ports of SPEC.md and
include/nf12/format.cuh: a word-level port of `encode_group_in_window` /
`decode_bulk` (vectorized, for exhaustive sweeps), a scalar loop encoder that
places bytes by the SPEC 4.2 table, and, when nvcc and a GPU are present, the
real device decode compiled from format.cuh.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import nf12.reference as ref  # noqa: E402

BASES = list(range(ref.MAX_EXP_BASE + 1))

# SPEC 4.1 table: slot -> (row offset added to g, col offset added to 2t).
SPEC_SLOT_OFFSETS = [(0, 0), (0, 1), (8, 0), (8, 1), (0, 8), (0, 9), (8, 8), (8, 9)]




def slot_row(lane, slot):  # format.cuh
    return (lane >> 2) + ((slot >> 1) & 1) * 8


def slot_col(lane, slot):  # format.cuh
    return 2 * (lane & 3) + (slot & 1) + (slot >> 2) * 8


def encode_group_scalar(slots, base):
    """Port of nf12::encode_group_in_window. Returns (meta, lo0, lo1) or None."""
    meta = lo0 = lo1 = 0
    lo_e = 2 * base
    for i, v in enumerate(slots):
        e = (v >> 7) & 0xFF
        if e < lo_e or e > lo_e + 15:
            return None
        s = v >> 15
        o1 = (e - lo_e) >> 1
        bits = ((s << 6) | (o1 << 3)) if i & 1 else ((s << 7) | o1)
        meta |= bits << (8 * (i >> 1))
        if i < 4:
            lo0 |= (v & 0xFF) << (8 * i)
        else:
            lo1 |= (v & 0xFF) << (8 * (i - 4))
    if meta == 0:
        return None
    return meta, lo0, lo1


def byte_perm_scalar(x, y, sel):
    b = (x | (y << 32)).to_bytes(8, "little")
    return sum(b[(sel >> (4 * i)) & 7] << (8 * i) for i in range(4))


def decode_group_scalar(meta, lo0, lo1, base, escapes):
    """Port of nf12::decode_group: returns R0..R3."""
    if meta == 0:
        e = [int(x) & 0xFFFF for x in escapes[lo0]]
        return [e[2 * j] | (e[2 * j + 1] << 16) for j in range(4)]
    bx4 = base * 0x01010101
    a = ((meta & 0x87878787) + bx4) & 0xFFFFFFFF
    b = ((((meta << 1) & 0x80808080) | ((meta >> 3) & 0x07070707)) + bx4) & 0xFFFFFFFF
    p0 = byte_perm_scalar(a, b, 0x5140)
    p1 = byte_perm_scalar(a, b, 0x7362)
    return [
        byte_perm_scalar(lo0, p0, 0x5140),
        byte_perm_scalar(lo0, p0, 0x7362),
        byte_perm_scalar(lo1, p1, 0x5140),
        byte_perm_scalar(lo1, p1, 0x7362),
    ]


def _byte_perm_vec(x, y, sel):
    out = torch.zeros_like(x)
    for i in range(4):
        src = (sel >> (4 * i)) & 7
        byte = (x >> (8 * src)) & 0xFF if src < 4 else (y >> (8 * (src - 4))) & 0xFF
        out |= byte << (8 * i)
    return out


def encode_groups_vec(slots, base):
    """Vectorized port of encode_group_in_window over int64 [G, 8] bit patterns.

    Returns (ok, meta, lo0, lo1) where ok is False for escaped groups.
    """
    lo_e = 2 * base
    e = (slots >> 7) & 0xFF
    ok = ((e >= lo_e) & (e <= lo_e + 15)).all(1)
    s = slots >> 15
    o1 = ((e - lo_e) >> 1) & 7
    meta = torch.zeros_like(slots[:, 0])
    lo0 = torch.zeros_like(meta)
    lo1 = torch.zeros_like(meta)
    for i in range(8):
        bits = ((s[:, i] << 6) | (o1[:, i] << 3)) if i & 1 else ((s[:, i] << 7) | o1[:, i])
        meta |= bits << (8 * (i >> 1))
        if i < 4:
            lo0 |= (slots[:, i] & 0xFF) << (8 * i)
        else:
            lo1 |= (slots[:, i] & 0xFF) << (8 * (i - 4))
    ok &= meta != 0
    return ok, meta, lo0, lo1


def decode_bulk_vec(meta, lo0, lo1, base):
    """Vectorized port of nf12::decode_bulk: int64 [G] words -> [G, 4] registers."""
    bx4 = base * 0x01010101
    a = ((meta & 0x87878787) + bx4) & 0xFFFFFFFF
    b = ((((meta << 1) & 0x80808080) | ((meta >> 3) & 0x07070707)) + bx4) & 0xFFFFFFFF
    p0 = _byte_perm_vec(a, b, 0x5140)
    p1 = _byte_perm_vec(a, b, 0x7362)
    return torch.stack(
        [
            _byte_perm_vec(lo0, p0, 0x5140),
            _byte_perm_vec(lo0, p0, 0x7362),
            _byte_perm_vec(lo1, p1, 0x5140),
            _byte_perm_vec(lo1, p1, 0x7362),
        ],
        dim=1,
    )


def words(byte_tensor):
    """uint8 [..., 4] -> int64 little-endian words [...]."""
    b = byte_tensor.to(torch.int64)
    return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)


def slow_encode(w, base):
    """Loop encoder placing bytes by the SPEC 4.1/4.2/4.3 tables."""
    n, k = w.shape
    bits = (w.view(torch.int16).to(torch.int32) & 0xFFFF).tolist()
    out = bytearray(n * k * 3 // 2)
    escapes = []
    for nb in range(n // 16):
        for kb in range(k // 64):
            sa = (nb * (k // 64) + kb) * 1536
            for j in range(4):
                for lane in range(32):
                    g, t = lane >> 2, lane & 3
                    slots = [bits[16 * nb + g + dr][64 * kb + 16 * j + 2 * t + dc] for dr, dc in SPEC_SLOT_OFFSETS]
                    enc = encode_group_scalar(slots, base)
                    if enc is None:
                        enc = (0, len(escapes), 0)
                        escapes.append(slots)
                    meta, lo0, lo1 = enc
                    out[sa + 16 * lane + 4 * j : sa + 16 * lane + 4 * j + 4] = meta.to_bytes(4, "little")
                    off = sa + 512 + 16 * lane + 8 * j if j < 2 else sa + 1024 + 16 * lane + 8 * (j - 2)
                    out[off : off + 4] = lo0.to_bytes(4, "little")
                    out[off + 4 : off + 8] = lo1.to_bytes(4, "little")
    packed = torch.frombuffer(out, dtype=torch.uint8).clone().view(n, k * 3 // 2)
    esc = torch.tensor(escapes, dtype=torch.int32).view(-1, 8)
    esc = (esc - ((esc >> 15) << 16)).to(torch.int16)
    return packed, esc




def bits_to_bf16(bits):
    bits = bits.to(torch.int32) & 0xFFFF
    return (bits - ((bits >> 15) << 16)).to(torch.int16).view(torch.bfloat16)


def random_bits(n, k, gen):
    return bits_to_bf16(torch.randint(0, 1 << 16, (n, k), generator=gen))


def weights(n, k, gen, scale=0.02):
    return (torch.randn(n, k, generator=gen) * scale).to(torch.bfloat16)


SPECIAL_BITS = [
    0x0000, 0x8000,  # +-0
    0x0001, 0x8001, 0x007F, 0x807F, 0x0040,  # subnormals
    0x0080, 0x8080,  # smallest normals
    0x7F80, 0xFF80,  # +-Inf
    0x7FC0, 0xFFC0, 0x7F81, 0xFFFF, 0x7FA5,  # NaNs with payloads
    0x7F7F, 0xFF7F,  # +-max finite
    0x3F80, 0xBF80,  # +-1
]


def with_specials(w, gen, count):
    bits = w.view(torch.int16).to(torch.int32).flatten().clone()
    pos = torch.randperm(bits.numel(), generator=gen)[:count]
    specials = torch.tensor(SPECIAL_BITS, dtype=torch.int32)
    bits[pos] = specials[torch.arange(count) % len(SPECIAL_BITS)]
    return bits_to_bf16(bits).view(w.shape)


def same_bits(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(1234)




def test_spec_slot_table_matches_format_cuh():
    for lane in range(32):
        for s, (dr, dc) in enumerate(SPEC_SLOT_OFFSETS):
            assert slot_row(lane, s) == (lane >> 2) + dr
            assert slot_col(lane, s) == 2 * (lane & 3) + dc


def test_group_coords_match_spec():
    n, k = 48, 192
    rows, cols = ref.group_coords(n, k)
    assert rows.shape == (n * k // 8, 8)
    for nb in range(n // 16):
        for kb in range(k // 64):
            for j in range(4):
                for lane in range(32):
                    gid = ((nb * (k // 64) + kb) * 4 + j) * 32 + lane
                    for s, (dr, dc) in enumerate(SPEC_SLOT_OFFSETS):
                        assert rows[gid, s] == 16 * nb + (lane >> 2) + dr
                        assert cols[gid, s] == 64 * kb + 16 * j + 2 * (lane & 3) + dc
    flat = ref.group_slots(n, k)
    assert torch.equal(flat, rows * k + cols)
    assert torch.equal(flat.flatten().sort().values, torch.arange(n * k))


def test_to_from_groups_inverse(gen):
    w = random_bits(32, 256, gen)
    assert same_bits(ref.from_groups(ref.to_groups(w), 32, 256), w)


def test_decoded_group_positions():
    """Decoding the bytes at SPEC 4.2 offsets yields the A fragment of SPEC 4.1."""
    n, k, base = 32, 128, 60
    # 4096 distinct in-window patterns: sign x 16 exponents x 128 mantissas.
    vals = torch.arange(n * k, dtype=torch.int32)
    bits = ((vals >> 11) << 15) | ((2 * base + ((vals >> 7) & 15)) << 7) | (vals & 0x7F)
    perm = torch.randperm(n * k, generator=torch.Generator().manual_seed(7))
    w = bits_to_bf16(bits[perm]).view(n, k)
    packed, escapes, b = ref.encode(w, exp_base=base)
    assert b == base
    raw = packed.flatten().tolist()
    wb = (w.view(torch.int16).to(torch.int32) & 0xFFFF).tolist()
    esc = escapes.tolist()

    def word(off):
        return int.from_bytes(bytes(raw[off : off + 4]), "little")

    for nb in range(n // 16):
        for kb in range(k // 64):
            sa = (nb * (k // 64) + kb) * 1536
            for j in range(4):
                for lane in range(32):
                    meta = word(sa + 16 * lane + 4 * j)
                    off = sa + 512 + 16 * lane + 8 * j if j < 2 else sa + 1024 + 16 * lane + 8 * (j - 2)
                    r = decode_group_scalar(meta, word(off), word(off + 4), base, esc)
                    g, t = lane >> 2, lane & 3
                    for s, (dr, dc) in enumerate(SPEC_SLOT_OFFSETS):
                        half = (r[s >> 1] >> (16 * (s & 1))) & 0xFFFF
                        assert half == wb[16 * nb + g + dr][64 * kb + 16 * j + 2 * t + dc]


def test_bf16t_layout(gen):
    n, k = 32, 128
    w = random_bits(n, k, gen)
    t = ref.pack_bf16t(w)
    assert t.shape == (n, 2 * k) and t.dtype == torch.uint8
    raw = t.flatten().tolist()
    wb = (w.view(torch.int16).to(torch.int32) & 0xFFFF).tolist()
    for nb in range(n // 16):
        for kb in range(k // 64):
            sa = (nb * (k // 64) + kb) * 2048
            for j in range(4):
                for lane in range(32):
                    g, tt = lane >> 2, lane & 3
                    off = sa + 512 * j + 16 * lane
                    for s, (dr, dc) in enumerate(SPEC_SLOT_OFFSETS):
                        v = raw[off + 2 * s] | (raw[off + 2 * s + 1] << 8)
                        assert v == wb[16 * nb + g + dr][64 * kb + 16 * j + 2 * tt + dc]
    assert same_bits(ref.unpack_bf16t(t), w)




@pytest.mark.parametrize(
    "kind,base",
    [("weights", None), ("weights", 0), ("weights", 120), ("specials", None), ("random", None), ("random", 63)],
)
def test_encode_matches_slow_encoder(kind, base, gen):
    n, k = 32, 192
    if kind == "weights":
        w = weights(n, k, gen)
    elif kind == "specials":
        w = with_specials(weights(n, k, gen), gen, 300)
    else:
        w = random_bits(n, k, gen)
    packed, escapes, b = ref.encode(w, exp_base=base)
    if base is not None:
        assert b == base
    slow_packed, slow_esc = slow_encode(w, b)
    assert packed.shape == (n, k * 3 // 2) and packed.dtype == torch.uint8
    assert escapes.dtype == torch.int16 and escapes.shape[1:] == (8,)
    assert torch.equal(packed, slow_packed)
    assert torch.equal(escapes, slow_esc)


def _rotating_groups(base, gen):
    """Every BF16 pattern once, at slot v % 8, among in-window fillers, plus
    1024 trailing all-positive groups with offsets 0/1 (sentinel collisions)."""
    v = torch.arange(1 << 16, dtype=torch.int64)
    off = torch.randint(0, 16, (v.numel(), 8), generator=gen)
    sign = torch.randint(0, 2, (v.numel(), 8), generator=gen)
    mant = torch.randint(0, 128, (v.numel(), 8), generator=gen)
    slots = (sign << 15) | ((2 * base + off) << 7) | mant
    slots[torch.arange(v.numel()), v % 8] = v
    sentinel = (((torch.arange(8) & 1) + 2 * base) << 7) | mant[:1024]
    return torch.cat([slots, sentinel])


@pytest.mark.parametrize("base", BASES)
def test_exhaustive_values_every_base(base, gen):
    """All 65536 patterns in every slot position class, at every base."""
    slots = _rotating_groups(base, gen)
    ok, meta, lo0, lo1 = encode_groups_vec(slots, base)
    g = slots.shape[0]
    n, k = 16, g * 8 // 16
    w = ref.from_groups(bits_to_bf16(slots), n, k)
    gb, escapes = ref.group_bytes(w, base)
    ref_meta, ref_lo0, ref_lo1 = words(gb[:, 0:4]), words(gb[:, 4:8]), words(gb[:, 8:12])

    assert torch.equal(ref_meta != 0, ok)
    assert torch.equal(ref_meta[ok], meta[ok])
    assert torch.equal(ref_lo0[ok], lo0[ok])
    assert torch.equal(ref_lo1[ok], lo1[ok])
    # Escaped groups: M = 0, L0 = running index, L1 = 0, payload verbatim.
    assert torch.equal(ref_lo0[~ok], torch.arange(int((~ok).sum())))
    assert bool((ref_lo1[~ok] == 0).all())
    assert torch.equal(escapes.to(torch.int64) & 0xFFFF, slots[~ok])
    # format.cuh decode_bulk reproduces every non-escaped group.
    regs = decode_bulk_vec(ref_meta[ok], ref_lo0[ok], ref_lo1[ok], base)
    s = slots[ok]
    assert torch.equal(regs, s[:, 0::2] | (s[:, 1::2] << 16))

    packed, esc2, _ = ref.encode(w, exp_base=base)
    assert torch.equal(esc2, escapes)
    assert same_bits(ref.decode(packed, escapes, base, n, k), w)


@pytest.mark.parametrize("base", [0, 1, 7, 59, 60, 63, 119, 120])
def test_exhaustive_pairs(base):
    """Every (sign, exponent) combination of a slot pair, in every pair position."""
    se = torch.arange(512, dtype=torch.int64)  # sign << 8 | exponent
    a, b = torch.meshgrid(se, se, indexing="ij")
    a, b = a.flatten(), b.flatten()
    g = a.numel()
    filler = (torch.tensor([1, 0, 1, 0, 1, 0, 1, 0]) << 15) | ((2 * base + torch.arange(8) % 16) << 7) | 0x2A
    slots = filler.repeat(g, 1)
    pair = torch.arange(g) % 4
    rows = torch.arange(g)
    to_bits = lambda x, m: ((x >> 8) << 15) | ((x & 0xFF) << 7) | m  # noqa: E731
    slots[rows, 2 * pair] = to_bits(a, 0x55)
    slots[rows, 2 * pair + 1] = to_bits(b, 0x33)
    ok, meta, lo0, lo1 = encode_groups_vec(slots, base)
    w = ref.from_groups(bits_to_bf16(slots), 16, g // 2)
    gb, _ = ref.group_bytes(w, base)
    ref_meta = words(gb[:, 0:4])
    assert torch.equal(ref_meta != 0, ok)
    assert torch.equal(ref_meta[ok], meta[ok])
    assert torch.equal(words(gb[:, 4:8])[ok], lo0[ok])
    assert torch.equal(words(gb[:, 8:12])[ok], lo1[ok])
    packed, escapes, _ = ref.encode(w, exp_base=base)
    assert same_bits(ref.decode(packed, escapes, base), w)




@pytest.mark.parametrize("shape", [(16, 64), (48, 320), (256, 1024)])
def test_roundtrip_random_bits(shape, gen):
    w = random_bits(*shape, gen)
    for base in (None, 0, 60, 120):
        packed, escapes, b = ref.encode(w, exp_base=base)
        assert same_bits(ref.decode(packed, escapes, b), w)


@pytest.mark.parametrize("scale", [1e-30, 1e-4, 0.02, 1.0, 1e4, 1e30])
def test_roundtrip_weights_with_specials(scale, gen):
    w = with_specials(weights(128, 512, gen, scale), gen, 500)
    packed, escapes, b = ref.encode(w)
    assert same_bits(ref.decode(packed, escapes, b), w)
    nan = torch.isnan(w.float())
    assert bool(nan.any())


def test_subnormal_matrix(gen):
    """Pure subnormals/zeros fit base 0 except all-positive groups (sentinel)."""
    bits = torch.randint(0, 128, (64, 256), generator=gen) | (torch.randint(0, 2, (64, 256), generator=gen) << 15)
    w = bits_to_bf16(bits)
    packed, escapes, b = ref.encode(w)
    assert b == 0
    assert same_bits(ref.decode(packed, escapes, b), w)


def test_zero_matrix_escapes_everything():
    """+0 groups collide with the sentinel at base 0 and are out of window above it."""
    w = torch.zeros(32, 128, dtype=torch.bfloat16)
    counts = ref.escape_counts_all_bases(w)
    assert bool((counts == 32 * 128 // 8).all())
    packed, escapes, b = ref.encode(w)
    assert b == 0 and escapes.shape == (32 * 128 // 8, 8)
    assert same_bits(ref.decode(packed, escapes, b), w)
    # -0 is fine: the sign bit makes the meta word non-zero.
    wn = torch.full((32, 128), -0.0, dtype=torch.bfloat16)
    packed, escapes, b = ref.encode(wn)
    assert b == 0 and escapes.shape[0] == 0
    assert same_bits(ref.decode(packed, escapes, b), wn)


def test_heavy_escape_order(gen):
    n, k = 64, 256
    w = random_bits(n, k, gen)
    packed, escapes, b = ref.encode(w, exp_base=60)
    groups = ref.to_groups(w.view(torch.int16))
    gb, _ = ref.group_bytes(w, 60)
    esc_mask = words(gb[:, 0:4]) == 0
    assert escapes.shape[0] == int(esc_mask.sum()) > 0.9 * groups.shape[0]
    assert torch.equal(escapes, groups[esc_mask])
    assert torch.equal(words(gb[esc_mask, 4:8]), torch.arange(escapes.shape[0]))
    assert same_bits(ref.decode(packed, escapes, b), w)


def test_forced_sentinel_collision(gen):
    n, k, base = 16, 64, 60
    # All groups in-window with a negative slot 0, so none collides...
    off = torch.randint(0, 16, (n * k // 8, 8), generator=gen)
    slots = ((2 * base + off) << 7) | torch.randint(0, 128, (n * k // 8, 8), generator=gen)
    slots[:, 0] |= 0x8000
    # ...except groups 5 and 77: positive, exponents 2B and 2B+1 only.
    for gid in (5, 77):
        slots[gid] = ((2 * base + (torch.arange(8) % 2)) << 7) | (torch.arange(8) * 9)
    # Group 6: same exponents but one negative sign -> meta != 0, not escaped.
    slots[6] = ((2 * base + (torch.arange(8) % 2)) << 7) | 0x11
    slots[6, 3] |= 0x8000
    w = ref.from_groups(bits_to_bf16(slots), n, k)
    assert ref.escape_counts_all_bases(w)[base] == 2
    packed, escapes, b = ref.encode(w, exp_base=base)
    gb, _ = ref.group_bytes(w, base)
    m = words(gb[:, 0:4])
    assert torch.equal((m == 0).nonzero().flatten(), torch.tensor([5, 77]))
    assert words(gb[5, 4:8]) == 0 and words(gb[77, 4:8]) == 1
    assert words(gb[5, 8:12]) == 0 and words(gb[77, 8:12]) == 0
    assert m[6] == 0x40 << 8  # slot 3 = b of pair 1: sign at bit 6 of meta byte 1
    assert torch.equal(escapes.to(torch.int64) & 0xFFFF, slots[[5, 77]])
    assert same_bits(ref.decode(packed, escapes, b), w)




def _brute_counts(w):
    return torch.tensor([ref.encode(w, exp_base=b)[1].shape[0] for b in BASES])


@pytest.mark.parametrize(
    "kind", ["w0.02", "w1", "w1e-6", "w1e20", "mixture", "zeros_mixed", "specials", "random", "sentinel_heavy"]
)
def test_choose_exp_base_brute_force(kind, gen):
    n, k = 32, 128
    if kind.startswith("w"):
        w = weights(n, k, gen, float(kind[1:]))
    elif kind == "mixture":
        scales = torch.tensor([1e-3, 0.02, 0.5, 8.0])[torch.randint(0, 4, (n, 1), generator=gen)]
        w = (torch.randn(n, k, generator=gen) * scales).to(torch.bfloat16)
    elif kind == "zeros_mixed":
        w = weights(n, k, gen)
        w[torch.rand(n, k, generator=gen) < 0.05] = 0
    elif kind == "specials":
        w = with_specials(weights(n, k, gen), gen, 200)
    elif kind == "random":
        w = random_bits(n, k, gen)
    else:
        # Tiny positive values right at the bottom of base 3's window.
        bits = ((6 + torch.randint(0, 2, (n, k), generator=gen)) << 7) | torch.randint(0, 128, (n, k), generator=gen)
        w = bits_to_bf16(bits)
    brute = _brute_counts(w)
    counts = ref.escape_counts_all_bases(w)
    assert torch.equal(counts, brute)
    best = int(brute.min())
    assert ref.choose_exp_base(w) == min(b for b in BASES if brute[b] == best)
    per_block = ref.escape_counts_all_bases(w, per_row_block=True)
    assert per_block.shape == (n // 16, len(BASES))
    assert torch.equal(per_block.sum(0), counts)
    for nb in range(n // 16):
        assert torch.equal(per_block[nb], ref.escape_counts_all_bases(w[16 * nb : 16 * nb + 16]))


def test_choose_exp_base_typical_weights(gen):
    w = weights(256, 1024, gen, 0.02)
    b = ref.choose_exp_base(w)
    # The tail of N(0, 0.02^2) reaches |w| ~ 0.1 (biased exponent 123), so the
    # 16-exponent window [2B, 2B+15] must reach 123: B = 54.
    assert b == 54
    assert int(ref.escape_counts_all_bases(w)[b]) < 0.01 * w.numel() / 8




def test_linear_reference(gen):
    x = torch.randn(3, 5, 128, generator=gen).to(torch.bfloat16)
    w = weights(64, 128, gen, 0.05)
    bias = torch.randn(64, generator=gen).to(torch.bfloat16)
    y = ref.linear(x, w, bias)
    assert y.dtype == torch.bfloat16 and y.shape == (3, 5, 64)
    exact = x.double() @ w.double().T + bias.double()
    assert torch.equal(y, exact.float().to(torch.bfloat16)) or torch.allclose(y.double(), exact, rtol=1e-2, atol=1e-3)
    assert torch.equal(ref.linear(x, w), (x.float() @ w.float().T).to(torch.bfloat16))


def test_shape_and_dtype_validation(gen):
    with pytest.raises(ValueError):
        ref.encode(weights(24, 64, gen))
    with pytest.raises(ValueError):
        ref.encode(weights(16, 96, gen))
    with pytest.raises(TypeError):
        ref.encode(torch.zeros(16, 64))
    with pytest.raises(ValueError):
        ref.encode(weights(16, 64, gen), exp_base=121)
    packed, escapes, b = ref.encode(random_bits(16, 64, gen), exp_base=60)
    with pytest.raises(ValueError):
        ref.decode(packed, escapes[:1], b)


def test_deterministic_and_non_contiguous(gen):
    w = weights(128, 256, gen)
    a = ref.encode(w)
    b = ref.encode(w.T.contiguous().T)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu(gen):
    w = with_specials(weights(256, 512, gen), gen, 400)
    p, e, b = ref.encode(w)
    pc, ec, bc = ref.encode(w.cuda())
    assert bc == b and torch.equal(pc.cpu(), p) and torch.equal(ec.cpu(), e)
    assert same_bits(ref.decode(pc, ec, bc).cpu(), w)
    assert torch.equal(ref.pack_bf16t(w.cuda()).cpu(), ref.pack_bf16t(w))



DEVICE_DECODE_SRC = r"""
#include <nf12/format.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>

// One warp per super-atom: decode all 4 atoms with nf12::decode_group and
// store R0..R3 in bf16t layout, which the test compares with pack_bf16t.
__global__ void decode_kernel(const uint4* packed, const uint4* escapes, uint32_t base, uint4* out) {
  const int sa = blockIdx.x, l = threadIdx.x;
  const uint4 p0 = packed[sa * 96 + l], p1 = packed[sa * 96 + 32 + l], p2 = packed[sa * 96 + 64 + l];
  const uint32_t meta[4] = {p0.x, p0.y, p0.z, p0.w};
  const uint32_t lo[8] = {p1.x, p1.y, p1.z, p1.w, p2.x, p2.y, p2.z, p2.w};
  const uint32_t bx4 = nf12::exp_base_x4(base);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t r[4];
    nf12::decode_group(nf12::Group{meta[j], lo[2 * j], lo[2 * j + 1]}, bx4, escapes, r);
    out[sa * 128 + j * 32 + l] = make_uint4(r[0], r[1], r[2], r[3]);
  }
}

static std::vector<char> slurp(const char* path) {
  FILE* f = fopen(path, "rb");
  fseek(f, 0, SEEK_END);
  std::vector<char> v(ftell(f));
  fseek(f, 0, SEEK_SET);
  if (!v.empty() && fread(v.data(), 1, v.size(), f) != v.size()) exit(2);
  fclose(f);
  return v;
}

int main(int argc, char** argv) {
  // argv: packed escapes out base
  const std::vector<char> packed = slurp(argv[1]), escapes = slurp(argv[2]);
  const uint32_t base = atoi(argv[4]);
  const int num_sa = packed.size() / 1536;
  void *d_packed, *d_esc, *d_out;
  cudaMalloc(&d_packed, packed.size());
  cudaMalloc(&d_esc, escapes.size() + 16);
  cudaMalloc(&d_out, num_sa * 2048);
  cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_esc, escapes.data(), escapes.size(), cudaMemcpyHostToDevice);
  decode_kernel<<<num_sa, 32>>>((const uint4*)d_packed, (const uint4*)d_esc, base, (uint4*)d_out);
  std::vector<char> out(num_sa * 2048);
  if (cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return 3;
  FILE* f = fopen(argv[3], "wb");
  fwrite(out.data(), 1, out.size(), f);
  fclose(f);
  return 0;
}
"""


def _nvcc():
    path = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    return path if os.path.exists(path) else None


@pytest.fixture(scope="module")
def device_decoder(tmp_path_factory):
    nvcc = _nvcc()
    if torch.version.hip is not None or nvcc is None or not torch.cuda.is_available():
        pytest.skip("CUDA format.cuh device helper requires an NVIDIA PyTorch build; HIP has tests/hip")
    d = tmp_path_factory.mktemp("nf12_device_decode")
    src = d / "decode.cu"
    src.write_text(DEVICE_DECODE_SRC)
    major, minor = torch.cuda.get_device_capability()
    exe = d / "decode"
    subprocess.run(
        [nvcc, "-std=c++17", "-O2", f"-arch=sm_{major}{minor}", "-I", str(ROOT / "include"), str(src), "-o", str(exe)],
        check=True,
    )

    def run(packed, escapes, base):
        (d / "packed.bin").write_bytes(packed.numpy().tobytes())
        (d / "esc.bin").write_bytes(escapes.numpy().tobytes())
        subprocess.run([str(exe), str(d / "packed.bin"), str(d / "esc.bin"), str(d / "out.bin"), str(base)], check=True)
        return torch.frombuffer(bytearray((d / "out.bin").read_bytes()), dtype=torch.uint8)

    return run


@pytest.mark.parametrize("kind", ["weights", "specials", "random", "sentinel"])
def test_device_decode_matches_reference(kind, device_decoder, gen):
    n, k = 64, 512
    base = None
    if kind == "weights":
        w = weights(n, k, gen)
    elif kind == "specials":
        w = with_specials(weights(n, k, gen), gen, 2000)
    elif kind == "random":
        w, base = random_bits(n, k, gen), 63
    else:
        slots = _rotating_groups(60, gen)[: n * k // 8]
        w, base = ref.from_groups(bits_to_bf16(slots), n, k), 60
    packed, escapes, b = ref.encode(w, exp_base=base)
    assert escapes.shape[0] > 0 or kind == "weights"
    out = device_decoder(packed, escapes, b)
    assert torch.equal(out, ref.pack_bf16t(w).flatten())
