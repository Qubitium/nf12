# SPDX-License-Identifier: Apache-2.0
"""Opt-in regression tests for index-range limits of the large-M kernels:
weights whose 16-row stride exceeds 2^31 bytes, and escape tables with
>= 2^31 entries. They need ~60 GB of device memory and synthesize the packed
bytes directly (the reference encoder would need far more), so they only run
with NF12_HUGE_TESTS=1:

    NF12_HUGE_TESTS=1 uv run --group test python tests/kernels/linear/large_m_huge_test.py [--family mma|tc|auto]
"""

import argparse
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import large_m_harness as H  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("NF12_HUGE_TESTS") != "1", reason="set NF12_HUGE_TESTS=1")

ONE, TWO = 0x3F80, 0x4000  # BF16 bit patterns


def _i32(v):
    return v - (1 << 32) if v >= 1 << 31 else v


def x_ones(m, k):
    """x = 1 on columns 0..15, 0 elsewhere: y[m, n] = sum of W[n, 0:16], exact in BF16."""
    x = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    x[:, :16] = 1.0
    return x


def wide_stride_weight(k):
    """N = 32: row block 0 all 1.0, row block 1 all 2.0, exp_base 57, no escapes.

    The 16-row stride is 16 * K * 1.5 bytes (NF12) / 16 * K * 2 bytes
    (bf16t), which exceeds 2^31 for K = 2^27."""
    n, kb = 32, k // 64
    # exp_base 57 gives window [114, 129]. For 1.0 (e=127), o>>1 = 6
    # and the low byte is 0x80. For 2.0 (e=128), they are 7 and 0x00.
    # Meta byte = (b.o>>1) << 3 | a.o>>1.
    planes = [(0x36363636, 0x80808080), (0x3F3F3F3F, 0x00000000)]
    packed = torch.empty(2, kb, 384, dtype=torch.int32, device="cuda")
    bf16t = torch.empty(2, kb, 512, dtype=torch.int32, device="cuda")
    for rb, (meta, lo) in enumerate(planes):
        packed[rb, :, :128] = _i32(meta)
        packed[rb, :, 128:] = _i32(lo)
        v = (ONE, TWO)[rb]
        bf16t[rb] = _i32(v | v << 16)
    return SimpleNamespace(n=n, k=k, exp_base=57, packed=packed.view(torch.uint8), bf16t=bf16t.view(torch.uint8),
                           escapes=torch.zeros(1, 4, dtype=torch.int32, device="cuda"), num_escapes=0)


def all_escaped_weight(n, k, chunk=1 << 20):
    """All weights 1.0 encoded with exp_base 63: every in-window meta byte
    would be 0 (sentinel collision), so every group escapes and the escape
    index equals the flat group id. No bf16t twin (it would not fit too)."""
    kb = k // 64
    num_sa = n // 16 * kb
    packed = torch.zeros(num_sa, 384, dtype=torch.int32, device="cuda")
    lane = torch.arange(32, device="cuda", dtype=torch.int64)
    for s0 in range(0, num_sa, chunk):
        sa = torch.arange(s0, min(num_sa, s0 + chunk), device="cuda", dtype=torch.int64).view(-1, 1)
        for j in range(4):
            idx = ((sa * 4 + j) * 32 + lane) & 0xFFFFFFFF  # [c, 32]
            idx = (idx - ((idx >> 31) << 32)).to(torch.int32)
            word = 128 * (1 + j // 2) + 4 * lane + 2 * (j % 2)  # L0 slot of atom j, lane l
            packed[s0:s0 + sa.shape[0], word] = idx
    num_escapes = n * k // 8
    escapes = torch.full((num_escapes, 4), _i32(ONE | ONE << 16), dtype=torch.int32, device="cuda")
    return SimpleNamespace(n=n, k=k, exp_base=63, packed=packed.view(torch.uint8), bf16t=None, escapes=escapes,
                           num_escapes=num_escapes)


def configs(family):
    s = H.Shim()
    if family == H.FAMILY_MMA:
        # heuristic, every loop at split 1, and a split-K run.
        return [-1] + list(range(s.num_configs(H.FAMILY_MMA))) + [16 * 3]
    if family == H.FAMILY_TC:
        return [-1] + list(range(s.num_configs(H.FAMILY_TC)))
    return [-1]


def run_wide_stride(family):
    shim = H.Shim()
    w = wide_stride_weight(1 << 27)
    x = x_ones(65, w.k)
    want = torch.tensor([16.0] * 16 + [32.0] * 16, dtype=torch.bfloat16, device="cuda").expand(65, 32)
    for cfg in configs(family):
        for src in ("nf12", "bf16t"):
            try:
                y = shim.linear(x, w, source=src, family=family, config=cfg)
            except H.Unsupported:
                continue
            torch.cuda.synchronize()
            assert torch.equal(y, want), f"wide stride: family {family} config {cfg} {src}: {y[0].tolist()}"
        print(f"  wide stride ok: config {cfg}")


def run_many_escapes(family):
    shim = H.Shim()
    # The last row block's escapes start at index 16384 * 1024 * 128 = 2^31.
    w = all_escaped_weight(262160, 65536)
    assert w.num_escapes == 2147614720
    x = x_ones(65, w.k)
    for cfg in configs(family):
        try:
            y = shim.linear(x, w, family=family, config=cfg)
        except H.Unsupported:
            continue
        torch.cuda.synchronize()
        bad = (y != 16.0).any(0).nonzero()
        assert bad.numel() == 0, f"2^31 escapes: family {family} config {cfg}: wrong columns {bad.flatten()[:8].tolist()}"
        print(f"  2^31 escapes ok: config {cfg}")


@pytest.mark.parametrize("family", [H.FAMILY_MMA, H.FAMILY_AUTO])
def test_wide_stride(family):
    run_wide_stride(family)


@pytest.mark.parametrize("family", [H.FAMILY_MMA, H.FAMILY_AUTO])
def test_many_escapes(family):
    run_many_escapes(family)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["mma", "tc", "auto"], default="mma")
    ap.add_argument("--only", choices=["stride", "escapes"], default=None)
    a = ap.parse_args()
    fam = {"mma": H.FAMILY_MMA, "tc": H.FAMILY_TC, "auto": H.FAMILY_AUTO}[a.family]
    if a.only != "escapes":
        run_wide_stride(fam)
        torch.cuda.empty_cache()
    if a.only != "stride":
        run_many_escapes(fam)
    print("huge tests passed")
