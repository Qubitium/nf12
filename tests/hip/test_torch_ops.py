# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
"""``torch.ops.nf12`` parity on ROCm for the core ops.

The CUDA torch extension is the same ``csrc/torch/ops.cpp``; on ROCm it links
``libnf12_hip.so`` through the ``nf12_*`` compatibility layer. These tests run
the Python API, not the C ABI.
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


def test_encode_decode_roundtrip():
    torch.manual_seed(0)
    w = (torch.randn(256, 4096, device="cuda") * 0.02).to(torch.bfloat16)
    nw = nf12.encode(w)
    assert nf12.decode(nw).view(torch.int16).equal(w.view(torch.int16))


def test_linear_matches_bf16_twin():
    torch.manual_seed(1)
    n, k = 1024, 4096
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    nw = nf12.encode(w)
    for m in (1, 4, 15, 16, 17, 64, 256, 512):
        x = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
        got = nf12.linear(x, nw)
        ref = F.linear(x, w)
        err = (got.double() - ref.double()).abs()
        tol = 2e-3 + 2e-3 * ref.double().abs()
        assert int(((err > tol) | ~torch.isfinite(got)).sum()) == 0, m


def test_embedding_matches_reference():
    torch.manual_seed(2)
    n, k = 512, 256
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    nw = nf12.encode(w)
    ids = torch.tensor([[0, 5, 511, -1, 999]], device="cuda", dtype=torch.int64)
    got = nf12.embedding(ids, nw)
    ref = F.embedding(ids.clamp(0, n - 1), w)
    valid = ((ids >= 0) & (ids < n)).unsqueeze(-1)
    ref = torch.where(valid, ref, torch.zeros_like(ref))
    assert got.view(torch.int16).equal(ref.view(torch.int16))


def test_pack_bf16t_matches_reference():
    torch.manual_seed(3)
    w = (torch.randn(128, 512, device="cuda") * 0.02).to(torch.bfloat16)
    assert nf12.pack_bf16t(w).cpu().equal(reference.pack_bf16t(w).cpu())


def test_encode_moe_matches_reference():
    torch.manual_seed(4)
    w = (torch.randn(2, 64, 128, device="cuda") * 0.02).to(torch.bfloat16)
    got = nf12.encode_moe(w)
    ref = reference.encode_moe(w)
    assert got.packed.cpu().equal(ref[0].cpu())
    assert got.escapes.cpu().equal(ref[1].cpu())
    assert got.escape_offsets.cpu().equal(ref[2].cpu())
    assert got.exp_base.cpu().equal(ref[3].cpu())
