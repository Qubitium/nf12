# SPDX-License-Identifier: Apache-2.0
"""PDL input dependencies through the Python, Torch and native linear APIs.

Run: uv run --group test pytest tests/kernels/linear/test_static_weights.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import small_m_harness as h  # noqa: E402

import nf12  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="PDL requires SM90+",
)


@pytest.fixture(scope="module")
def lib():
    return h.Lib()


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 144, 7168),
        (1, 400, 1280),
        (8, 400, 1280),
        (16, 400, 1280),
        (1, 4096, 4096),
        (8, 4096, 4096),
        (16, 16640, 4096),
        (3, 6160, 1280),
        (8, 1536, 128),
        (8, 2304, 1536),
        (8, 7168, 768),
        (8, 6144, 7168),
        (5, 128, 7168),
        (8, 256, 8192),
    ],
    ids=[
        "m1-narrow", "m1-small", "m8-small", "m16-small", "m1-wide", "m8-wide", "m16-wide",
        "m3-padded", "m8-short-k", "m8-medium", "m8-short-wide", "m8-long-wide",
        "m5-cluster-narrow", "m8-cluster-narrow",
    ],
)
@pytest.mark.parametrize(
    "produced,static_weights",
    [("packed", False), ("escapes", False), ("weights", False), ("bf16t", False),
     ("activation", False), ("activation", True), ("bias", True)],
)
def test_pdl_linear_dependencies(lib, m, n, k, produced, static_weights):
    enc = h.encode(h.random_weight(n, k, seed=42, escapes=0.002))
    assert enc.num_escapes > 0
    weight = nf12.NF12Weight(enc.packed.clone(), enc.escapes_u4.clone(), enc.exp_base)
    bf16t = enc.bf16t.clone()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    inputs = {
        "packed": [(enc.packed, weight.packed)],
        "escapes": [(enc.escapes_u4, weight.escapes)],
        "weights": [(enc.packed, weight.packed), (enc.escapes_u4, weight.escapes)],
        "bf16t": [(enc.bf16t, bf16t)],
        "activation": [(x.clone(), x)],
        "bias": [(bias.clone(), bias)],
    }[produced]

    # Omit the keyword on dynamic paths to exercise the published default.
    kwargs = {"static_weights": True} if static_weights else {}
    nf12._library.load()

    def linear():
        if produced == "bf16t":
            return torch.ops.nf12.linear_bf16t(x, bf16t, bias)
        return nf12.linear(x, weight, bias, **kwargs)

    expected = linear()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            linear()
    stream.synchronize()

    # Capture removes host launch latency from the producer/consumer overlap.
    # Only one producer CTA occupies the device during its delayed writes.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        lib.pdl_copy(*inputs[0], *(inputs[1] if len(inputs) == 2 else (None, None)))
        actual = linear()

    for _ in range(3):
        for _, dst in inputs:
            dst.zero_()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        for src, dst in inputs:
            assert torch.equal(src, dst), "PDL producer did not finish its copy"
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
            f"{produced=}, {static_weights=}, {m=}: linear read a pending PDL input"
        )
