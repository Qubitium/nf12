# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the backend-neutral API suites."""

import torch


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def assert_bitwise(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(bits(a), bits(b))
