# SPDX-License-Identifier: Apache-2.0
"""Format parametrization for the shared CUDA/ROCm API contract tests."""

import pytest


@pytest.fixture(params=["nf12", "bf16t"])
def dense_ops(request):
    import nf12

    return ((nf12.encode, nf12.linear) if request.param == "nf12"
            else (nf12.pack_bf16t, nf12.linear_bf16t))


@pytest.fixture(params=["nf12", "bf16t"])
def moe_pack(request):
    import nf12

    return nf12.encode_moe if request.param == "nf12" else nf12.pack_bf16t_moe
