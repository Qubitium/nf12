# SPDX-License-Identifier: Apache-2.0
"""NF12: lossless 12-bit encoding of BF16 weights with fused CUDA kernels.

The native libraries load on first use, so ``import nf12.reference`` (pure
PyTorch) works without a compiled build.
"""

from ._library import library_path, load
from .api import NF12Weight, decode, embedding, encode, linear, linear_bf16t, pack_bf16t
from .moe import (
    NF12MoEWeight,
    encode_moe,
    fused_moe,
    moe_align,
    moe_combine,
    moe_linear,
    moe_linear_bf16t,
    moe_silu_mul,
    pack_bf16t_moe,
)
from .nn import NF12Embedding, NF12Linear

__version__ = "1.0.0"

__all__ = [
    "NF12Embedding",
    "NF12Linear",
    "NF12MoEWeight",
    "NF12Weight",
    "decode",
    "embedding",
    "encode",
    "encode_moe",
    "fused_moe",
    "library_path",
    "linear",
    "linear_bf16t",
    "load",
    "moe_align",
    "moe_combine",
    "moe_linear",
    "moe_linear_bf16t",
    "moe_silu_mul",
    "pack_bf16t",
    "pack_bf16t_moe",
]
