# SPDX-License-Identifier: Apache-2.0
"""NF12: lossless 12-bit encoding of BF16 weights with fused CUDA and Metal kernels.

The top-level functions accept PyTorch tensors (CUDA or MPS) and, on macOS, MLX
arrays, which they route to :mod:`nf12.mlx`. Submodules load on first use, so
``import nf12.reference`` (pure PyTorch) works without a compiled build and
``import nf12.mlx`` works without PyTorch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # static exports for type checkers and IDEs; loaded lazily at runtime
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
    from .nn import NF12Embedding, NF12Linear, verify_weights

__version__ = "1.1.0"

_EXPORTS = {
    "library_path": "._library",
    "load": "._library",
    "NF12Weight": ".api",
    "decode": ".api",
    "embedding": ".api",
    "encode": ".api",
    "linear": ".api",
    "linear_bf16t": ".api",
    "pack_bf16t": ".api",
    "NF12MoEWeight": ".moe",
    "encode_moe": ".moe",
    "fused_moe": ".moe",
    "moe_align": ".moe",
    "moe_combine": ".moe",
    "moe_linear": ".moe",
    "moe_linear_bf16t": ".moe",
    "moe_silu_mul": ".moe",
    "pack_bf16t_moe": ".moe",
    "NF12Embedding": ".nn",
    "NF12Linear": ".nn",
    "verify_weights": ".nn",
}

__all__ = sorted(_EXPORTS)


_SUBMODULES = {"api", "moe", "nn", "reference", "mlx"}


def __getattr__(name: str):
    if name in _SUBMODULES:  # nf12.nn etc. after a plain `import nf12`
        return importlib.import_module(f".{name}", __name__)
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'nf12' has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS) | _SUBMODULES)
