# SPDX-License-Identifier: Apache-2.0
"""Architecture gating for the HIP suite.

The C++ dispatcher selects kernels from device *traits* (wave width, matrix
core, XCCs, smem, L2), never from an ISA string. The tests cannot call those
traits directly, so they gate on the set of ISAs that expose the same matrix
core generation the suite was written against: RDNA4 WMMA (gfx1200/gfx1201) and
CDNA3/CDNA4 MFMA (gfx942/gfx950). Adding an ISA here is only a test gate; the
runtime decision is still trait-based.
"""

from __future__ import annotations

import torch

MATRIX_CORE_ARCHS = ("gfx1200", "gfx1201", "gfx942", "gfx950")


def arch() -> str:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
