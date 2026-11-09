# SPDX-License-Identifier: Apache-2.0
"""Gate every suite before collection on unsupported ROCm/PyTorch runtimes."""

from __future__ import annotations

import re

import pytest

# Validated on R9700: ROCm SDK 10.1.0 / HIP 7.16.26385. See README.md.
MIN_ROCM = (10, 1, 0)
MIN_HIP = (7, 16, 26385)


def _version(value):
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:$|[-+])", str(value))
    return tuple(int(part or 0) for part in match.groups()) if match else None


def pytest_configure(config):
    import torch

    hip = torch.version.hip
    if hip is None:  # Existing per-module skips handle CPU/CUDA environments.
        return
    rocm = getattr(torch.version, "rocm", None)
    if rocm is None:
        tag = re.search(r"\+rocm(\d+\.\d+(?:\.\d+)?)", torch.__version__)
        rocm = tag.group(1) if tag else None
    # HIP has its own release numbering; never interpret it as a ROCm version.
    rocm_version, hip_version = _version(rocm), _version(hip)
    if (rocm_version is None or rocm_version < MIN_ROCM
            or hip_version is None or hip_version < MIN_HIP):
        raise pytest.UsageError(
            "NF12 HIP tests require ROCm >= 10.1.0 and HIP >= 7.16.26385; "
            f"detected ROCm {rocm or 'unknown'}, HIP {hip}, torch {torch.__version__}. "
            "Install a supported ROCm PyTorch build before running tests/hip."
        )
