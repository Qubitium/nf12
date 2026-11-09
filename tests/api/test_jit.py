# SPDX-License-Identifier: Apache-2.0
"""JIT cache identity checks; no GPU or native build required."""

import pytest
from nf12 import _jit


@pytest.mark.parametrize("hip", [False, True])
def test_hipflags_cache_identity(monkeypatch, tmp_path, hip):
    (tmp_path / "CMakeLists.txt").write_text("# test source\n")
    monkeypatch.setattr(_jit, "_is_hip", lambda: hip)
    monkeypatch.setattr(_jit, "_hip_compiler", lambda: "/test/amdclang++")
    archs = ["gfx1201"] if hip else [89]
    monkeypatch.delenv("HIPFLAGS", raising=False)
    original = _jit._cache_key(tmp_path, archs)
    monkeypatch.setenv("HIPFLAGS", "-fno-fast-math -ffp-contract=off")
    changed = _jit._cache_key(tmp_path, archs)
    assert (changed != original) == hip
    monkeypatch.delenv("HIPFLAGS")
    assert _jit._cache_key(tmp_path, archs) == original
