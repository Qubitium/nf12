# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the shared HIP test gate, including early failure."""

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

GATE = Path(__file__).parents[1] / "conftest.py"
configure = runpy.run_path(str(GATE))["pytest_configure"]


@pytest.mark.parametrize("rocm,hip,accepted", [
    ("10.1.0", "7.16.26385", True),
    ("10.1", "7.16.26385-0000000", True),
    ("10.2.0", "7.17.0", True),
    ("11.0.0", "8.0.0", True),
    ("10.0.9", "7.16.26385", False),
    ("9.9.0", "7.17.0", False),
    ("10.1.0", "7.16.26384", False),
    ("10.1.0", "7.15.99999", False),
    ("10.1.0", "7.16", False),
    ("invalid", "7.16.26385", False),
    ("10.1.0", "invalid", False),
    (None, "7.16.26385", False),
])
def test_version_boundary(monkeypatch, rocm, hip, accepted):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        version=SimpleNamespace(rocm=rocm, hip=hip), __version__="2.16.0"))
    if accepted:
        configure(None)
    else:
        with pytest.raises(pytest.UsageError, match="require ROCm >= 10.1.0"):
            configure(None)


@pytest.mark.parametrize("tag,accepted", [
    ("2.16.0+rocm10.1", True),
    ("2.16.0+rocm10.1.0", True),
    ("2.16.0+rocm10.0", False),
])
def test_wheel_tag_fallback(monkeypatch, tag, accepted):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        version=SimpleNamespace(hip="7.16.26385"), __version__=tag))
    if accepted:
        configure(None)
    else:
        with pytest.raises(pytest.UsageError):
            configure(None)


def test_runtime_release_takes_precedence(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        version=SimpleNamespace(rocm="10.0.0", hip="7.16.26385"),
        __version__="2.16.0+rocm10.1"))
    with pytest.raises(pytest.UsageError):
        configure(None)


def test_non_rocm_is_unaffected(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        version=SimpleNamespace(hip=None)))
    configure(None)


def test_unsupported_runtime_fails_before_test_import(tmp_path):
    (tmp_path / "conftest.py").write_text(GATE.read_text())
    (tmp_path / "torch.py").write_text(
        'from types import SimpleNamespace\n'
        'version = SimpleNamespace(rocm="10.0.0", hip="7.16.26385")\n'
        '__version__ = "2.16.0+rocm10.0"\n')
    (tmp_path / "test_probe.py").write_text(
        'from pathlib import Path\nPath("collected").touch()\n'
        'def test_probe(): pass\n')
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", str(tmp_path)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout + result.stderr
    assert "detected ROCm 10.0.0" in result.stderr
    assert not (tmp_path / "collected").exists()
