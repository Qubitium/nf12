# SPDX-License-Identifier: Apache-2.0
"""The raw HIP benchmark must launch and time on its selected device."""

from pathlib import Path

import pytest
import torch


def test_linear_benchmark_selects_requested_device(monkeypatch, capsys):
    if torch.version.hip is None or torch.cuda.device_count() < 2:
        pytest.skip("requires two ROCm GPUs")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "bench"))
    import hip_linear

    monkeypatch.setattr(hip_linear.common, "SHAPES", [
        hip_linear.common.Shape(64, 16, None, "device-selection regression"),
    ])
    monkeypatch.setattr("sys.argv", [
        "hip_linear.py", "--device", "cuda:1", "--m", "1,16", "--copies", "2",
    ])
    bind = hip_linear._bind

    def bind_on_target(path):
        # Check before any raw launch: wrong-device kernels can corrupt results.
        assert torch.cuda.current_device() == 1
        return bind(path)

    monkeypatch.setattr(hip_linear, "_bind", bind_on_target)
    with torch.cuda.device(0):
        hip_linear.main()
        assert torch.cuda.current_device() == 1
    rows = [line.split("|") for line in capsys.readouterr().out.splitlines()
            if line.lstrip().startswith("64 ")]
    assert len(rows) == 2
    assert all(int(row[-1].split()[0]) == 0 for row in rows)
