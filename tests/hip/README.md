# ROCm test environment

Validated on Radeon AI PRO R9700 (`gfx1201`) with:

| Component | Validated version |
| --- | --- |
| ROCm SDK (`rocm`, `rocm-sdk-core`, `rocm-sdk-devel`) | 10.1.0 |
| HIP | 7.16.26385 (`hipconfig --version`: `7.16.26385-0000000`) |
| PyTorch | 2.16.0.dev20261025+rocm10.1 |
| Python | 3.14, free-threaded (`PYTHON_GIL=0`) |

The shared `conftest.py` requires **ROCm >= 10.1.0 and HIP >= 7.16.26385**.
On a ROCm PyTorch build it fails before collecting test modules if either
version is older or cannot be identified. It reads `torch.version.rocm`
(or the wheel's `+rocm` tag) separately from `torch.version.hip`: HIP 7.16
is the component version accompanying ROCm 10.1, not a ROCm 7.16 release.
Newer versions pass the minimum gate; that does not imply they have been
validated. CUDA/CPU builds retain the HIP modules' normal skip behavior.

Run from the repository root with a matching ROCm PyTorch environment:

```sh
hipconfig --version
python -c 'import torch; print(torch.__version__, getattr(torch.version, "rocm", None), torch.version.hip)'
PYTHON_GIL=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=python python -m pytest tests/hip -q
```

Record the SDK, HIP, PyTorch and GPU versions with new validation or benchmark
results. Native libraries are built from source; no precompiled library is
required by this test setup.
