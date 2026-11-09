# ROCm test environment

Validated on Radeon AI PRO R9700 (`gfx1201`, RDNA4) and Instinct MI300X
(`gfx942`, CDNA3) with:

| Component | Validated version |
| --- | --- |
| ROCm SDK (`rocm`, `rocm-sdk-core`, `rocm-sdk-devel`) | 10.1.0 |
| HIP | 7.16.26385 (`hipconfig --version`: `7.16.26385-0000000`) |
| PyTorch | 2.16.0.dev20261025+rocm10.1 |
| Python | 3.14, free-threaded (`PYTHON_GIL=0`) |

The suite gates on the ISAs that expose a matrix core the kernels drive
(`_gating.py`: `gfx1200`, `gfx1201`, `gfx942`, `gfx950`). That is only a test
filter — the library itself selects a kernel from device traits. Tests whose
expectations are specific to one family (the WMMA plan and grid tests) still
skip on the other, so a run reports them as skips rather than failures.

The repository-level `tests/conftest.py` requires **ROCm >= 10.1.0 and HIP >= 7.16.26385**.
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
  PYTHONPATH=python python -m pytest tests/api tests/hip -q
```

## Where a test belongs

`tests/api` holds the contract tests for the public C, Python and torch APIs.
They are backend-neutral: the same cases run on CUDA and ROCm, and format
coverage is parametrized with the `dense_ops` and `moe_pack` fixtures in
`tests/api/conftest.py` rather than duplicated per backend. `tests/hip` holds
only what has no CUDA counterpart: the native `nf12_hip_*` entry points,
RDNA4 route selection and native graph capture. Add a shared case to
`tests/api` first; add to `tests/hip` only when the assertion needs a
HIP-only symbol or architecture.

Record the SDK, HIP, PyTorch and GPU versions with new validation or benchmark
results. Native libraries are built from source; no precompiled library is
required by this test setup.
