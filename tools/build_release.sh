#!/usr/bin/env bash
# Run in the CUDA Rocky Linux 8 release container, with the checkout at /io.
set -euo pipefail

: "${WHEEL_ARCH:?Set WHEEL_ARCH to x86_64 or aarch64}"
test "$(uname -m)" = "$WHEEL_ARCH"
export NF12_BUILD_NATIVE=ON
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export PATH="/tmp/nf12-build/bin:$PATH"

uv venv --python 3.10 /tmp/nf12-build
uv pip install --python /tmp/nf12-build/bin/python \
  scikit-build-core==1.1.2 cmake==4.4.4 ninja==1.13.2 \
  auditwheel==6.8.2 patchelf==0.19.3.0 tomli==2.5.0
uv pip install --python /tmp/nf12-build/bin/python \
  'torch==2.10.0+cu130' --index https://download.pytorch.org/whl/cu130

# Use the package's complete target list, not the build host's GPU.
NF12_CUDA_ARCHS=$(uv run --no-project --python /tmp/nf12-build/bin/python \
  python -c 'import tomli; print(tomli.load(open("pyproject.toml", "rb"))["tool"]["scikit-build"]["cmake"]["define"]["NF12_CUDA_ARCHS"]["default"])')
export NF12_CUDA_ARCHS
CUDA_RUNTIME_LICENSE=$(uv run --no-project --python /tmp/nf12-build/bin/python \
  python -c 'from importlib.metadata import distribution; d = distribution("nvidia-cuda-runtime"); print(d.locate_file(next(p for p in d.files if p.name == "License.txt")))')

uv build --wheel --no-sources --no-build-isolation \
  --python /tmp/nf12-build/bin/python --out-dir /tmp/nf12-unrepaired \
  --config-setting build-dir=/tmp/nf12-native-build \
  --config-setting "cmake.define.NF12_CUDA_RUNTIME_LICENSE=$CUDA_RUNTIME_LICENSE"

# Torch supplies its own libraries. Bundle our CUDA runtime so loading NF12
# does not depend on a system toolkit or on Torch's CUDA runtime search paths.
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
uv run --no-project --python /tmp/nf12-build/bin/python \
  auditwheel repair --plat "manylinux_2_28_${WHEEL_ARCH}" \
  --exclude libtorch_cpu.so --exclude libtorch_cuda.so \
  --exclude libc10.so --exclude libc10_cuda.so \
  --wheel-dir /dist /tmp/nf12-unrepaired/*.whl
uv run --no-project --python /tmp/nf12-build/bin/python \
  python tools/check_release_wheel.py /dist/*.whl --arches "$NF12_CUDA_ARCHS" \
  --platform "manylinux_2_28_${WHEEL_ARCH}"

# Exercise the installed artifact, not the source tree, without a GPU or JIT.
uv pip install --python /tmp/nf12-build/bin/python --no-deps /dist/*.whl
uv run --no-project --python /tmp/nf12-build/bin/python \
  python tools/check_release_wheel.py --load

# Reuse the same wheel on a newer Python without keeping two Torch installs.
rm -rf /tmp/nf12-build /tmp/nf12-native-build /tmp/nf12-unrepaired
uv venv --python 3.14 /tmp/nf12-smoke
uv pip install --python /tmp/nf12-smoke/bin/python \
  'torch==2.10.0+cu130' --index https://download.pytorch.org/whl/cu130
uv pip install --python /tmp/nf12-smoke/bin/python --no-deps /dist/*.whl
uv run --no-project --python /tmp/nf12-smoke/bin/python \
  python tools/check_release_wheel.py --load
