# SPDX-License-Identifier: Apache-2.0
"""Host-only ABI compatibility checks, run under either CUDA or ROCm Torch."""

from __future__ import annotations

import ctypes as C
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


class Weight(C.Structure):
    _fields_ = [
        ("packed", C.c_void_p), ("escapes", C.c_void_p),
        ("num_escapes", C.c_uint32), ("n", C.c_int64),
        ("k", C.c_int64), ("exp_base", C.c_uint8),
    ]


class MoeWeight(C.Structure):
    _fields_ = [
        ("packed", C.c_void_p), ("escapes", C.c_void_p),
        ("escape_offsets", C.c_void_p), ("exp_base", C.c_void_p),
        ("num_experts", C.c_int64), ("n", C.c_int64), ("k", C.c_int64),
    ]


@pytest.fixture(scope="module")
def header_flags():
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    is_hip = torch.version.hip is not None
    home = ROCM_HOME if is_hip else CUDA_HOME
    if home is None:
        pytest.skip("active runtime toolkit headers required")
    flags = [f"-I{ROOT / 'include'}", f"-I{ROOT / 'csrc'}",
             "-isystem", str(Path(home) / "include")]
    if is_hip:
        flags += ["-DNF12_HIP", "-D__HIP_PLATFORM_AMD__"]
    return flags


def compile_and_run(tmp_path, source, header_flags, *, cxx=False):
    variable, fallback = ("CXX", "c++") if cxx else ("CC", "cc")
    compiler = shlex.split(os.environ.get(variable) or fallback)
    if not shutil.which(compiler[0]):
        pytest.skip(f"{variable} compiler required")
    path = tmp_path / ("probe.cpp" if cxx else "probe.c")
    path.write_text(source)
    executable = tmp_path / "probe"
    cmd = compiler + ["-std=c++17" if cxx else "-std=c99", "-Wall", "-Wextra", "-Werror"]
    if cxx:
        cmd += ["-pthread"]
    cmd += header_flags + [str(path), "-o", str(executable)]
    built = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, f"{shlex.join(cmd)}\n{built.stdout}{built.stderr}"
    run = subprocess.run([str(executable)], check=False, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    return run.stdout


def test_public_descriptor_layout_lp64(tmp_path, header_flags):
    if C.sizeof(C.c_void_p) != 8 or C.sizeof(C.c_long) != 8:
        pytest.skip("literal ABI fingerprint targets LP64")
    # Literal baseline, independent of both the header and ctypes declarations.
    layouts = [
        ("nf12_weight_t", Weight, 48,
         {"packed": 0, "escapes": 8, "num_escapes": 16, "n": 24, "k": 32, "exp_base": 40}),
        ("nf12_moe_weight_t", MoeWeight, 56,
         {"packed": 0, "escapes": 8, "escape_offsets": 16, "exp_base": 24,
          "num_experts": 32, "n": 40, "k": 48}),
    ]
    statements, expected = [], []
    for name, structure, size, offsets in layouts:
        assert C.sizeof(structure) == size
        assert {field: getattr(structure, field).offset for field in offsets} == offsets
        expressions = [f"sizeof({name})"] + [f"offsetof({name}, {field})" for field in offsets]
        statements += [f'printf("%zu\\n", {expression});' for expression in expressions]
        expected += [size, *offsets.values()]
    source = '#include <nf12/nf12.h>\n#include <stdio.h>\nint main(void) {\n'
    source += "\n".join(statements) + "\nreturn 0;\n}\n"
    actual = compile_and_run(tmp_path, source, header_flags)
    assert [int(value) for value in actual.split()] == expected


def test_runtime_and_codec_status_categories(tmp_path, header_flags):
    # No runtime library or GPU is needed: exercise the actual inline helpers.
    source = r'''
#include "abi_runtime.h"
#include <cassert>
#include <initializer_list>
#include <thread>

int main() {
  using namespace nf12::abi;
  using nf12::backend::kInvalidValue;
  using nf12::backend::kNoKernelImage;
#ifdef NF12_HIP
  constexpr auto allocation_error = hipErrorOutOfMemory;
#else
  constexpr auto allocation_error = cudaErrorMemoryAllocation;
#endif
  assert(last_error == cudaSuccess);
  assert(from_runtime(allocation_error) == NF12_ERROR_CUDA);
  assert(last_error == allocation_error);
  assert(from_device(cudaSuccess, false) == NF12_ERROR_UNSUPPORTED);
  assert(last_error == allocation_error);
  assert(from_device(cudaSuccess, true) == NF12_OK);
  assert(last_error == allocation_error);
  // The same numeric runtime code has different public meanings by context.
  assert(from_codec(kInvalidValue) == NF12_ERROR_INVALID_ARGUMENT);
  assert(last_error == allocation_error);
  assert(from_runtime(kInvalidValue) == NF12_ERROR_CUDA);
  assert(last_error == kInvalidValue);
  for (auto error : {cudaErrorNotSupported, kNoKernelImage}) {
    assert(from_runtime(error) == NF12_ERROR_UNSUPPORTED);
    assert(last_error == error);
    assert(from_codec(error) == NF12_ERROR_UNSUPPORTED);
    assert(last_error == error);
  }
  assert(from_codec(allocation_error) == NF12_ERROR_CUDA);
  assert(from_runtime(cudaSuccess) == NF12_OK);
  assert(from_codec(cudaSuccess) == NF12_OK);
  assert(last_error == allocation_error);
  std::thread other([] {
    assert(last_error == cudaSuccess);
    assert(from_codec(nf12::backend::kInvalidValue) == NF12_ERROR_INVALID_ARGUMENT);
    assert(last_error == cudaSuccess);
    assert(from_runtime(cudaErrorNotSupported) == NF12_ERROR_UNSUPPORTED);
    assert(last_error == cudaErrorNotSupported);
  });
  other.join();
  assert(last_error == allocation_error);
}
'''
    compile_and_run(tmp_path, source, header_flags, cxx=True)


def test_two_devices_preserve_caller_stream_and_bf16t_bits():
    from concurrent.futures import ThreadPoolExecutor

    if torch.cuda.device_count() < 2:
        pytest.skip("two visible CUDA or ROCm devices required")

    import nf12

    def run(device):
        with torch.cuda.device(device):
            producer = torch.cuda.Stream(device=device)
            consumer = torch.cuda.Stream(device=device)
            generator = torch.Generator(device=device).manual_seed(1234 + device)
            with torch.cuda.stream(producer):
                # Dyadic inputs give exact FP32 sums even when M=272 selects
                # external GEMM rather than the native BF16t launch plan.
                w = torch.randint(-4, 5, (256, 256), device=device, generator=generator).to(torch.bfloat16) / 8
                x = torch.randint(-4, 5, (272, 256), device=device, generator=generator).to(torch.bfloat16) / 8
                weight = nf12.encode(w)
                twin = nf12.pack_bf16t(w)
            consumer.wait_stream(producer)
            with torch.cuda.stream(consumer):
                decoded = nf12.decode(weight)
                results = [(nf12.linear(x[:m], weight), nf12.linear_bf16t(x[:m], twin))
                           for m in (16, 272)]
                assert decoded.device == w.device
                assert torch.equal(decoded.view(torch.int16), w.view(torch.int16))
                for actual, expected in results:
                    assert actual.device == expected.device == w.device
                    assert actual.dtype == expected.dtype == torch.bfloat16
                    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            consumer.synchronize()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, (0, 1)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU library required")
def test_grouped_expert_count_rejects_before_launch():
    import nf12

    lib = C.CDLL(str(nf12.library_path()))
    fn = lib.nf12_moe_grouped_operands
    p, i = C.c_void_p, C.c_int64
    fn.argtypes = [p, i, p, i, p, i, i, i, p, i, p, p]
    fn.restype = C.c_int
    lib.nf12_last_cuda_error.restype = C.c_int
    previous = lib.nf12_last_cuda_error()
    # Aligned sentinels are never dereferenced: expert IDs must fit in int32.
    assert fn(None, 8, None, 1, 16, 0, 2**31, 8, None, 8, 32, None) == 1
    assert lib.nf12_last_cuda_error() == previous
