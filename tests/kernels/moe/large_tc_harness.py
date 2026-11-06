# SPDX-License-Identifier: Apache-2.0
"""Standalone harness for the tcgen05 grouped-GEMM moe_linear family.

Builds csrc/moe/sm100/large_tc.cu, the shared MoE schedule/grouped sources,
csrc/device_info.cu and the sibling large_tc_shim.cu with nvcc, independently
of CMake. ctypes exposes forced moe_linear_large_tc() configs. Expert
encoding and the routing reference come from small_harness.py.

Env: NF12_MOE_TC_BUILD_DIR (private output dir when agents share the tree),
NF12_NVCC_FLAGS, NVCC.
"""

import ctypes
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSRC = os.path.join(ROOT, "csrc")
sys.path.insert(0, HERE)

from small_harness import NF12, BF16T, align, encode_moe, random_experts  # noqa: E402,F401


class Unsupported(RuntimeError):
    """The requested config cannot run on this device."""


def build(force=False):
    out_dir = os.environ.get("NF12_MOE_TC_BUILD_DIR", os.path.join(ROOT, "build", "moe_tc_tests"))
    os.makedirs(out_dir, exist_ok=True)
    lib = os.path.join(out_dir, "libmoe_tc_shim.so")
    srcs = [os.path.join(CSRC, "moe", "sm100", "large_tc.cu"), os.path.join(CSRC, "moe", "schedule.cu"),
            os.path.join(CSRC, "moe", "grouped.cu"), os.path.join(CSRC, "device_info.cu"),
            os.path.join(HERE, "large_tc_shim.cu")]
    deps = srcs + [
        os.path.join(directory, f)
        for family in ("linear", "moe")
        for directory, _, files in os.walk(os.path.join(CSRC, family))
        for f in files if f.endswith((".h", ".cuh"))
    ] + [os.path.join(ROOT, "include", "nf12", "format.cuh"), os.path.join(CSRC, "internal.h")]
    if not force and os.path.exists(lib):
        mtime = os.path.getmtime(lib)
        if all(os.path.getmtime(d) <= mtime for d in deps):
            return lib
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    arch = f"{cc}a" if cc in (100, 103, 110) else str(cc)
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    cmd = [nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "--expt-relaxed-constexpr", "-lineinfo",
           "-t", "0", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-Xlinker", "--no-undefined",
           "-I", os.path.join(ROOT, "include"), "-I", CSRC, "-o", lib, *srcs]
    extra = os.environ.get("NF12_NVCC_FLAGS")
    if extra:
        cmd[1:1] = extra.split()
    subprocess.run(cmd, check=True)
    return lib


class Lib:
    def __init__(self, lib_path=None):
        self.lib = ctypes.CDLL(lib_path or build())
        L = self.lib
        i, i64, vp, sz = ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_size_t
        L.nf12_moe_tc_supported.restype = i
        L.nf12_moe_tc_num_configs.restype = i
        L.nf12_moe_tc_config_name.restype = ctypes.c_char_p
        L.nf12_moe_tc_config_name.argtypes = [i]
        L.nf12_moe_tc_workspace_bytes.restype = sz
        L.nf12_moe_tc_workspace_bytes.argtypes = [i64, i64, i64, i64, i]
        L.nf12_moe_tc_linear.restype = i
        L.nf12_moe_tc_linear.argtypes = [i, vp, i64, vp, i64, vp, vp, i64, i, vp, vp, vp, vp, i64, i64, i64, vp, vp,
                                         i64, vp, sz, vp]
        self._ws = None

    def supported(self):
        return bool(self.lib.nf12_moe_tc_supported())

    def num_configs(self):
        return self.lib.nf12_moe_tc_num_configs()

    def config_name(self, config):
        return "auto" if config < 0 else self.lib.nf12_moe_tc_config_name(config).decode()

    def workspace(self, config, A, E, n, k):
        nbytes = self.lib.nf12_moe_tc_workspace_bytes(A, E, n, k, config)
        if self._ws is None or self._ws.numel() < nbytes:
            self._ws = torch.empty(max(nbytes, 256), dtype=torch.uint8, device="cuda")
        return self._ws, nbytes

    def moe_linear(self, config, x, enc, offsets, max_assignments, y, gather=None, gather_div=1, scatter=None,
                   row_scale=None, source=NF12, ws=None):
        """y[out(p)] = x[in(p)] @ W_e.T (* row_scale[out(p)]). x, y: 2-D views with unit column stride.
        `ws` is a (tensor, bytes) workspace that requires no initialization."""
        ptr = lambda t: None if t is None else t.data_ptr()  # noqa: E731
        weight = enc.packed if source == NF12 else enc.bf16t
        wst, nbytes = ws if ws is not None else self.workspace(config, max_assignments, enc.num_experts, enc.n,
                                                                enc.k)
        err = self.lib.nf12_moe_tc_linear(
            config, x.data_ptr(), x.stride(0), ptr(gather), gather_div, ptr(scatter), offsets.data_ptr(),
            max_assignments, source, weight.data_ptr(), ptr(enc.escapes) if source == NF12 else None,
            ptr(enc.escape_offsets) if source == NF12 else None, ptr(enc.exp_base) if source == NF12 else None,
            enc.num_experts, enc.n, enc.k, ptr(row_scale), y.data_ptr(), y.stride(0), wst.data_ptr(), nbytes,
            torch.cuda.current_stream().cuda_stream)
        if err == 801:
            raise Unsupported(f"config {config} unsupported on this device")
        if err != 0:
            raise RuntimeError(f"moe_linear_large_tc(config={config}, A={max_assignments}, E={enc.num_experts}, "
                               f"n={enc.n}, k={enc.k}) failed: cuda error {err}")
        return y
