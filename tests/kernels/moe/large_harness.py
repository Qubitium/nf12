# SPDX-License-Identifier: Apache-2.0
"""Standalone harness for the grouped (MoE) prefill kernel families.

Builds csrc/moe/{mma,sm120,sm90}/large_*.cu and the sibling large_shim.cu
with nvcc, independently of CMake. Expert encoding uses the torch encoder in
tests/kernels/linear/large_m_harness.py (SPEC.md 9.1, local escape indices).
Routing uses a torch reference of moe_align (SPEC.md 9.2). ctypes exposes
forced kernel configs.
"""

import ctypes
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOE = os.path.join(ROOT, "csrc", "moe")
sys.path.insert(0, os.path.join(ROOT, "tests", "kernels", "linear"))
import large_m_harness as LH  # noqa: E402

FAMILY_MMA = 0
FAMILY_SM120 = 1
FAMILY_SM90 = 2
FAMILIES = {"mma": FAMILY_MMA, "sm120": FAMILY_SM120, "sm90": FAMILY_SM90}


class Unsupported(RuntimeError):
    """The requested family/config cannot run on this device."""


def build(force=False):
    """Compile the shim library for the local GPU and return its path.

    NF12_MOE_BUILD_DIR overrides the output directory (concurrent users),
    NF12_NVCC_FLAGS adds nvcc flags, NVCC picks the compiler.
    """
    out_dir = os.environ.get("NF12_MOE_BUILD_DIR", os.path.join(ROOT, "build", "moe_large_tests"))
    os.makedirs(out_dir, exist_ok=True)
    lib = os.path.join(out_dir, "libmoe_large_shim.so")
    srcs = [os.path.join(MOE, family, name) for family, name in
            (("sm80", "large_mma.cu"), ("sm120", "large_sm120.cu"), ("sm90", "large_sm90.cu"))]
    srcs.append(os.path.join(HERE, "large_shim.cu"))
    deps = srcs + [
        os.path.join(directory, f)
        for family in ("moe", "linear")
        for directory, _, files in os.walk(os.path.join(ROOT, "csrc", family))
        for f in files if f.endswith((".h", ".cuh"))
    ] + [os.path.join(ROOT, "include", "nf12", "format.cuh"), os.path.join(ROOT, "csrc", "internal.h")]
    if not force and os.path.exists(lib):
        mtime = os.path.getmtime(lib)
        if all(os.path.getmtime(d) <= mtime for d in deps):
            return lib
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    cmd = [
        nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-lineinfo", "--expt-relaxed-constexpr",
        *LH._arch_flags(), "-I", os.path.join(ROOT, "include"), "-I", os.path.join(ROOT, "csrc"),
        "-o", lib, *srcs,
    ]
    extra = os.environ.get("NF12_NVCC_FLAGS")
    if extra:
        cmd[1:1] = extra.split()
    subprocess.run(cmd, check=True)
    return lib


class MoeWeights:
    """E experts [E, n, k] bf16 -> NF12 (SPEC.md 9.1) plus the bf16t twin."""

    def __init__(self, w):
        assert w.dim() == 3 and w.dtype == torch.bfloat16
        self.E, self.n, self.k = w.shape
        packs = [LH.NF12Packed(w[e]) for e in range(self.E)]
        self.packed = torch.stack([p.packed for p in packs]).contiguous()
        self.bf16t = torch.stack([p.bf16t for p in packs]).contiguous()
        counts = [p.num_escapes for p in packs]
        self.num_escapes = sum(counts)
        offs = [0]
        for c in counts:
            offs.append(offs[-1] + c)
        self.escape_offsets = torch.tensor(offs, dtype=torch.int64, device=w.device)
        self.escapes = (torch.cat([p.escapes[:c] for p, c in zip(packs, counts)]) if self.num_escapes
                        else None)
        self.exp_base = torch.tensor([p.exp_base for p in packs], dtype=torch.uint8, device=w.device)

    def escape_fraction(self):
        return self.num_escapes / (self.E * self.n * self.k / 8)


def align(topk_ids, num_experts):
    """Torch reference of moe_align: (sorted [A] int32, expert_offsets [E + 1] int64)."""
    ids = topk_ids.flatten().to(torch.int64)
    valid = (ids >= 0) & (ids < num_experts)
    key = torch.where(valid, ids, torch.full_like(ids, num_experts))
    order = torch.sort(key, stable=True).indices
    counts = torch.bincount(ids[valid], minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=ids.device)
    offsets[1:] = torch.cumsum(counts, 0)
    return order.to(torch.int32), offsets


class Shim:
    def __init__(self, lib_path=None):
        self.lib = ctypes.CDLL(lib_path or build())
        L = self.lib
        i64, vp, ci = ctypes.c_int64, ctypes.c_void_p, ctypes.c_int
        L.nf12_test_moe_large.restype = ci
        L.nf12_test_moe_large.argtypes = [
            ci, ci,                  # family, config
            vp, i64, vp, i64, vp,    # x, ldx, gather, gather_div, scatter
            vp, i64,                 # expert_offsets, max_assignments
            ci, vp, vp, vp, vp,      # source, weight, escapes, escape_offsets, exp_base
            i64, i64, i64,           # E, n, k
            vp, vp, i64, vp,         # row_scale, y, ldy, stream
        ]
        L.nf12_test_moe_supported.restype = ci
        L.nf12_test_moe_supported.argtypes = [ci, i64]
        L.nf12_test_moe_num_configs.restype = ci
        L.nf12_test_moe_num_configs.argtypes = [ci]
        L.nf12_test_moe_config_name.restype = ctypes.c_char_p
        L.nf12_test_moe_config_name.argtypes = [ci, ci]
        L.nf12_test_moe_heuristic.restype = ci
        L.nf12_test_moe_heuristic.argtypes = [ci, i64, i64, i64, i64]

    def supported(self, family, num_experts=1):
        return bool(self.lib.nf12_test_moe_supported(family, num_experts))

    def num_configs(self, family):
        return self.lib.nf12_test_moe_num_configs(family)

    def config_name(self, family, config):
        return "auto" if config < 0 else self.lib.nf12_test_moe_config_name(family, config).decode()

    def heuristic(self, family, max_assignments, num_experts, n, k):
        return self.lib.nf12_test_moe_heuristic(family, max_assignments, num_experts, n, k)

    def run(self, x, mw, expert_offsets, max_assignments, y, gather=None, gather_div=1, scatter=None,
            row_scale=None, source="nf12", family=FAMILY_MMA, config=-1):
        """y[out(p)] = x[in(p)] . W_e^T (* row_scale[out(p)]) for unit-column-stride x/y."""
        assert x.stride(1) == 1 and y.stride(1) == 1 and x.shape[1] == mw.k and y.shape[1] == mw.n
        ptr = lambda t: t.data_ptr() if t is not None else None  # noqa: E731
        nf12 = source == "nf12"
        rc = self.lib.nf12_test_moe_large(
            family, config, x.data_ptr(), x.stride(0), ptr(gather), gather_div, ptr(scatter),
            expert_offsets.data_ptr(), max_assignments, 0 if nf12 else 1,
            (mw.packed if nf12 else mw.bf16t).data_ptr(), ptr(mw.escapes) if nf12 else None,
            mw.escape_offsets.data_ptr() if nf12 else None, mw.exp_base.data_ptr() if nf12 else None,
            mw.E, mw.n, mw.k, ptr(row_scale), y.data_ptr(), y.stride(0), torch.cuda.current_stream().cuda_stream)
        if rc == 801:  # cudaErrorNotSupported
            raise Unsupported(f"family {family} config {config} unsupported")
        if rc != 0:
            raise RuntimeError(f"moe large kernel failed: cudaError {rc}")
        return y
