# SPDX-License-Identifier: Apache-2.0
"""Standalone harness for the MoE decode kernels and moe_linear dispatch.

Builds csrc/moe/{small,dispatch,schedule}.cu, csrc/linear/small_m.cu,
csrc/device_info.cu and the sibling small_shim.cu with nvcc, independently of
CMake. Uses python/nf12/reference.py for expert encoding and ctypes for
forced kernel configs.

NF12_MOE_WITH_LARGE=1 links the grouped-GEMM families under csrc/moe and their
linear dependencies. The default diagnostic build shadows their headers
with stand-ins returning cudaErrorNotSupported. Dispatch above the crossover
raises LargeStub.
"""

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSRC = os.path.join(ROOT, "csrc")
sys.path.insert(0, os.path.join(ROOT, "python"))

from nf12 import reference  # noqa: E402

NF12 = 0
BF16T = 1
WITH_LARGE = os.environ.get("NF12_MOE_WITH_LARGE", "0") == "1"


@dataclass(frozen=True)
class Config:
    mt: int
    warps: int
    kc: int
    slice: int = 0
    pdl: int = 0
    kp: int = 1  # K parts per CTA (below SM90)

    def __str__(self):
        kp = f"_kp{self.kp}" if self.kp != 1 else ""
        return f"mt{self.mt}_w{self.warps}_kc{self.kc}_s{self.slice}_p{self.pdl}{kp}"


AUTO = Config(0, 0, 0, 0, 0)  # production dispatch (nf12::detail::moe_linear)
LARGE_MMA = Config(-1, 0, 0)  # grouped-GEMM families forced, heuristic configs
LARGE_SM120 = Config(-2, 0, 0)
LARGE_TC = Config(-3, 0, 0)


class LargeStub(RuntimeError):
    """Dispatch chose a grouped-GEMM family but the library links the stand-in."""


_STUB_HEADERS = {
    "sm80/large_mma.h": """#pragma once
#include "internal.h"
namespace nf12::detail {
size_t moe_linear_large_mma_workspace_bytes(int64_t, int64_t, int64_t, int64_t, const DeviceInfo&, int config = -1);
cudaError_t moe_linear_large_mma(const MoeLinearArgs&, int config = -1);
}
""",
    "sm120/large_sm120.h": """#pragma once
#include "internal.h"
namespace nf12::detail {
bool moe_large_sm120_supported(const DeviceInfo&, int64_t);
size_t moe_linear_large_sm120_workspace_bytes(int64_t, int64_t, int64_t, int64_t, const DeviceInfo&, int config = -1);
cudaError_t moe_linear_large_sm120(const MoeLinearArgs&, int config = -1);
}
""",
    "sm90/large_sm90.h": """#pragma once
#include "internal.h"
namespace nf12::detail {
bool moe_large_sm90_supported(const DeviceInfo&, int64_t);
size_t moe_linear_large_sm90_workspace_bytes(int64_t, int64_t, int64_t, int64_t, const DeviceInfo&, int config = -1);
cudaError_t moe_linear_large_sm90(const MoeLinearArgs&, int config = -1);
}
""",
    "sm100/large_tc.h": """#pragma once
#include "internal.h"
namespace nf12::detail {
bool moe_large_tc_supported(const DeviceInfo&);
size_t moe_large_tc_workspace_bytes(int64_t, int64_t, int64_t, int64_t, const DeviceInfo&, int config = -1);
cudaError_t moe_linear_large_tc(const MoeLinearArgs&, int config = -1);
}
""",
}

_STUB_SOURCE = """#include "moe/sm80/large_mma.h"
#include "moe/sm120/large_sm120.h"
#include "moe/sm90/large_sm90.h"
#include "moe/sm100/large_tc.h"
namespace nf12::detail {
size_t moe_linear_large_mma_workspace_bytes(int64_t, int64_t, int64_t, int64_t, const DeviceInfo&, int) { return 0; }
cudaError_t moe_linear_large_mma(const MoeLinearArgs&, int) { return cudaErrorNotSupported; }
bool moe_large_sm120_supported(const DeviceInfo&, int64_t) { return false; }
size_t moe_linear_large_sm120_workspace_bytes(int64_t, int64_t, int64_t, int64_t, const DeviceInfo&, int) { return 0; }
cudaError_t moe_linear_large_sm120(const MoeLinearArgs&, int) { return cudaErrorNotSupported; }
bool moe_large_sm90_supported(const DeviceInfo&, int64_t) { return false; }
size_t moe_linear_large_sm90_workspace_bytes(int64_t, int64_t, int64_t, int64_t, const DeviceInfo&, int) { return 0; }
cudaError_t moe_linear_large_sm90(const MoeLinearArgs&, int) { return cudaErrorNotSupported; }
bool moe_large_tc_supported(const DeviceInfo&) { return false; }
size_t moe_large_tc_workspace_bytes(int64_t, int64_t, int64_t, int64_t, const DeviceInfo&, int) { return 0; }
cudaError_t moe_linear_large_tc(const MoeLinearArgs&, int) { return cudaErrorNotSupported; }
}
"""


def _arch_flags():
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    suffix = "a" if cc in (90, 100, 103, 110, 120, 121) else ""
    return [f"-gencode=arch=compute_{cc}{suffix},code=sm_{cc}{suffix}"]


def build(force=False):
    """Compile the shim library for the local GPU and return its path."""
    out_dir = os.path.join(ROOT, "build", "moe_small_tests" + ("_full" if WITH_LARGE else ""))
    os.makedirs(out_dir, exist_ok=True)
    lib = os.path.join(out_dir, "libmoe_small_shim.so")
    includes = ["-I", os.path.join(ROOT, "include"), "-I", CSRC]
    if WITH_LARGE:
        # The linear decode-then-GEMM route links the codec's decode kernels.
        srcs = sorted(
            os.path.join(directory, f)
            for family in ("linear", "moe", "codec")
            for directory, _, files in os.walk(os.path.join(CSRC, family))
            for f in files if f.endswith(".cu")
        )
    else:
        srcs = [os.path.join(CSRC, "linear", "small_m.cu")] + [
            os.path.join(CSRC, "moe", f) for f in ("small.cu", "dispatch.cu", "schedule.cu")]
        stub_inc = os.path.join(out_dir, "include")
        for name, text in _STUB_HEADERS.items():
            os.makedirs(os.path.dirname(os.path.join(stub_inc, "moe", name)), exist_ok=True)
            with open(os.path.join(stub_inc, "moe", name), "w") as f:
                f.write(text)
        stub = os.path.join(out_dir, "large_stub.cu")
        with open(stub, "w") as f:
            f.write(_STUB_SOURCE)
        srcs.append(stub)
        includes[2:2] = ["-I", stub_inc]  # before csrc, so the stand-in headers win
    srcs += [os.path.join(CSRC, "device_info.cu"), os.path.join(HERE, "small_shim.cu")]
    deps = srcs + [
        os.path.join(directory, f)
        for family in ("linear", "moe", "codec")
        for directory, _, files in os.walk(os.path.join(CSRC, family))
        for f in files if f.endswith((".h", ".cuh"))
    ] + [os.path.join(ROOT, "include", "nf12", "format.cuh"), os.path.join(CSRC, "internal.h")]
    if not force and os.path.exists(lib):
        mtime = os.path.getmtime(lib)
        if all(os.path.getmtime(d) <= mtime for d in deps):
            return lib
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    cmd = [nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "--expt-relaxed-constexpr", "-lineinfo",
           *_arch_flags(), *includes, "-o", lib, *srcs]
    extra = os.environ.get("NF12_NVCC_FLAGS")
    if extra:
        cmd[1:1] = extra.split()
    if WITH_LARGE:
        cmd[1:1] = ["-t", "0", "-Xlinker", "--no-undefined"]
    subprocess.run(cmd, check=True)
    return lib


@dataclass
class MoeEncoded:
    w: torch.Tensor  # distinct BF16 experts [D, n, k], expert e is w[e % D]
    packed: torch.Tensor  # uint8 [E, n * k * 3 / 2]
    escapes: torch.Tensor  # int16 [sum, 8]
    escape_offsets: torch.Tensor  # int64 [E + 1]
    exp_base: torch.Tensor  # uint8 [E]
    bf16t: torch.Tensor  # uint8 [E, n * 2k]
    num_experts: int
    n: int
    k: int

    @property
    def num_escapes(self):
        return int(self.escape_offsets[-1] - self.escape_offsets[0])

    def expert(self, e):
        return self.w[e % self.w.shape[0]]


def encode_moe(w, exp_bases=None, escape_pad=0, with_bf16t=True, num_experts=None):
    """Encodes each expert of w [D, n, k] standalone (SPEC.md 9.1).
    escape_pad: dummy escape rows before expert 0's, so escape_offsets[0] != 0.
    num_experts: tile the D encoded experts to this many (benchmarks: DRAM
    traffic does not depend on content)."""
    d, n, k = w.shape
    E = d if num_experts is None else num_experts
    packed, escs, bases, bf16t = [], [], [], []
    for e in range(d):
        base = None if exp_bases is None else int(exp_bases[e])
        p, esc, b = reference.encode(w[e], base)
        packed.append(p.reshape(-1))
        escs.append(esc.reshape(-1, 8) if esc.numel() else torch.zeros(0, 8, dtype=torch.int16, device=w.device))
        bases.append(b)
        if with_bf16t:
            bf16t.append(reference.pack_bf16t(w[e]).reshape(-1))
    idx = [e % d for e in range(E)]
    offs = [escape_pad]
    for i in idx:
        offs.append(offs[-1] + escs[i].shape[0])
    pad = torch.full((escape_pad, 8), 0x7fc0, dtype=torch.int16, device=w.device)
    escapes = torch.cat([pad] + [escs[i] for i in idx]).contiguous()
    if escapes.shape[0] == 0:
        escapes = torch.zeros(1, 8, dtype=torch.int16, device=w.device)
    return MoeEncoded(
        w, torch.stack([packed[i] for i in idx]).contiguous(), escapes,
        torch.tensor(offs, dtype=torch.int64, device=w.device),
        torch.tensor([bases[i] for i in idx], dtype=torch.uint8, device=w.device),
        torch.stack([bf16t[i] for i in idx]).contiguous() if with_bf16t else None, E, n, k)


def random_experts(E, n, k, seed=0, escapes=0.0, device="cuda"):
    """Gaussian BF16 experts with an `escapes` fraction of out-of-window values."""
    g = torch.Generator(device=device).manual_seed(seed)
    w = (torch.randn(E, n, k, generator=g, device=device) * 0.02).to(torch.bfloat16)
    if escapes > 0:
        mask = torch.rand(E, n, k, generator=g, device=device) < escapes
        big = torch.where(torch.rand(E, n, k, generator=g, device=device) < 0.5, 3.0e4, -1.0e-30)
        w = torch.where(mask, big.to(torch.bfloat16), w)
    return w


def align(topk_ids, E, pad=0):
    """Torch reference of moe_align: stable sort of assignment ids by expert,
    ids outside [0, E) dropped. Positions >= off[E] (and `pad` extra) hold
    out-of-range garbage so any read of them shows."""
    flat = topk_ids.reshape(-1).to(torch.int64)
    valid = (flat >= 0) & (flat < E)
    key = torch.where(valid, flat, torch.full_like(flat, E))
    order = torch.sort(key, stable=True).indices
    count = int(valid.sum())
    sorted_ids = torch.full((flat.numel() + pad,), -(1 << 30), dtype=torch.int32, device=flat.device)
    sorted_ids[:count] = order[:count].to(torch.int32)
    hist = torch.bincount(key[valid], minlength=E)
    offsets = torch.zeros(E + 1, dtype=torch.int64, device=flat.device)
    offsets[1:] = torch.cumsum(hist, 0)
    return sorted_ids, offsets


class Lib:
    def __init__(self, lib_path=None):
        self.lib = ctypes.CDLL(lib_path or build())
        L = self.lib
        i, i64, vp, sz = ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_size_t
        L.nf12_moe_small_supported.restype = i
        L.nf12_moe_small_supported.argtypes = [i, i, i, i, i, i, i64, i64, i64]
        L.nf12_moe_small_heuristic.restype = None
        L.nf12_moe_small_heuristic.argtypes = [i64, i64, i64, i64, ctypes.POINTER(ctypes.c_int)]
        L.nf12_moe_workspace_bytes.restype = sz
        L.nf12_moe_workspace_bytes.argtypes = [i64, i64, i64, i64]
        L.nf12_moe_large_workspace_bytes.restype = sz
        L.nf12_moe_large_workspace_bytes.argtypes = [i, i64, i64, i64, i64]
        L.nf12_moe_linear.restype = i
        L.nf12_moe_linear.argtypes = [i, i, i, i, i, i, vp, i64, vp, i64, vp, vp, i64, i, vp, vp, vp, vp, i64, i64, i64,
                                      vp, vp, i64, vp, sz, vp]
        L.nf12_moe_schedule.restype = i
        L.nf12_moe_schedule.argtypes = [vp, i64, i64, i, vp, vp]
        L.nf12_moe_schedule_bytes.restype = sz
        L.nf12_moe_schedule_bytes.argtypes = [i64, i64, i]
        L.nf12_moe_schedule_row_end_offset.restype = sz
        L.nf12_moe_schedule_row_end_offset.argtypes = [i64, i64, i]
        self._ws = None

    def supported(self, cfg, E, n, k):
        return bool(self.lib.nf12_moe_small_supported(cfg.mt, cfg.warps, cfg.kc, cfg.slice, cfg.pdl, cfg.kp, E, n, k))

    def heuristic(self, A, E, n, k):
        out = (ctypes.c_int * 6)()
        self.lib.nf12_moe_small_heuristic(A, E, n, k, out)
        return Config(*out)

    def workspace(self, cfg, A, E, n, k):
        if cfg.mt > 0:
            return None, 0
        nbytes = (self.lib.nf12_moe_workspace_bytes(A, E, n, k) if cfg.mt == 0 else
                  self.lib.nf12_moe_large_workspace_bytes(-cfg.mt, A, E, n, k))
        if nbytes and (self._ws is None or self._ws.numel() < nbytes):
            self._ws = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        return (self._ws.data_ptr() if nbytes else None), nbytes

    def moe_linear(self, cfg, x, enc, offsets, max_assignments, y, gather=None, gather_div=1, scatter=None,
                   row_scale=None, source=NF12, raw=False):
        """y[out(p)] = x[in(p)] @ W_e.T (* row_scale). x, y: 2-D views with unit column stride."""
        ptr = lambda t: None if t is None else t.data_ptr()  # noqa: E731
        weight = enc.packed if source == NF12 else enc.bf16t
        ws, ws_bytes = self.workspace(cfg, max_assignments, enc.num_experts, enc.n, enc.k)
        err = self.lib.nf12_moe_linear(
            cfg.mt, cfg.warps, cfg.kc, cfg.slice, cfg.pdl, cfg.kp, x.data_ptr(), x.stride(0), ptr(gather), gather_div,
            ptr(scatter), offsets.data_ptr(), max_assignments, source, weight.data_ptr(),
            ptr(enc.escapes) if source == NF12 else None, ptr(enc.escape_offsets) if source == NF12 else None,
            ptr(enc.exp_base) if source == NF12 else None, enc.num_experts, enc.n, enc.k, ptr(row_scale),
            y.data_ptr(), y.stride(0), ws, ws_bytes, torch.cuda.current_stream().cuda_stream)
        if raw:
            return err
        if err == 801 and cfg.mt <= 0 and not WITH_LARGE:
            raise LargeStub(f"A={max_assignments} dispatches to a grouped-GEMM stand-in (NF12_MOE_WITH_LARGE=0)")
        if err != 0:
            raise RuntimeError(f"nf12_moe_linear({cfg}, A={max_assignments}, E={enc.num_experts}, n={enc.n}, "
                               f"k={enc.k}) failed: cuda error {err}")
        return y

    def schedule(self, offsets, E, A, bm):
        """moe_schedule's list: (tiles [count, 4] int32, row_end [E] int32), on the CPU."""
        nbytes = self.lib.nf12_moe_schedule_bytes(A, E, bm)
        ws = torch.full((nbytes,), 0xA5, dtype=torch.uint8, device="cuda")  # garbage: must not matter
        err = self.lib.nf12_moe_schedule(offsets.data_ptr(), E, A, bm, ws.data_ptr(),
                                         torch.cuda.current_stream().cuda_stream)
        if err != 0:
            raise RuntimeError(f"moe_schedule failed: {err}")
        ws = ws.cpu()
        count = int(ws[:4].view(torch.int32)[0])
        end = self.lib.nf12_moe_schedule_row_end_offset(A, E, bm)
        assert end + 4 * E == nbytes and 16 + 16 * count <= end
        return ws[16:16 + 16 * count].view(torch.int32).view(count, 4), ws[end:end + 4 * E].view(torch.int32)
