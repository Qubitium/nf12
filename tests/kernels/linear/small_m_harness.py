# SPDX-License-Identifier: Apache-2.0
"""Standalone harness for the small-M linear kernels and linear dispatch.

Builds the small-M kernels, dispatch and large-M families under csrc/linear,
csrc/device_info.cu, codec decode and the sibling small_m_shim.cu with nvcc,
independently of CMake. Uses python/nf12/reference.py for weight encoding and
ctypes for forced kernel configs.
"""

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LINEAR = os.path.join(ROOT, "csrc", "linear")
sys.path.insert(0, os.path.join(ROOT, "python"))

from nf12 import reference  # noqa: E402

NF12 = 0
BF16T = 1


@dataclass(frozen=True)
class Config:
    mt: int
    warps: int
    kc: int
    slice: int = 0
    kp: int = 1  # K parts per CTA (below SM90)

    def __str__(self):
        return f"mt{self.mt}_w{self.warps}_kc{self.kc}_s{self.slice}" + (f"_kp{self.kp}" if self.kp != 1 else "")


AUTO = Config(0, 0, 0, 0)  # production dispatch (nf12::detail::linear)


def _arch_flags():
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    suffix = "a" if cc in (90, 100, 103, 110) else ""
    return [f"-gencode=arch=compute_{cc}{suffix},code=sm_{cc}{suffix}"]


WITH_LARGE_M = os.environ.get("NF12_SM_WITH_LARGE_M", "1") != "0"


class LargeMStub(RuntimeError):
    """Dispatch chose linear_large_m() but the library links the stand-in."""


def build(force=False):
    """Compile the shim library for the local GPU and return its path.

    NF12_SM_WITH_LARGE_M=0 isolates small-M diagnostics by linking a large-M
    stand-in that returns cudaErrorNotSupported above the dispatch crossover.
    """
    with_large = WITH_LARGE_M
    out_dir = os.path.join(ROOT, "build", "small_m_tests" + ("_full" if with_large else ""))
    os.makedirs(out_dir, exist_ok=True)
    lib = os.path.join(out_dir, "libsmall_m_shim.so")
    prefixes = ("small_m", "large_m") if with_large else ("small_m",)
    srcs = sorted(
        os.path.join(directory, f)
        for directory, _, files in os.walk(LINEAR)
        for f in files
        if f.endswith(".cu") and (f.startswith(prefixes) or f == "dispatch.cu")
    ) + [
        # dispatch.cu's decode-then-GEMM route calls decode_bf16t.
        os.path.join(ROOT, "csrc", "codec", "decode.cu"),
        os.path.join(ROOT, "csrc", "device_info.cu"),
        os.path.join(HERE, "small_m_shim.cu"),
    ]
    if not with_large:
        stub = os.path.join(out_dir, "large_m_stub.cu")
        text = ('#include "linear/large_m.h"\n#include "linear/sm90/large_m_sm90.h"\nnamespace nf12::detail {\n'
                "size_t linear_large_m_workspace_bytes(int64_t, int64_t, int64_t, const DeviceInfo&) "
                "{ return 0; }\ncudaError_t linear_large_m(const LinearArgs&) "
                "{ return cudaErrorNotSupported; }\n"
                "bool large_m_sm90_supported(const DeviceInfo&) { return false; }\n}\n")
        if not os.path.exists(stub) or open(stub).read() != text:
            with open(stub, "w") as f:
                f.write(text)
        srcs.append(stub)
    deps = srcs + [
        os.path.join(directory, f)
        for directory, _, files in os.walk(LINEAR)
        for f in files if f.endswith((".h", ".cuh"))
    ] + [os.path.join(ROOT, "include", "nf12", "format.cuh"), os.path.join(ROOT, "csrc", "internal.h")] + [
        os.path.join(ROOT, "csrc", "codec", f) for f in ("codec.h", "tile.cuh")
    ]
    if not force and os.path.exists(lib):
        mtime = os.path.getmtime(lib)
        if all(os.path.getmtime(d) <= mtime for d in deps):
            return lib
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    cmd = [
        nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "--expt-relaxed-constexpr",
        "-lineinfo", *_arch_flags(),
        "-I", os.path.join(ROOT, "include"), "-I", os.path.join(ROOT, "csrc"),
        "-o", lib, *srcs,
    ]
    extra = os.environ.get("NF12_NVCC_FLAGS")
    if extra:
        cmd[1:1] = extra.split()
    subprocess.run(cmd, check=True)
    return lib


class Lib:
    def __init__(self, lib_path=None):
        self.lib = ctypes.CDLL(lib_path or build())
        L = self.lib
        i, i64, u32, vp, sz = ctypes.c_int, ctypes.c_int64, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t
        L.nf12_sm_linear.restype = i
        L.nf12_sm_linear.argtypes = [i, i, i, i, i, vp, i64, i64, i, vp, vp, u32, u32, i64, i64, vp, vp, i64, vp, sz,
                                     vp]
        L.nf12_sm_supported.restype = i
        L.nf12_sm_supported.argtypes = [i, i, i, i, i, i64, i64, i64]
        L.nf12_sm_resident_ctas.restype = i
        L.nf12_sm_resident_ctas.argtypes = [i, i, i, i, i, i64, i64, i64]
        L.nf12_sm_stages.restype = i
        L.nf12_sm_stages.argtypes = [i, i64, i64]
        L.nf12_sm_slice_step.restype = i
        L.nf12_sm_slice_step.argtypes = [i, i64, i64]
        L.nf12_sm_heuristic.restype = None
        L.nf12_sm_heuristic.argtypes = [i64, i64, i64, ctypes.POINTER(ctypes.c_int)]
        L.nf12_sm_workspace_bytes.restype = sz
        L.nf12_sm_workspace_bytes.argtypes = [i64, i64, i64]
        L.nf12_sm_config_workspace_bytes.restype = sz
        L.nf12_sm_config_workspace_bytes.argtypes = [i, i, i, i, i, i64, i64]
        L.nf12_sm_heuristic_slice.restype = i
        L.nf12_sm_heuristic_slice.argtypes = [i, i, i, i, i64, i64, i64]
        L.nf12_sm_stream_read.restype = i
        L.nf12_sm_stream_read.argtypes = [vp, sz, i, i, vp, vp]
        self._ws = None

    def supported(self, cfg, m, n, k):
        return bool(self.lib.nf12_sm_supported(cfg.mt, cfg.warps, cfg.kc, cfg.slice, cfg.kp, m, n, k))

    def stages(self, mt, n, k):
        """Weight ring of mt-tile configs on an n x k weight: 0 = register ring, else shared-memory stages."""
        return self.lib.nf12_sm_stages(mt, n, k)

    def slice_step(self, mt, n, k):
        """Granularity of Config.slice per K part for mt-tile configs on an n x k weight on this device."""
        return self.lib.nf12_sm_slice_step(mt, n, k)

    def resident_ctas(self, cfg, m, n, k):
        return self.lib.nf12_sm_resident_ctas(cfg.mt, cfg.warps, cfg.kc, cfg.slice, cfg.kp, m, n, k)

    def heuristic(self, m, n, k):
        out = (ctypes.c_int * 5)()
        self.lib.nf12_sm_heuristic(m, n, k, out)
        return Config(*out)

    def heuristic_slice(self, mt, warps, kc, kp, m, n, k):
        """Return the heuristic x slice for this config/shape, or -1 if unsupported."""
        return self.lib.nf12_sm_heuristic_slice(mt, warps, kc, kp, m, n, k)

    def workspace(self, m, n, k, cfg=None):
        """Workspace of the production dispatch (cfg None) or of the forced cfg."""
        nbytes = (self.lib.nf12_sm_workspace_bytes(m, n, k) if cfg is None else
                  self.lib.nf12_sm_config_workspace_bytes(cfg.mt, cfg.warps, cfg.kc, cfg.slice, cfg.kp, m, n))
        if nbytes and (self._ws is None or self._ws.numel() < nbytes):
            self._ws = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        return (self._ws.data_ptr() if nbytes else None), nbytes

    def linear(self, cfg, x, w, y=None, bias=None, source=NF12, m=None):
        """y[:m, :n] = x[:m] @ W.T (+ bias) for x [m, ldx] and Encoded w."""
        m = x.shape[0] if m is None else m
        if y is None:
            y = torch.empty(m, w.n, dtype=torch.bfloat16, device=x.device)
        weight = w.packed if source == NF12 else w.bf16t
        esc = w.escapes_u4.data_ptr() if (source == NF12 and w.escapes_u4.numel()) else None
        ws, ws_bytes = self.workspace(m, w.n, w.k, None if cfg.mt == 0 else cfg)
        err = self.lib.nf12_sm_linear(
            cfg.mt, cfg.warps, cfg.kc, cfg.slice, cfg.kp,
            x.data_ptr(), m, x.stride(0), source, weight.data_ptr(), esc, w.num_escapes if esc else 0, w.exp_base,
            w.n, w.k,
            None if bias is None else bias.data_ptr(), y.data_ptr(), y.stride(0), ws, ws_bytes,
            torch.cuda.current_stream().cuda_stream)
        if err == 801 and cfg.mt == 0 and not WITH_LARGE_M:
            raise LargeMStub(f"m={m} dispatches to the large-M stand-in (NF12_SM_WITH_LARGE_M=0)")
        if err != 0:
            raise RuntimeError(f"nf12_sm_linear({cfg}, m={m}, n={w.n}, k={w.k}) failed: cuda error {err}")
        return y

    def stream_read(self, buf, blocks, depth, out):
        err = self.lib.nf12_sm_stream_read(buf.data_ptr(), buf.numel(), blocks, depth, out.data_ptr(),
                                           torch.cuda.current_stream().cuda_stream)
        if err != 0:
            raise RuntimeError(f"stream_read failed: {err}")


@dataclass
class Encoded:
    w: torch.Tensor  # original BF16 [n, k]
    packed: torch.Tensor
    escapes_u4: torch.Tensor  # int16 [E, 8], 16-byte rows
    exp_base: int
    bf16t: torch.Tensor
    n: int
    k: int

    @property
    def num_escapes(self):
        return self.escapes_u4.shape[0]


def encode(w, exp_base=None, with_bf16t=True):
    n, k = w.shape
    packed, escapes, base = reference.encode(w, exp_base)
    if escapes.numel() == 0:
        escapes = torch.zeros(0, 8, dtype=torch.int16, device=w.device)
    return Encoded(w, packed.contiguous(), escapes.contiguous(), base,
                   reference.pack_bf16t(w).contiguous() if with_bf16t else None, n, k)


def random_weight(n, k, seed=0, escapes=0.0, device="cuda"):
    """Gaussian BF16 weights with an `escapes` fraction of values replaced
    by out-of-window magnitudes to force escaped groups."""
    g = torch.Generator(device=device).manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device=device) * 0.02).to(torch.bfloat16)
    if escapes > 0:
        mask = torch.rand(n, k, generator=g, device=device) < escapes
        big = torch.where(torch.rand(n, k, generator=g, device=device) < 0.5, 3.0e4, -1.0e-30)
        w = torch.where(mask, big.to(torch.bfloat16), w)
    return w
