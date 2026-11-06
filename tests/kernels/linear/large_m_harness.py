# SPDX-License-Identifier: Apache-2.0
"""Standalone harness for the large-M linear kernels.

Builds the large-M CUDA sources under csrc/linear and the sibling
large_m_shim.cu with nvcc, independently of CMake. Weights use a vectorized
torch encoder for SPEC.md sections 2-4. ctypes exposes forced kernel configs.
"""

import ctypes
import os
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LINEAR = os.path.join(ROOT, "csrc", "linear")

FAMILY_MMA = 0   # mma.sync kernels (SM80+)
FAMILY_TC = 1    # tcgen05/UMMA kernels (SM100/SM110)
FAMILY_AUTO = 2  # linear_large_m(): production family + config choice


class Unsupported(RuntimeError):
    """The requested family/config cannot run on this device."""


def _arch_flags():
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    suffix = "a" if cc in (90, 100, 103, 110, 120, 121) else ""
    return [f"-gencode=arch=compute_{cc}{suffix},code=sm_{cc}{suffix}"]


def build(force=False):
    """Compile the shim library for the local GPU and return its path.

    Env overrides: NF12_LARGE_M_BUILD_DIR (output dir, so concurrent users
    don't clobber each other), NF12_LARGE_M_EXCLUDE (comma-separated
    csrc/linear basenames to skip), NF12_EXTRA_SRCS (extra .cu files),
    NF12_SHIM_DECODE=1 (link Codec's decode for the decode+cuBLAS baseline),
    NF12_NVCC_FLAGS, NVCC.
    """
    out_dir = os.environ.get("NF12_LARGE_M_BUILD_DIR", os.path.join(ROOT, "build", "large_m_tests"))
    os.makedirs(out_dir, exist_ok=True)
    extra_srcs = list(filter(None, os.environ.get("NF12_EXTRA_SRCS", "").split(",")))
    # Extra entry points require a separate library from the base shim.
    tag = "".join("+" + os.path.splitext(os.path.basename(s))[0] for s in extra_srcs)
    lib = os.path.join(out_dir, f"liblarge_m_shim{tag}.so")
    exclude = set(filter(None, os.environ.get("NF12_LARGE_M_EXCLUDE", "").split(",")))
    srcs = sorted(
        os.path.join(directory, f)
        for directory, _, files in os.walk(LINEAR)
        for f in files
        if f.startswith("large_m") and f.endswith(".cu") and f not in exclude
    ) + [os.path.join(HERE, "large_m_shim.cu")]
    srcs += extra_srcs
    defines = []
    if os.environ.get("NF12_SHIM_DECODE") == "1":
        srcs.append(os.path.join(ROOT, "csrc", "codec", "decode.cu"))
        defines.append("-DNF12_SHIM_DECODE")
        lib = lib.replace(".so", "_decode.so")
    deps = srcs + [
        os.path.join(directory, f)
        for directory, _, files in os.walk(LINEAR)
        for f in files if f.endswith((".h", ".cuh"))
    ] + [os.path.join(ROOT, "include", "nf12", "format.cuh"), os.path.join(ROOT, "csrc", "internal.h")]
    if not force and os.path.exists(lib):
        mtime = os.path.getmtime(lib)
        if all(os.path.getmtime(d) <= mtime for d in deps):
            return lib
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    flags = ["-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-lineinfo", *_arch_flags(), *defines,
             "-I", os.path.join(ROOT, "include"), "-I", os.path.join(ROOT, "csrc")]
    extra = os.environ.get("NF12_NVCC_FLAGS")
    if extra:
        flags = extra.split() + flags
    # Compile independent kernel translation units concurrently.
    obj_dir = os.path.splitext(lib)[0] + ".obj"
    os.makedirs(obj_dir, exist_ok=True)
    objs = [os.path.join(obj_dir, os.path.splitext(os.path.basename(s))[0] + ".o") for s in srcs]
    procs = [subprocess.Popen([nvcc, *flags, "-c", "-o", o, s]) for s, o in zip(srcs, objs)]
    if any([p.wait() != 0 for p in procs]):
        raise subprocess.CalledProcessError(1, "nvcc")
    subprocess.run([nvcc, "-shared", *_arch_flags(), "-o", lib, *objs], check=True)
    return lib


class Shim:
    def __init__(self, lib_path=None):
        self.lib = ctypes.CDLL(lib_path or build())
        L = self.lib
        i64, u32, vp, sz = ctypes.c_int64, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t
        L.nf12_test_large_m.restype = ctypes.c_int
        L.nf12_test_large_m.argtypes = [
            ctypes.c_int, ctypes.c_int,  # family, config (-1 = heuristic)
            vp, i64, i64,                # x, m, ldx
            ctypes.c_int, vp, vp, u32, u32,  # source, weight, escapes, num_escapes, exp_base
            i64, i64, vp, vp, i64,       # n, k, bias, y, ldy
            vp, sz, vp,                  # workspace, bytes, stream
        ]
        L.nf12_test_large_m_ws.restype = sz
        L.nf12_test_large_m_ws.argtypes = [ctypes.c_int, ctypes.c_int, i64, i64, i64]
        L.nf12_test_large_m_num_configs.restype = ctypes.c_int
        L.nf12_test_large_m_num_configs.argtypes = [ctypes.c_int]
        L.nf12_test_large_m_config_name.restype = ctypes.c_char_p
        L.nf12_test_large_m_config_name.argtypes = [ctypes.c_int, ctypes.c_int]
        L.nf12_test_large_m_supported.restype = ctypes.c_int
        L.nf12_test_large_m_supported.argtypes = [ctypes.c_int]
        L.nf12_test_large_m_heuristic.restype = ctypes.c_int
        L.nf12_test_large_m_heuristic.argtypes = [i64, i64, i64]
        self.has_decode = hasattr(L, "nf12_test_decode")
        if self.has_decode:
            L.nf12_test_decode.restype = ctypes.c_int
            L.nf12_test_decode.argtypes = [vp, vp, u32, i64, i64, vp, vp]
        self._ws = None

    def decode(self, w, out=None):
        """Codec decode of an NF12Packed to [n, k] bf16 (requires NF12_SHIM_DECODE=1 build)."""
        if out is None:
            out = torch.empty(w.n, w.k, dtype=torch.bfloat16, device="cuda")
        rc = self.lib.nf12_test_decode(w.packed.data_ptr(), w.escapes.data_ptr(), w.exp_base, w.n, w.k,
                                       out.data_ptr(), torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"decode failed: cudaError {rc}")
        return out

    def mma_heuristic(self, m, n, k):
        """Encoded mma-family config the heuristic picks (index + 16 * (splits - 1))."""
        return self.lib.nf12_test_large_m_heuristic(m, n, k)

    def supported(self, family):
        return bool(self.lib.nf12_test_large_m_supported(family))

    def num_configs(self, family):
        return self.lib.nf12_test_large_m_num_configs(family)

    def config_name(self, family, config):
        return self.lib.nf12_test_large_m_config_name(family, config).decode()

    def workspace(self, family, config, m, n, k):
        nbytes = self.lib.nf12_test_large_m_ws(family, config, m, n, k)
        if self._ws is None or self._ws.numel() < nbytes:
            self._ws = torch.empty(max(nbytes, 256), dtype=torch.uint8, device="cuda")
        return self._ws, nbytes

    def linear(self, x, w, bias=None, y=None, source="nf12", family=FAMILY_MMA, config=-1, ws=None):
        """Return y [m, n] for bf16 x [m, k] (possibly row-padded) and NF12Packed w."""
        m, k = x.shape
        n = w.n
        assert k == w.k and x.stride(1) == 1
        if y is None:
            y = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
        if ws is None:
            ws, nbytes = self.workspace(family, config, m, n, k)
        else:
            nbytes = ws.numel()
        wt = w.packed if source == "nf12" else w.bf16t
        # A table without entries is passed as NULL, as the C API allows.
        esc = w.escapes.data_ptr() if w.num_escapes else None
        rc = self.lib.nf12_test_large_m(
            family, config, x.data_ptr(), m, x.stride(0), 0 if source == "nf12" else 1, wt.data_ptr(), esc,
            w.num_escapes, w.exp_base,
            n, k, bias.data_ptr() if bias is not None else None, y.data_ptr(), y.stride(0), ws.data_ptr(), nbytes,
            torch.cuda.current_stream().cuda_stream,
        )
        if rc != 0:
            if rc == 801:  # cudaErrorNotSupported
                raise Unsupported(f"config {config} unsupported on this device")
            raise RuntimeError(f"large_m kernel failed: cudaError {rc}")
        return y


# Torch encoder (SPEC.md 2-4), vectorized on GPU.

def _slot_maps(device):
    lane = torch.arange(32, device=device).view(32, 1)
    slot = torch.arange(8, device=device).view(1, 8)
    row = (lane >> 2) + ((slot >> 1) & 1) * 8
    col = 2 * (lane & 3) + (slot & 1) + (slot >> 2) * 8
    return row, col


def _to_i32(x):
    x = x & 0xFFFFFFFF
    return (x - ((x >> 31) << 32)).to(torch.int32)


def gather_groups(w):
    """[N, K] bf16 -> [NB, KB, 4, 32, 8] int64 slot bit patterns (flat group order)."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    t = bits.view(n // 16, 16, k // 64, 4, 16).permute(0, 2, 3, 1, 4)
    row, col = _slot_maps(w.device)
    return t[..., row, col]


def choose_exp_base(w):
    e = (w.contiguous().view(torch.int16).to(torch.int32) >> 7) & 0xFF
    hist = torch.bincount(e.flatten(), minlength=256).to(torch.int64)
    win = torch.stack([hist[2 * b: 2 * b + 16].sum() for b in range(121)])
    return int(torch.argmax(win).item())


class NF12Packed:
    """NF12 (+ bf16t twin) encoding of w, computed in row chunks to bound memory."""

    def __init__(self, w, exp_base=None, bf16t=True, chunk_rows=1024):
        assert w.dtype == torch.bfloat16 and w.dim() == 2
        n, k = w.shape
        assert n % 16 == 0 and k % 64 == 0
        self.n, self.k = n, k
        self.exp_base = choose_exp_base(w) if exp_base is None else exp_base
        self.packed = torch.empty(n, k * 3 // 2, dtype=torch.uint8, device=w.device)
        self.bf16t = torch.empty(n, k * 2, dtype=torch.uint8, device=w.device) if bf16t else None
        escapes = []
        self.num_escapes = 0
        for r0 in range(0, n, chunk_rows):
            r1 = min(n, r0 + chunk_rows)
            p, t, e = self._encode_rows(w[r0:r1], self.num_escapes)
            self.packed[r0:r1] = p
            if bf16t:
                self.bf16t[r0:r1] = t
            escapes.append(e)
            self.num_escapes += e.shape[0]
        self.escapes = (torch.cat(escapes) if self.num_escapes
                        else torch.zeros(1, 4, dtype=torch.int32, device=w.device)).contiguous()

    def _encode_rows(self, w, esc_base):
        n, k = w.shape
        g = gather_groups(w)  # [NB, KB, 4, 32, 8]
        nb, kb = n // 16, k // 64
        B = self.exp_base
        e = (g >> 7) & 0xFF
        s = g >> 15
        inwin = (e >= 2 * B) & (e <= 2 * B + 15)
        o1 = ((e - 2 * B) >> 1) & 7
        a_s, b_s = s[..., 0::2], s[..., 1::2]
        a_o, b_o = o1[..., 0::2], o1[..., 1::2]
        mbyte = (a_s << 7) | (b_s << 6) | (b_o << 3) | a_o  # [..., 4]
        sh = torch.tensor([0, 8, 16, 24], device=w.device)
        meta = (mbyte << sh).sum(-1)
        lo = g & 0xFF
        l0 = (lo[..., 0:4] << sh).sum(-1)
        l1 = (lo[..., 4:8] << sh).sum(-1)
        esc = (~inwin.all(-1)) | (meta == 0)
        flat_esc = esc.flatten()
        idx = (torch.cumsum(flat_esc.to(torch.int64), 0) - 1 + esc_base).view_as(meta)
        meta = torch.where(esc, torch.zeros_like(meta), meta)
        l0 = torch.where(esc, idx, l0)
        l1 = torch.where(esc, torch.zeros_like(l1), l1)
        # Escape table entries and the bf16t twin share the R0..R3 packing.
        r = g[..., 0::2] | (g[..., 1::2] << 16)  # [NB, KB, 4, 32, 4]
        escv = _to_i32(r.reshape(-1, 4)[flat_esc])
        # Plane 0: word 4l + j = meta of atom j.
        p0 = meta.permute(0, 1, 3, 2).reshape(nb, kb, 128)
        # Planes 1/2: word 128*(1+half) + 4l + 2jj + {0: L0, 1: L1}, atom j = 2*half + jj.
        lw = torch.stack([l0, l1], -1).view(nb, kb, 2, 2, 32, 2).permute(0, 1, 2, 4, 3, 5).reshape(nb, kb, 256)
        packed = _to_i32(torch.cat([p0, lw], -1)).contiguous().view(torch.uint8).view(n, k * 3 // 2)
        bf16t = _to_i32(r).contiguous().view(torch.uint8).view(n, k * 2)
        return packed, bf16t, escv

    def bytes_nf12(self):
        return self.packed.numel() + self.num_escapes * 16


def force_escapes(w, frac=0.01, seed=0, specials=False):
    """Copy w with ~frac of groups holding out-of-window magnitudes.

    Finite values allow comparison with an FP32 reference. specials=True
    uses Inf/NaN for bitwise NF12-vs-bf16t checks only.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = w.clone()
    flat = w.view(-1)
    cnt = max(1, int(flat.numel() / 8 * frac))
    pos = torch.randint(0, flat.numel(), (cnt,), generator=gen).to(w.device)
    choices = [float("inf"), float("-inf"), float("nan"), -0.0] if specials else [0.0, 1e-20, -6.0e4, 2.5e5]
    vals = torch.tensor(choices, dtype=torch.bfloat16, device=w.device)
    flat[pos] = vals[torch.arange(cnt, device=w.device) % 4]
    return w
