# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
"""RDNA4 HIP linear benchmark on the CUDA benchmark shape grid.

Uses the same ``k x n`` rows as ``bench/common.py`` so the HIP numbers are
directly comparable with the CUDA results. Each cell is warm/cold: warm reuses
one weight, cold cycles copies until the working set exceeds 96 MiB.

    HIP_VISIBLE_DEVICES=0 python bench/hip_linear.py
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

_VOID, _I64, _SIZE, _U8, _U32, _INT = (
    ctypes.c_void_p, ctypes.c_int64, ctypes.c_size_t, ctypes.c_uint8,
    ctypes.c_uint32, ctypes.c_int)


def _bind(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.nf12_hip_encode_workspace_bytes.restype = _SIZE
    lib.nf12_hip_encode_workspace_bytes.argtypes = [_I64, _I64]
    lib.nf12_hip_encode_analyze.restype = _INT
    lib.nf12_hip_encode_analyze.argtypes = [
        _VOID, _I64, _I64, _I64, _VOID, _SIZE,
        ctypes.POINTER(_U8), ctypes.POINTER(_U32), _VOID]
    lib.nf12_hip_encode.restype = _INT
    lib.nf12_hip_encode.argtypes = [
        _VOID, _I64, _I64, _I64, _U8, _VOID, _VOID, _U32, _VOID, _SIZE, _VOID]
    lib.nf12_hip_decode.restype = _INT
    lib.nf12_hip_decode.argtypes = [
        _VOID, _VOID, _U8, _I64, _I64, _VOID, _I64, _VOID]
    lib.nf12_hip_linear_fused_small_m_supported.restype = _INT
    lib.nf12_hip_linear_fused_small_m_supported.argtypes = [_I64]
    lib.nf12_hip_linear_fused.restype = _INT
    lib.nf12_hip_linear_fused.argtypes = [
        _VOID, _VOID, _U8, _VOID, _VOID, _I64, _I64, _I64, _VOID, _VOID]
    lib.nf12_hip_wmma_max_m.restype = _INT
    lib.nf12_hip_wmma_max_m.argtypes = []
    lib.nf12_hip_wmma_supported.restype = _INT
    lib.nf12_hip_wmma_supported = lib.nf12_hip_wmma_supported
    lib.nf12_hip_wmma_supported.argtypes = [_I64]
    lib.nf12_hip_linear_wmma_workspace_bytes.restype = _SIZE
    lib.nf12_hip_linear_wmma_workspace_bytes.argtypes = [_I64, _I64, _I64]
    lib.nf12_hip_linear_wmma.restype = _INT
    lib.nf12_hip_linear_wmma.argtypes = [
        _VOID, _VOID, _U8, _VOID, _VOID, _I64, _I64, _I64, _VOID,
        _VOID, _SIZE, _VOID]
    return lib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--m", default="1,16,32,64,128")
    ap.add_argument("--copies", type=int, default=0, help="cold copies (0 = auto)")
    args = ap.parse_args()
    torch.cuda.init()
    dev = torch.device(args.device)
    # Raw HIP launches and timing events use the current device, not tensor.device.
    torch.cuda.set_device(dev)
    import nf12
    lib = _bind(nf12.library_path())
    stream = torch.cuda.current_stream().cuda_stream
    max_m = int(lib.nf12_hip_wmma_max_m())

    def encode(w):
        n, k = w.shape
        ws = torch.empty(max(lib.nf12_hip_encode_workspace_bytes(n, k), 1),
                         dtype=torch.uint8, device=dev)
        eb, ne = _U8(0), _U32(0)
        rc = lib.nf12_hip_encode_analyze(w.data_ptr(), n, k, k, ws.data_ptr(),
                                         ws.numel(), ctypes.byref(eb), ctypes.byref(ne), stream)
        assert rc == 0, rc
        packed = torch.empty((n, k * 3 // 2), dtype=torch.uint8, device=dev)
        escapes = torch.empty((int(ne.value), 8), dtype=torch.int16, device=dev)
        rc = lib.nf12_hip_encode(w.data_ptr(), n, k, k, eb, packed.data_ptr(),
                                 escapes.data_ptr(), int(ne.value), ws.data_ptr(),
                                 ws.numel(), stream)
        assert rc == 0, rc
        return packed, escapes, int(eb.value)

    def linear(x, packed, escapes, eb, ws, ws_bytes):
        m, k = x.shape
        n = packed.shape[0]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        if m < 16 and lib.nf12_hip_linear_fused_small_m_supported(m):
            rc = lib.nf12_hip_linear_fused(packed.data_ptr(), escapes.data_ptr(), eb,
                                           x.data_ptr(), out.data_ptr(), n, k, m, None, stream)
        elif m <= max_m and lib.nf12_hip_wmma_supported(m):
            rc = lib.nf12_hip_linear_wmma(packed.data_ptr(), escapes.data_ptr(), eb,
                                          x.data_ptr(), out.data_ptr(), n, k, m, None,
                                          ws.data_ptr(), ws_bytes, stream)
        else:
            wd = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
            rc = lib.nf12_hip_decode(packed.data_ptr(), escapes.data_ptr(), eb, n, k,
                                     wd.data_ptr(), k, stream)
            assert rc == 0, rc
            return torch.matmul(x, wd.t())
        assert rc == 0, rc
        return out

    def best(fn, iters=50, warmup=10, reps=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        b = 1e18
        for _ in range(reps):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            e.synchronize()
            b = min(b, s.elapsed_time(e) / iters * 1e3)
        return b

    ms = [int(v) for v in args.m.split(",")]
    mib = 1024 * 1024
    print(f"{'k':>6} {'n':>6} {'m':>4} | {'warm':>7} {'bf16':>7} {'r':>5} | "
          f"{'cold':>7} {'bf16':>7} {'r':>5} | {'viol':>4} {'rms':>9}", flush=True)
    for shape in common.SHAPES:
        k, n = shape.k, shape.n
        packed_bytes = n * k * 3 // 2
        copies = args.copies or max(2, (96 * mib) // packed_bytes + 1)
        sets = []
        for c in range(copies):
            g = torch.Generator(device="cpu").manual_seed(c)
            w = (torch.randn(n, k, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(dev)
            sets.append((w, *encode(w)))
        # Exactly-decoded weight: the operand of the BF16 twin the spec (6.2)
        # compares against. Decoding is outside every timed region.
        w0, p0, e0, b0 = sets[0]
        wd = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
        rc = lib.nf12_hip_decode(p0.data_ptr(), e0.data_ptr(), b0, n, k,
                                 wd.data_ptr(), k, stream)
        assert rc == 0, rc
        for m in ms:
            g = torch.Generator(device="cpu").manual_seed(100 + m)
            x = (torch.randn(m, k, generator=g, dtype=torch.float32) * 0.05).to(torch.bfloat16).to(dev)
            # Caller-owned split-K scratch, allocated once per shape and reused
            # so the timed loop measures the kernel, not the allocator.
            ws_bytes = lib.nf12_hip_linear_wmma_workspace_bytes(n, k, m)
            ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device=dev)
            got = linear(x, p0, e0, b0, ws, ws_bytes)
            ref = F.linear(x, wd)
            err = (got.double() - ref.double()).abs()
            viol = int(((err > 2e-3 + 2e-3 * ref.double().abs()) | ~torch.isfinite(got)).sum())
            rms = (err.pow(2).mean().sqrt() / ref.double().pow(2).mean().sqrt()).item()
            tn = best(lambda x=x, p0=p0, e0=e0, b0=b0, ws=ws, ws_bytes=ws_bytes:
                      linear(x, p0, e0, b0, ws, ws_bytes))
            tb = best(lambda x=x, w0=w0: F.linear(x, w0))

            def cold_n(sets=sets, x=x, ws=ws, ws_bytes=ws_bytes):
                for (_, p_, e_, b_) in sets:
                    linear(x, p_, e_, b_, ws, ws_bytes)

            def cold_b(sets=sets, x=x):
                for (w_, _, _, _) in sets:
                    F.linear(x, w_)

            cn = best(cold_n, iters=10, warmup=2, reps=3) / copies
            cb = best(cold_b, iters=10, warmup=2, reps=3) / copies
            print(f"{k:>6} {n:>6} {m:>4} | {tn:>7.1f} {tb:>7.1f} {tb/tn:>5.2f} | "
                  f"{cn:>7.1f} {cb:>7.1f} {cb/cn:>5.2f} | {viol:>4} {rms:>9.3e}", flush=True)
        del sets
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
