# SPDX-License-Identifier: Apache-2.0
"""Codec check on real checkpoints: the CUDA encoder must produce exactly the
bytes of the Python reference (python/nf12/reference.py), decode must round
trip without changes, and pack_bf16t must match the reference twin layout.

The CUDA codec is driven through libcodec_shim.so (built by the Makefile next
to this file), so this runs without the library build.

    HF_HOME=/workspace/hf uv run --group bench python tests/kernels/codec/test_real_weights.py [repo_id ...]
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import pathlib
import sys
import time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Load the reference by path to avoid requiring nf12's compiled library.
_spec = importlib.util.spec_from_file_location("nf12_reference", ROOT / "python" / "nf12" / "reference.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

# unsloth/Llama-3.2-1B mirrors the gated meta-llama/Llama-3.2-1B weights.
DEFAULT_MODELS = ("Qwen/Qwen3-0.6B", "unsloth/Llama-3.2-1B")
LDW_PAD = 64  # elements of row padding for the strided-input variant
LDO_PAD = 32


class Codec:
    def __init__(self, path: pathlib.Path):
        lib = ctypes.CDLL(str(path))
        i64, u32, vp = ctypes.c_int64, ctypes.c_uint32, ctypes.c_void_p
        lib.codec_encode_workspace_bytes.restype = ctypes.c_size_t
        lib.codec_encode_workspace_bytes.argtypes = [i64, i64]
        lib.codec_encode_analyze.argtypes = [vp, i64, i64, i64, vp, ctypes.c_size_t,
                                             ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(u32)]
        lib.codec_encode.argtypes = [vp, i64, i64, i64, ctypes.c_uint8, vp, vp, u32, vp, ctypes.c_size_t]
        lib.codec_decode.argtypes = [vp, vp, u32, i64, i64, vp, i64]
        lib.codec_pack_bf16t.argtypes = [vp, i64, i64, i64, vp]
        self.lib = lib

    @staticmethod
    def _check(err: int, what: str) -> None:
        if err != 0:
            raise RuntimeError(f"{what} failed with cudaError {err}")

    def encode(self, w: torch.Tensor):
        """w: [N, K] bf16 CUDA view with unit column stride (row stride may exceed K)."""
        n, k = w.shape
        ldw = w.stride(0)
        ws_bytes = self.lib.codec_encode_workspace_bytes(n, k)
        ws = torch.empty(ws_bytes, dtype=torch.uint8, device=w.device)
        b, e = ctypes.c_uint8(), ctypes.c_uint32()
        torch.cuda.synchronize()
        self._check(self.lib.codec_encode_analyze(w.data_ptr(), n, k, ldw, ws.data_ptr(), ws_bytes,
                                                  ctypes.byref(b), ctypes.byref(e)), "encode_analyze")
        packed = torch.empty(n, k * 3 // 2, dtype=torch.uint8, device=w.device)
        escapes = torch.empty(max(e.value, 1), 8, dtype=torch.int16, device=w.device)
        self._check(self.lib.codec_encode(w.data_ptr(), n, k, ldw, b.value, packed.data_ptr(), escapes.data_ptr(),
                                          e.value, ws.data_ptr(), ws_bytes), "encode")
        torch.cuda.synchronize()
        return packed, escapes[: e.value], b.value

    def decode(self, packed, escapes, exp_base, n, k, out: torch.Tensor) -> None:
        self._check(self.lib.codec_decode(packed.data_ptr(), escapes.data_ptr(), exp_base, n, k, out.data_ptr(),
                                          out.stride(0)), "decode")

    def pack_bf16t(self, w: torch.Tensor) -> torch.Tensor:
        n, k = w.shape
        out = torch.empty(n, 2 * k, dtype=torch.uint8, device=w.device)
        self._check(self.lib.codec_pack_bf16t(w.data_ptr(), n, k, w.stride(0), out.data_ptr()), "pack_bf16t")
        return out


def padded(w: torch.Tensor, pad: int) -> torch.Tensor:
    n, k = w.shape
    buf = torch.full((n, k + pad), float("nan"), dtype=w.dtype, device=w.device)
    buf[:, :k] = w
    return buf[:, :k]


def check_tensor(codec: Codec, name: str, w: torch.Tensor) -> tuple[bool, int]:
    n, k = w.shape
    r_packed, r_esc, r_b = ref.encode(w)
    ok = True
    for variant, src in (("dense", w), ("ldw+pad", padded(w, LDW_PAD))):
        packed, esc, b = codec.encode(src)
        problems = []
        if b != r_b:
            problems.append(f"exp_base {b} vs ref {r_b}")
        if esc.shape != r_esc.shape or not torch.equal(esc, r_esc):
            problems.append(f"escapes {tuple(esc.shape)} vs ref {tuple(r_esc.shape)}")
        if not torch.equal(packed, r_packed):
            nbad = int((packed != r_packed).sum())
            problems.append(f"{nbad} packed bytes differ")
        out_buf = torch.full((n, k + LDO_PAD), -7.0, dtype=torch.bfloat16, device=w.device)
        codec.decode(packed, esc if esc.numel() else torch.empty(1, 8, dtype=torch.int16, device=w.device), b, n, k,
                     out_buf)
        torch.cuda.synchronize()
        if not torch.equal(out_buf[:, :k].view(torch.int16), w.view(torch.int16)):
            problems.append("decode mismatch")
        if not bool((out_buf[:, k:] == -7.0).all()):
            problems.append("decode wrote padding")
        if not torch.equal(codec.pack_bf16t(src), ref.pack_bf16t(w)):
            problems.append("bf16t mismatch")
        if problems:
            ok = False
            print(f"    FAIL {name} [{variant}] {n}x{k}: {'; '.join(problems)}")
    return ok, int(r_esc.shape[0])


def safetensors_files(model: str) -> list[pathlib.Path]:
    """`model` is an HF repo id (resolved in the local cache only), a directory, or a .safetensors file."""
    p = pathlib.Path(model)
    if p.is_file():
        return [p]
    if not p.is_dir():
        p = pathlib.Path(snapshot_download(model, local_files_only=True, allow_patterns=["*.safetensors"]))
    files = sorted(p.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors weights found for {model} (in {p})")
    return files


def run_model(codec: Codec, model: str) -> bool:
    files = safetensors_files(model)
    tensors = elems = groups_escaped = skipped = 0
    ok = True
    t0 = time.time()
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for name in sf.keys():
                t = sf.get_tensor(name)
                if t.dtype != torch.bfloat16 or t.dim() != 2 or t.shape[0] % 16 or t.shape[1] % 64:
                    skipped += 1
                    continue
                t_ok, e = check_tensor(codec, name, t.cuda())
                ok &= t_ok
                tensors += 1
                elems += t.numel()
                groups_escaped += e
    status = "ok" if ok else "FAIL"
    print(f"  {status:4} {model}: {tensors} matrices, {elems / 1e6:.1f}M weights, "
          f"{groups_escaped} escaped groups ({100 * groups_escaped * 8 / max(elems, 1):.4f}%), "
          f"{skipped} non-matrix tensors skipped, {time.time() - t0:.1f}s")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--lib", default=str(HERE / "libcodec_shim.so"))
    args = ap.parse_args()
    codec = Codec(pathlib.Path(args.lib))
    results = [run_model(codec, m) for m in args.models]
    print("PASSED" if all(results) else "FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
