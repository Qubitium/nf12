# SPDX-License-Identifier: Apache-2.0
"""Header-level C API checks: a C99 program compiled against include/nf12/nf12.h
and linked against libnf12.so, plus the library's exported symbol set."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import nf12

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include" / "nf12" / "nf12.h"


def header_functions() -> set[str]:
    text = re.sub(r"/\*.*?\*/", "", HEADER.read_text(), flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return set(re.findall(r"\b(nf12_\w+)\s*\(", text))


def cuda_home() -> Path | None:
    for candidate in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if candidate and (Path(candidate) / "include" / "cuda_runtime_api.h").is_file():
            return Path(candidate)
    nvcc = shutil.which("nvcc")
    return Path(nvcc).resolve().parents[1] if nvcc else None


def cudart_dir(home: Path) -> Path:
    for sub in ("lib64", "lib", "targets/x86_64-linux/lib", "targets/sbsa-linux/lib"):
        if (home / sub / "libcudart.so").exists():
            return home / sub
    pytest.skip(f"libcudart.so not found under {home}")


def test_header_declares_the_c_api():
    assert header_functions() == {
        "nf12_status_string", "nf12_last_cuda_error", "nf12_packed_bytes", "nf12_bf16t_bytes",
        "nf12_encode_workspace_bytes", "nf12_encode_analyze", "nf12_encode", "nf12_decode",
        "nf12_linear_workspace_bytes", "nf12_linear", "nf12_linear_decode_gemm_rows", "nf12_pack_bf16t",
        "nf12_linear_static_weights",
        "nf12_linear_bf16t", "nf12_moe_align_workspace_bytes", "nf12_moe_align", "nf12_moe_linear_workspace_bytes",
        "nf12_moe_linear", "nf12_moe_linear_bf16t", "nf12_moe_combine", "nf12_moe_silu_mul", "nf12_moe_decode",
        "nf12_moe_linear_decode_gemm_rows", "nf12_moe_grouped_operands", "nf12_embedding", "nf12_embedding_i64",
    }


def test_exports_match_header():
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm not found")
    out = subprocess.run([nm, "-D", "--defined-only", str(nf12.library_path())], check=True,
                         capture_output=True, text=True).stdout
    exported = {line.split()[-1] for line in out.splitlines() if line.strip()}
    assert exported == header_functions()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_c99_consumer(tmp_path):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler found")
    home = cuda_home()
    if home is None:
        pytest.skip("CUDA toolkit headers not found")
    lib_dir = nf12.library_path().parent
    rt_dir = cudart_dir(home)
    exe = tmp_path / "c_consumer"
    cmd = [
        cc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-Werror=implicit-function-declaration",
        f"-I{ROOT / 'include'}", "-isystem", str(home / "include"),
        str(ROOT / "tests" / "api" / "c_consumer.c"), "-o", str(exe),
        f"-L{lib_dir}", "-l:libnf12.so", f"-L{rt_dir}", "-lcudart", "-lm",
        f"-Wl,-rpath,{lib_dir}", f"-Wl,-rpath,{rt_dir}",
    ]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, f"{' '.join(cmd)}\n{build.stdout}{build.stderr}"
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "c_consumer: OK" in run.stdout
