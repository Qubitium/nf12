# SPDX-License-Identifier: Apache-2.0
"""Build missing native libraries in a per-user, source-keyed cache."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import torch

# The HIP port currently ships only the C library; there is no ROCm torch
# extension yet, so the two backends produce different sets of artifacts.
_LIBRARIES_CUDA = ("libnf12.so", "libnf12_torch.so")
_LIBRARIES_HIP = ("libnf12_hip.so", "libnf12_torch.so")
_BUILD_ENV = (
    "CUDACXX", "CUDA_HOME", "CUDA_PATH", "CUDAHOSTCXX", "CXX", "CXXFLAGS",
    "CUDAFLAGS", "LDFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
)
_lock = threading.Lock()
_directory: Path | None = None


def _is_hip() -> bool:
    """True on an ROCm build."""
    return torch.version.hip is not None


def _libraries() -> tuple[str, ...]:
    return _LIBRARIES_HIP if _is_hip() else _LIBRARIES_CUDA


def _hip_compiler() -> str | None:
    """CMake needs amdclang++ directly; it rejects the hipcc wrapper."""
    explicit = os.environ.get("CMAKE_HIP_COMPILER")
    if explicit:
        return explicit
    found = shutil.which("amdclang++")
    if found:
        return found
    for root in (os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH"), "/opt/rocm"):
        if not root:
            continue
        candidate = Path(root) / "bin" / "amdclang++"
        if candidate.is_file():
            return str(candidate)
    return None


def _hip_architectures() -> list[str]:
    """Visible gfx targets, e.g. ``gfx1201`` (feature suffixes stripped)."""
    archs = sorted(
        {
            torch.cuda.get_device_properties(i).gcnArchName.split(":", 1)[0]
            for i in range(torch.cuda.device_count())
        }
    )
    if not archs:
        raise RuntimeError("nf12: JIT compilation requires a visible ROCm GPU")
    return archs


def _source_root() -> Path:
    package = Path(__file__).resolve().parent
    for root in (package / "_src", package.parent.parent):
        if ((root / "CMakeLists.txt").is_file() and (root / "csrc").is_dir()
                and (root / "include").is_dir()):
            return root
    raise ImportError(
        "nf12: native binaries and CUDA sources are missing; reinstall the package "
        "or use a source checkout"
    )


def _architectures() -> list[int]:
    archs = sorted({major * 10 + minor for major, minor in (
        torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
    )})
    if not archs:
        raise RuntimeError("nf12: JIT compilation requires a visible CUDA GPU")
    if archs[0] < 80:
        raise RuntimeError(
            f"nf12: JIT compilation requires SM80 or newer, found SM{archs[0]}; "
            "select supported GPUs with CUDA_VISIBLE_DEVICES"
        )
    return archs


def _cache_key(source: Path, archs: list[int]) -> str:
    digest = hashlib.sha256()
    identity = {
        "backend": "hip" if _is_hip() else "cuda",
        "archs": archs,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_path": str(Path(torch.__file__).resolve().parent),
        "python": sys.executable,
        "platform": (sys.platform, platform.machine()),
        "environment": {name: os.environ.get(name) for name in _BUILD_ENV},
        "source_path": str(source.resolve()),
    }
    if _is_hip():
        # A different ROCm compiler emits different device code, so it must
        # invalidate the cache exactly like CUDACXX does on the CUDA path.
        # ``_hip_compiler`` resolves the same precedence the build uses
        # (CMAKE_HIP_COMPILER, then PATH, then ROCM_PATH/HIP_PATH).
        identity["hip_compiler"] = _hip_compiler()
        identity["hip_toolkit"] = torch.version.hip
    digest.update(json.dumps(identity, sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    files = [source / "CMakeLists.txt"]
    for directory in ("csrc", "include"):
        files.extend(path for path in (source / directory).rglob("*") if path.is_file())
    for path in sorted(files):
        digest.update(str(path.relative_to(source)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _ready(directory: Path) -> bool:
    return (directory / "complete").is_file() and all(
        (directory / "build" / name).is_file() for name in _libraries()
    )


def _build(source: Path, directory: Path, archs) -> None:
    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    if not cmake or not ninja:
        raise RuntimeError("nf12: JIT compilation requires CMake 3.26+ and Ninja on PATH")

    if _is_hip():
        compiler = _hip_compiler()
        if not compiler:
            raise RuntimeError(
                "nf12: JIT compilation requires amdclang++ from the ROCm toolkit; "
                "set CMAKE_HIP_COMPILER or ROCM_PATH to locate it"
            )
        configure = [
            "-DNF12_BUILD_TORCH=ON",
            "-DNF12_ENABLE_HIP=ON",
            "-DNF12_HIP_ARCHS=" + ";".join(map(str, archs)),
            f"-DCMAKE_HIP_COMPILER={compiler}",
            f"-DPython_EXECUTABLE={sys.executable}",
        ]
        targets = ["nf12_hip", "nf12_torch"]
    else:
        # Import CUDA toolkit discovery only on a cache miss.
        from torch.utils.cpp_extension import CUDA_HOME

        nvcc = os.environ.get("CUDACXX")
        if not nvcc:
            nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc") if CUDA_HOME else "nvcc"
        nvcc = shutil.which(nvcc)
        if not nvcc:
            raise RuntimeError(
                "nf12: JIT compilation requires CUDA toolkit 13.0+ (nvcc); "
                "set CUDA_HOME or CUDACXX to locate CUDA"
            )
        configure = [
            "-DNF12_BUILD_TORCH=ON",
            f"-DCMAKE_CUDA_COMPILER={nvcc}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DNF12_CUDA_ARCHS=" + ";".join(map(str, archs)),
        ]
        targets = ["nf12", "nf12_torch"]

    build = directory / "build"
    log = directory / "build.log"
    env = os.environ.copy()
    # This is a runtime build for visible devices, not the wheel's target list.
    env.pop("NF12_CUDA_ARCHS", None)
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    commands = (
        [cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DNF12_BUILD_NATIVE=ON",
         f"-DCMAKE_MAKE_PROGRAM={ninja}", *configure],
        [cmake, "--build", str(build), "--target", *targets],
    )
    logging.getLogger(__name__).info("Building NF12 for %s; log: %s", archs, log)
    try:
        with log.open("w") as output:
            for command in commands:
                subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"nf12: JIT compilation failed; see {log}") from error
    if not all((build / name).is_file() for name in _libraries()):
        raise RuntimeError(
            f"nf12: JIT build did not produce {', '.join(_libraries())}; see {log}"
        )
    (directory / "complete").touch()


def library_directory() -> Path:
    global _directory
    with _lock:
        if _directory is not None:
            return _directory
        source = _source_root()
        archs = _hip_architectures() if _is_hip() else _architectures()
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        directory = cache.expanduser().resolve() / "nf12" / _cache_key(source, archs)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Keep the lock file: unlinking it lets waiters lock a different inode.
        with (directory / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not _ready(directory):
                (directory / "complete").unlink(missing_ok=True)
                _build(source, directory, archs)
        _directory = directory / "build"
        return _directory
