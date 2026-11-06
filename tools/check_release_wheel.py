#!/usr/bin/env python3
"""Check release wheel payloads, or smoke-load an installed wheel without a GPU."""

from __future__ import annotations

import argparse
import ctypes
from email.parser import BytesParser
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


def check_wheel(wheel: Path, arches: str, platform: str) -> None:
    from packaging.specifiers import SpecifierSet
    from packaging.utils import parse_wheel_filename
    import tomli

    root = Path(__file__).resolve().parents[1]
    project = tomli.loads((root / "pyproject.toml").read_text())["project"]
    name, version, _, tags = parse_wheel_filename(wheel.name)
    assert name == "nf12" and str(version) == project["version"], wheel.name
    assert all(tag.interpreter == "py3" and tag.abi == "none" for tag in tags), tags
    assert platform in {tag.platform for tag in tags}, tags
    assert wheel.stat().st_size < 100 * 1024 * 1024, "Wheel exceeds PyPI's default size limit"
    expected = {f"{arch}a" if arch in {"90", "100", "103", "110", "120", "121"} else arch
                for arch in arches.split(";")}
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as temporary:
        names = set(archive.namelist())
        metadata_name, = (name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        python = SpecifierSet(metadata["Requires-Python"])
        assert "3.9" not in python and all(v in python for v in ("3.10", "3.14", "3.15"))
        assert "nf12/_jit.py" in names and "nf12/_src/CMakeLists.txt" in names
        assert "nf12/licenses/NVIDIA_CUDA_RUNTIME.txt" in names
        for directory in ("csrc", "include"):
            for source in (root / directory).rglob("*"):
                if source.suffix in {".cu", ".cpp", ".cuh", ".h", ".map"}:
                    assert f"nf12/_src/{source.relative_to(root)}" in names, source
        for library in ("libnf12.so", "libnf12_torch.so"):
            path = Path(archive.extract(f"nf12/{library}", temporary))
            elf = subprocess.check_output(["cuobjdump", "--list-elf", str(path)], text=True)
            found = set(re.findall(r"\.sm_(\d+a?)\.", elf))
            assert expected <= found, f"{library}: missing SASS targets {expected - found}"
            ptx = subprocess.check_output(["cuobjdump", "--list-ptx", str(path)], text=True)
            assert f".sm_{max(map(int, arches.split(';')))}." in ptx, f"{library}: missing forward PTX"
            symbols = subprocess.check_output(["nm", "-D", "--undefined-only", str(path)], text=True)
            assert not any(line.split()[-1].startswith(("Py", "_Py")) for line in symbols.splitlines()), \
                f"{library}: depends on the Python C API"
            print(f"{library}: SASS {', '.join(sorted(found, key=lambda arch: int(arch.rstrip('a'))))}; PTX present")
    print(f"Checked {wheel.name} ({wheel.stat().st_size / 1024**2:.1f} MiB)")


def smoke_load() -> None:
    import torch
    import nf12
    from nf12._library import library_path

    nf12.load()
    assert "nf12._jit" not in sys.modules, "Installed wheel tried to JIT compile"
    native = ctypes.CDLL(str(library_path()))
    native.nf12_packed_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64]
    native.nf12_packed_bytes.restype = ctypes.c_size_t
    assert native.nf12_packed_bytes(16, 64) == 1536
    x = torch.empty((2, 64), dtype=torch.bfloat16, device="meta")
    packed = torch.empty((16, 96), dtype=torch.uint8, device="meta")
    escapes = torch.empty((0, 8), dtype=torch.int16, device="meta")
    y = torch.ops.nf12.linear(x, packed, escapes, 0)
    assert y.shape == (2, 16) and y.dtype == torch.bfloat16
    print(f"Loaded native C API and Torch ops: Python {sys.version.split()[0]}, Torch {torch.__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, nargs="?")
    parser.add_argument("--arches")
    parser.add_argument("--platform")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    if args.load:
        smoke_load()
    elif args.wheel and args.arches and args.platform:
        check_wheel(args.wheel, args.arches, args.platform)
    else:
        parser.error("supply WHEEL --arches TARGETS --platform TAG, or --load")
