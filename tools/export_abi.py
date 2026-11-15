#!/usr/bin/env python3
"""Export the public NF12 ABI as ``abi/nf12-abi.json``.

The manifest is derived, never hand-edited: the C entry points and descriptor
layouts from ``include/nf12/nf12.h``, the torch operator schemas from
``csrc/torch/ops.cpp``, the exported symbol globs from ``csrc/nf12.map`` and the
public Python names from ``python/nf12/__init__.py``. Downstream consumers pin
the digest instead of parsing headers, and ``tests/api/test_abi_manifest.py``
fails when the committed file stops matching the sources.

    python tools/export_abi.py                  # rewrite abi/nf12-abi.json
    python tools/export_abi.py --check          # exit 1 when it is stale
    python tools/export_abi.py --stdout         # print without writing
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "abi" / "nf12-abi.json"
HEADER = ROOT / "include" / "nf12" / "nf12.h"
TORCH_OPS = ROOT / "csrc" / "torch" / "ops.cpp"
EXPORT_MAP = ROOT / "csrc" / "nf12.map"
PACKAGE = ROOT / "python" / "nf12"

# LP64 sizes and alignments of the field types the descriptors use.
FIELD_LAYOUT = {
    "const void*": (8, 8),
    "const int64_t*": (8, 8),
    "const uint8_t*": (8, 8),
    "int64_t": (8, 8),
    "uint32_t": (4, 4),
    "uint8_t": (1, 1),
}


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def c_api() -> dict:
    text = HEADER.read_text()
    body = strip_comments(text)
    functions = {}
    for match in re.finditer(
        r"\b(nf12_\w+|size_t|int64_t|const char\*)\s+(nf12_\w+)\s*\(([^;{]*?)\)\s*;", body, re.S
    ):
        functions[match.group(2)] = f"{match.group(1)} {match.group(2)}(" + " ".join(match.group(3).split()) + ")"
    descriptors = {}
    for match in re.finditer(r"typedef struct (\w+) \{(.*?)\} (\w+);", body, re.S):
        fields = [" ".join(field.split()) for field in match.group(2).split(";") if field.strip()]
        offset, size, alignment = 0, 0, 1
        layout = {}
        for field in fields:
            kind, name = field.rsplit(" ", 1)
            width, align = FIELD_LAYOUT[kind]
            offset = (offset + align - 1) // align * align
            layout[name] = {"offset": offset, "type": kind}
            offset += width
            size = offset
            alignment = max(alignment, align)
        size = (size + alignment - 1) // alignment * alignment
        descriptors[match.group(3)] = {"size": size, "align": alignment, "fields": layout}
    version = {
        "major": int(re.search(r"#define NF12_VERSION_MAJOR (\d+)", text).group(1)),
        "minor": int(re.search(r"#define NF12_VERSION_MINOR (\d+)", text).group(1)),
    }
    status = {name: int(value) for name, value in re.findall(r"(NF12_\w+) = (\d+)", body)}
    return {"version": version, "status": status, "functions": functions, "descriptors": descriptors}


def torch_ops() -> dict:
    text = TORCH_OPS.read_text()
    namespaces = {}
    for block in re.finditer(r"STABLE_TORCH_LIBRARY\((\w+), m\)\s*\{(.*?)\n\}", text, re.S):
        schemas = []
        for call in re.finditer(r"\bm\.def\((.*?)\)\s*;", block.group(2), re.S):
            schema = " ".join("".join(re.findall(r'"([^"]*)"', call.group(1))).split())
            if schema:
                schemas.append(schema)
        namespaces[block.group(1)] = schemas
    return {"stable_abi": "STABLE_TORCH_LIBRARY" in text, "namespaces": namespaces}


def symbol_exports() -> dict:
    return {"version_script": re.findall(r"(nf12_\*)", EXPORT_MAP.read_text())}


def python_api() -> dict:
    init = ast.parse((PACKAGE / "__init__.py").read_text())
    exports, version = {}, None
    for node in init.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target == "__version__":
                version = ast.literal_eval(node.value)
            elif target == "_EXPORTS":
                exports = {ast.literal_eval(key): ast.literal_eval(value) for key, value in zip(node.value.keys, node.value.values)}
    signatures = {}
    for name, module in sorted(exports.items()):
        tree = ast.parse((PACKAGE / f"{module.lstrip('.')}.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                signatures[name] = f"({ast.unparse(node.args)})"
                break
    return {"version": version, "exports": sorted(exports), "signatures": signatures}


def build() -> dict:
    manifest = {
        "schema": 1,
        "repository": "dphnAI/nf12",
        "c_api": c_api(),
        "torch": torch_ops(),
        "exports": symbol_exports(),
        "python": python_api(),
    }
    body = json.dumps({key: value for key, value in manifest.items() if key != "digest"}, sort_keys=True)
    manifest["digest"] = hashlib.sha256(body.encode()).hexdigest()[:16]
    return manifest


def render(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail when abi/nf12-abi.json is stale")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args()
    text = render(build())
    if args.stdout:
        sys.stdout.write(text)
        return 0
    if args.check:
        if not MANIFEST.is_file() or MANIFEST.read_text() != text:
            print(f"{MANIFEST.relative_to(ROOT)} is stale; run tools/export_abi.py")
            return 1
        return 0
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(text)
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
