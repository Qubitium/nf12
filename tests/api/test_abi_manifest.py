# SPDX-License-Identifier: Apache-2.0
"""``abi/nf12-abi.json`` must stay derived from the sources it describes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "abi" / "nf12-abi.json"

# LP64 baseline, independent of the exporter's own layout arithmetic. The same
# numbers are checked against a compiled header in test_abi_contract.py.
DESCRIPTOR_BASELINE = {
    "nf12_weight_t": (48, {"packed": 0, "escapes": 8, "num_escapes": 16, "n": 24, "k": 32, "exp_base": 40}),
    "nf12_moe_weight_t": (
        56,
        {"packed": 0, "escapes": 8, "escape_offsets": 16, "exp_base": 24, "num_experts": 32, "n": 40, "k": 48},
    ),
}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_manifest_is_current():
    exported = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "export_abi.py"), "--stdout"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(exported.stdout) == json.loads(MANIFEST.read_text()), "run tools/export_abi.py"


def test_descriptor_layouts_match_the_lp64_contract(manifest):
    for name, (size, offsets) in DESCRIPTOR_BASELINE.items():
        descriptor = manifest["c_api"]["descriptors"][name]
        assert descriptor["size"] == size
        assert {field: entry["offset"] for field, entry in descriptor["fields"].items()} == offsets


def test_python_version_matches_the_header(manifest):
    major, minor = manifest["c_api"]["version"]["major"], manifest["c_api"]["version"]["minor"]
    assert manifest["python"]["version"].split(".")[:2] == [str(major), str(minor)]


def test_every_public_c_symbol_is_exported(manifest):
    version_script = manifest["exports"]["version_script"]
    assert version_script == ["nf12_*"]
    assert all(name.startswith("nf12_") for name in manifest["c_api"]["functions"])


def test_torch_registration_uses_the_stable_abi(manifest):
    assert manifest["torch"]["stable_abi"] is True
    assert set(manifest["torch"]["namespaces"]) == {"nf12"}
    assert manifest["torch"]["namespaces"]["nf12"], "no operator schemas exported"


def test_digest_covers_the_manifest(manifest):
    body = json.dumps({key: value for key, value in manifest.items() if key != "digest"}, sort_keys=True)
    assert manifest["digest"] == hashlib.sha256(body.encode()).hexdigest()[:16]
