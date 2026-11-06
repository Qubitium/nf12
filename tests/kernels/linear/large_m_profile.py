# SPDX-License-Identifier: Apache-2.0
"""Run one large-M linear call per source for profilers (ncu/nsys):
uv run --group bench python tests/kernels/linear/large_m_profile.py K N M [config] [--family mma|tc|auto]."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import large_m_harness as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("m", type=int)
    ap.add_argument("config", type=int, nargs="?", default=-1)
    ap.add_argument("--family", choices=["mma", "tc", "auto"], default="mma")
    ap.add_argument("--sources", default="nf12,bf16t")
    a = ap.parse_args()
    fam = {"mma": H.FAMILY_MMA, "tc": H.FAMILY_TC, "auto": H.FAMILY_AUTO}[a.family]
    shim = H.Shim()
    w = (torch.randn(a.n, a.k, device="cuda") * 0.02).to(torch.bfloat16)
    nw = H.NF12Packed(w)
    x = torch.randn(a.m, a.k, device="cuda", dtype=torch.bfloat16)
    for src in a.sources.split(","):
        shim.linear(x, nw, source=src, family=fam, config=a.config)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
