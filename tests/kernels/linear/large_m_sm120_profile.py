# SPDX-License-Identifier: Apache-2.0
"""One SM120-family call per source for Nsight Compute, e.g.
  uv run --group bench ncu --kernel-name regex:large_m_sm120_kernel --set full -o prof \\
      python tests/kernels/linear/large_m_sm120_profile.py --m 8192 --n 8192 --k 8192 --config 1
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import large_m_harness as H  # noqa: E402
import large_m_sm120_harness as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=8192)
ap.add_argument("--n", type=int, default=8192)
ap.add_argument("--k", type=int, default=8192)
ap.add_argument("--config", type=int, default=-1)
ap.add_argument("--escapes", type=float, default=0.0)
ap.add_argument("--sources", default="nf12,bf16t")
a = ap.parse_args()
shim = S.Sm120Shim()
w = (torch.randn(a.n, a.k, device="cuda") * 0.02).to(torch.bfloat16)
if a.escapes:
    w = H.force_escapes(w, a.escapes)
nw = H.NF12Packed(w)
x = torch.randn(a.m, a.k, device="cuda", dtype=torch.bfloat16)
for src in a.sources.split(","):
    shim.sm120(x, nw, source=src, config=a.config)
torch.cuda.synchronize()
