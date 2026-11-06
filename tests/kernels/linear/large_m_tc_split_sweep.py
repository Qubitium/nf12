# SPDX-License-Identifier: Apache-2.0
"""Sweep of the tcgen05 cluster split-K configs against the heuristic and cuBLAS (SM100).

For every (shape, M): cuBLAS, the heuristic config, and each cluster split-K
config (tm1_bn256_cs, tm1_bn64_cs, tm1_bn128_cs) at every K split kc that
keeps tiles x kc within one resident wave, graph-timed over rotated weight
copies (bench/common.py). The K split is forced through NF12_TC_KC, which
large_m_tc.cu reads only in builds made with -DNF12_TC_SWEEP (the harness
passes NF12_NVCC_FLAGS through). Results go to a JSON list for fitting.

Usage: NF12_NVCC_FLAGS=-DNF12_TC_SWEEP NF12_LARGE_M_BUILD_DIR=build/sweep \\
       uv run --group bench python tests/kernels/linear/large_m_tc_split_sweep.py --shapes 4096x4096,... --ms 16,32,64 --out sweep.json
"""

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "bench"))
import common as C  # noqa: E402
import large_m_harness as H  # noqa: E402

SPLIT_CONFIGS = {5: 256, 7: 64, 8: 128}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True, help="comma list of KxN")
    ap.add_argument("--ms", required=True)
    ap.add_argument("--configs", default="5,7,8")
    ap.add_argument("--max-kc", type=int, default=8)
    ap.add_argument("--budget-s", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock", help="flock file taken per (shape, M)")
    args = ap.parse_args()
    shim = H.Shim()
    fam = H.FAMILY_TC
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = []
    for shape in args.shapes.split(","):
        k, n = (int(v) for v in shape.split("x"))
        torch.manual_seed(0)
        w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
        P = H.NF12Packed(w)
        copies = C.cold_copies(P.bytes_nf12())
        ps, wts = [P], [w]
        for _ in range(copies - 1):
            c = copy.copy(P)
            c.packed, c.escapes, c.bf16t = P.packed.clone(), P.escapes.clone(), None
            ps.append(c)
            wts.append(w.clone())
        for m in (int(v) for v in args.ms.split(",")):
            with C.gpu_lock(args.lock):
                rows.append(measure(shim, fam, sms, args, k, n, m, ps, wts, copies))
            with open(args.out, "w") as f:
                json.dump(rows, f)


def measure(shim, fam, sms, args, k, n, m, ps, wts, copies):
    x = torch.randn(m, k, device="cuda").bfloat16()
    y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    row = {"k": k, "n": n, "m": m}
    row["cublas"] = C.time_graph(lambda i: torch.matmul(x, wts[i].T), copies,
                                 budget_s=args.budget_s)["median_us"]

    def run(cfg):
        ws, _ = shim.workspace(fam, cfg, m, n, k)
        return C.time_graph(lambda i: shim.linear(x, ps[i], None, y, family=fam, config=cfg, ws=ws), copies,
                            budget_s=args.budget_s)["median_us"]

    os.environ.pop("NF12_TC_KC", None)
    row["auto"] = run(-1)
    for cfg in (int(c) for c in args.configs.split(",")):
        bn = SPLIT_CONFIGS[cfg]
        tiles = -(-n // 128) * -(-m // bn)
        for kc in range(1, min(args.max_kc, k // 64) + 1):
            if tiles * kc > sms and kc > 1:
                break
            os.environ["NF12_TC_KC"] = str(kc)
            row[f"{cfg}/{kc}"] = run(cfg)
    os.environ.pop("NF12_TC_KC", None)
    best = min((v, key) for key, v in row.items() if key not in ("k", "n", "m", "cublas"))
    print(f"{k}x{n} M={m}: cublas {row['cublas']:.1f} auto {row['auto']:.1f} best {best[1]} {best[0]:.1f} "
          f"({row['cublas'] / best[0]:.2f}x)", flush=True)
    return row


if __name__ == "__main__":
    main()
