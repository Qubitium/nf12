# SPDX-License-Identifier: Apache-2.0
"""Sample nvidia-smi clocks, power and throttle reasons while running
compute-bound NF12 / bf16t / cuBLAS GEMMs for --seconds each.
Reports median call time alongside median SM clock and power to investigate
decode issue-slot costs and board power limits.

    CUDA_VISIBLE_DEVICES=0 flock /tmp/nf12-sm120.lock uv run --group bench python tests/kernels/linear/power_probe.py --shape 4096x14336 --m 8192
"""

import argparse
import os
import statistics
import subprocess
import sys
import threading
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "bench"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
import common as BC  # noqa: E402
import nf12  # noqa: E402


def sample(stop, out, index):
    q = "clocks.sm,power.draw,clocks_throttle_reasons.sw_power_cap"
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", "-i", index, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            clk, pw, cap = (v.strip() for v in r.stdout.strip().split(","))
            out.append((float(clk), float(pw), cap))
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="4096x14336")
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--gpu-index", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                    help="nvidia-smi index of the GPU under test")
    a = ap.parse_args()
    shape = BC.shape_by_key(a.shape)
    w, _ = BC.make_weight(shape, "real", torch.device("cuda"))
    nw = nf12.encode(w)
    bt = nf12.pack_bf16t(w)
    x = torch.randn(a.m, shape.k, device="cuda", dtype=torch.bfloat16)
    fns = {"nf12": lambda: nf12.linear(x, nw), "bf16t": lambda: nf12.linear_bf16t(x, bt),
           "cublas": lambda: x @ w.T}
    for name, fn in fns.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(10):
                fn()
        samples, stop = [], threading.Event()
        th = threading.Thread(target=sample, args=(stop, samples, a.gpu_index))
        times = []
        th.start()
        t_end = time.time() + a.seconds
        while time.time() < t_end:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e) * 100.0)  # us per call (10 calls per replay)
        stop.set()
        th.join()
        steady = samples[len(samples) // 3:]
        print(f"{a.shape} M={a.m} {name:>6}: {statistics.median(times):9.1f} us/call  "
              f"SM {statistics.median(c for c, _, _ in steady):6.0f} MHz  "
              f"{statistics.median(p for _, p, _ in steady):5.0f} W  "
              f"power-capped {sum(cap == 'Active' for _, _, cap in steady)}/{len(steady)} samples", flush=True)


if __name__ == "__main__":
    main()
