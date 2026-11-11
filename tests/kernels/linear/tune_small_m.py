# SPDX-License-Identifier: Apache-2.0
"""Fits the constants of small_m_heuristic()'s time model (SmallMModel in
linear/small_m.cu) to sweep results.

  uv run --group bench python tests/kernels/linear/bench_small_m.py --sweep --model-slices --json sweep.json
  uv run --group bench python tests/kernels/linear/tune_small_m.py sweep.json [--init sat,wave,tail,cover,split,cta,loc,x,slice,reduce,warp,chain]

Replays the C++ model over each point's swept configs in the heuristic's
(warps, kp, kc) order with its paired x slice. Run on the GPU that produced
the sweep because occupancy comes from the library. Coordinate descent
starts from --init and minimizes geomean regret (chosen time / best time).
Prints fitted constants and points above 3% regret. Fits each weight ring
(small_m_stages()) separately because SM90 uses both rings with distinct
constants.
"""

import argparse
import json
import math
import os
import random
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import small_m_harness as h  # noqa: E402

NAMES = ["saturation", "wave", "tail", "coverage", "split", "cta", "locality", "x", "slice", "reduce", "warp",
         "chain"]
# Values tried per constant in each descent pass (besides the current one).
STEPS = {
    "saturation": [0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 4.0, 6.0, 8.0],
    "wave": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "tail": [0.0, 0.05, 0.1, 0.2, 0.3],
    "coverage": [0.5, 0.6, 0.75, 0.85, 1.0],
    "split": [0.0, 0.05, 0.1, 0.2, 0.4, 0.8],
    "cta": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "locality": [0.0, 0.1, 0.2, 0.3, 0.5, 0.8],
    "x": [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "slice": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "reduce": [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0, 160.0, 240.0],
    "warp": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "chain": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
}

ap = argparse.ArgumentParser()
ap.add_argument("sweeps", nargs="+")
ap.add_argument("--init", default="3.0,4.0,0.0,0.5,0.4,4.0,0.0,0.0,0.0,0.0,0.0,0.0")
ap.add_argument("--fix", default="", help="comma-separated constants to keep at --init")
ap.add_argument("--restarts", type=int, default=6, help="extra descents from random starting points")
a = ap.parse_args()

pts = [p for path in a.sweeps for p in json.load(open(path))["results"]]
lib = h.Lib()
sms = torch.cuda.get_device_properties(0).multi_processor_count
sm8x = torch.cuda.get_device_capability()[0] < 9
sm103 = torch.cuda.get_device_capability() == (10, 3)
CFG = re.compile(r"mt(\d+)_w(\d+)_kc(\d+)_s(\d+)(?:_kp(\d+))?$")

# Group model inputs by weight ring, retaining the heuristic's iteration
# order (warps 16, 8, 4, then ascending kp and kc). Strict improvement keeps
# the first candidate in a tie.
groups = {}
for p in pts:
    m, n, k = p["m"], p["n"], p["k"]
    kb, nb = k // 64, n // 16
    table = {c: t for t, c in p["all"]}
    cands = []
    for name in table:
        mt, warps, kc, sl, kp = (int(v or 1) for v in CFG.match(name).groups())
        cnt = -(-kb // kc)
        if (warps == 4 and mt == 8 and cnt // kp > 16) or (warps == 16 and nb < 16):
            continue
        if sm103:
            if warps > 8 or (k <= 128 and (warps != 4 or kc != 1)):
                continue
            if kp > 1 and (m > 8 or (n >= 4096 and k >= 4096)):
                continue
        o = lib.resident_ctas(h.Config(mt, warps, kc, sl, kp), m, n, k)
        if o <= 0:
            continue
        total = -(-nb // (warps // kp)) * kc
        streamed_nb = -(-nb // (warps // kp)) * (warps // kp) if sm103 else nb
        slices = 1 if sl == 0 else -(-cnt // sl)
        cands.append(((-warps, kp, kc), name, dict(cap=o * sms, total=total, per=streamed_nb * kb / total, cnt=cnt,
                                                  warps=warps, kp=kp, kc=kc, mt=mt, slices=slices)))
    cands.sort(key=lambda c: c[0])
    groups.setdefault(lib.stages(1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8, n, k), []).append(
        (p, [(name, c) for _, name, c in cands], table, min(table.values())))


def model(c, m, n, prm):
    sat, wave, tail, cover, split, cta, loc, xc, slc, red, wrp, chain = prm
    t, rem = 0.0, c["total"]
    cov = 0.85 * cover if (c["mt"] == 8 and c["cnt"] > 16) else cover
    while rem > 0:
        w = min(rem, c["cap"])
        rate = min(1.0, w * c["warps"] * min(1.0, c["cnt"] / c["kp"]) / 8.0 / (sat * sms), min(w, sms) / (cov * sms))
        stream = w * c["per"] / rate * (1.0 + loc / c["kp"])
        t += max(stream, chain * (-(-c["cnt"] // c["kp"])) * sms) + wave * (1.0 + m / 8.0) * sms
        t += split * m * (c["kc"] * c["kp"] - 1) * sms
        t += slc * c["slices"] * sms
        rem -= w
    if c["total"] > c["cap"]:
        t += tail * c["per"] * c["cap"]
    t += (cta + wrp * c["warps"]) * c["total"]
    t += xc * c["total"] * m * c["cnt"] * 128.0 / 1536
    if sm8x and c["kc"] > 1:
        t += red * sms + 8.0 * c["kc"] * m * n / 1536
    return t


def choose(cands, m, n, prm):
    best, bt = None, None
    for name, c in cands:
        t = model(c, m, n, prm)
        if bt is None or t < bt:
            best, bt = name, t
    return best


def score(points, prm):
    regrets = [table[choose(cands, p["m"], p["n"], prm)] / best for p, cands, table, best in points]
    return math.exp(sum(map(math.log, regrets)) / len(regrets)), max(regrets), regrets


fixed = set(filter(None, a.fix.split(",")))


def descend(points, prm):
    geo, worst, _ = score(points, prm)
    improved = True
    while improved:
        improved = False
        for i, name in enumerate(NAMES):
            if name in fixed:
                continue
            for v in STEPS[name]:
                trial = prm[:i] + [v] + prm[i + 1:]
                g, wst, _ = score(points, trial)
                if g < geo - 1e-6 or (abs(g - geo) <= 1e-6 and wst < worst - 1e-6):
                    prm, geo, worst, improved = trial, g, wst, True
    return geo, worst, prm


rng = random.Random(0)
for stages, points in sorted(groups.items()):
    print(f"# {'register ring' if stages == 0 else f'shared-memory ring ({stages} stages)'}: {len(points)} points")
    init = [float(v) for v in a.init.split(",")]
    geo, worst, _ = score(points, init)
    print(f"init geo {geo:.4f} worst {worst:.3f}")
    starts = [init] + [[v if n in fixed else rng.choice(STEPS[n]) for n, v in zip(NAMES, init)]
                       for _ in range(a.restarts)]
    geo, worst, prm = min(descend(points, s) for s in starts)
    geo, worst, regrets = score(points, prm)
    print(f"fit  geo {geo:.4f} worst {worst:.3f} " + ", ".join(f"{n}={v:g}" for n, v in zip(NAMES, prm)))
    for (p, cands, table, best), r in zip(points, regrets):
        if r > 1.03:
            print(f"K={p['k']:6d} N={p['n']:6d} M={p['m']:3d} regret {r:.3f} chose {choose(cands, p['m'], p['n'], prm)} "
                  f"best {p['best_cfg']}")
