# SPDX-License-Identifier: Apache-2.0
"""Exactness suite for tcgen05/UMMA large-M kernels (csrc/linear/sm100/large_m_tc.cu).

For every config of the family (and the heuristic, config=-1):
  * NF12 output matches the same config on the bf16t twin without tolerance,
  * the output is within a rigorous FP32-accumulation + BF16-rounding bound of
    an FP32 reference,
  * runs are deterministic, padded rows/columns of x/y (ldx > k, ldy > n) are
    honored and left untouched,
  * forced escapes (finite: bound-checked; Inf/NaN: bitwise vs bf16t),
  * CUDA-graph capture and replay.

Usage: uv run --group test python tests/kernels/linear/large_m_tc_test.py [--quick]
Set NF12_LARGE_M_BUILD_DIR to a private directory for concurrent runs.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import large_m_harness as H  # noqa: E402

FAM = H.FAMILY_TC
U = 2.0**-24  # FP32 unit roundoff
BF16_HALF_ULP_REL = 2.0**-8  # |round_bf16(v) - v| <= 2^-8 |v| for normal v


class Checker:
    def __init__(self, shim):
        self.s = shim
        self.cfgs = list(range(shim.num_configs(FAM))) + [-1]
        self.failures = []
        self.cases = 0

    def fail(self, msg):
        self.failures.append(msg)
        print("FAIL", msg, flush=True)

    def bound(self, x, w, bias):
        """Elementwise worst-case |y - ref| for any FP32 summation order.

        y is round_bf16(s) with s an FP32 sum of k products (+ bias), so
        |s - exact| <= k*u*A with A = |x| @ |w|^T + |bias|. The FP32 reference
        has the same bound, and BF16 rounding adds 2^-8 |s|.
        """
        k = x.shape[1]
        a = x.float().abs() @ w.float().abs().T
        if bias is not None:
            a = a + bias.float().abs()
        acc = 2 * (k + 1) * U * a
        return acc, a

    def run(self, x, P, w, bias, y, cfg, src):
        return self.s.linear(x, P, bias, y, source=src, family=FAM, config=cfg)

    def case(self, m, n, k, *, cfgs=None, bias=True, ldx_pad=0, ldy_pad=0, escapes=0.0, specials=False,
             seed=0, tag=""):
        g = torch.Generator(device="cuda").manual_seed(seed)
        w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).bfloat16()
        if escapes:
            w = H.force_escapes(w, escapes, seed=seed, specials=specials)
        xb = torch.randn(m, k + ldx_pad, device="cuda", generator=g).bfloat16()
        x = xb[:, :k]
        b = (torch.randn(n, device="cuda", generator=g) * 0.5).bfloat16() if bias else None
        P = H.NF12Packed(w)
        sentinel = torch.tensor(-12345.0, dtype=torch.bfloat16)
        ref = abs_bound = absval = None
        if not specials:
            torch.backends.cuda.matmul.allow_tf32 = False
            ref = x.float() @ w.float().T
            if b is not None:
                ref = ref + b.float()
            abs_bound, absval = self.bound(x, w, b)
        for cfg in cfgs if cfgs is not None else self.cfgs:
            self.cases += 1
            name = f"{tag} m={m} n={n} k={k} cfg={cfg} bias={bias} ldx+{ldx_pad} ldy+{ldy_pad} esc={escapes}"
            outs = {}
            for src in ("nf12", "bf16t"):
                yb = torch.full((m, n + ldy_pad), sentinel.item(), dtype=torch.bfloat16, device="cuda")
                y = yb[:, :n]
                self.run(x, P, w, b, y, cfg, src)
                if ldy_pad and not torch.equal(yb[:, n:].view(torch.int16), torch.full_like(yb[:, n:], sentinel.item()).view(torch.int16)):
                    self.fail(f"{name} {src}: wrote into ldy padding")
                outs[src] = y.clone()
            if not torch.equal(outs["nf12"].view(torch.int16), outs["bf16t"].view(torch.int16)):
                diff = (outs["nf12"].view(torch.int16) != outs["bf16t"].view(torch.int16)).sum().item()
                self.fail(f"{name}: NF12 != bf16t in {diff} elements")
            again = self.run(x, P, w, b, torch.empty(m, n, dtype=torch.bfloat16, device="cuda"), cfg, "nf12")
            if not torch.equal(again.view(torch.int16), outs["nf12"].view(torch.int16)):
                self.fail(f"{name}: nondeterministic")
            if ref is not None:
                y = outs["nf12"].float()
                tol = abs_bound + BF16_HALF_ULP_REL * (ref.abs() + abs_bound) + 1e-30
                bad = ~((y - ref).abs() <= tol)
                if bad.any():
                    i = bad.nonzero()[0].tolist()
                    self.fail(f"{name}: {bad.sum().item()} elements outside bound, e.g. {i}: "
                              f"y={y[i[0], i[1]].item()} ref={ref[i[0], i[1]].item()} tol={tol[i[0], i[1]].item()}")
                rel = ((y - ref).norm() / ref.norm().clamp_min(1e-30)).item()
                if rel > 4e-3:  # BF16 output rounding alone is ~1.1e-3 rms relative
                    self.fail(f"{name}: rms relative error {rel:.2e}")

    def graph(self, m, n, k, cfg):
        self.cases += 1
        torch.manual_seed(1)
        w = (torch.randn(n, k, device="cuda") * 0.05).bfloat16()
        P = H.NF12Packed(w)
        x = torch.randn(m, k, device="cuda").bfloat16()
        b = torch.randn(n, device="cuda").bfloat16()
        y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        nbytes = self.s.lib.nf12_test_large_m_ws(FAM, cfg, m, n, k)
        # Arbitrary contents on first use must be valid (no host-side init).
        ws = torch.randint(0, 256, (max(nbytes, 256),), dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.s.linear(x, P, b, y, family=FAM, config=cfg, ws=ws)  # warm-up (attribute setup) outside capture
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.s.linear(x, P, b, y, family=FAM, config=cfg, ws=ws)
        for trial in range(3):
            x.copy_(torch.randn(m, k, device="cuda").bfloat16())
            ws.fill_(0xA5)  # stale workspace contents must not matter
            graph.replay()
            torch.cuda.synchronize()
            eager = self.s.linear(x, P, b, torch.empty_like(y), family=FAM, config=cfg)
            if not torch.equal(eager.view(torch.int16), y.view(torch.int16)):
                self.fail(f"graph m={m} n={n} k={k} cfg={cfg} trial={trial}: replay != eager")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller M sweep")
    args = ap.parse_args()
    shim = H.Shim()
    if not shim.supported(FAM):
        print("tcgen05 family not supported on this device/build; nothing to test")
        return 0
    c = Checker(shim)
    print("configs:", [shim.config_name(FAM, i) for i in range(shim.num_configs(FAM))], flush=True)

    # Every M in 1..300 on a shape with a partial 128-row tile and odd k-block count.
    ms = range(1, 301, 7 if args.quick else 1)
    for m in ms:
        c.case(m, 272, 576, bias=(m % 2 == 0), seed=m, tag="sweep")
    print(f"M sweep done ({c.cases} cases, {len(c.failures)} failures)", flush=True)

    # Sparse M up to 16384, several shapes (stream-K splits, multi-wave, tails).
    for m in (301, 383, 512, 640, 1000, 1024, 2047, 2048, 4096, 4097, 8192, 16384):
        c.case(m, 1040, 1024, seed=m, tag="sparse")
    for m in (16, 96, 256, 1000):
        c.case(m, 2048, 4096, seed=m, tag="deep-k")

    for m in (1, 40, 257, 600):
        c.case(m, 400, 448, ldx_pad=64, ldy_pad=24, bias=True, seed=m, tag="strides")
        c.case(m, 400, 448, ldx_pad=8, ldy_pad=8, bias=False, seed=m + 1, tag="strides")

    # Finite escapes use the FP32 bound, specials use bitwise comparison.
    # Decoders prefetch 128 escape entries past an escaped stage's last entry.
    # Fractions 0.10 / 0.30 / 0.80 leave ~12 / ~33 / ~70 escaped groups per
    # super-atom, so prefetch reaches ~10 / ~4 / ~2 stages ahead.
    for m in (5, 128, 700):
        c.case(m, 528, 1024, escapes=0.02, seed=m, tag="esc")
        c.case(m, 528, 1024, escapes=0.02, specials=True, seed=m, tag="esc-special")
        c.case(m, 528, 1024, escapes=0.10, seed=m, tag="esc-heavy")
        c.case(m, 528, 1024, escapes=0.30, specials=True, seed=m, tag="esc-30-special")
        c.case(m, 528, 1024, escapes=0.80, seed=m, tag="esc-dense")
    for m in (512, 3000):
        c.case(m, 4096, 4096, escapes=0.10, specials=True, seed=m, tag="esc-heavy-special")
        c.case(m, 4096, 4096, escapes=0.30, seed=m, tag="esc-30")

    # Real-model shapes (K x N): Llama-3-8B/70B, Qwen-32B-ish.
    for k, n in ((4096, 6144), (4096, 4096), (4096, 28672), (14336, 4096), (5120, 25600), (25600, 5120),
                 (8192, 8192)):
        for m in (33, 512, 2000):
            c.case(m, n, k, seed=k + n + m, tag="model")

    # CUDA graphs: stream-K (tag memset node + kernel) and data-parallel configs.
    for cfg in c.cfgs:
        c.graph(200, 4096, 4096, cfg)
        c.graph(3000, 1024, 1024, cfg)

    if os.environ.get("NF12_HUGE_TESTS") == "1":
        huge(c)

    for cfg in range(shim.num_configs(FAM)):
        for m, n, k in ((200, 4096, 4096), (40, 1040, 2048), (600, 2048, 1024)):
            prefilled_tags(c, cfg, m, n, k)

    print(f"{c.cases} cases, {len(c.failures)} failures")
    return 1 if c.failures else 0


def prefilled_tags(c, cfg, m, n, k):
    """Prefill stream-K workspace with this launch's tags and NaN partials.

    Reproduce the launcher's plan and check its size against
    linear_large_m_tc_workspace_bytes. Per-launch tag reset must make stale
    contents irrelevant, so NF12 and bf16t must each match a clean-workspace run.
    """
    name = c.s.config_name(FAM, cfg)
    if name.endswith("_cs"):
        # Every reduction input must be overwritten before it is read.
        nbytes = c.s.lib.nf12_test_large_m_ws(FAM, cfg, m, n, k)
        if nbytes == 0:
            return
        torch.manual_seed(m + n + k)
        P = H.NF12Packed((torch.randn(n, k, device="cuda") * 0.05).bfloat16())
        x = torch.randn(m, k, device="cuda").bfloat16()
        ref = c.s.linear(x, P, None, None, family=FAM, config=cfg,
                         ws=torch.zeros(nbytes, dtype=torch.uint8, device="cuda"))
        c.cases += 1
        for src in ("nf12", "bf16t"):
            ws = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            ws.view(torch.float32).fill_(float("nan"))
            out = c.s.linear(x, P, None, None, source=src, family=FAM, config=cfg, ws=ws)
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                c.fail(f"prefilled workspace {name} m={m} n={n} k={k}: {src} differs from the clean-workspace run")
        return
    ctas, bn = int(name[2]), int(name.split("_bn")[1].split("_")[0])
    if bn == 512 and c.s.lib.nf12_test_large_m_ws(FAM, cfg, m, n, k) == 0:
        return  # BN = 512 runs data-parallel where its time model says so: no workspace
    units = torch.cuda.get_device_properties(0).multi_processor_count // ctas
    n_tiles, m_tiles, kb = -(-n // (128 * ctas)), -(-m // bn), k // 64
    tiles = n_tiles * m_tiles
    if tiles % units == 0:
        return
    dp = max(0, tiles // units - 1) * units
    sk_units = (tiles - dp) * kb
    grid = units * ctas
    partial_bytes = grid * bn * 128 * 4
    nbytes = c.s.lib.nf12_test_large_m_ws(FAM, cfg, m, n, k)
    assert nbytes == partial_bytes + ((grid * 8 + 255) & ~255), (name, nbytes)
    tags = torch.zeros(grid, dtype=torch.int64)
    published = 0
    for u in range(units):
        a, e = sk_units * u // units, sk_units * (u + 1) // units
        if a < e and a % kb:
            tags[u * ctas:(u + 1) * ctas] = ((dp + a // kb + 1) << 32) | (a % kb)
            published += 1
    assert published, name
    torch.manual_seed(m + n + k)
    w = (torch.randn(n, k, device="cuda") * 0.05).bfloat16()
    P = H.NF12Packed(w)
    x = torch.randn(m, k, device="cuda").bfloat16()
    ref = c.s.linear(x, P, None, None, family=FAM, config=cfg,
                     ws=torch.zeros(nbytes, dtype=torch.uint8, device="cuda"))
    c.cases += 1
    for src in ("nf12", "bf16t"):
        ws = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        ws[:partial_bytes].view(torch.float32).fill_(float("nan"))
        ws[partial_bytes:partial_bytes + grid * 8].view(torch.int64).copy_(tags)
        out = c.s.linear(x, P, None, None, source=src, family=FAM, config=cfg, ws=ws)
        if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
            c.fail(f"prefilled tags {name} m={m} n={n} k={k}: {src} differs from the clean-workspace run")


def encode_in_window(w, exp_base):
    """NF12-encode a big in-window matrix 16-row block by block (bounded memory).

    Each 16-row block's super-atoms are contiguous, so per-block encodings
    concatenate. Escapes are forbidden because their indices would restart.
    """
    n, k = w.shape
    packed = torch.empty(n, k * 3 // 2, dtype=torch.uint8, device=w.device)
    bf16t = torch.empty(n, k * 2, dtype=torch.uint8, device=w.device)
    rows = max(16, (1 << 25) // k // 16 * 16)
    for r0 in range(0, n, rows):
        p = H.NF12Packed(w[r0:r0 + rows], exp_base=exp_base)
        assert p.num_escapes == 0
        packed[r0:r0 + rows] = p.packed
        bf16t[r0:r0 + rows] = p.bf16t
    P = H.NF12Packed.__new__(H.NF12Packed)
    P.n, P.k, P.exp_base, P.num_escapes = n, k, exp_base, 0
    P.packed, P.bf16t = packed, bf16t
    P.escapes = torch.zeros(1, 4, dtype=torch.int32, device=w.device)
    return P


def huge(c):
    """Opt-in (NF12_HUGE_TESTS=1, ~25 GB): byte offsets beyond 2^31.

    Weight stream N*K*3/2 = 3.2 GB has row-block offsets > 2^31 bytes
    at K = 2^21. Output M*ldy exceeds 2^31 elements.
    """
    def run(m, n, k, w, P, cfg, tag, chunk_k=1 << 18, chunk_m=8192):
        x = torch.randn(m, k, device="cuda").bfloat16()
        y = {}
        for src in ("nf12", "bf16t"):
            y[src] = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            c.s.linear(x, P, None, y[src], source=src, family=FAM, config=cfg)
        torch.cuda.synchronize()
        c.cases += 1
        if not torch.equal(y["nf12"].view(torch.int16), y["bf16t"].view(torch.int16)):
            c.fail(f"huge {tag} cfg={cfg}: NF12 != bf16t")
        bad = 0
        for m0 in range(0, m, chunk_m):
            xs = x[m0:m0 + chunk_m]
            ref = torch.zeros(xs.shape[0], n, dtype=torch.float64, device="cuda")
            mag = torch.zeros_like(ref)
            for k0 in range(0, k, chunk_k):
                xf, wf = xs[:, k0:k0 + chunk_k].float(), w[:, k0:k0 + chunk_k].float()
                ref += (xf @ wf.T).double()
                mag += (xf.abs() @ wf.abs().T).double()
            tol = 2 * (k + 1) * U * mag + BF16_HALF_ULP_REL * (ref.abs() + 2 * (k + 1) * U * mag)
            bad += int(((y["nf12"][m0:m0 + chunk_m].double() - ref).abs() > tol).sum().item())
        if bad:
            c.fail(f"huge {tag} cfg={cfg}: {bad} elements outside bound")
        print(f"huge {tag} cfg={cfg} m={m} n={n} k={k}: {'ok' if not bad else 'FAIL'}", flush=True)

    exp_base = 60  # window: biased exponents 120..135, i.e. |w| in [2^-7, 2^9)

    def in_window(n, k):
        mag = torch.exp2(torch.empty(n, k, device="cuda").uniform_(-7.0, 8.9))
        sign = torch.where(torch.rand(n, k, device="cuda") < 0.5, -1.0, 1.0)
        return (mag * sign).bfloat16()

    n, k = 1024, 1 << 21
    w = in_window(n, k)
    P = encode_in_window(w, exp_base)
    for cfg in (-1, 3):
        run(64, n, k, w, P, cfg, "weights>2^31B")
    del w, P
    torch.cuda.empty_cache()

    n, k = 32768, 64
    w = in_window(n, k)
    P = encode_in_window(w, exp_base)
    run(70000, n, k, w, P, -1, "y>2^31elem")


if __name__ == "__main__":
    sys.exit(main())
