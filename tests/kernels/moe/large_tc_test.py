# SPDX-License-Identifier: Apache-2.0
"""Exactness suite for tcgen05 grouped-GEMM moe_linear (csrc/moe/sm100/large_tc.cu).

For every config of the family (and the heuristic, config=-1), over routings
(uniform, skewed onto one expert, most experts empty, dropped ids), topk 1..8,
and the gather / scatter / row_scale combinations of SPEC.md 9.2:
  * NF12 output matches the same config on the bf16t twin without tolerance,
  * the output is within a rigorous FP32-accumulation + BF16-rounding bound of
    an FP32 reference,
  * rows of y no valid position writes, and padding columns (ldy > n), are
    untouched, and ldx > k is honored,
  * runs are deterministic, with the workspace filled with garbage each time,
  * forced escapes (heavy finite: bound-checked; Inf/NaN: bitwise vs bf16t),
  * a captured CUDA graph replays correctly after the routing changes in place.
--huge adds byte offsets beyond 2^31 in x, y and the weights.

Usage: uv run --group test python tests/kernels/moe/large_tc_test.py [--quick] [--huge]
Run on Thor under `flock /tmp/nf12-thor.lock`. Set NF12_MOE_TC_BUILD_DIR to a
private directory for concurrent runs.
"""

import argparse
import ctypes
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import large_tc_harness as H  # noqa: E402

U = 2.0**-24  # FP32 unit roundoff
SENTINEL = -12345.0


def routing(T, topk, E, kind, gen):
    """topk_ids [T, topk] int32: distinct experts per token."""
    if kind == "skew":  # every token's first slot on expert E // 2, the rest spread
        ids = torch.rand(T, E, generator=gen, device="cuda").argsort(1)[:, :topk]
        hot = E // 2
        ids[:, 0] = hot
        ids[:, 1:] = torch.where(ids[:, 1:] == hot, (hot + 1) % E if E > 1 else -1, ids[:, 1:])
        if E == 1:
            ids[:, 1:] = -1
    elif kind == "sparse":  # only ~min(E, topk + 1) experts receive rows
        live = torch.randperm(E, generator=torch.Generator().manual_seed(int(T)))[:min(E, topk + 1)].cuda()
        ids = live[torch.rand(T, live.numel(), generator=gen, device="cuda").argsort(1)[:, :topk]]
    else:
        ids = torch.rand(T, E, generator=gen, device="cuda").argsort(1)[:, :topk]
    ids = ids.to(torch.int32)
    if kind == "dropped":  # expert-parallel style: ids of other ranks are -1, some out of range
        r = torch.rand(T, topk, generator=gen, device="cuda")
        ids = torch.where(r < 0.3, torch.full_like(ids, -1), ids)
        ids = torch.where((r >= 0.3) & (r < 0.35), torch.full_like(ids, E + 3), ids)
    return ids


class Case:
    """One routing and operand set. reference() is the FP32 SPEC.md 9.2 result."""

    def __init__(self, enc, T, topk, kind, mode, seed, ldx_pad=0, ldy_pad=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        E, n, k = enc.num_experts, enc.n, enc.k
        self.enc, self.T, self.topk, self.mode = enc, T, topk, mode
        self.A = T * topk
        ids = routing(T, topk, E, kind, gen)
        self.sorted, self.offsets = H.align(ids, E)
        self.count = int(self.offsets[-1])
        gathered = mode in ("w13", "both")
        scattered = mode in ("w2", "both")
        x_rows = T if gathered else self.A
        self.xb = torch.randn(x_rows, k + ldx_pad, generator=gen, device="cuda").bfloat16()
        self.x = self.xb[:, :k]
        self.gather = self.sorted if gathered else None
        self.gather_div = topk if gathered else 1
        self.scatter = self.sorted if scattered else None
        self.row_scale = (torch.rand(self.A, generator=gen, device="cuda") + 0.25) if scattered else None
        self.y_rows = self.A
        self.ldy_pad = ldy_pad

    def new_y(self):
        return torch.full((self.y_rows, self.enc.n + self.ldy_pad), SENTINEL, dtype=torch.bfloat16, device="cuda")

    def maps(self):
        p = torch.arange(self.count, device="cuda")
        expert = torch.searchsorted(self.offsets[1:], p, right=True)
        src = self.gather[:self.count].long() // self.gather_div if self.gather is not None else p
        dst = self.scatter[:self.count].long() if self.scatter is not None else p
        return expert, src, dst

    def reference(self):
        """(ref, bound) for the written rows dst, FP32."""
        expert, src, dst = self.maps()
        w = self.enc.w.float()
        xs = self.x[src].float()
        ref = torch.empty(self.count, self.enc.n, device="cuda")
        mag = torch.empty_like(ref)
        for e in range(self.enc.num_experts):
            lo, hi = int(self.offsets[e]), int(self.offsets[e + 1])
            if hi > lo:
                ref[lo:hi] = xs[lo:hi] @ w[e].T
                mag[lo:hi] = xs[lo:hi].abs() @ w[e].abs().T
        if self.row_scale is not None:
            s = self.row_scale[dst].unsqueeze(1)
            ref, mag = ref * s, mag * s
        bound = 2 * (self.enc.k + 2) * U * mag + 2.0**-8 * ref.abs() + 1e-30
        return dst, ref, bound

    def run(self, lib, cfg, source, y=None, ws=None):
        y = self.new_y() if y is None else y
        lib.moe_linear(cfg, self.x, self.enc, self.offsets, self.A, y[:, :self.enc.n], gather=self.gather,
                       gather_div=self.gather_div, scatter=self.scatter, row_scale=self.row_scale, source=source,
                       ws=ws)
        return y


class Checker:
    def __init__(self, lib, cfgs):
        self.lib = lib
        self.cfgs = cfgs
        self.failures = []
        self.cases = 0

    def fail(self, msg):
        self.failures.append(msg)
        print("FAIL", msg, flush=True)

    def garbage_ws(self, cfg, c):
        enc = c.enc
        nbytes = self.lib.lib.nf12_moe_tc_workspace_bytes(c.A, enc.num_experts, enc.n, enc.k, cfg)
        ws = torch.randint(0, 256, (max(nbytes, 256),), dtype=torch.uint8, device="cuda")
        return ws, nbytes

    def check(self, c, tag, specials=False):
        enc = c.enc
        ref = None if specials else c.reference()
        written = torch.zeros(c.y_rows, dtype=torch.bool, device="cuda")
        written[c.maps()[2]] = True
        for cfg in self.cfgs:
            self.cases += 1
            name = (f"{tag} E={enc.num_experts} n={enc.n} k={enc.k} T={c.T} topk={c.topk} mode={c.mode} "
                    f"cfg={self.lib.config_name(cfg)}")
            try:
                ynf = c.run(self.lib, cfg, H.NF12, ws=self.garbage_ws(cfg, c))
                ybt = c.run(self.lib, cfg, H.BF16T, ws=self.garbage_ws(cfg, c))
                yre = c.run(self.lib, cfg, H.NF12, ws=self.garbage_ws(cfg, c))
            except H.Unsupported:
                continue
            torch.cuda.synchronize()
            bits = lambda t: t.view(torch.int16)  # noqa: E731
            if not torch.equal(bits(ynf), bits(ybt)):
                diff = (bits(ynf) != bits(ybt)).nonzero()
                self.fail(f"{name}: NF12 != bf16t at {diff.shape[0]} elements, first {diff[0].tolist()}")
            if not torch.equal(bits(ynf), bits(yre)):
                self.fail(f"{name}: not deterministic")
            untouched = ynf[~written]
            pad = ynf[written][:, enc.n:]
            if untouched.numel() and not bool((untouched == SENTINEL).all()):
                self.fail(f"{name}: rows without a valid position were written")
            if pad.numel() and not bool((pad == SENTINEL).all()):
                self.fail(f"{name}: ldy padding written")
            if ref is not None:
                dst, r, bound = ref
                got = ynf[dst, :enc.n].float()
                err = (got - r).abs()
                bad = ~(err <= bound)
                if bool(bad.any()):
                    i = bad.nonzero()[0].tolist()
                    self.fail(f"{name}: {int(bad.sum())} outside bound, e.g. {i}: got {got[i[0], i[1]].item()} "
                              f"ref {r[i[0], i[1]].item()} bound {bound[i[0], i[1]].item()}")

    def graph(self, enc, T, topk, mode, seed):
        """A captured graph must match eager execution after in-place routing changes."""
        a = Case(enc, T, topk, "uniform", mode, seed)
        b = Case(enc, T, topk, "skew", mode, seed + 1)
        for cfg in self.cfgs:
            self.cases += 1
            name = f"graph E={enc.num_experts} n={enc.n} k={enc.k} T={T} mode={mode} cfg={self.lib.config_name(cfg)}"
            try:
                want = b.run(self.lib, cfg, H.NF12)
            except H.Unsupported:
                continue
            y = a.new_y()
            ws, nbytes = self.garbage_ws(cfg, a)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                a.run(self.lib, cfg, H.NF12, y=y, ws=(ws, nbytes))  # warm-up outside capture
                with torch.cuda.graph(g, stream=s):
                    a.run(self.lib, cfg, H.NF12, y=y, ws=(ws, nbytes))
            torch.cuda.current_stream().wait_stream(s)
            a.sorted.copy_(b.sorted)
            a.offsets.copy_(b.offsets)
            a.x.copy_(b.x)
            if a.row_scale is not None:
                a.row_scale.copy_(b.row_scale)
            y.fill_(SENTINEL)
            ws.random_(0, 256)
            g.replay()
            torch.cuda.synchronize()
            if not torch.equal(y.view(torch.int16), want.view(torch.int16)):
                self.fail(f"{name}: replay after re-routing differs from eager")


def huge(lib, cfgs, chk):
    """Byte offsets beyond 2^31: weights (E * n * k * 3/2 > 2 GiB, the last
    experts hold the rows), x rows past 2 GiB (gathered) and y rows past 2 GiB
    (scattered). Few routed rows keep the reference cheap."""
    E, n, k = 48, 2048, 16384  # NF12 2.25 GiB, bf16t 3 GiB
    w = torch.empty(E, n, k, dtype=torch.bfloat16, device="cuda")
    for e in range(E):
        w[e] = H.random_experts(1, n, k, seed=e)[0]
    enc = H.encode_moe(w)
    del w
    T, topk = 70000, 2  # x [T, k] bf16 = 2.1 GiB; y [T * topk, n] = 0.55 GiB
    gen = torch.Generator(device="cuda").manual_seed(7)
    ids = torch.full((T, topk), -1, dtype=torch.int32, device="cuda")
    tokens = torch.cat([torch.arange(T - 300, T, device="cuda"), torch.arange(0, 40, device="cuda")])
    ids[tokens, 0] = E - 1
    ids[tokens, 1] = torch.randint(E - 4, E - 1, (tokens.numel(),), generator=gen, device="cuda").int()
    c = Case.__new__(Case)
    c.enc, c.T, c.topk, c.mode = enc, T, topk, "both"
    c.A = T * topk
    c.sorted, c.offsets = H.align(ids, E)
    c.count = int(c.offsets[-1])
    c.xb = torch.randn(T, k, generator=gen, device="cuda").bfloat16()
    c.x = c.xb
    c.gather, c.gather_div, c.scatter = c.sorted, topk, c.sorted
    c.row_scale = torch.rand(c.A, generator=gen, device="cuda") + 0.5
    c.y_rows, c.ldy_pad = c.A, 0
    # y rows past 2 GiB: widen y's row stride instead of allocating more rows.
    big = torch.full((c.A, 4 * n), SENTINEL, dtype=torch.bfloat16, device="cuda")  # 2.1 GiB, ldy = 4n
    for cfg in cfgs:
        chk.cases += 1
        name = f"huge cfg={lib.config_name(cfg)}"
        try:
            outs = []
            for src in (H.NF12, H.BF16T):
                big.fill_(SENTINEL)
                lib.moe_linear(cfg, c.x, enc, c.offsets, c.A, big[:, :n], gather=c.gather, gather_div=topk,
                               scatter=c.scatter, row_scale=c.row_scale, source=src)
                torch.cuda.synchronize()
                outs.append(big[:, :n].clone())
        except H.Unsupported:
            continue
        dst, r, bound = c.reference()
        if not torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)):
            chk.fail(f"{name}: NF12 != bf16t")
        err = (outs[0][dst].float() - r).abs()
        if not bool((err <= bound).all()):
            chk.fail(f"{name}: {int((err > bound).sum())} outside bound")
        written = torch.zeros(c.A, dtype=torch.bool, device="cuda")
        written[dst] = True
        if not bool((outs[0][~written] == SENTINEL).all()):
            chk.fail(f"{name}: unwritten rows modified")


def check_raster_arithmetic(lib):
    """Check host/kernel arithmetic near int32 limits without allocating
    >48 GiB weights or >1 TiB activations.

    Raster band must not exceed one expert's m-tile bound, keeping
    band * n_tiles within max_tiles. E = 1, 8 assignments, N = 2^29,
    K = 64 would overflow with band 1024 * 2^22 n-tiles.
    Ceiling division must remain exact through INT32_MAX rows."""
    L = lib.lib
    i64 = ctypes.c_int64
    L.nf12_test_moe_tc_band.restype = ctypes.c_int
    L.nf12_test_moe_tc_band.argtypes = [i64, i64, i64, ctypes.c_int]
    L.nf12_test_moe_max_mtiles.restype = i64
    L.nf12_test_moe_max_mtiles.argtypes = [i64, i64, ctypes.c_int]
    L.nf12_test_moe_ceil_div.restype = ctypes.c_int
    L.nf12_test_moe_ceil_div.argtypes = [ctypes.c_int, ctypes.c_int]
    budget = 16 << 20
    n_tiles = (1 << 29) // 128
    band = L.nf12_test_moe_tc_band(budget, 8, 64, 128)
    assert band == 1 and band * n_tiles <= n_tiles * L.nf12_test_moe_max_mtiles(8, 1, 128) < 2**31
    # Uncapped prefill bands are unchanged: half of L2 in 128 x 64 panels.
    assert L.nf12_test_moe_tc_band(budget, 1 << 20, 64, 128) == budget // (128 * 64 * 2)
    for A in (1, 7, 129, 4096, 2**31 - 1):
        for E in (1, 8, 256):
            for k, bn in ((64, 128), (64, 256), (7168, 256)):
                band = L.nf12_test_moe_tc_band(budget, A, k, bn)
                assert 1 <= band <= L.nf12_test_moe_max_mtiles(A, E, bn), (A, E, k, bn, band)
    imax = 2**31 - 1
    for count, bm in [(imax, 256), (imax, 128), (imax - 255, 256), (0, 256), (257, 256)]:
        assert L.nf12_test_moe_ceil_div(count, bm) == -(-count // bm), (count, bm)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--huge", action="store_true")
    ap.add_argument("--configs", default=None, help="comma list (default: all + heuristic -1)")
    args = ap.parse_args()
    lib = H.Lib()
    check_raster_arithmetic(lib)
    if not lib.supported():
        print("tcgen05 MoE family not supported on this device")
        return 0
    cfgs = ([int(c) for c in args.configs.split(",")] if args.configs
            else list(range(lib.num_configs())) + [-1])
    chk = Checker(lib, cfgs)
    torch.backends.cuda.matmul.allow_tf32 = False

    # (E, n, k): pair tiles with a dead CTA (n = 384), partial 128-row tiles
    # (n = 144), a single k-block, E = 1.
    shapes = [(8, 256, 256), (16, 384, 128), (8, 144, 64), (1, 256, 192), (64, 128, 128)]
    if args.quick:
        shapes = shapes[:2]
    for si, (E, n, k) in enumerate(shapes):
        enc = H.encode_moe(H.random_experts(E, n, k, seed=si), escape_pad=3)
        for T, topk in [(1, 1), (37, 2), (300, 4), (513, 8)]:
            topk = min(topk, E)
            for kind in ("uniform", "skew", "sparse", "dropped"):
                for mode in ("w13", "w2", "both", "plain"):
                    if args.quick and kind in ("sparse",) and mode != "both":
                        continue
                    c = Case(enc, T, topk, kind, mode, seed=si * 1000 + T, ldx_pad=8 * (T % 3),
                             ldy_pad=8 * ((T + 1) % 3))
                    chk.check(c, f"{kind}")
    # Escapes: heavy finite (bound-checked) and specials (bitwise only).
    for esc in (0.002, 0.03):
        enc = H.encode_moe(H.random_experts(16, 256, 512, seed=11, escapes=esc), escape_pad=5)
        for mode in ("w13", "both"):
            chk.check(Case(enc, 700, 6, "uniform", mode, seed=5), f"esc{esc}")
    w = H.random_experts(8, 256, 256, seed=12)
    flat = w.view(-1)
    pos = torch.randint(0, flat.numel(), (flat.numel() // 200,), device="cuda")
    vals = torch.tensor([float("inf"), float("-inf"), float("nan"), -0.0], dtype=torch.bfloat16, device="cuda")
    flat[pos] = vals[torch.arange(pos.numel(), device="cuda") % 4]
    chk.check(Case(H.encode_moe(w), 200, 4, "uniform", "w13", seed=6), "specials", specials=True)
    # A larger mixed shape: several m-tiles per expert (bands), K > ring depth.
    enc = H.encode_moe(H.random_experts(8, 512, 1024, seed=21, escapes=0.001))
    chk.check(Case(enc, 1200, 2, "skew", "both", seed=9), "multi-mtile")
    for mode in ("w13", "w2"):
        chk.graph(H.encode_moe(H.random_experts(16, 256, 256, seed=31)), 260, 4, mode, seed=40)
    if args.huge:
        huge(lib, cfgs, chk)

    print(f"{chk.cases} cases, {len(chk.failures)} failures")
    return 1 if chk.failures else 0


if __name__ == "__main__":
    sys.exit(main())
