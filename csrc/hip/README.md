# RDNA4 (gfx1200 / gfx1201) HIP kernels

A HIP port of the NF12 codec and fused linear kernels for AMD RDNA4. It
consumes and produces the **identical tiled byte stream** as the CUDA kernels,
so a checkpoint is portable in both directions.

Status: **decode, encode and the matrix-core fused path are implemented and
validated on a Radeon AI PRO R9700 (`gfx1201`)**. Not yet wired into the
`python/nf12` API or the torch extension — see *Follow-ups*.

## What is here

| File | Contents |
| --- | --- |
| `format_hip.cuh` | SPEC §4 geometry, AMD `v_perm_b32` selector conversion, `decode_bulk`/`decode_group`, tiling helpers, wmma fragment helpers. |
| `decode.hip` | `nf12_hip_decode` (NF12 → exact BF16), `nf12_hip_linear_fused` (small-M fused decode+GEMM; the dispatcher uses it for `m <= 15` and sends `m = 16` to the matrix-core path), `nf12_hip_selftest`. |
| `encode.hip` | Native analyze/count/scan/write encoder (`nf12_hip_encode_workspace_bytes`, `nf12_hip_encode_analyze`, `nf12_hip_encode`). |
| `wmma.hip` | `nf12_hip_linear_wmma`, the matrix-core fused path for `16 <= m <= 256`. |

## Build

Standalone, with no CUDA toolkit required:

```sh
cmake -S csrc/hip -B build-hip -DNF12_HIP_ARCHS="gfx1201"
cmake --build build-hip -j
```

produces `libnf12_hip.so`. Or enable it from the top-level build with
`-DNF12_ENABLE_HIP=ON`, which is additive and does not alter any CUDA target.
On a host with no CUDA compiler the top-level build now skips the CUDA targets
instead of failing, so this is the normal path on ROCm:

```sh
cmake -S . -B build -G Ninja -DNF12_ENABLE_HIP=ON -DNF12_BUILD_TORCH=OFF \
      -DNF12_HIP_ARCHS=gfx1201 -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build --target nf12_hip -j
```

`nf12._jit` uses exactly that configuration: on a ROCm build it detects the
visible `gfx` targets, builds `libnf12_hip.so` into the per-user cache, and
`nf12.library_path()` resolves to it. Note for CMake >= 4.2:
`CMAKE_HIP_COMPILER` must point at `amdclang++` directly; the `hipcc` wrapper
is rejected.

## Interface

Plain C ABI, declared in [`include/nf12/nf12_hip.h`](../../include/nf12/nf12_hip.h).
`hipStream_t` is passed explicitly and `__hip_bfloat16` never crosses the
boundary (buffers are `void*`):

```c
size_t     nf12_hip_encode_workspace_bytes(int64_t n, int64_t k);

hipError_t nf12_hip_encode_analyze(const void* w_bf16, int64_t n, int64_t k,
                                   int64_t ldw, void* workspace,
                                   size_t workspace_bytes, uint8_t* exp_base_out,
                                   uint32_t* num_escapes_out, hipStream_t stream);

hipError_t nf12_hip_encode(const void* w_bf16, int64_t n, int64_t k, int64_t ldw,
                           uint8_t exp_base, uint8_t* packed_out,
                           uint4* escapes_out, uint32_t escapes_capacity,
                           void* workspace, size_t workspace_bytes,
                           hipStream_t stream);

hipError_t nf12_hip_decode(const uint8_t* packed, const uint4* escapes,
                           uint32_t exp_base, int64_t n, int64_t k,
                           void* out_bf16, int64_t ldo, hipStream_t stream);

int        nf12_hip_linear_fused_small_m_supported(int64_t m);
hipError_t nf12_hip_linear_fused(const uint8_t* packed, const uint4* escapes,
                                 uint32_t exp_base, const void* x, void* out,
                                 int64_t n, int64_t k, int64_t m,
                                 const void* bias, hipStream_t stream);

int        nf12_hip_wmma_max_m(void);
int        nf12_hip_wmma_supported(int64_t m);
hipError_t nf12_hip_linear_wmma(const uint8_t* packed, const uint4* escapes,
                                uint32_t exp_base, const void* x, void* out,
                                int64_t n, int64_t k, int64_t m,
                                const void* bias, hipStream_t stream);

hipError_t nf12_hip_selftest(uint32_t* out, hipStream_t stream);
```

`analyze` and `encode` synchronize the supplied stream and cannot be graph
captured, matching the CUDA contract.

Route selection by token count: `m <= 15` uses the fused small-M kernel,
`16 <= m <= 256` the matrix-core path, and larger `m` falls back to
decode-then-rocBLAS. `m=16` is a single 16-token tile and is faster than the
fused kernel on every measured shape. `gfx1250` has no `wmma-128b-insts`;
`wmma.hip` compiles to an empty translation unit there and
`nf12_hip_wmma_supported` returns 0.

## Correctness

* **Decode** is bit-exact: `decode(encode(w))` reproduces every input BF16 bit,
  including ±0, subnormals, Inf and NaN payloads.
* **Encode** is byte-identical to the reference codec — same `exp_base`, same
  packed bytes, same escape order — verified over 30 cases spanning
  `16x64 … 11008x4096` and five weight distributions.
* **Fused linear** was checked against an FP64 oracle over 21 shapes
  (256², 1024×4096, 4096×4096 × `m ∈ {1,16,17,32,33,48,64,96,128,256}`):
  **0 violations** of `|err| <= 2e-3 + 2e-3*|ref|`, `rms_rel ≈ 1.66e-3` at
  every `m`, which is the oracle's own BF16 rounding floor. NF12 itself adds
  no error; the residual is FP32 accumulation order.

## Performance (R9700, `gfx1201`, 32 CUs, wave32)

Versus `torch.nn.functional.linear` on the dense BF16 weight, on the same
`k × n` grid as the CUDA benchmarks (`bench/common.py`). Each cell is
**warm / cold**: *warm* reuses one weight in a loop, *cold* cycles several
weights so the working set exceeds 96 MiB, which is what a real multi-layer
model does.

| k × n | m=1 | m=8 | m=16 | m=32 | m=64 | m=128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096×4096 | 1.20 / 1.26 | 0.94 / 1.17 | 0.88 / 1.08 | 0.76 / 1.08 | 0.68 / 0.96 | 0.69 / 0.69 |
| 4096×14336 | 1.33 / 1.33 | 1.29 / 1.29 | 1.23 / 1.23 | 1.17 / 1.16 | 1.16 / 1.15 | 0.82 / 0.74 |
| 14336×4096 | 1.35 / 1.33 | 1.31 / 1.30 | 1.32 / 1.32 | 1.24 / 1.25 | 1.20 / 1.21 | 1.11 / 1.05 |
| 4096×28672 | 1.34 / 1.34 | 1.30 / 1.31 | 1.14 / 1.14 | 1.07 / 1.06 | 0.96 / 0.96 | 0.75 / 0.69 |
| 5120×25600 | 1.38 / 1.38 | 1.36 / 1.36 | 1.21 / 1.21 | 1.14 / 1.14 | 1.04 / 1.04 | 0.79 / 0.77 |
| 25600×5120 | 1.33 / 1.33 | 1.31 / 1.30 | 1.35 / 1.35 | 1.34 / 1.34 | 1.21 / 1.20 | 1.18 / 1.10 |
| 8192×8192 | 1.71 / 1.69 | 1.27 / 1.27 | 1.29 / 1.28 | 1.28 / 1.28 | 1.24 / 1.24 | 0.80 / 0.73 |
| 8192×57344 | 1.40 / 1.41 | 1.32 / 1.32 | 1.22 / 1.22 | 1.21 / 1.20 | 1.16 / 1.14 | 0.77 / 0.74 |
| 2048×8192 | 1.38 / 1.36 | 0.97 / 1.22 | 1.01 / 1.23 | 0.93 / 1.20 | 0.73 / 1.11 | 0.65 / 0.74 |
| 896×4864 | 1.69 / 1.82 | 0.97 / 1.03 | 1.27 / 1.26 | 1.20 / 1.64 | 0.80 / 1.38 | 0.52 / 0.74 |

Accuracy on every row: `rms_rel ≈ 1.66e-3` (1.64–1.67e-3), the BF16
output-rounding floor. The fused linear has **0 violations** of
`|err| <= 2e-3 + 2e-3*|ref|` on the project's scaled accuracy harness.

The win is bandwidth: NF12 reads 25% fewer weight bytes, so it pays off from
`m=8` upward on weights large enough that the packed read dominates. Across the
seven 14336/25600/57344-wide shapes it is 1.13–1.35× at `m=8..32` and
0.96–1.24× at `m=64`. At `m=1` every shape is now above parity (1.20–1.71×).

### Small batches: the fused path is specialized on the token count

`m <= 15` runs the fused kernel, which decodes each 16×64 super-atom once and
dots it against every token in flight. It used to take `m` as a runtime
argument, so `acc[16][2]` was indexed dynamically — which keeps the array out of
VGPRs — and the warp-partial plane was always sized for 16 tokens. Making the
token count a template parameter fixes both and fully unrolls the token loop.

At `m=1` that raises occupancy enough to hurt: the kernel then runs eight CTAs
per CU and 256 concurrent packed streams, which thrashes the 64 MiB LLC.
Over-allocating the partial plane to 32 KiB throttles it to two CTAs per CU,
which is worth 1.40× on `4096×4096` and 1.05× on `2048×8192`.

Same-process A/B of the old and new fused kernel (µs, lower is better):

| k × n | m | before | after | gain |
| --- | ---: | ---: | ---: | ---: |
| 4096×4096 | 1 | 28.57 | 20.45 | **1.40×** |
| 4096×4096 | 8 | 32.58 | 24.15 | **1.35×** |
| 2048×8192 | 1 | 19.44 | 18.49 | **1.05×** |
| 2048×8192 | 8 | 31.95 | 23.87 | **1.34×** |
| 896×4864 | 1 | 8.36 | 8.08 | 1.03× |
| 896×4864 | 8 | 15.40 | 11.57 | **1.33×** |
| 4096×14336 | 8 | 161.67 | 149.18 | **1.08×** |
| 25600×5120 | 8 | 364.93 | 323.03 | **1.13×** |

Warm ratio against dense BF16, before → after:

| k × n | m=1 | m=8 |
| --- | ---: | ---: |
| 4096×4096 | 0.84 → **1.20** | 0.67 → **0.94** |
| 2048×8192 | 1.33 → **1.38** | 0.80 → **0.97** |
| 896×4864 | 1.63 → **1.69** | 0.83 → **0.97** |
| 4096×28672 | 1.33 → **1.34** | 1.22 → **1.30** |
| 25600×5120 | 1.33 → **1.33** | 1.15 → **1.31** |
| 8192×8192 | 1.64 → **1.71** | 1.09 → **1.27** |

`m=16` still routes to the matrix-core kernel, which is 1.2–2.3× faster than the
fused one there on every measured shape.

### Small weights: four fixes that took them from ~0.5× to parity

`4096×4096`, `2048×8192` and `896×4864` are the shapes whose dense BF16 weight
fits in the 64 MiB Infinity Cache, so BF16 runs at ~1.4 TB/s and NF12 has to
beat it while decoding.

1. **No transpose in the store.** `quad_transpose` is a 4×4 dword transpose
   inside each 16-byte chunk: 16 `ds_bpermute_b32` plus 32 selects per
   super-atom, all occupying the same LDS crossbar as the store and the
   B-fragment load. It is self-inverse, so undoing it on the *address* is
   equivalent — dword `d` of lane `l`'s chunk `(h,q)` belongs to dword
   `lane&3` of chunk `d+4q` in row `(lane>>2)+8h`. That is 16 `STS.32` whose
   bank is `4*(lane>>2) + (lane&3) + const`, i.e. all 32 banks exactly once.
   The shuffles disappear and the store becomes conflict-free.
2. **Padded shared stride.** The decoded super-atom is 16×64 BF16 — a
   128-byte row — so with a plain 64 stride the B-fragment load hits 16-way.
   Eight BF16 of padding (144-byte rows, 36 banks) brings it to the 4-cycle
   hardware minimum. It costs 4 KiB of LDS per CTA (4→3 CTAs per CU), so it is
   a template variant selected only when the launch fits in three CTAs per CU.
3. **Load hoisted over the MMA.** The next super-atom's loads are issued
   *before* the matrix-core work and decoded afterwards, so the L3 latency is
   covered by the mma instead of stalling the decode. At `m=16` a 4096×4096
   launch is only two warps per SIMD, which is not enough to hide it any other
   way. This alone took 4096×4096 from 32.1 to 26.7 µs.
4. **The reduce pass is capped at 128 blocks.** The split-K reducer was
   launched with `(total+255)/256` blocks, and on gfx1201 exactly 256
   blocks is pathological: 14.2 µs against 3.3 µs at 64, 128 or 512
   (measured standalone; total=65536 with two splits). Capping the grid
   at 128 removes that cliff for every split shape without slowing the
   small outputs, which keep their natural, smaller grid.

Split-K is picked per shape from the measured optima (see `pick_splits`): a
narrow N splits 8 ways, a very wide N splits 2 ways at eight token tiles, and
everything else stays unsplit. A launch of one wave or less splits only when
K is long enough to give every split a real slice, so `4096×4096` takes 4
ways at `m=16` and 2 above it while `896×4864` — 14 k-blocks — stays unsplit
rather than paying four prologues for three super-atoms each.

| k × n | m=16 before | m=16 after | m=64 before | m=64 after |
| --- | ---: | ---: | ---: | ---: |
| 4096×4096 | 0.48× | **0.89×** (cold 1.10×) | 0.45× | **0.65×** |
| 2048×8192 | 0.66× | **1.03×** (cold 1.23×) | 0.59× | **0.74×** |
| 896×4864 | 0.62× | **1.25×** (cold 1.25×) | 0.50× | **0.80×** |
| 25600×5120 | 1.15× | **1.35×** | 0.98× | **1.21×** |

`4096×4096` at `m=16` is 0.89× warm but 1.10× cold: the 24 MiB packed weight
fits in the 64 MiB Infinity Cache, so warm BF16 runs at ~1.4 TB/s while cold
BF16 falls back to the ~633 GB/s HBM ceiling. Its 4096 output columns give only
32 CTAs — one per CU, two warps per SIMD — so the warm case is
memory-latency-bound rather than bandwidth-bound.

Cold — the regime a real multi-layer model sees — all three small shapes are at
or above parity at `m=16` (1.10× / 1.23× / 1.25×). Above that the gap is
0.51–1.21× warm and 0.69–1.63× cold: eight token tiles per CTA halve the CTA
count, and the extra packed reads are only paid back once the weight leaves the
cache. Still below parity cold: `4096×4096` at `m=64` (0.93×) and all three at
`m=128` (0.69× / 0.74× / 0.74×).

## Porting notes (NVIDIA → AMD)

All verified on `gfx1201` hardware:

1. **`__byte_perm` → `v_perm_b32` is not a transliteration.** NVIDIA numbers a
   4-bit nibble per output byte over `[x.b0..x.b3, y.b0..y.b3]`. AMD uses one
   *full byte* per output byte, where `0..3` selects `src1` (the **second**
   argument) and `4..7` `src0` (the **first**) — reversed. `nv_sel_to_amd()`
   converts NVIDIA selectors at compile time, so the decode expressions are
   copied verbatim from `include/nf12/format.cuh`.
2. **`hip_bfloat16` is not valid in a hipified `.hip` TU.** A bare
   `hip_bfloat16` silently degrades to `int`. Device code uses
   `__hip_bfloat16`; the host boundary uses `void*`.
3. **`__gfx1201__` is defined only during the device pass.** A host-side guard
   on it silently reports "unsupported". The architecture test is a runtime
   `hipGetDeviceProperties().gcnArchName` check. For the same reason a guard
   must not change a *declaration* between the host and device passes —
   different mangled names fail with "Cannot find Symbol" — so
   `NF12_HIP_HAS_WMMA` gates only executable bodies.
4. **`__builtin_amdgcn_fdot2` is FP16, not BF16.** It compiles, runs, and
   produces silently wrong results on BF16 data. The BF16 dot is
   `__builtin_amdgcn_fdot2_f32_bf16` → `v_dot2_f32_bf16`.
5. **Bias must join the FP32 accumulator**, not the rounded BF16 output;
   adding after rounding double-rounds and measured 1.00 ULP versus 0.64 ULP.

## Follow-ups

* Port the torch extension and the `nf12.nn` / `torch.ops.nf12` Python API. The
  C library, the JIT build and `nf12.library_path()` are wired today;
  `nf12._library.load()` raises a clear error on ROCm rather than failing
  obscurely.
* Unify the geometry constants with `include/nf12/format.cuh`. `format_hip.cuh`
  currently restates them and `static_assert`s that they match, so a format
  change breaks the build rather than silently diverging.
* `m > 128` still falls back to decode-then-rocBLAS; a larger-tile matrix-core
  kernel is the next step.
* Only `gfx1200`/`gfx1201` are validated. `gfx1250` builds but takes the
  non-wmma route.

## Credits

* **NF12 format, reference codec and CUDA kernels:** [AlpinDale](https://x.com/AlpinDale),
  [dphnAI/nf12](https://github.com/dphnAI/nf12), Apache-2.0. This HIP port reads
  and writes the identical tiled byte stream and follows the same tiling and
  decode expressions; the SPDX headers in each `.hip`/`.cuh` file carry the
  original attribution.
* **NyoomFloat12 format and theory:** *"NyoomFloat12: Accelerating LLM Inference
  via Lossless 12-bit Weight Compression"*, Liberman, Fang, Zhang, Dao, and Fu —
  [Together AI](https://www.together.ai/) (2026).
