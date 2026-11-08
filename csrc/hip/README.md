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
| 4096×4096 | 0.83 / 1.15 | 0.68 / 0.97 | 0.48 / 0.90 | 0.45 / 0.83 | 0.45 / 0.75 | 0.61 / 0.66 |
| 4096×14336 | 1.33 / 1.33 | 1.19 / 1.19 | 1.25 / 1.23 | 1.25 / 1.23 | 1.20 / 1.20 | 0.84 / 0.85 |
| 14336×4096 | 1.32 / 1.33 | 1.21 / 1.21 | 1.20 / 1.20 | 1.12 / 1.13 | 1.11 / 1.12 | 0.83 / 0.86 |
| 4096×28672 | 1.34 / 1.34 | 1.22 / 1.22 | 1.26 / 1.26 | 1.24 / 1.24 | 1.19 / 1.21 | 0.79 / 0.75 |
| 5120×25600 | 1.38 / 1.39 | 1.30 / 1.30 | 1.28 / 1.28 | 1.26 / 1.26 | 1.22 / 1.23 | 0.83 / 0.78 |
| 25600×5120 | 1.33 / 1.33 | 1.15 / 1.15 | 1.15 / 1.17 | 1.09 / 1.08 | 0.98 / 0.98 | 0.83 / 0.77 |
| 8192×8192 | 1.68 / 1.66 | 1.09 / 1.10 | 1.21 / 1.20 | 1.15 / 1.15 | 1.09 / 1.10 | 0.89 / 0.93 |
| 8192×57344 | 1.40 / 1.40 | 1.23 / 1.23 | 1.29 / 1.28 | 1.28 / 1.28 | 1.21 / 1.20 | 0.80 / 0.74 |
| 2048×8192 | 1.32 / 1.30 | 0.77 / 1.05 | 0.66 / 1.07 | 0.65 / 1.04 | 0.59 / 0.98 | 0.60 / 0.74 |
| 896×4864 | 1.62 / 1.86 | 0.71 / 0.94 | 0.62 / 0.89 | 0.68 / 0.95 | 0.50 / 0.74 | 0.50 / 0.70 |

Accuracy on every row: `rms_rel ≈ 1.66e-3` (1.64–1.67e-3), the BF16
output-rounding floor. The fused linear has **0 violations** of
`|err| <= 2e-3 + 2e-3*|ref|` on the project's scaled accuracy harness.

The win is bandwidth: NF12 reads 25% fewer weight bytes, so it pays off from
`m=8` upward on weights large enough that the packed read dominates:
1.09–1.30× at `m=8..64` for the 14336/25600/57344-wide shapes and 1.20–1.29×
at `m=16`. `m=128` is covered below. The remaining losses are the small
weights (`4096×4096`, `2048×8192`, `896×4864`), where the dense BF16 weight
fits in the 64 MiB Infinity Cache (L3/MALL) and runs at ~1.4 TB/s while NF12
stays decode-bound at ~0.3–0.5 TB/s of packed data.

### `m=128`: shape-aware M split

`kMt=8` costs 64 VGPRs for the accumulator and lands at 167 VGPRs / 9 waves per
SIMD, which under-fills the machine at `m=128`. When `full_tiles > 4`,
`k_blocks >= 128` and `n_blocks <= 64`, the launcher instead runs the 4-tile
kernel over two M blocks: measured `8192×8192` 0.68 → 0.89× and
`14336×4096` 0.79 → 0.83×. Short-K and wide-N shapes keep the 8-tile kernel
and do not regress. Above `m=128` there are already two M blocks, so the split
is disabled and the packed weight is read twice rather than four times.

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
