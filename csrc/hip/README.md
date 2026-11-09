# RDNA4 (gfx1200 / gfx1201) HIP kernels

A HIP port of the NF12 codec and fused linear kernels for AMD RDNA4. It
consumes and produces the **identical tiled byte stream** as the CUDA kernels,
so a checkpoint is portable in both directions.

Status: **decode, encode, the matrix-core fused path, the bf16t twin and the
full MoE op set are implemented and validated on a Radeon AI PRO R9700
(`gfx1201`)**, and wired into `python/nf12` / `torch.ops.nf12` — the torch
extension builds on ROCm and shares `csrc/torch/ops.cpp` with the CUDA build.

Validated toolchain: **ROCm SDK 10.1.0**, **HIP 7.16.26385**
(`hipconfig --version`: `7.16.26385-0000000`), and
**PyTorch 2.16.0.dev20261025+rocm10.1**, with free-threaded Python.
ROCm release numbers and HIP component versions are distinct;
`torch.version.hip` reports HIP. See [test setup](../../tests/hip/README.md).

## What is here

| File | Contents |
| --- | --- |
| `format_hip.cuh` | SPEC §4 geometry, AMD `v_perm_b32` selector conversion, `decode_bulk`/`decode_group`, tiling helpers, wmma fragment helpers. |
| `decode.hip` | `nf12_hip_decode` (NF12 → exact BF16), `nf12_hip_linear_fused` (small-M fused decode+GEMM; the dispatcher uses it for `m <= 15` and sends `m = 16` to the matrix-core path), `nf12_hip_selftest`. |
| `encode.hip` | Native analyze/count/scan/write encoder (`nf12_hip_encode_workspace_bytes`, `nf12_hip_encode_analyze`, `nf12_hip_encode`). |
| `wmma.hip` | `nf12_hip_linear_wmma` (matrix cores, `16 <= m <= 256`), `nf12_hip_linear_wmma_bf16t`, and the routed `nf12_hip_moe_linear{,_bf16t}`. |
| `moe.hip` | `moe_align` (stable counting sort), `moe_combine`, `moe_silu_mul`, `moe_decode`, `moe_grouped_operands`. |
| `fused_decode.hip` | The `nf12::torch_detail::fused_decode` seam for `_fused_moe` at `T == 1` (gate/SiLU and down/combine in two launches), so the torch op layer and the C ABI are identical on both backends. |
| `embedding.hip` | `nf12_hip_embedding` for int32/int64 ids (bit-exact vs CUDA). |
| `pack_bf16t.hip` | BF16 → bf16t twin packing (bit-exact vs CUDA). |
| `backend.hip` | HIP launches and workspace policy behind the shared `csrc/api.cpp` frontend. |

## Build

Standalone, with no CUDA toolkit required:

```sh
cmake -S csrc/hip -B build-hip -DNF12_HIP_ARCHS="gfx1201"
cmake --build build-hip -j
```

produces `libnf12_hip.so`. A top-level build with ROCm PyTorch selects HIP by
default, even if `nvcc` is installed. With `NF12_BUILD_TORCH=OFF`, enable HIP
explicitly:

```sh
cmake -S . -B build -G Ninja -DNF12_ENABLE_HIP=ON -DNF12_BUILD_TORCH=OFF \
      -DNF12_HIP_ARCHS=gfx1201 -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build --target nf12_hip -j
```

On ROCm, `nf12._jit` detects visible `gfx` targets and builds both
`libnf12_hip.so` and `libnf12_torch.so` into the per-user cache.
`nf12.library_path()` resolves the HIP library. Note for CMake >= 4.2:
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
size_t     nf12_hip_linear_wmma_workspace_bytes(int64_t n, int64_t k,
                                                int64_t m);
hipError_t nf12_hip_linear_wmma(const uint8_t* packed, const uint4* escapes,
                                uint32_t exp_base, const void* x, void* out,
                                int64_t n, int64_t k, int64_t m,
                                const void* bias, void* workspace,
                                size_t workspace_bytes, hipStream_t stream);

hipError_t nf12_hip_selftest(uint32_t* out, hipStream_t stream);
```

`analyze` and `encode` synchronize the supplied stream and cannot be graph
captured, matching the CUDA contract.

Inference supports HIP graph capture/replay through both the torch API and
native entry points. Load/JIT, encode weights, and warm up the chosen route
before capture. Native callers must retain the captured input/output addresses
and workspace until the graph is released; concurrent graphs need independent
writable buffers. Torch manages its captured scratch through the graph allocator.
See [graph validation](../../docs/ROCM_VALIDATION.md#inference-graph-capture)
for the tested routes and replay counts. Encoding/packing is outside the
inference graph requirement.

The library holds no mutable device state. The split-K reduction scratch for
`nf12_hip_linear_wmma` is owned by the caller: query
`nf12_hip_linear_wmma_workspace_bytes(n, k, m)` and pass a buffer of at least
that size (0 bytes means the chosen launch does not split K). An earlier
revision cached a single process-wide pointer, which was neither per device nor
per stream; do not reintroduce it.

This mirrors the CUDA build exactly (`csrc/api.cpp`, `run_linear`): the library
owns no scratch, allocates nothing, and keeps no per-device or per-stream cache.
Seven of the forty benchmark `shape × m` cells split K; a null or undersized
buffer is rejected and `nf12_linear` returns `NF12_ERROR_INSUFFICIENT_WORKSPACE`,
the same status the CUDA build returns, so a caller ported from CUDA behaves
identically.

**Where the reusable buffer comes from.** The caller's own caching allocator.
`csrc/torch/ops.cpp` calls `nf12_linear_workspace_bytes(...)` and takes the
block from PyTorch's caching allocator, which is per device, stream ordered and
graph safe, so repeated launches reuse one block instead of allocating. A raw
C-ABI caller does the same with whatever allocator it already has.

The library deliberately does not allocate this itself, for the same reasons the
CUDA build does not: a single per-device buffer is shared by every stream, so
two concurrent launches would write the same FP32 partial planes and corrupt
both results; a per-stream arena would need a lifecycle, because serving loops
create and destroy streams; and `hipMallocAsync`/`hipFreeAsync` around each
launch measured ~11 µs of host time, more than the split saves.

Route selection by token count: `m <= 15` uses the fused small-M kernel,
`16 <= m <= 256` the matrix-core path, and larger `m` falls back to
decode-then-rocBLAS in the torch layer. Decoded slices follow CUDA's 256 MiB
external-buffer budget (at least one 16-row tile), with storage local to the
current call. The raw C API uses native M chunks and a caller-owned workspace.
`m=16` is a single 16-token tile and is faster than the
fused kernel on every measured shape. `gfx1250` has no `wmma-128b-insts`;
`wmma.hip` compiles to an empty translation unit there and
`nf12_hip_wmma_supported` returns 0.

## torch extension on ROCm

See [the shared ABI design](../../docs/ABI_BACKENDS.md) for ownership and compatibility checks.

`csrc/torch/ops.cpp` is the op schema, argument checking, workspace allocation
and dispatch layer for the CUDA build. Instead of writing a second op layer for
ROCm, the HIP build compiles **the same `ops.cpp`** and backs it with
`csrc/api.cpp`, the same public C ABI frontend used by CUDA. Common validation,
empty-input behavior, size arithmetic, and error semantics are implemented once;
`backend.hip` supplies HIP device queries, workspace policy, and launches. `nf12/nf12.h` supplies HIP aliases for the handful of CUDA
spellings the shared layer uses (`cudaStream_t`, `cudaError_t`,
`cudaMemcpyAsync`, …), so there is one op implementation, not two.

```sh
cmake -S . -B build/hip -G Ninja -DNF12_ENABLE_HIP=ON -DNF12_BUILD_TORCH=ON \
      -DNF12_HIP_ARCHS=gfx1201 -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build/hip --target nf12_torch -j
```

`nf12_torch` links `libnf12_hip.so` and `torch_hip`. It relies on
`aoti_torch_get_current_cuda_stream` and `aoti_torch_cuda_{mm,addmm}_out`,
which exist in ROCm's `libtorch_hip.so`; `USE_ROCM` exposes the stream shim in
torch's stable-ABI header.

Coverage, all exercised by the shared `tests/api/test_torch.py`,
`tests/api/test_moe.py` and `tests/api/test_codec_contracts.py` suites, which run
against whichever backend is loaded:

| op | ROCm status |
| --- | --- |
| `encode`, `decode`, `encode_moe` | native |
| `linear`, `linear_bf16t` | native (`linear_bf16t` for any `m`) |
| `pack_bf16t` | native |
| `embedding` (int32/int64) | native |
| `moe_align`, `moe_combine`, `moe_silu_mul`, `moe_decode` | native |
| `moe_linear`, `moe_linear_bf16t` | native |
| `_fused_moe` | native: `T == 1` takes the fused two-kernel route below, `T > 1` the composed route |

### `_fused_moe` at `T == 1`

`ops.cpp` asks `nf12::torch_detail::fused_decode_supported` before it allocates
anything and, when that says yes, calls `fused_decode`, which must not fall back
afterwards. On ROCm that seam is `fused_decode.hip`: a gate kernel (`x -> 2I`,
BF16 round, `SiLU(gate) * up`, one BF16 round) and a down kernel (per-assignment
scale in FP32 plus BF16 round, top-k summed in FP32 in increasing `j`, one final
round), which is `moe_silu_mul` / `round_assignment` / `moe_combine` evaluated
inline. Both resolve the expert from `topk_ids` on device, so there is no host
synchronisation and no workspace -- the activation plane is the caller's
`[topk, I]` buffer.

The CUDA original (`csrc/torch/fused_decode.cu`) is not transliterable: it needs
thread-block clusters, PDL and the `small_m_core` bulk ring, none of which exist
on RDNA4. The down kernel therefore keeps the whole top-k walk inside one CTA,
which is what preserves the ordered reduction without atomics.

Warm `T = 1`, `E = max(2, topk)`, distinct expert ids, median of 100 launches,
against the composed `moe_*` route:

| K | I | n | topk | fused (µs) | composed (µs) | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 14336 | 4096 | 2 | 842.8 | 945.2 | 1.12× |
| 4096 | 14336 | 4096 | 1 | 428.3 | 491.8 | 1.15× |
| 4096 | 14336 | 4096 | 4 | 1672.1 | 1800.2 | 1.08× |
| 4096 | 4096 | 4096 | 2 | 253.7 | 288.7 | 1.14× |
| 2048 | 8192 | 2048 | 2 | 248.0 | 307.5 | 1.24× |
| 4096 | 1024 | 4096 | 2 | 38.5 | 97.5 | 2.53× |
| 5120 | 25600 | 5120 | 2 | 1868.9 | 2029.8 | 1.09× |
| 4096 | 7168 | 4096 | 2 | 427.8 | 463.3 | 1.08× |

The fused route is faster at every shape measured (1.08× to 2.53×): it drops
`moe_align`, `moe_silu_mul` and `moe_combine` plus the two intermediate BF16
round-trips. It is *not* bit-identical to the composed route -- the K walk has a
different FP32 association order, exactly like the CUDA fused kernel -- so
`tests/api/test_moe.py` asserts the SPEC 6.2 tolerance against both the
reference and the composed path, plus determinism and multi-stream independence.

Two deliberate choices:

* **No `aten::_grouped_mm` dependency.** `nf12_moe_linear_decode_gemm_rows`
  returns 0, so the decode + grouped-GEMM shortcut is never selected and MoE
  never needs a ROCm grouped GEMM.
* **The bf16t twin is a template parameter, not a second kernel.**
  `WeightSource::{kNf12, kBf16t}` selects how a super-atom's planes become atom
  registers; decode, matrix cores and the epilogue are shared. The twin is the
  exactness reference for the nf12 route (SPEC §6.2), so it must run the same
  arithmetic. It now also runs the same `(tiles, splits, m_blocks)` plan, which
  means it needs the split-K partial plane whenever the nf12 route does; the two
  are then bitwise identical, which `tests/api/test_moe.py` asserts.

`moe_linear` reuses the dense matrix-core kernel with a row map enabled
(`kRouted`): one CTA per (expert, 16-row weight block, 16-assignment tile), the
expert resolved on device from `expert_offsets`, `gather`/`scatter`/`row_scale`
applied around the same FP32 accumulator. Per-expert counts stay device-side,
so the launch is host-synchronisation free and graph-capturable.

### MoE throughput (R9700, `gfx1201`, E=8, N=K=4096, topk=2)

Against a naive dense-BF16 baseline that loops over experts with
`torch.arange`/`F.linear`-style FP32 matmuls and BF16 rounding — i.e. what a
straightforward implementation does, not a tuned grouped GEMM:

| tokens | nf12 `moe_linear` µs | dense BF16 loop µs | speedup |
| ---: | ---: | ---: | ---: |
| 16 | 375.9 | 2431.2 | 6.47× |
| 64 | 485.5 | 2461.2 | 5.07× |
| 128 | 695.5 | 2556.4 | 3.68× |
| 512 | 2423.7 | 3990.5 | 1.65× |
| 1024 | 4948.2 | 6020.8 | 1.22× |

The advantage is the 25% smaller weight read, so it shrinks as the activation
side grows. Above ~1k tokens a real grouped GEMM would win; that is the
follow-up, and it is why the MoE GEMM is not claimed to be tensor-core optimal.

## Correctness

* **Decode** is bit-exact: `decode(encode(w))` reproduces every input BF16 bit,
  including ±0, subnormals, Inf and NaN payloads.
* **Encode** is byte-identical to the reference codec — same `exp_base`, same
  packed bytes, same escape order — verified over 30 cases spanning
  `16x64 … 11008x4096` and five weight distributions.
* **Fused linear** must match the same-configuration BF16t kernel bit for bit
  (SPEC.md §6.2). Dense BF16 GEMM is a separate §6.3 comparison, using
  `|err| <= 2e-3 + 2e-3*|ref|` and rejecting non-finite results. Decode/GEMM
  routes match the same GEMMs on exactly decoded BF16 slices (§6.5).
* **C ABI** preserves padded input/output row strides, normalizes malformed
  MoE offsets like CUDA, and checks workspace requirements for both full
  M chunks and their tails before launching. Empty routing returns zero
  offsets without scratch storage.

See [ROCm validation and performance](../../docs/ROCM_VALIDATION.md) for
review regression coverage, fallback timings and encoder measurements.

## Performance (R9700, `gfx1201`, 32 CUs, wave32)

Versus `torch.nn.functional.linear` on the dense BF16 weight, on the same
`k × n` grid as the CUDA benchmarks (`bench/common.py`). Each cell is
**warm / cold**: *warm* reuses one weight in a loop, *cold* cycles several
weights so the working set exceeds 96 MiB, which is what a real multi-layer
model does.

| k × n | m=1 | m=8 | m=16 | m=32 | m=64 | m=128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096×4096 | 1.19 / 1.27 | 0.95 / 1.18 | 1.04 / 1.13 | 0.79 / 1.03 | 0.64 / 0.95 | 0.82 / 0.89 |
| 4096×14336 | 1.33 / 1.34 | 1.29 / 1.28 | 1.25 / 1.24 | 1.18 / 1.17 | 1.15 / 1.16 | 0.82 / 0.78 |
| 14336×4096 | 1.34 / 1.33 | 1.30 / 1.30 | 1.33 / 1.33 | 1.24 / 1.24 | 1.19 / 1.21 | 1.08 / 1.06 |
| 4096×28672 | 1.34 / 1.34 | 1.31 / 1.31 | 1.26 / 1.22 | 1.27 / 1.21 | 1.28 / 1.29 | 0.74 / 0.68 |
| 5120×25600 | 1.38 / 1.38 | 1.36 / 1.36 | 1.21 / 1.21 | 1.21 / 1.18 | 1.20 / 1.22 | 0.79 / 0.73 |
| 25600×5120 | 1.33 / 1.34 | 1.31 / 1.31 | 1.35 / 1.35 | 1.34 / 1.34 | 1.21 / 1.20 | 1.20 / 1.10 |
| 8192×8192 | 1.67 / 1.67 | 1.27 / 1.27 | 1.28 / 1.28 | 1.28 / 1.28 | 1.24 / 1.24 | 0.79 / 0.71 |
| 8192×57344 | 1.40 / 1.41 | 1.32 / 1.32 | 1.23 / 1.22 | 1.25 / 1.25 | 1.25 / 1.22 | 0.76 / 0.74 |
| 2048×8192 | 1.36 / 1.36 | 0.95 / 1.22 | 1.02 / 1.23 | 0.91 / 1.19 | 0.74 / 1.11 | 0.64 / 0.74 |
| 896×4864 | 1.71 / 1.82 | 1.03 / 1.06 | 1.34 / 1.26 | 1.21 / 1.64 | 0.81 / 1.38 | 0.77 / 1.32 |

Accuracy on the 60-cell grid: **0 violations** of `|err| <= 2e-3 + 2e-3*|ref|`
against the BF16 twin, median `rms_rel` 5.5e-5 and maximum 1.5e-4 (the
difference between two BF16 output roundings). Against the FP64 oracle the
same runs report `rms_rel ≈ 1.66e-3`, the oracle's own BF16 floor.

The win is bandwidth: NF12 reads 25% fewer weight bytes, so it pays off from
`m=8` upward on weights large enough that the packed read dominates. Across the
14336/25600/57344-wide shapes it is 1.18–1.36× at `m=8..32` and 1.15–1.25× at
`m=64`. At `m=1` every shape is above parity (1.19–1.71×).

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

`m=16` still routes to the matrix-core kernel, which is 1.2–2.3× faster than the
fused one there on every measured shape.

**Split-K only when the grid is starved.** `pick_splits` used to hand four K
splits to any shape with at least 128 column blocks, which included launches
that were already 6–7 waves deep. There the extra FP32 partial plane (7–28 MB
written *and* read) costs more than the additional CTAs buy. Dropping that rule,
measured in one process against the same binary otherwise unchanged:

| k × n | m | split 4 | one K pass | delta |
| --- | ---: | ---: | ---: | ---: |
| 4096×28672 | 16 | 333.6 | 310.2 | **−7.0%** |
| 4096×28672 | 32 | 362.4 | 321.7 | **−11.2%** |
| 4096×28672 | 64 | 417.2 | 317.9 | **−23.8%** |
| 5120×25600 | 16 | 363.3 | 375.8 | +3.4% |
| 5120×25600 | 32 | 388.1 | 382.0 | −1.6% |
| 5120×25600 | 64 | 433.9 | 385.2 | **−11.2%** |

One cell loses 3.4% (and still runs at 1.20× BF16); the other five gain, up to
23.8%. Those two shapes no longer need split-K scratch at all, so
`nf12_hip_linear_wmma_workspace_bytes` returns 0 for them. `8192×57344` keeps
its two-way split: it is 14 waves deep, and there the split still helps at
m=128 (2601 → 2414 µs).

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
| 4096×4096 | 0.48× | **1.05×** (cold 1.13×) | 0.45× | **0.67×** |
| 2048×8192 | 0.66× | **1.01×** (cold 1.23×) | 0.59× | **0.74×** |
| 896×4864 | 0.62× | **1.24×** (cold 1.26×) | 0.50× | **0.80×** |
| 25600×5120 | 1.15× | **1.35×** | 0.98× | **1.22×** |

`4096×4096` at `m=16` is 1.05× warm and 1.13× cold: the 24 MiB packed weight
fits in the 64 MiB Infinity Cache, so warm BF16 runs at ~1.4 TB/s while cold
BF16 falls back to the ~633 GB/s HBM ceiling. Its 4096 output columns give only
32 CTAs, so the kernel is latency-bound rather than bandwidth-bound, which is
why the occupancy tuning above is worth 1.20× there.

Cold — the regime a real multi-layer model sees — all three small shapes are at
or above parity at `m=16` (1.13× / 1.23× / 1.26×). Above that the gap is
0.64–1.35× warm and 0.67–1.65× cold: from `kMt = 4` the accumulator costs 64
VGPRs and three CTAs per CU is the ceiling. Still below parity cold: the wide
shapes at `m=128` (0.67–0.76×), `2048×8192` at `m=128` (0.75×), and
`4096×4096` at `m=64`/`m=128` (0.97× / 0.94×).

### `m=128`: the accumulator is the whole occupancy story

At `kMt=8` the eight 16-token accumulators are 64 FP32 VGPRs on their own, and
the kernel lands at 167 — about **one CTA per CU**. Halving the tile count and
splitting M across CTAs doubles the packed read but restores the occupancy, and
that trade only pays where the second read is cheap. Two cases qualify: long K,
where the matrix-core work amortises it (14336×4096 **1.08×**, 25600×5120
**1.16×**, 8192×8192 **0.79×**), and a narrow N, where the doubled grid is still
only ~2.5 waves (`4096×4096` 0.69 → **0.87×**, `896×4864` 0.51 → **0.76×** warm
and **1.29×** cold).

`2048×8192` is the case that fails: 64 column blocks means splitting M makes 128
CTAs *and* reads the weight twice, measured 58.6 → 85.2 µs. It stays on 8 tiles
and sits at 0.64×. The wide-N shapes (0.74–0.82×) are still on 8 tiles for the
same reason — their grid is already 112–896 CTAs, so there is nothing to buy by
doubling the read.

### Occupancy: the buffer and the split are both count tests

Two knobs were tuned by hand and are now derived from the launch shape.

**Double buffering.** The decode-into-the-idle-buffer trick costs 18 KiB of
shared memory per CTA (36 KiB against 18 KiB single-buffered), and a CU has
128 KiB. At `kMt <= 2` the accumulator is small enough that the VGPR budget
allows well over three CTAs, so that LDS footprint is what caps occupancy; from
`kMt = 3` up the ~135 VGPRs already cap a CU at three CTAs and dropping the
buffer buys nothing (25600×5120 m=128: 447 µs buffered against 479 µs not).

Where LDS does bind, dropping the buffer only helps in a narrow band: the
launch must be starved enough that one extra resident CTA matters, but not so
starved that the buffer's own latency hiding is the bigger win. `4096×4096`
m=16 is 128 CTAs — four waves against a cap of three — and the buffer costs
28.4 → 23.5 µs. The wide shapes are 14–28 waves deep, where the cap never bites
and removing the buffer only loses.
`2048×8192` m=16 is two waves, the three still fit, and the buffer is worth
30.5 → 25.9 µs.

**Split-K.** Splitting K multiplies the CTA count without duplicating any read,
so it is free parallelism — but only while the extra CTAs can actually be
resident. Four ways pays up to `m=32` (`4096×4096`: 30.1 → 27.1 µs) and loses
from `kMt = 4` on (`4096×4096` m=64: 38.2 → 43.3 µs, m=128: 62.9 → 79.8 µs),
because three CTAs per CU is already the ceiling.

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

* Unify the geometry constants with `include/nf12/format.cuh`. `format_hip.cuh`
  currently restates them and `static_assert`s that they match, so a format
  change breaks the build rather than silently diverging.
* `m > 128` on the dense nf12 route still falls back to decode-then-rocBLAS; a
  larger-tile matrix-core kernel is the next step.
* The MoE route runs one 16-assignment tile per CTA, so the packed weight is
  re-read once per 16 assignments of an expert. That wins from 16 to ~1k tokens
  against a dense-BF16 loop, but a multi-tile CTA (or split-K) is the next step
  for large token counts.
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
