# ROCm validation and performance

Validated on Radeon AI PRO R9700 (`gfx1201`, 8 MiB L2, 64 MiB L3), ROCm SDK
10.1.0, HIP 7.16.26385 (`7.16.26385-0000000`), PyTorch
2.16.0.dev20261025+rocm10.1, and free-threaded Python 3.14 (`PYTHON_GIL=0`).
OMP/MKL threads were limited to one. Libraries were built from source.

## Test results

| Suite | Device | Passed | Skipped | Deselected |
| --- | --- | ---: | ---: | ---: |
| HIP + shared API | R9700 | 641 | 12 | 1 |
| Shared API | RTX 4090 | 553 | 16 | 1 |

The 641-case full HIP run was on `8e1835a`; head `40761fe` adds three
out-of-range workspace-query cases, all covered by the latest-head focused
run (10 native align tests and 19 shared align tests passed). CUDA runtime code
is unchanged since its full run; the CUDA-only build still configures.

Both runs use the shared ABI frontend and source-built native and torch libraries,
with two visible GPUs to exercise concurrent device/stream propagation. The
backend-neutral cases live in `tests/api` and run on either runtime; `tests/hip`
holds native `nf12_hip_*`, RDNA4 dispatch and HIP graph cases that have no
CUDA counterpart. The 4 extra CUDA skips are decode-slice tests whose device
selects the fused route instead. The known CUDA MoE
reference-tolerance case tracked separately in PR #7 is explicitly deselected;
these counts do not claim it passes. The CUDA-only format-header helper is
skipped under ROCm; HIP codec/format tests run separately in the same suite.
RTX 5090 is not claimed validated by these runs.

## Review regression coverage

`tests/hip/test_review_contracts.py` covers negative/decreasing/extreme MoE
offsets with output guards; tail-dependent workspace sizing and rejection
before output writes; zero-token routing on a non-default stream; padded
strides for both packed and BF16t inputs; bias alignment; slice byte budgets;
exact BF16 GEMM results for single- and multi-slice fallbacks; and WMMA launch
grids above the old 32767-block ceiling. The latter admits Y up to 65535
and now spans larger routed workloads across Y/Z (see the contract follow-up below).
`tests/api/test_jit.py` verifies that changing `HIPFLAGS` invalidates the HIP
build cache without affecting CUDA identity. The MoE twin test requires exact
BF16 bit equality and finite outputs.
`tests/hip/test_hip.py` checks matched BF16t kernels bitwise and uses dense
BF16 only for the separate accumulation-order comparison. The C99 consumer runs against the active CUDA/HIP runtime and checks the
shared exported API. Status strings come from one shared `csrc/status.cpp`;
runtime error state is thread-local and survives successful/validation calls.
CMake uses `hip::host` for runtime headers, including LLVM-subdirectory compiler paths.
The native HIP codec and linear entry points now share the C ABI's buffer,
alignment, stride and workspace checks; 26 negative calls return
`hipErrorInvalidValue` before a GPU launch. A default source build selects HIP
when the installed PyTorch wheel uses ROCm, even with `nvcc` available; an
explicit `NF12_ENABLE_HIP=OFF` fails at configure time. The composed fused-MoE
route zero-initializes the final assignment buffer before combining dropped
slots, including when its torch op is called directly with `zero_fill=False`.

### Independent review follow-up

Two independent reviews examined head `9a5074e`. Concurrency probes passed
776 tensor comparisons across CUDA/HIP, including GIL=0, multiple streams and
devices, graph replay, failed-build retries, and thread-local error isolation.
The separate correctness review passed 171 exact NF12/BF16t comparisons and
an all-65536-BF16-pattern codec check. These probes did not use a sanitizer
or test hardware faults; they are evidence for the exercised paths.

The review found the WMMA grid limit and missing `HIPFLAGS` cache identity.
Both are corrected. The post-fix suites above include the new regression
tests, using two visible GPUs per backend. On R9700, one 16-by-64 all-one
expert produces exactly 64 per element and matches its BF16t twin:

| Assignments | Reserved Y blocks | Result | Mismatches / changed guards |
| ---: | ---: | --- | --- |
| 524256 | 32767 | Success | 0 / 0 |
| 524272 | 32768 | Success | 0 / 0 |
| 1048544 | 65535 | Success | 0 / 0 |
| 1048560 | 65536 total tiles across Y/Z | Success | 0 / 0 |

### Fresh-context review

Two new agents independently reviewed `526b89d` without inherited review
context. All three confirmed findings are fixed:

- `bench/hip_linear.py --device cuda:1` now selects that device before native
  launches, stream acquisition, and timing. The regression test failed before
  the fix and passes on two R9700s, with zero incorrect outputs at M=1 and 16.
- Mixed CUDA+HIP C-library builds now pass `-Wall -Wextra -Werror`; both
  `nf12` (SM89) and `nf12_hip` (gfx1201) build in the same CMake tree.
- `nf12_hip_linear_wmma_bf16t` now uses the existing shared validation and
  chunk dispatcher. Native-symbol tests cover M=256, 257, 271, 272, 288 and
  512, including non-default streams, small tails, tail-dependent scratch,
  undersized workspace rejection before writes, and guarded allocations.
  All 1581056 output elements match the NF12 C-API twin bitwise; all outputs
  are finite and guards are unchanged.

The latest full HIP run took 469.64 s while this host ran other work; the
earlier full CUDA run took 22.07 s. Both used GIL-disabled Python and two
visible devices. These are test wall times, not kernel benchmarks.

A separate integration checkout at `8f31489` applied PR #7's test-only correction
while loading the same native sources (before the additional graph tests). With GIL=0
and two GPUs, the full suites passed with **zero deselections**:

| Suite with PR #7 test correction | Device | Passed | Skipped | Seconds |
| --- | --- | ---: | ---: | ---: |
| HIP + shared API | R9700 | 659 | 12 | 26.26 |
| Shared API | RTX 4090 | 459 | 13 | 20.23 |

PR #7 remains separate; these integration results do not claim the unchanged
reference-tolerance test passes in PR #3 alone.

### Native MoE bounds follow-up

A further pair of fresh-context agents reviewed `c6c4da0`. No new concurrency
issue was confirmed: 1280 concurrent dense calls and 840 graph replays with
2840 output comparisons passed on gfx1201 and SM89 with GIL disabled.
The independent correctness probes included 468 exact linear/twin cases and
121 explicit codec bases per backend; CUDA base/head fingerprints matched.
No race sanitizer, SM90+ hardware, or large-allocation opt-in tests were run.

The correctness review found missing validation in native HIP MoE decode and
SiLU. The fixed native entry points also validate the adjacent combine and
gather buffers. Shape/alignment helpers are shared with the C ABI. Twenty-four
new cases verify slice limits, row strides, alignment and overflow rejection,
with **zero output/guard writes**. A 16-row slice starting at N now returns
`hipErrorInvalidValue` instead of reading the trailing encoded matrix. SiLU
with output stride zero now returns that error instead of overwriting output.

The `NF12_BUILD_NATIVE=OFF` source-only wheel retains all 78 required source
files byte-for-byte, including `csrc/validation.h`, and contains no compiled
libraries. Before the bounds fix, installed-wheel JIT probes on both backends
passed all 65536 BF16 bit patterns and 78080 exact twin elements each. Packed bytes (98304), escapes
(131072 bytes), and exponent base matched across CUDA and HIP.

R9700 valid-call A/B against `c6c4da0`: same-process interleaved calls, 20 warmups,
101 host-call samples and 15 event-timed groups of 50 launches per version.
The event loop includes host submission gaps. Medians in microseconds;
all outputs were bitwise equal. The largest host increase was 0.030 us;
timing differences below are measurements, not claims of kernel speedups.

| Operation | Shape | Host before | Host after | Event-loop before | Event-loop after |
| --- | --- | ---: | ---: | ---: | ---: |
| silu_mul | rows=128, inter=4096 | 2.654 | 2.664 | 15.088 | 15.283 |
| combine | tokens=128, topk=8, N=4096 | 2.724 | 2.754 | 14.780 | 14.789 |
| decode | E=4, N=256, K=512 | 3.556 | 3.575 | 4.434 | 4.450 |
| grouped_operands | assignments=512, E=4, K=4096 | 3.275 | 3.295 | 14.398 | 14.131 |

## Inference graph capture

The inference audit on `8f31489` found no capture blocker on R9700. Explicit
host synchronization is confined to encoding; inference launchers allocate no
device scratch internally. Encoding and packing are outside this requirement.
JIT/loading, weight preparation and route warmup happen before capture.

With GIL disabled and two R9700s, changed-input replay covered decode, embedding,
small-M and WMMA linear, BF16t twins, large-M decode/GEMM, MoE alignment,
routed linear, activation/combine, and single-token/composed fused MoE.
The targeted graph suite passed **32 tests, 1 skipped, 233 deselected in 6.00 s**;
the skipped CUDA grouped-GEMM route is unavailable on RDNA4.

| Replay check | Graphs | Replays | Incorrect outputs / changed guards |
| --- | ---: | ---: | ---: |
| Native small-M, split-K and chunked dense/BF16t | 8 | 40 | 0 |
| Native decode, BF16t unpack and selftest | 1 | 5 | 0 |
| Native grouped operands, with/without gather | 2 | 10 | 0 |
| Inference families, changed activations/routing on two devices | 2 | 20 | 0 |
| Concurrent graphs, two streams per device | 4 | 400 | 0 |

The last two rows checked 620 and 800 output tensors respectively, exactly
against eager execution. Concurrent graphs used independent pools, outputs
and scratch; these results do not permit concurrently replaying graphs that
share writable buffers. Native callers retain captured pointers and workspace
for the graph lifetime. Torch owns its captured allocations. Only gfx1201 was
runtime-tested here; this is not coverage of every architecture/configuration.

The eleven native regression cases are in `tests/hip/test_inference_graphs.py`;
the existing shared API graph tests also cover a multi-slice decode/GEMM route.
Reproduce the targeted suite with the ROCm environment above:

```sh
python -m pytest tests/hip/test_inference_graphs.py tests/api/test_torch.py \
  tests/api/test_embedding.py tests/api/test_moe.py tests/api/test_decode_route.py \
  -k graph -q
```

## CUDA contract follow-up

An independent reviewer started without prior review context at `434ff38` and
reproduced four contract differences. Commits `317998c`, `c16e287`, `edb7af4`
and `1c6e4c1` correct them individually. The reviewer rechecked the arithmetic,
routed indexing and native validation; the shared/native regression also
verifies the corrected adapter's error category.

| Contract case | Before | Fixed |
| --- | --- | --- |
| Cancelling split-K dot product, bias 1 | 0 | 1, exact |
| Cancelling split-K dot product, bias 2^-25 | 0 | 2^-25, exact |
| Finite cancelling dot product, bias 2^127 | Infinity | 2^127, finite and exact |
| MoE with 1048545 through 2097120 assignments | Unsupported above Y limit | Correct outputs; zero changed guards |
| Native MoE K=0 or K=65 | Success, truncated computation | Invalid argument, no writes |
| Native MoE empty work | Invalid argument | Success, no writes |
| Shared MoE decode E=65536 | Unsupported (2) | Invalid argument (1), matching CUDA |

Bias joins the complete FP32 reduction before BF16 rounding. Both NF12 and
BF16t previously shared the incorrect early-bias epilogue, so twin equality
alone could not detect it. Regression expectations use exactly cancelling
products and their analytically known bias, without relaxing tolerances.
The shared C frontend and native HIP launcher now reuse the buffer validator.
Routed tiles extend into grid.z while processing all K; dense split-K retains
its separate Z interpretation. Boundary graph replay also passes.

After these changes, the suites at the top of this report pass with GIL disabled.
The unchanged PR #7 reference-tolerance case remains explicitly deselected.
The C99 consumers and descriptor-layout checks pass on both backends. Before
these repairs, direct fingerprints matched all 25 shared C symbols, 14 torch
schemas and eight status-string cases; no signatures or schemas changed.

### Contract-fix timing

R9700 gfx1201; old `434ff38` library and repaired library in one process,
interleaved order. Twenty warmups, 101 host-call samples, 21 GPU samples with
32 native calls per captured graph; warm resident weights, median microseconds.
These eight cells are a targeted regression check, not the full shape grid.

| Operation | Shape | Host before | Host after | GPU before | GPU after |
| --- | --- | ---: | ---: | ---: | ---: |
| dense | M=16,N=256,K=2048 | 4.787 | 4.807 | 8.102 | 8.207 |
| dense+bias | M=16,N=256,K=2048 | 5.097 | 5.067 | 8.295 | 8.390 |
| dense | M=16,N=4096,K=4096 | 4.927 | 4.998 | 22.301 | 22.336 |
| dense+bias | M=16,N=4096,K=4096 | 5.178 | 5.288 | 21.361 | 21.537 |
| dense | M=64,N=4096,K=4096 | 4.937 | 4.937 | 35.895 | 36.197 |
| dense+bias | M=64,N=4096,K=4096 | 5.338 | 5.388 | 36.365 | 37.513 |
| moe | A=16,E=4,N=128,K=512 | 4.627 | 4.617 | 10.190 | 10.369 |
| moe | A=512,E=4,N=128,K=512 | 4.577 | 4.587 | 10.597 | 10.769 |

The largest measured GPU increase is 3.16% (M=64, N=K=4096 with bias);
maximum host increase is 0.110 us. Unbiased outputs are bitwise unchanged.
Biased outputs can change because the previous bias ordering was incorrect;
the largest random-data difference in this sample was 6.103515625e-5.
No claim of universal performance parity is made.

## Large-batch fallback

Baseline: PR #3 at `bacb6a1`, fixed 64-row slices. Revised code: the same
256 MiB external decode-buffer budget as CUDA, with tile-aligned balanced
slices. Decode storage exists only during the call; no expanded weights are
retained. Smaller batches retain the native fused route.

Synthetic BF16 weights, repeated warm inputs; HIP-event time for public
`nf12.linear`. Five warmups, seven groups of ten calls; median below. These
are sequential baseline/revised runs, not a cold-cache bandwidth benchmark.
Full-decode plus BF16 GEMM is shown as a performance reference and includes
decoding on every call. Times are microseconds.

| K × N | M | Before µs | After µs | Speedup | Full decode + GEMM µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 × 4096 | 257 | 986.08 | 123.98 | 7.95× | 123.98 |
| 4096 × 4096 | 512 | 1007.84 | 212.38 | 4.75× | 211.44 |
| 4096 × 4096 | 1024 | 1097.27 | 338.44 | 3.24× | 336.93 |
| 896 × 4864 | 257 | 1227.42 | 44.72 | 27.45× | 44.73 |
| 896 × 4864 | 512 | 1214.24 | 55.41 | 21.91× | 55.32 |
| 896 × 4864 | 1024 | 1226.67 | 87.20 | 14.07× | 86.17 |
| 2048 × 8192 | 257 | 2026.16 | 129.96 | 15.59× | 130.45 |
| 2048 × 8192 | 512 | 2050.40 | 179.74 | 11.41× | 183.17 |
| 2048 × 8192 | 1024 | 2019.86 | 346.71 | 5.83× | 338.28 |
| 4096 × 14336 | 257 | 3578.24 | 762.08 | 4.70× | 769.49 |
| 4096 × 14336 | 512 | 3661.60 | 826.83 | 4.43× | 828.19 |
| 4096 × 14336 | 1024 | 3867.88 | 1198.35 | 3.23× | 1193.79 |

## Native small-batch regression check

Before/after raw HIP kernels run in the same process with identical weights
and activations. Five warmups, 12 alternating groups of 30 calls per kernel;
HIP-event medians below. Every paired output was bitwise equal. The stride
support change preserves the contiguous ABI and arithmetic order. Reported
small differences include timing variation; no claim of a speedup is made.

| K × N | M | Before µs | After µs | Before / after |
| --- | ---: | ---: | ---: | ---: |
| 4096 × 4096 | 1 | 19.42 | 19.33 | 1.005× |
| 4096 × 4096 | 8 | 24.21 | 24.13 | 1.003× |
| 4096 × 4096 | 16 | 24.12 | 24.12 | 1.000× |
| 4096 × 4096 | 32 | 28.29 | 28.38 | 0.997× |
| 4096 × 4096 | 64 | 37.48 | 37.52 | 0.999× |
| 4096 × 4096 | 128 | 55.12 | 54.73 | 1.007× |
| 896 × 4864 | 1 | 7.53 | 7.69 | 0.980× |
| 896 × 4864 | 8 | 10.98 | 10.95 | 1.003× |
| 896 × 4864 | 16 | 8.96 | 8.96 | 1.000× |
| 896 × 4864 | 32 | 10.50 | 10.49 | 1.001× |
| 896 × 4864 | 64 | 14.46 | 14.42 | 1.003× |
| 896 × 4864 | 128 | 20.70 | 20.67 | 1.002× |
| 2048 × 8192 | 1 | 16.83 | 16.93 | 0.994× |
| 2048 × 8192 | 8 | 23.27 | 23.23 | 1.002× |
| 2048 × 8192 | 16 | 23.36 | 23.45 | 0.996× |
| 2048 × 8192 | 32 | 25.13 | 25.78 | 0.975× |
| 2048 × 8192 | 64 | 30.59 | 30.94 | 0.989× |
| 2048 × 8192 | 128 | 60.90 | 61.08 | 0.997× |
| 4096 × 14336 | 1 | 143.33 | 143.37 | 1.000× |
| 4096 × 14336 | 8 | 148.76 | 148.73 | 1.000× |
| 4096 × 14336 | 16 | 153.56 | 153.53 | 1.000× |
| 4096 × 14336 | 32 | 165.09 | 165.30 | 0.999× |
| 4096 × 14336 | 64 | 178.72 | 178.74 | 1.000× |
| 4096 × 14336 | 128 | 277.38 | 275.94 | 1.005× |

## Packing

Public `nf12.encode` wall-clock medians include allocation and synchronization;
eight warmups and 31 alternating samples per implementation on R9700. Baseline
is `bacb6a1`; the revised encoder changes scheduling, not the file format.
All 28 measured normal/special-value cases improved end-to-end.

| K × N | Before µs | After µs | Speedup |
| --- | ---: | ---: | ---: |
| 2048 × 128 | 156.8 | 138.0 | 1.14× |
| 2048 × 64 | 155.7 | 136.8 | 1.14× |
| 4096 × 128 | 156.4 | 137.7 | 1.14× |
| 4096 × 4096 | 301.3 | 226.3 | 1.33× |
| 4096 × 14336 | 1175.3 | 884.5 | 1.33× |
| 14336 × 4096 | 1176.7 | 887.7 | 1.33× |
| 4096 × 28672 | 1989.5 | 1630.1 | 1.22× |
| 5120 × 25600 | 2268.8 | 1810.3 | 1.25× |
| 25600 × 5120 | 2236.3 | 1830.2 | 1.22× |
| 8192 × 8192 | 1332.4 | 995.6 | 1.34× |
| 8192 × 57344 | 6483.7 | 6076.2 | 1.07× |
| 2048 × 8192 | 328.7 | 243.2 | 1.35× |
| 896 × 4864 | 167.6 | 160.8 | 1.04× |

The largest tested analysis workspace grows from 448 KiB to 1.75 MiB.
Its scan stage grows from 36.1 to 137.4 µs, while total encoding improves.
`pack_bf16t` is unchanged. Tests check all 65,536 BF16 bit patterns, scan
boundaries, exact packed bytes and escape order, streams, graphs, and GIL=0.
Reproduce with `bench/hip_encode.py --help`.

## Encode pass order (L3 reuse)

Encoding streams the weight three times (analyze, count, write). The 64 MiB
Infinity Cache retains only the tail of the previous pass, so reading every
pass front-to-back made each pass start on the evicted end. Counting
back-to-front now starts where analyze finished and leaves the head resident,
which is exactly where the write pass begins. Counts are still stored by flat
unit id, so scan order, escape order and packed bytes are unchanged.

Public `nf12.encode` wall clock, 8 warmups and 31 alternating samples per
implementation, median, same process:

| K × N | Before µs | After µs | Speedup |
| --- | ---: | ---: | ---: |
| 4096 × 14336 | 886.6 | 761.3 | 1.165× |
| 14336 × 4096 | 887.8 | 758.6 | 1.170× |
| 8192 × 8192 | 986.1 | 872.5 | 1.130× |
| 4096 × 28672 | 1627.8 | 1510.6 | 1.078× |
| 5120 × 25600 | 1806.0 | 1681.6 | 1.074× |
| 25600 × 5120 | 1804.5 | 1679.3 | 1.075× |
| 8192 × 57344 | 6071.5 | 5971.2 | 1.017× |
| 4096 × 4096 | 220.2 | 219.9 | 1.002× |
| 2048 × 128, 2048 × 64, 896 × 4864 | 137.6, 135.7, 158.5 | 137.8, 136.0, 158.4 | ~1.00× |

The saving is a fixed 110–127 µs for weights above 64 MiB, consistent with two
cache hits. Flushing 512 MiB of L3 before every call still measures 1.152×
(4096 × 14336) and 1.108× (8192 × 8192), so this is not a warm-benchmark
artifact. Kernel-only `encode_gpu` improves 1.212×/1.224×. Shapes at or below
L3 size are unchanged, and the mandated host synchronizations still dominate
them: at 2048 × 128 the public call is 129.3 µs against 21.3 µs of kernel time.

`pack_bf16t` remains a pure byte shuffle at the copy ceiling (569–571 GB/s).
A 16-warp variant measured 2.5–4.2% faster cold but ~10% slower on the
L3-resident 4096 × 4096 case, so it was not adopted.

## Decode occupancy and MoE decode parallelism

Dense decode and `nf12_hip_unpack_bf16t` already ran at the streaming ceiling
(560–568 GB/s against a 574 GB/s 512 MiB copy), so the remaining cost was
occupancy. One super-atom per warp instead of two drops the kernel from 48 to
32 VGPRs and doubles the CTA count; the arithmetic is untouched.

Cold interleaved A/B, event medians:

| K × N | Before µs | After µs | Change |
| --- | ---: | ---: | ---: |
| 896 × 4864 | 30.14 | 28.42 | +6.06% |
| 2048 × 8192 | 105.32 | 102.06 | +3.20% |
| 4096 × 4096 | 105.16 | 102.03 | +3.06% |
| 14336 × 4096 | 360.28 | 350.63 | +2.75% |
| 4096 × 14336 | 359.35 | 352.04 | +2.08% |
| 8192 × 8192 | 411.66 | 403.27 | +2.08% |

`moe_decode` was genuinely under-parallelised: its grid was (row blocks,
experts) and each warp serially walked `k_blocks / kWarpsPerBlock` super-atoms,
reaching only 435–496 GB/s. `blockIdx.x` now packs (row block, K slice), so each
CTA owns a disjoint contiguous run of whole super-atoms and each warp owns one.
Every output element is still written exactly once and there is no reduction.

| E | K × N | Before µs | After µs | Change |
| ---: | --- | ---: | ---: | ---: |
| 4 | 14336 × 4096 | 1891.6 | 1426.6 | +32.6% |
| 4 | 4096 × 4096 | 485.4 | 405.6 | +19.7% |
| 8 | 4096 × 4096 | 947.6 | 818.5 | +15.8% |
| 1 | 4096 × 4096 | 39.8 | 35.7 | +11.5% |
| 4 | 2048 × 8192 | 424.2 | 404.3 | +4.9% |

MoE decode now reaches 574–581 GB/s. The matrix-core route for m = 16..256 is
untouched: `bench/hip_linear.py` gives a post/baseline median ratio of 1.000
(min 0.965, max 1.040) across the shape grid, with zero correctness violations.
A `quad_transpose` shared-store scheme for the decode-into-shared stage helped
wide-N/large-K by up to 7% but regressed narrow-N by 12–19%, so it was not
adopted. Both changes are bit-exact for all 65,536 BF16 patterns, for forced
`exp_base` 0 and 120, through `pack_bf16t`/`unpack`, and for MoE decode at
K = 4096 with sliced `row0`/`rows`.
