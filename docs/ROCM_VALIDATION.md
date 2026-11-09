# ROCm validation and performance

Validated on Radeon AI PRO R9700 (`gfx1201`, 8 MiB L2, 64 MiB L3), ROCm SDK
10.1.0, HIP 7.16.26385 (`7.16.26385-0000000`), PyTorch
2.16.0.dev20261025+rocm10.1, and free-threaded Python 3.14 (`PYTHON_GIL=0`).
OMP/MKL threads were limited to one. Libraries were built from source.

## Test results

| Suite | Device | Passed | Skipped | Deselected |
| --- | --- | ---: | ---: | ---: |
| HIP + shared API | R9700 | 647 | 12 | 1 |
| Shared API | RTX 4090 | 447 | 13 | 1 |

Both runs use the shared ABI frontend and source-built native and torch libraries,
with two visible GPUs to exercise concurrent device/stream propagation. The known CUDA MoE
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
and rejects larger grids as unsupported before output writes.
`tests/api/test_jit.py` verifies that changing `HIPFLAGS` invalidates the HIP
build cache without affecting CUDA identity. The MoE twin test requires exact
BF16 bit equality and finite outputs.
`tests/hip/test_hip.py` checks matched BF16t kernels bitwise and uses dense
BF16 only for the separate accumulation-order comparison. The C99 consumer runs against the active CUDA/HIP runtime and checks the
shared exported API. Status strings come from one shared `csrc/status.cpp`;
runtime error state is thread-local and survives successful/validation calls.
CMake uses `hip::host` for runtime headers, including LLVM-subdirectory compiler paths.

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
| 1048560 | 65536 | Unsupported | No output writes |

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

The current PR suites above passed in 28.25 s (HIP) and 74.53 s (CUDA), with
GIL disabled and two visible devices. The benchmark/device selection fix does
not establish new performance numbers for the canonical shape grid.

A separate integration checkout applied PR #7's test-only correction to the
current tests while loading these same source-built libraries. With GIL=0
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

The source-only wheel retains all 78 required source files byte-for-byte,
including `csrc/validation.h`, and contains no compiled libraries. Before the
bounds fix, installed-wheel JIT probes on both backends passed all 65536 BF16
bit patterns and 78080 exact twin elements each. Packed bytes (98304), escapes
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
