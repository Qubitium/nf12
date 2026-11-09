# ROCm validation and performance

Validated on Radeon AI PRO R9700 (`gfx1201`, 8 MiB L2, 64 MiB L3), ROCm SDK
10.1.0, HIP 7.16.26385 (`7.16.26385-0000000`), PyTorch
2.16.0.dev20261025+rocm10.1, and free-threaded Python 3.14 (`PYTHON_GIL=0`).
OMP/MKL threads were limited to one. Libraries were built from source.

## Test results

| Suite | Device | Passed | Skipped | Deselected |
| --- | --- | ---: | ---: | ---: |
| HIP + shared API | R9700 | 611 | 12 | 1 |
| Shared API | RTX 4090 | 445 | 13 | 1 |

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
and exact BF16 GEMM results for single- and multi-slice fallbacks.
`tests/hip/test_hip.py` checks matched BF16t kernels bitwise and uses dense
BF16 only for the separate accumulation-order comparison. The C99 consumer runs against the active CUDA/HIP runtime and checks the
shared exported API. Status strings come from one shared `csrc/status.cpp`;
runtime error state is thread-local and survives successful/validation calls.
CMake uses `hip::host` for runtime headers, including LLVM-subdirectory compiler paths.

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
