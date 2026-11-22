# NF12 on Apple GPUs (Metal)

On macOS, nf12 builds a Metal backend instead of CUDA, for two frameworks:

- **PyTorch**: pass MPS tensors (`device="mps"`) to `nf12.encode`, `nf12.linear`,
  `nf12.fused_moe` and the rest. Every `torch.ops.nf12.*` op has a Metal implementation.
- **MLX**: pass `mlx.core.array`s to the same top-level functions, or use
  `nf12.mlx` directly (same function names, `NF12Weight`/`NF12MoEWeight` holding MLX
  arrays, and `mlx.nn` modules `NF12Linear` and `NF12Embedding`). `import nf12.mlx`
  does not import PyTorch.

Both frameworks share the Metal shaders and launch code. A matching fused
route gives bit-identical results. The encoded format (docs/SPEC.md) is byte-for-byte
the same as on CUDA, so weights move freely between all three.

```python
import mlx.core as mx
import nf12.mlx as nm

nw = nm.encode(w)                      # w: bf16 [N, K]; synchronizes once
y = nm.linear(x, nw, bias)             # lazy, like any MLX op
layer = nm.NF12Linear.from_linear(lin) # mlx.nn.Linear -> NF12
mx.save_safetensors("w.safetensors", nw.state_dict("w."))
nm.load_weights(model, "model.safetensors")  # a model containing NF12 modules
```

Load NF12 checkpoints into a model with `nm.load_weights(model, ...)`, not
`model.load_weights(...)`. MLX's strict load requires every array to keep its
shape, but the number of escape rows differs per weight. `nm.load_weights`
makes the same checks except for that, and leaves the model unchanged if it
rejects the checkpoint (wrong dtypes or shapes, exp_base out of range, or a
weight whose packed array and escape rows disagree).

With `strict=False` a checkpoint may supply part of a weight, as when one
weight's tensors are split across shards: supplied entries merge with the
current ones (in PyTorch, `load_state_dict(strict=False)` does the same). Such a
weight is only consistent once all of its tensors are loaded, so after the last
shard call `nm.verify_weights(model)` (or `nf12.verify_weights(model)`, which
handles both frameworks). It checks every NF12 weight with one sync, and in
PyTorch also reports weights still on the meta device: a module built on meta
needs all three of a weight's tensors in one `load_state_dict` call.

## Requirements

- Apple Silicon Mac, macOS 14 or newer.
- Apple Command Line Tools for source or first-use native builds (`xcode-select
  --install`). Metal compiles the embedded shaders on the first GPU operation.
- CMake 3.26+, Ninja, uv, Python 3.10+ and PyTorch 2.10+ with MPS.
- For MLX: MLX 0.32+ and the nanobind version MLX was built with (3.1.0 for MLX
  0.32), installed by the `mlx` extra.

## Build

```bash
uv sync --locked --group test --extra mlx  # libnf12_mps.dylib and nf12/_mlx_ext*.so
NF12_BUILD_NATIVE=OFF uv sync --locked      # or defer the build to first use (python/nf12/_jit.py)
```

CMake chooses the backend from the host: `cmake/Metal.cmake` on macOS, the CUDA
rules in `CMakeLists.txt` elsewhere. `NF12_BUILD_MLX=AUTO` (default) builds the MLX
module when the build interpreter can import `mlx` and `nanobind`; `ON` requires
it, `OFF` skips it. For a standalone build:

```bash
cmake -S . -B build/metal -G Ninja -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build build/metal   # build/metal/{libnf12_mps.dylib,_mlx_ext*.so}
```

`LDFLAGS`/`CPPFLAGS` that point at a non-Apple LLVM (for example
`-L/opt/llvm/lib`) make the extension link against a different libc++ than
PyTorch and fail with undefined `std::length_error` symbols; unset them for the build.

Neither framework has a stable ABI for this, so each extension is built against one
version and records it. When the installed version differs, nf12 builds a matching
copy in the JIT cache (`~/.cache/nf12/<key>/build`) on first use:

- The torch library uses MPS stream internals. CMake writes the torch version next to
  it (`libnf12_mps.torch_version`), and the loader uses a packaged copy only under that
  torch. CUDA's `libnf12_torch.so` uses the torch stable ABI and needs no such check.
- The MLX module records its MLX version (`nf12._mlx_ext.mlx_version`); the MLX
  rebuild is MLX-only and does not need torch. It needs the nanobind version the
  installed MLX uses: MLX's bindings share nanobind state, and so accept `mx.array`
  arguments, only with the same nanobind ABI. `nf12.mlx` checks this when it loads
  the module and names the mismatch instead of failing every call. uv builds
  (`[tool.uv.extra-build-dependencies]`) pin MLX 0.32.4 with nanobind 3.1.0; other
  installers build the module on first use, so mlx is never a build requirement.

A macOS wheel is CPython specific because of the MLX module.

## Layout

| File | Contents |
|---|---|
| `csrc/metal/nf12_metal.h` | MSL format helpers: group decode/encode, BF16 round-to-nearest-even, NF12 and bf16t weight sources |
| `csrc/metal/linear_core.h` | Fused decode + multiply cores shared by dense and MoE kernels |
| `csrc/metal/codec.metal` | Encode (analyze, count, write), decode (dense and stacked experts), bf16t packing, embedding |
| `csrc/metal/linear.metal` | Dense `linear` / `linear_bf16t` kernels |
| `csrc/metal/linear_tensor.metal` | Metal 4 TensorOps dense prefill kernels |
| `csrc/metal/moe.metal` | `moe_align`, tile schedule, grouped linear, `moe_combine`, `moe_silu_mul` |
| `csrc/metal/launch.h` | Host ABI shared by both frameworks: kernel names, buffer indices, parameter structs, grid geometry, routing |
| `csrc/metal/ops_mps.mm` | `torch.ops.nf12` schemas and MPS implementations |
| `csrc/mlx/nf12_mlx.cpp` | MLX primitives and the `nf12._mlx_ext` nanobind module |
| `python/nf12/mlx.py` | MLX Python API and `mlx.nn` modules |
| `csrc/metal/shader_library.mm` | Loads or compiles the Metal shader library shared by both extensions |
| `cmake/Metal.cmake` | Builds the MPS and MLX extensions with embedded shader source |

Each `launch_*` function in `launch.h` is written once against a small encoder
interface (`pipeline`, `in`, `out`, `bytes`, `dispatch`, `barrier`), as in
QuixiCore-Metal's `tk_launch.h`. The two adapters are:

- **PyTorch** (`TorchEncoder`): binds `MTLBuffer`s on PyTorch's current MPS stream
  through its shared compute encoder, so kernels order with surrounding torch work
  without extra synchronization.
- **MLX** (`MlxEncoder`): one MLX `Primitive` per op whose `eval_gpu` binds through
  `set_input_array`/`set_output_array` on MLX's command encoder. That keeps MLX's
  dependency tracking and buffer lifetimes, so ops stay lazy and compose with other
  MLX ops in one command buffer. Because MLX encoders dispatch concurrently, launches
  that write a buffer twice (zeroing, then accumulating) call `barrier()`, which is a
  no-op on PyTorch's serial encoder.

`NF12_METALLIB` can point to a compatible precompiled shader library. By default,
each backend compiles the required embedded libraries on first use.

## Kernels

A simdgroup (32 lanes) plays the role of a CUDA warp: lane `l = 4g + t` owns group
`l` of each atom (SPEC.md 4.1), so one coalesced 16-byte load per plane gives a lane
its four groups of a super-atom, exactly as on CUDA.

- **Encode** follows `csrc/codec/encode.cu`: per-group exponent intervals feed three
  histograms (the host picks `exp_base`), escapes are counted per super-atom,
  prefix-summed, and written in flat group id order. Output bytes equal
  `nf12.reference.encode`.
- **Small M (scalar)**: one simdgroup streams the super-atoms of a 16-row block,
  decoding in registers and accumulating rows `g` and `g + 8` per lane in FP32.
  When K is split across simdgroups, partials are summed in a fixed order.
  Without a split, each simdgroup writes its result directly.
- **Medium M (register MMA)**: Apple's 8x8 `simdgroup_matrix` fragment gives lane `l`
  the pair `(fm, fn), (fm, fn + 1)`. A decoded NF12 register is a bf16 pair of an
  8x8 quarter of an atom, so each A fragment is one `simd_shuffle` from lane
  `4 fm + fn / 2`; activations load straight into B fragments. Threadgroup memory
  is used only to combine split-K partials.
- **Fused GEMM**: a 64-row weight tile is decoded into threadgroup memory.
  Dense linear stages BF16 operands and MoE stages FP32 operands. Both
  accumulate in FP32.
- **Decode-then-GEMM** (SPEC.md 6.5): NF12 `linear` at large M decodes row slices
  of up to 64 MiB by default, then calls the framework's BF16 matmul or addmm
  on each slice.
- **MoE**: `moe_align` is a deterministic stable counting sort (chunk counts, chunk
  bases, in-chunk ranks; a stable device sort above 1024 experts). A one-threadgroup
  schedule turns device-side expert offsets into tiles sized to the expected rows
  per expert, so nothing synchronizes with the host. Grouped linear reuses the
  scalar, MMA and GEMM cores with gather, scatter and FP32 row scaling.

## Numerics

The kernels disable fast math and follow SPEC.md section 6: FP32 accumulation, one
BF16 round-to-nearest-even, fixed reduction orders, and a single kernel template
per route for both weight sources, so NF12 output is bit-identical to the bf16t twin
on every fused route. MLX and MPS outputs are bit-identical when both use
the same fused route. Each framework's decode-then-GEMM route matches its own
matmul on the decoded weight (as on CUDA).

## Routing

Outside the prefill cases below, linear uses scalar kernels up to 4 tokens and
register MMA up to 16. Larger inputs use decode-then-GEMM for NF12 or fused GEMM
for bf16t. MoE uses scalar kernels up to 4 expected rows per expert and register
MMA up to 8, then fused GEMM.

- At 17–32 tokens, both backends use fused GEMM for `N >= 4096` and
  `2048 <= K <= 16384`. MLX also uses it for `N >= 512` and
  `2048 <= K <= 4096`.
- On macOS 27 or later with Apple10 GPU support, Metal 4.1 TensorOps handles
  selected dense shapes at 16–128 tokens. Long reductions use ordered split-K.
  Explicit route overrides and `NF12_METALLIB` select the standard kernels.

`NF12_MPS_LINEAR=small|mma|gemm|decode` and `NF12_MPS_MOE=small|mma|gemm` force a
route for both frameworks (tests run the suites under each; the MLX tests set them
per test, since MLX reads them when an op is built). `torch.ops.nf12._mps_decode_gemm_rows(m, n, k)`
reports the decode-then-GEMM slice height (0 for fused routes).

## Performance (M5 Max, macOS 26.6, PyTorch 2.15)

Median of 7 timed batches after a GPU warm-up; run-to-run variation is about ±10-20%.

Dense `linear`, 14336 x 4096 (117 MB BF16, larger than the GPU's system cache):

| M | torch BF16 | NF12 | |
|---|---:|---:|---|
| 1 | 199 µs | 152 µs | 1.31x faster |
| 4 | 200 µs | 168 µs | 1.19x faster |
| 8 | 212 µs | 189 µs | 1.12x faster |
| 16 | 214 µs | 281 µs | 1.31x slower |
| 128 | 321 µs | ~700 µs | decode-then-GEMM |

At M <= 8 NF12 streams 25% fewer bytes at DRAM bandwidth. Beyond that, matmul is
compute bound and BF16 matmul on MPS (which uses the GPU's matrix units) wins;
NF12 then pays the decode pass.

MLX (`nf12.mlx.linear` against `x @ w.T` in bf16, same shape, 30 independent ops per timed batch):

| M | MLX BF16 | NF12 | |
|---|---:|---:|---|
| 1 | 205 µs | 147 µs | 1.39x faster |
| 4 | 167 µs | 148 µs | 1.13x faster |
| 8 | 334 µs | 162 µs | 2.05x faster (MLX switches GEMV to GEMM here) |
| 16 | 215 µs | 229 µs | 1.06x slower |
| 128 | 438 µs | ~800 µs | decode-then-GEMM |

`fused_moe` (128 experts, top-8, K = 2048, I = 768): 150 µs at T = 1 against a
bytes-only BF16 floor of about 69 µs (75 MB of active experts at 550 GB/s), so the
small-T MoE path still has headroom; see below.

## Testing

`tools/run_tests.py` runs the whole matrix for the machine it is on and prints a
summary. On macOS that is every suite (PyTorch MPS and MLX), then the suites under
each forced route. On Linux it is the CUDA API suite and the release smoke load. A
configuration that runs no tests fails. Changes are tested on every supported
platform before they are pushed: an Apple silicon Mac, a CUDA x86-64 machine and a
CUDA aarch64 machine (for example an RTX 5090 host and a DGX Spark).

What the suites check:

- **Kernels against references:** codec bytes equal `nf12.reference`, NF12 output
  equals the bf16t twin bit for bit, and MLX output equals MPS output bit for bit
  on nf12's own kernels.
- **Module state** (`tests/api/test_modules.py`, `tests/mlx/test_mlx_modules.py`):
  NF12 modules inside parent models get their weights replaced through every loading
  path users take (a parent's or a child's update or load, files, partial and rejected
  checkpoints). After each, every observable is checked: the forward pass, the weight
  and its decode, a compiled forward captured beforehand, and a save/reload. Testing
  module methods directly missed paths that never call them.
- **Cross-framework checkpoints:** a checkpoint saved by either framework's modules
  loads into the other's with bit-identical decoded weights, and bit-identical outputs
  on nf12's own kernels.

Test data comes from `tests/portable_random.py`, so it is bit-identical on every
machine and device. A new test should fail on the code it guards; check that by
running it against the unfixed code.

Bit-exact assertions are only made where nf12's code fixes the arithmetic order:
nf12's own kernels under the same configuration with fast math disabled, or the same framework call
at the same shape on the same inputs. The decode route's GEMM belongs to the
framework (MPS or MLX), which chooses its kernel by GPU and shape, so it is
compared with a tolerance against anything else. A bit-exact pass on one Apple GPU
does not carry over to another, so Metal changes should also be run on a second
Apple GPU generation when one is available.

## Limitations and next steps

- Small-T MoE runs 9 dispatches (align, schedule, two grouped linears, SiLU-mul,
  combine); fusing align with the schedule and SiLU-mul into the w13 epilogue, as
  CUDA's single-token route does, would cut most of the remaining overhead.
- `moe_linear` supports up to 1024 experts (as on CUDA); `moe_align` supports any count.
- The C API (`include/nf12/nf12.h`) and CUDA graphs are CUDA only.
- MLX: ops have no `vjp`/`vmap` (inference only) and run on the GPU stream only.
  The `mlx.nn` modules work under `mx.compile` with their state captured: a dense
  weight's `exp_base` reaches the kernels as an array, so a reused trace stays correct
  after the weight is replaced, however it is replaced (an update or load on the
  module or on any parent). `mx.compile(..., shapeless=True)` is not supported:
  the kernels are specialized to the input's row count, so it fails with
  `cannot infer output shapes`. Inputs must start 4-byte aligned (16 for packed
  weights); a row-contiguous slice at an odd offset is rejected when evaluated,
  so copy it first (`mx.array(x)`). The package still depends on PyTorch.
