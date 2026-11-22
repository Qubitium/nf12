// SPDX-License-Identifier: Apache-2.0
// NF12 for MLX (Apple GPUs). Every op builds lazy MLX graph nodes whose
// primitives encode the shared Metal kernels on MLX's own command
// encoder through the launches in csrc/metal/launch.h, the same host ABI the
// PyTorch MPS backend (csrc/metal/ops_mps.mm) uses. Binding through
// set_input_array/set_output_array keeps MLX's dependency tracking, so
// nothing synchronizes except encoding (which reads histograms on the host).
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "launch.h"
#include "shader_library.h"
#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;
using mx::array;
using std::optional;
using namespace nf12::metal;

namespace nf12_mlx {
namespace {

// ---------------------------------------------------------------------------
// Metal plumbing.

// Encoder adapter for the shared launches in launch.h.
struct MlxEncoder {
  using tensor = array;
  using out_t = array&;
  mx::metal::Device& d;
  mx::metal::CommandEncoder& ce;
  MTL::Library* lib;

  ThreadgroupLimit limit{};

  void pipeline(const std::string& name) {
    MTL::ComputePipelineState* pso = d.get_kernel(name, lib);
    ce.set_compute_pipeline_state(pso);
    limit.set(name, pso->maxTotalThreadsPerThreadgroup());
  }
  void in(const array& a, int index) { ce.set_input_array(a, index); }
  void out(array& a, int index) { ce.set_output_array(a, index); }
  template <typename T>
  void bytes(const T& v, int index) {
    ce.set_bytes(v, index);
  }
  void dispatch(uint64_t gx, uint64_t gy, uint64_t gz, uint64_t tx, uint64_t ty, uint64_t tz) {
    if (gx == 0 || gy == 0 || gz == 0) return;
    limit.check(tx * ty * tz);
    ce.dispatch_threadgroups(MTL::Size(gx, gy, gz), MTL::Size(tx, ty, tz));
  }
  // MLX encodes concurrent dispatches; order writes to the same buffer.
  void barrier() { ce.barrier(); }
};

using Body = std::function<void(MlxEncoder&, const std::vector<array>&, std::vector<array>&)>;

// One primitive type for every NF12 kernel family: `body` calls the shared
// launch with scalar parameters captured at graph-build time (never arrays).
class Kernel : public mx::Primitive {
 public:
  Kernel(mx::Stream stream, const char* name, Body body) : mx::Primitive(stream), name_(name), body_(std::move(body)) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(std::string("nf12.") + name_ + " runs on the GPU only");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    for (const array& a : inputs) {
      // Kernels read bf16 pairs and 16-byte packed vectors from row-contiguous
      // operands (ops below pass them through mx::contiguous).
      if (!a.flags().row_contiguous)
        throw std::invalid_argument(std::string("nf12.") + name_ + ": operands must be row contiguous");
      // Packed weights and escapes are read as 16-byte vectors, bf16 as pairs.
      const auto address = reinterpret_cast<uintptr_t>(a.data<char>());
      const bool vectors = (a.dtype() == mx::uint8 || a.dtype() == mx::int16) && a.ndim() >= 2;
      // bf16 matrices are read as element pairs; a 1-D bf16 bias as single elements.
      const uintptr_t need = vectors ? 16 : a.dtype() == mx::bfloat16 ? (a.ndim() >= 2 ? 4 : 2) : a.itemsize();
      if (a.nbytes() > 0 && address % need != 0)
        throw std::invalid_argument(std::string("nf12.") + name_ +
                                    ": operands must be aligned (copy sliced inputs with mx.array(x))");
    }
    for (array& out : outputs) out.set_data(mx::allocator::malloc(out.nbytes()));
    const mx::Stream& s = stream();
    mx::metal::Device& d = mx::metal::device(s.device);
    static const auto lib = NS::TransferPtr(static_cast<MTL::Library*>(new_shader_library(d.mtl_device())));
    MlxEncoder e{d, mx::metal::get_command_encoder(s), lib.get()};
    body_(e, inputs, outputs);
  }

  const char* name() const override { return name_; }

 private:
  const char* name_;
  Body body_;
};

mx::Stream gpu_stream() { return mx::default_stream(mx::Device::gpu); }

std::vector<array> kernel(const char* name, std::vector<mx::Shape> shapes, std::vector<mx::Dtype> dtypes,
                          std::vector<array> inputs, Body body) {
  return array::make_arrays(std::move(shapes), dtypes, std::make_shared<Kernel>(gpu_stream(), name, std::move(body)),
                            std::move(inputs));
}

array kernel1(const char* name, mx::Shape shape, mx::Dtype dtype, std::vector<array> inputs, Body body) {
  return array(std::move(shape), dtype, std::make_shared<Kernel>(gpu_stream(), name, std::move(body)),
               std::move(inputs));
}

array contig(const array& a) { return mx::contiguous(a, false, gpu_stream()); }

// ---------------------------------------------------------------------------
// Validation (mirrors csrc/metal/ops_mps.mm).

template <typename... Args>
[[noreturn]] void fail(const Args&... args) {
  std::ostringstream os;
  (os << ... << args);
  throw std::invalid_argument(os.str());
}

std::string shape_str(const array& a) {
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < a.ndim(); ++i) os << (i ? ", " : "") << a.shape(int(i));
  os << ']';
  return os.str();
}

void check_shape(int64_t n, int64_t k, const char* what) {
  if (n <= 0 || n % 16 != 0) fail("nf12: ", what, " needs N (out_features) to be a positive multiple of 16, got ", n);
  if (k <= 0 || k % 64 != 0) fail("nf12: ", what, " needs K (in_features) to be a positive multiple of 64, got ", k);
  // bf16t rows are 2K bytes: every dimension must fit MLX's int32 shapes.
  if (2 * k > INT32_MAX) fail("nf12: ", what, " K (", k, ") is too large for MLX's int32 dimensions");
}

array check_weight_matrix(const array& w, const char* op) {
  if (w.dtype() != mx::bfloat16 || w.ndim() != 2) fail("nf12.", op, ": weight must be a bfloat16 [N, K] array");
  check_shape(w.shape(0), w.shape(1), op);
  return contig(w);
}

// A dense exponent base: a Python int, or a uint8 array whose element 0 is the
// base (e.g. an nn.Module's nf12_exp_base). Arrays are graph inputs, so a
// compiled function stays correct when the weight it captured is replaced.
using ExpBaseArg = std::variant<int64_t, array>;

struct WeightView {
  array packed, escapes;
  int64_t n, k, num_escapes;
  uint32_t exp_base;              // when base_array is empty
  optional<array> base_array;     // uint8 [1]
};

// Adds the weight's optional inputs (escapes, exp_base array) to `inputs` and
// records where they are, for the launch inside eval_gpu.
struct WeightInputs {
  int esc = -1, base = -1;
  uint32_t value = 0;
  const array* escapes(const std::vector<array>& in) const { return esc >= 0 ? &in[esc] : nullptr; }
  ExpBase<MlxEncoder> exp_base(const std::vector<array>& in) const {
    return base >= 0 ? ExpBase<MlxEncoder>{&in[base], 0} : ExpBase<MlxEncoder>{nullptr, value};
  }
};

WeightInputs add_weight_inputs(const WeightView& w, std::vector<array>& inputs) {
  WeightInputs r;
  r.value = w.exp_base;
  if (w.num_escapes > 0) {
    r.esc = int(inputs.size());
    inputs.push_back(w.escapes);
  }
  if (w.base_array) {
    r.base = int(inputs.size());
    inputs.push_back(*w.base_array);
  }
  return r;
}

WeightView weight_view(const array& packed, const array& escapes, const ExpBaseArg& exp_base_arg) {
  if (packed.dtype() != mx::uint8 || packed.ndim() != 2) fail("nf12: packed must be a uint8 [N, K*3/2] array");
  if (packed.shape(1) % 3 != 0) fail("nf12: packed row length must be K*3/2, got ", packed.shape(1));
  if (escapes.dtype() != mx::int16 || escapes.ndim() != 2 || escapes.shape(1) != 8)
    fail("nf12: escapes must be an int16 [E, 8] array");
  uint32_t exp_base = 0;
  optional<array> base_array;
  if (const int64_t* v = std::get_if<int64_t>(&exp_base_arg)) {
    if (*v < 0 || *v > kMaxExpBase) fail("nf12: exp_base must be in [0, 120], got ", *v);
    exp_base = uint32_t(*v);
  } else {
    const array& b = std::get<array>(exp_base_arg);
    if (b.dtype() != mx::uint8 || b.size() != 1) fail("nf12: an exp_base array must be uint8 with one element");
    base_array = contig(mx::reshape(b, {1}, gpu_stream()));
  }
  const int64_t n = packed.shape(0), k = int64_t(packed.shape(1)) / 3 * 2;
  check_shape(n, k, "packed weight");
  return {contig(packed), contig(escapes), n, k, int64_t(escapes.shape(0)), exp_base, base_array};
}

// ---------------------------------------------------------------------------
// Codec.

// Pass 1 synchronizes to read the histograms.
std::pair<uint32_t, uint64_t> analyze(const array& w, int64_t n, int64_t k) {
  const uint64_t groups = uint64_t(n) * uint64_t(k) / 8;
  if (groups > 0xffffffffull) fail("nf12.encode: more than 2^32 - 1 groups");
  array hist = kernel1("encode_analyze", {int(kHistWords)}, mx::uint32, {w},
                       [n, k](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                         launch_encode_analyze(e, in[0], uint64_t(k), n, k, out[0]);
                       });
  mx::eval(hist);
  return best_exp_base(hist.data<uint32_t>(), groups);
}

// Encodes contiguous `w` [n, k]; returns (packed, escapes).
std::pair<array, array> encode_matrix(const array& w, int64_t n, int64_t k, uint32_t exp_base, uint64_t count) {
  array counts = kernel1("encode_count", {int(super_atoms(n, k))}, mx::int32, {w},
                         [n, k, exp_base](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                           launch_encode_count(e, in[0], uint64_t(k), n, k, exp_base, out[0]);
                         });
  const mx::Stream s = gpu_stream();
  array offsets = mx::cumsum(mx::astype(counts, mx::int64, s), /*reverse=*/false, /*inclusive=*/false, s);
  if (count > uint64_t(INT32_MAX)) fail("nf12.encode: ", count, " escaped groups exceed MLX's int32 dimensions");
  // The write pass places escapes at these offsets: check the total against
  // the analysis before sizing the escape buffer (encoding synchronizes anyway).
  array total = mx::add(mx::slice(offsets, {-1}, {int(offsets.size())}, s),
                        mx::astype(mx::slice(counts, {-1}, {int(counts.size())}, s), mx::int64, s), s);
  mx::eval(total);
  if (uint64_t(total.item<int64_t>()) != count)
    throw std::runtime_error("nf12.encode: escape count mismatch (" + std::to_string(total.item<int64_t>()) +
                             " vs " + std::to_string(count) + ")");
  auto outs = kernel("encode_write", {{int(n), int(k * 3 / 2)}, {int(count), 8}}, {mx::uint8, mx::int16}, {w, offsets},
                     [n, k, exp_base, count](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                       launch_encode_write(e, in[0], uint64_t(k), n, k, exp_base, in[1], out[0],
                                           count > 0 ? &out[1] : nullptr);
                     });
  return {outs[0], outs[1]};
}

std::tuple<array, array, int64_t> encode(const array& w_in) {
  const array w = check_weight_matrix(w_in, "encode");
  const int64_t n = w.shape(0), k = w.shape(1);
  const auto [exp_base, count] = analyze(w, n, k);
  auto [packed, escapes] = encode_matrix(w, n, k, exp_base, count);
  return {packed, escapes, int64_t(exp_base)};
}

array decode_view(const WeightView& w) {
  std::vector<array> inputs{w.packed};
  const WeightInputs wi = add_weight_inputs(w, inputs);
  const int64_t n = w.n, k = w.k;
  return kernel1("decode", {int(n), int(k)}, mx::bfloat16, std::move(inputs),
                 [=](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                   launch_decode(e, in[0], wi.escapes(in), nullptr, wi.exp_base(in), 1, n, k, out[0], uint64_t(k));
                 });
}

array decode(const array& packed, const array& escapes, const ExpBaseArg& exp_base) {
  return decode_view(weight_view(packed, escapes, exp_base));
}

array pack_bf16t(const array& w_in) {
  const array w = check_weight_matrix(w_in, "pack_bf16t");
  const int64_t n = w.shape(0), k = w.shape(1);
  return kernel1("pack_bf16t", {int(n), int(2 * k)}, mx::uint8, {w},
                 [n, k](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                   launch_pack_bf16t(e, in[0], uint64_t(k), n, k, out[0]);
                 });
}

array embedding(const array& ids_in, const array& packed, const array& escapes, const ExpBaseArg& exp_base) {
  const WeightView w = weight_view(packed, escapes, exp_base);
  if (ids_in.dtype() != mx::int32 && ids_in.dtype() != mx::int64)
    fail("nf12.embedding: ids must be an int32 or int64 array");
  mx::Shape shape = ids_in.shape();
  shape.push_back(int(w.k));
  const int64_t num_ids = int64_t(ids_in.size());
  if (num_ids == 0) return mx::zeros(shape, mx::bfloat16, gpu_stream());
  if (num_ids > int64_t(INT32_MAX)) fail("nf12.embedding: more than 2^31 - 1 ids");  // MLX dimensions are int32
  const array ids = contig(mx::reshape(ids_in, {int(num_ids)}, gpu_stream()));
  const bool ids64 = ids_in.dtype() == mx::int64;
  std::vector<array> inputs{w.packed, ids};
  const WeightInputs wi = add_weight_inputs(w, inputs);
  const int64_t n = w.n, k = w.k;
  array out = kernel1("embedding", {int(num_ids), int(k)}, mx::bfloat16, std::move(inputs),
                      [=](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& o) {
                        launch_embedding(e, in[0], wi.escapes(in), wi.exp_base(in), in[1], ids64, uint64_t(num_ids), n,
                                         k, o[0]);
                      });
  return mx::reshape(out, shape, gpu_stream());
}

// ---------------------------------------------------------------------------
// Linear (SPEC.md sections 1 and 6).

// `nf12_view` is null for bf16t weights.
array run_linear(const array& x, const array& packed, int64_t n, int64_t k, const optional<array>& bias_in,
                 const char* op, const WeightView* nf12_view) {
  if (x.dtype() != mx::bfloat16) fail("nf12.", op, ": x must be bfloat16");
  if (x.ndim() < 1 || x.shape(-1) != k) fail("nf12.", op, ": x must have shape [..., ", k, "], got ", shape_str(x));
  mx::Shape out_shape = x.shape();
  out_shape.back() = int(n);
  const int64_t m = int64_t(x.size()) / k;
  const mx::Stream s = gpu_stream();
  if (m == 0) return mx::zeros(out_shape, mx::bfloat16, s);
  if (m > INT32_MAX) fail("nf12.", op, ": too many rows");
  const array x2 = contig(mx::reshape(x, {int(m), int(k)}, s));
  optional<array> bias;
  if (bias_in.has_value()) {
    if (bias_in->dtype() != mx::bfloat16 || bias_in->ndim() != 1 || bias_in->shape(0) != n)
      fail("nf12.", op, ": bias must be a bfloat16 [", n, "] array");
    bias = contig(*bias_in);
  }
  const bool nf12 = nf12_view != nullptr;
  // A single 32-token tile can use the packed weights directly without
  // writing and rereading a decoded BF16 slice for MLX's matmul.
  const bool fused_prefill = m > 16 && m <= kGemmBM && k >= 2048 &&
                             ((n >= 512 && k <= 4096) || (n >= 4096 && k <= 16384));
  const Route route = linear_route(m, nf12, fused_prefill);
  array y = [&]() -> array {
    if (route == Route::kDecode) {
      // Decode-then-GEMM (SPEC.md 6.5): MLX's matmul on decoded row slices of at
      // most decode_slice_bytes() (as on MPS), so a large weight never exists
      // decoded in full. Escapes are indexed absolutely, so slices share them.
      auto gemm = [&](const WeightView& w, const optional<array>& b) {
        const array wt = mx::transpose(decode_view(w), s);
        return b ? mx::addmm(*b, x2, wt, 1.0f, 1.0f, s) : mx::matmul(x2, wt, s);
      };
      const int64_t rows = decode_slice_rows(n, k);
      if (rows >= n) return gemm(*nf12_view, bias);
      const int cols = int(nf12_view->packed.shape(1));
      std::vector<array> parts;
      for (int64_t n0 = 0; n0 < n; n0 += rows) {
        const int64_t r = std::min(rows, n - n0);
        WeightView slice = *nf12_view;
        slice.packed = mx::slice(nf12_view->packed, {int(n0), 0}, {int(n0 + r), cols}, s);
        slice.n = r;
        optional<array> b;
        if (bias) b = mx::slice(*bias, {int(n0)}, {int(n0 + r)}, s);
        parts.push_back(gemm(slice, b));
      }
      return mx::concatenate(parts, 1, s);
    }
    std::vector<array> inputs{x2, packed};
    const WeightInputs wi = nf12 ? add_weight_inputs(*nf12_view, inputs) : WeightInputs{};
    int bias_i = -1;
    if (bias) {
      bias_i = int(inputs.size());
      inputs.push_back(*bias);
    }
    return kernel1(nf12 ? "linear" : "linear_bf16t", {int(m), int(n)}, mx::bfloat16, std::move(inputs),
                   [=](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                     launch_linear(e, route, in[0], uint64_t(k), in[1], nf12, wi.escapes(in), wi.exp_base(in),
                                   bias_i >= 0 ? &in[bias_i] : nullptr, out[0], m, n, k);
                   });
  }();
  return mx::reshape(y, out_shape, s);
}

array linear(const array& x, const array& packed, const array& escapes, const ExpBaseArg& exp_base,
             const optional<array>& bias) {
  const WeightView w = weight_view(packed, escapes, exp_base);
  return run_linear(x, w.packed, w.n, w.k, bias, "linear", &w);
}

array linear_bf16t(const array& x, const array& w_bf16t, const optional<array>& bias) {
  if (w_bf16t.dtype() != mx::uint8 || w_bf16t.ndim() != 2 || w_bf16t.shape(1) % 2 != 0)
    fail("nf12.linear_bf16t: w_bf16t must be a uint8 [N, 2*K] array from pack_bf16t");
  const int64_t n = w_bf16t.shape(0), k = w_bf16t.shape(1) / 2;
  check_shape(n, k, "linear_bf16t");
  return run_linear(x, contig(w_bf16t), n, k, bias, "linear_bf16t", nullptr);
}

// ---------------------------------------------------------------------------
// Mixture of experts (SPEC.md section 9).

array index_vector(const array& t, mx::Dtype dtype, const char* op, const char* name, const char* type) {
  if (t.dtype() != dtype || t.ndim() != 1) fail("nf12.", op, ": ", name, " must be a 1-D ", type, " array");
  return contig(t);
}

struct MoeWeight {
  array packed;
  optional<array> escapes, escape_offsets, exp_base;  // NF12 only
  int64_t experts, n, k;
  bool nf12() const { return escapes.has_value(); }
};

MoeWeight moe_weight_view(const array& packed, const array& escapes, const array& escape_offsets,
                          const array& exp_base) {
  if (packed.dtype() != mx::uint8 || packed.ndim() != 3 || packed.shape(0) < 1)
    fail("nf12: MoE packed must be a uint8 [E, N, K*3/2] array with E >= 1");
  if (packed.shape(2) % 3 != 0) fail("nf12: packed row length must be K*3/2, got ", packed.shape(2));
  if (escapes.dtype() != mx::int16 || escapes.ndim() != 2 || escapes.shape(1) != 8)
    fail("nf12: escapes must be an int16 [S, 8] array");
  const int64_t e = packed.shape(0);
  if (escape_offsets.dtype() != mx::int64 || escape_offsets.ndim() != 1 || escape_offsets.shape(0) != e + 1)
    fail("nf12: escape_offsets must be an int64 [E + 1] array");
  if (exp_base.dtype() != mx::uint8 || exp_base.ndim() != 1 || exp_base.shape(0) != e)
    fail("nf12: exp_base must be a uint8 [E] array");
  const int64_t n = packed.shape(1), k = int64_t(packed.shape(2)) / 3 * 2;
  check_shape(n, k, "MoE packed weight");
  return {contig(packed), contig(escapes), contig(escape_offsets), contig(exp_base), e, n, k};
}

MoeWeight moe_bf16t_view(const array& packed) {
  if (packed.dtype() != mx::uint8 || packed.ndim() != 3 || packed.shape(0) < 1 || packed.shape(2) % 2 != 0)
    fail("nf12.moe_linear_bf16t: w_bf16t must be a uint8 [E, N, 2*K] array from pack_bf16t");
  const int64_t n = packed.shape(1), k = packed.shape(2) / 2;
  check_shape(n, k, "moe_linear_bf16t");
  return {contig(packed), std::nullopt, std::nullopt, std::nullopt, int64_t(packed.shape(0)), n, k};
}

std::tuple<array, array> moe_align(const array& topk_ids_in, int64_t num_experts) {
  if (topk_ids_in.dtype() != mx::int32 || topk_ids_in.ndim() != 2 || topk_ids_in.shape(1) < 1)
    fail("nf12.moe_align: topk_ids must be an int32 [T, topk] array with topk >= 1");
  if (num_experts < 1 || num_experts > INT32_MAX) fail("nf12.moe_align: num_experts must be in [1, 2^31 - 1]");
  const int64_t t = topk_ids_in.shape(0), topk = topk_ids_in.shape(1);
  if (t > INT32_MAX / topk) fail("nf12.moe_align: more than 2^31 - 1 assignments");
  const mx::Stream s = gpu_stream();
  const int a = int(t * topk);
  const array ids = contig(mx::reshape(topk_ids_in, {a}, s));
  if (num_experts > kMoeMaxExperts) {
    // Beyond the kernels' threadgroup tables: sort unique (expert, assignment)
    // keys (stable by construction), count experts with a scatter-add.
    const array flat = mx::astype(ids, mx::int64, s);
    const array valid = mx::logical_and(mx::greater_equal(flat, mx::array(int64_t(0)), s),
                                        mx::less(flat, mx::array(int64_t(num_experts)), s), s);
    const array keys = mx::where(valid, flat, mx::array(int64_t(num_experts)), s);
    const array order = mx::argsort(mx::add(mx::multiply(keys, mx::array(int64_t(a)), s),
                                            mx::arange(0, a, mx::int64, s), s), s);
    // GPU scatter has no int64; counts fit int32 (A < 2^31).
    const array counts = mx::scatter_add(mx::zeros({int(num_experts) + 1}, mx::int32, s), keys,
                                         mx::ones({a, 1}, mx::int32, s), 0, s);
    const array offsets =
        mx::cumsum(mx::astype(counts, mx::int64, s), /*reverse=*/false, /*inclusive=*/false, s);
    return {mx::astype(order, mx::int32, s), offsets};
  }
  const uint32_t a32 = uint32_t(a), e32 = uint32_t(num_experts);
  // Scratch is [chunks, E] (at least one word): each dimension fits MLX's int32
  // shapes even when the 64-bit word count (align_scratch_words) does not.
  const mx::Shape scratch = {int(std::max<uint64_t>(1, align_chunks(a32))), int(num_experts)};
  auto outs = kernel("moe_align", {{a}, {int(num_experts) + 1}, scratch},
                     {mx::int32, mx::int64, mx::uint32}, {ids},
                     [=](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                       launch_moe_align(e, in[0], a32, e32, out[2], out[0], out[1]);
                     });
  return {outs[0], outs[1]};
}

array run_moe_linear(const array& x_in, const MoeWeight& w, const array& expert_offsets,
                     const optional<array>& gather, int64_t gather_div, const optional<array>& scatter,
                     const optional<array>& row_scale, bool zero_fill, const char* op) {
  if (w.experts > kMoeMaxExperts) fail("nf12.", op, ": at most ", kMoeMaxExperts, " experts");
  if (gather_div < 1 || gather_div > int64_t(UINT32_MAX)) fail("nf12.", op, ": gather_div must be >= 1, got ", gather_div);
  if (x_in.dtype() != mx::bfloat16 || x_in.ndim() != 2 || x_in.shape(1) != w.k)
    fail("nf12.", op, ": x must be a bfloat16 [rows, ", w.k, "] array, got ", shape_str(x_in));
  const array offsets = index_vector(expert_offsets, mx::int64, op, "expert_offsets", "int64");
  if (offsets.shape(0) != w.experts + 1)
    fail("nf12.", op, ": expert_offsets must have E + 1 = ", w.experts + 1, " entries, got ", offsets.shape(0));
  optional<array> g, sc, r;
  if (gather) g = index_vector(*gather, mx::int32, op, "gather", "int32");
  if (scatter) sc = index_vector(*scatter, mx::int32, op, "scatter", "int32");
  if (g && sc && g->shape(0) != sc->shape(0)) fail("nf12.", op, ": gather and scatter must have the same length");
  const int64_t a = g ? g->shape(0) : sc ? sc->shape(0) : x_in.shape(0);
  if (row_scale) {
    r = index_vector(*row_scale, mx::float32, op, "row_scale", "float32");
    if (r->shape(0) != a) fail("nf12.", op, ": row_scale must have ", a, " entries, got ", r->shape(0));
  }
  const mx::Stream s = gpu_stream();
  if (a == 0) return mx::zeros({0, int(w.n)}, mx::bfloat16, s);
  if (x_in.shape(0) == 0) fail("nf12.", op, ": x has no rows");
  const array x = contig(x_in);
  const MoeRoute route = moe_route(a, w.experts);
  const uint32_t e32 = uint32_t(w.experts), a32 = uint32_t(a);
  const array schedule =
      // [1 + max_tiles, 4] int32 (the flat int4 layout launch_moe_schedule writes): a flat
      // dimension of 4 * (1 + max_tiles) would overflow MLX's int32 shapes for large A.
      kernel1("moe_schedule", {int(moe_schedule_ints(a, w.experts, route.bm) / 4), 4}, mx::int32, {offsets},
              [=](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                launch_moe_schedule(e, in[0], e32, a32, route.bm, out[0]);
              });
  // Inputs: x, packed, schedule, then optional escapes, escape_offsets, exp_base, gather, scatter, row_scale.
  std::vector<array> inputs{x, w.packed, schedule};
  auto add = [&](const optional<array>& v) -> int {
    if (!v) return -1;
    inputs.push_back(*v);
    return int(inputs.size()) - 1;
  };
  const bool nf12 = w.nf12();
  const int esc_i = nf12 && w.escapes->shape(0) > 0 ? add(w.escapes) : -1;
  const int off_i = nf12 ? add(w.escape_offsets) : -1, base_i = nf12 ? add(w.exp_base) : -1;
  const int g_i = add(g), s_i = add(sc), r_i = add(r);
  const int64_t experts = w.experts, n = w.n, k = w.k;
  const uint32_t div = uint32_t(gather_div);
  return kernel1(nf12 ? "moe_linear" : "moe_linear_bf16t", {int(a), int(n)}, mx::bfloat16, std::move(inputs),
                 [=](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                   auto at = [&](int i) { return i >= 0 ? &in[i] : nullptr; };
                   const MoeWeightArgs<MlxEncoder> wa{in[1], nf12, at(esc_i), at(off_i), at(base_i), experts, n, k};
                   launch_moe_linear(e, route, in[0], uint64_t(k), wa, at(g_i), div, at(s_i), at(r_i), in[2], out[0],
                                     a, zero_fill);
                 });
}

array moe_linear(const array& x, const array& packed, const array& escapes, const array& escape_offsets,
                 const array& exp_base, const array& expert_offsets, const optional<array>& gather,
                 int64_t gather_div, const optional<array>& scatter, const optional<array>& row_scale,
                 bool zero_fill) {
  const MoeWeight w = moe_weight_view(packed, escapes, escape_offsets, exp_base);
  return run_moe_linear(x, w, expert_offsets, gather, gather_div, scatter, row_scale, zero_fill, "moe_linear");
}

array moe_linear_bf16t(const array& x, const array& w_bf16t, const array& expert_offsets,
                       const optional<array>& gather, int64_t gather_div, const optional<array>& scatter,
                       const optional<array>& row_scale, bool zero_fill) {
  const MoeWeight w = moe_bf16t_view(w_bf16t);
  return run_moe_linear(x, w, expert_offsets, gather, gather_div, scatter, row_scale, zero_fill, "moe_linear_bf16t");
}

array moe_decode(const array& packed, const array& escapes, const array& escape_offsets, const array& exp_base) {
  const MoeWeight w = moe_weight_view(packed, escapes, escape_offsets, exp_base);
  const bool has_esc = w.escapes->shape(0) > 0;
  std::vector<array> inputs{w.packed, *w.escape_offsets, *w.exp_base};
  if (has_esc) inputs.push_back(*w.escapes);
  const int64_t experts = w.experts, n = w.n, k = w.k;
  return kernel1("moe_decode", {int(experts), int(n), int(k)}, mx::bfloat16, std::move(inputs),
                 [=](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                   launch_decode(e, in[0], has_esc ? &in[3] : nullptr, &in[1], ExpBase<MlxEncoder>{&in[2], 0}, experts, n, k, out[0],
                                 uint64_t(k));
                 });
}

// Per-expert analysis synchronizes, as on CUDA and MPS.
std::tuple<array, array, array, array> encode_moe(const array& w_in) {
  if (w_in.dtype() != mx::bfloat16 || w_in.ndim() != 3 || w_in.shape(0) < 1)
    fail("nf12.encode_moe: weight must be a bfloat16 [E, N, K] array with E >= 1");
  const int64_t e = w_in.shape(0), n = w_in.shape(1), k = w_in.shape(2);
  check_shape(n, k, "encode_moe");
  const mx::Stream s = gpu_stream();
  const array w = contig(w_in);
  std::vector<array> packed, escapes;
  std::vector<int64_t> offsets(e + 1, 0);
  std::vector<uint8_t> bases(e);
  for (int64_t i = 0; i < e; ++i) {
    const array we = contig(mx::slice(w, {int(i), 0, 0}, {int(i) + 1, int(n), int(k)}, s));
    const array w2 = mx::reshape(we, {int(n), int(k)}, s);
    const auto [base, count] = analyze(w2, n, k);
    auto [p, esc] = encode_matrix(w2, n, k, base, count);
    packed.push_back(p);
    escapes.push_back(esc);
    bases[i] = uint8_t(base);
    offsets[i + 1] = offsets[i] + int64_t(count);
  }
  return {mx::stack(packed, 0, s), mx::concatenate(escapes, 0, s),
          array(offsets.data(), {int(e) + 1}, mx::int64), array(bases.data(), {int(e)}, mx::uint8)};
}

array moe_combine(const array& y_in, int64_t topk) {
  if (y_in.dtype() != mx::bfloat16 || y_in.ndim() != 2) fail("nf12.moe_combine: y must be a bfloat16 [T * topk, N] array");
  if (topk < 1 || y_in.shape(0) % topk != 0)
    fail("nf12.moe_combine: rows (", y_in.shape(0), ") must be a multiple of topk (", topk, ")");
  const int64_t n = y_in.shape(1), t = y_in.shape(0) / topk;
  if (n % 8 != 0) fail("nf12.moe_combine: N (", n, ") must be a multiple of 8");
  if (t == 0 || n == 0) return mx::zeros({int(t), int(n)}, mx::bfloat16, gpu_stream());
  const uint32_t k32 = uint32_t(topk);
  return kernel1("moe_combine", {int(t), int(n)}, mx::bfloat16, {contig(y_in)},
                 [=](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                   launch_moe_combine(e, in[0], uint64_t(n), k32, t, n, out[0]);
                 });
}

array moe_silu_mul(const array& h_in) {
  if (h_in.dtype() != mx::bfloat16 || h_in.ndim() != 2) fail("nf12.moe_silu_mul: h must be a bfloat16 [rows, 2 * I] array");
  const int64_t inter = h_in.shape(1) / 2;
  if (h_in.shape(1) != 2 * inter || inter <= 0 || inter % 8 != 0)
    fail("nf12.moe_silu_mul: h width (", h_in.shape(1), ") must be 2 * I with I a positive multiple of 8");
  const int64_t rows = h_in.shape(0);
  if (rows == 0) return mx::zeros({0, int(inter)}, mx::bfloat16, gpu_stream());
  return kernel1("moe_silu_mul", {int(rows), int(inter)}, mx::bfloat16, {contig(h_in)},
                 [=](MlxEncoder& e, const std::vector<array>& in, std::vector<array>& out) {
                   launch_moe_silu_mul(e, in[0], uint64_t(2 * inter), rows, inter, out[0]);
                 });
}

MoeWeight fused_weight_view(const array& packed, const optional<array>& escapes, const optional<array>& escape_offsets,
                            const optional<array>& exp_base) {
  if (escapes.has_value() != escape_offsets.has_value() || escapes.has_value() != exp_base.has_value())
    fail("nf12.fused_moe: escapes, escape_offsets and exp_base must all be present or all be None");
  if (!escapes) return moe_bf16t_view(packed);
  return moe_weight_view(packed, *escapes, *escape_offsets, *exp_base);
}

// align -> w13 (gathered into sorted order) -> SiLU-mul -> w2 (scattered to
// assignment order, scaled by router weights) -> combine.
array fused_moe(const array& x, const array& packed13, const optional<array>& escapes13,
                const optional<array>& escape_offsets13, const optional<array>& exp_base13, const array& packed2,
                const optional<array>& escapes2, const optional<array>& escape_offsets2,
                const optional<array>& exp_base2, const array& topk_weights, const array& topk_ids) {
  // y is always zeroed, as in torch.ops.nf12._fused_moe: dropped assignments
  // (id -1, with or without an expert_map) are never written.
  const MoeWeight w13 = fused_weight_view(packed13, escapes13, escape_offsets13, exp_base13);
  const MoeWeight w2 = fused_weight_view(packed2, escapes2, escape_offsets2, exp_base2);
  if (w13.experts != w2.experts || w13.n != 2 * w2.k) fail("nf12.fused_moe: weights need the same E and N13 == 2 * K2");
  if (x.dtype() != mx::bfloat16 || x.ndim() != 2 || x.shape(1) != w13.k)
    fail("nf12.fused_moe: x must be a bfloat16 [T, ", w13.k, "] array");
  if (topk_ids.dtype() != mx::int32 || topk_ids.ndim() != 2 || topk_ids.shape(1) < 1 || topk_ids.shape(0) != x.shape(0))
    fail("nf12.fused_moe: topk_ids must be an int32 [T, topk] array with topk >= 1");
  if (topk_weights.dtype() != mx::float32 || topk_weights.shape() != topk_ids.shape())
    fail("nf12.fused_moe: topk_weights must be a float32 [T, topk] array");
  const int64_t t = x.shape(0), topk = topk_ids.shape(1);
  if (t > INT32_MAX / topk) fail("nf12.fused_moe: more than 2^31 - 1 assignments");
  const mx::Stream s = gpu_stream();
  if (t == 0) return mx::zeros({0, int(w2.n)}, mx::bfloat16, s);
  auto [sorted, offsets] = moe_align(topk_ids, w13.experts);
  array h = run_moe_linear(x, w13, offsets, sorted, topk, std::nullopt, std::nullopt, false, "fused_moe");
  h = moe_silu_mul(h);
  const array row_scale = mx::reshape(topk_weights, {int(t * topk)}, s);
  const array y = run_moe_linear(h, w2, offsets, std::nullopt, 1, sorted, row_scale, true, "fused_moe");
  return moe_combine(y, topk);
}

}  // namespace
}  // namespace nf12_mlx

NB_MODULE(_mlx_ext, m) {
  using namespace nf12_mlx;
  m.doc() = "NF12 kernels for MLX arrays (see nf12.mlx)";
  m.attr("mlx_version") = NF12_MLX_VERSION;  // the MLX this module was built against
  // nf12.mlx calls this once: it raises TypeError if this module's nanobind ABI
  // differs from MLX's, so mx.array arguments would never convert.
  m.def("_accepts_array", [](const array&) { return true; });
  m.def("encode", &encode, "w"_a);
  m.def("decode", &decode, "packed"_a, "escapes"_a, "exp_base"_a);
  m.def("pack_bf16t", &pack_bf16t, "w"_a);
  m.def("embedding", &embedding, "ids"_a, "packed"_a, "escapes"_a, "exp_base"_a);
  m.def("linear", &linear, "x"_a, "packed"_a, "escapes"_a, "exp_base"_a, "bias"_a = nb::none());
  m.def("linear_bf16t", &linear_bf16t, "x"_a, "w_bf16t"_a, "bias"_a = nb::none());
  m.def("encode_moe", &encode_moe, "w"_a);
  m.def("moe_align", &moe_align, "topk_ids"_a, "num_experts"_a);
  m.def("moe_decode", &moe_decode, "packed"_a, "escapes"_a, "escape_offsets"_a, "exp_base"_a);
  m.def("moe_linear", &moe_linear, "x"_a, "packed"_a, "escapes"_a, "escape_offsets"_a, "exp_base"_a,
        "expert_offsets"_a, "gather"_a = nb::none(), "gather_div"_a = 1, "scatter"_a = nb::none(),
        "row_scale"_a = nb::none(), "zero_fill"_a = false);
  m.def("moe_linear_bf16t", &moe_linear_bf16t, "x"_a, "w_bf16t"_a, "expert_offsets"_a, "gather"_a = nb::none(),
        "gather_div"_a = 1, "scatter"_a = nb::none(), "row_scale"_a = nb::none(), "zero_fill"_a = false);
  m.def("moe_combine", &moe_combine, "y"_a, "topk"_a);
  m.def("moe_silu_mul", &moe_silu_mul, "h"_a);
  m.def("fused_moe", &fused_moe, "x"_a, "packed13"_a, "escapes13"_a.none(), "escape_offsets13"_a.none(),
        "exp_base13"_a.none(), "packed2"_a, "escapes2"_a.none(), "escape_offsets2"_a.none(), "exp_base2"_a.none(),
        "topk_weights"_a, "topk_ids"_a);
}
