// SPDX-License-Identifier: Apache-2.0
// torch.ops.nf12 on Apple GPUs (MPS). The schemas match csrc/torch/ops.cpp, so
// the Python API is unchanged. Kernels are compiled by Metal on first use
// and are encoded on PyTorch's current MPS stream through its shared compute
// encoder, so they order with surrounding torch work without extra syncs.
#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/library.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "launch.h"
#include "shader_library.h"

namespace nf12_mps {
namespace {

using at::Tensor;
using namespace nf12::metal;

// ---------------------------------------------------------------------------
// Metal library and pipelines.

id<MTLDevice> mtl_device() { return at::mps::getCurrentMPSStream()->device(); }

id<MTLComputePipelineState> pipeline(const char* name) {
  static std::mutex mu;
  static id<MTLLibrary> library = nil;
  static id<MTLLibrary> tensor_library = nil;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  std::lock_guard<std::mutex> lock(mu);
  if (auto it = cache.find(name); it != cache.end()) return it->second;
  const bool tensor = is_tensor_kernel(name);
  id<MTLLibrary>& lib = tensor ? tensor_library : library;
  if (lib == nil) lib = static_cast<id<MTLLibrary>>(new_shader_library(mtl_device(), tensor));
  id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
  TORCH_CHECK(fn != nil, "nf12: kernel ", name, " not found in the Metal library");
  NSError* error = nil;
  id<MTLComputePipelineState> pso = [mtl_device() newComputePipelineStateWithFunction:fn error:&error];
  [fn release];
  TORCH_CHECK(pso != nil, "nf12: failed to build pipeline ", name, ": ",
              error ? [[error localizedDescription] UTF8String] : "unknown error");
  cache.emplace(name, pso);
  return pso;
}

id<MTLBuffer> mtl_buffer(const Tensor& t) { return __builtin_bit_cast(id<MTLBuffer>, t.storage().data()); }

// Encoder adapter for the shared launches in launch.h.
struct TorchEncoder {
  using tensor = Tensor;
  using out_t = const Tensor&;
  id<MTLComputeCommandEncoder> enc;
  ThreadgroupLimit limit{};
  void pipeline(const std::string& name) {
    id<MTLComputePipelineState> pso = nf12_mps::pipeline(name.c_str());
    [enc setComputePipelineState:pso];
    limit.set(name, pso.maxTotalThreadsPerThreadgroup);
  }
  void in(const Tensor& t, int index) {
    [enc setBuffer:mtl_buffer(t) offset:t.storage_offset() * t.element_size() atIndex:index];
  }
  void out(const Tensor& t, int index) { in(t, index); }
  template <typename T>
  void bytes(const T& v, int index) {
    [enc setBytes:&v length:sizeof(T) atIndex:index];
  }
  void dispatch(uint64_t gx, uint64_t gy, uint64_t gz, uint64_t tx, uint64_t ty, uint64_t tz) {
    if (gx == 0 || gy == 0 || gz == 0) return;
    limit.check(tx * ty * tz);
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
  }
  // PyTorch's MPS compute encoder dispatches serially.
  void barrier() {}
};

// Runs `body(TorchEncoder&)` on the current MPS stream (PyTorch's serial queue
// and coalesced compute encoder). Exceptions propagate to the caller.
template <typename Body>
void run(const Body& body) {
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  const Body* b = &body;
  __block std::exception_ptr error;
  dispatch_sync(stream->queue(), ^{
    @autoreleasepool {
      try {
        TorchEncoder e{stream->commandEncoder()};
        (*b)(e);
      } catch (...) {
        error = std::current_exception();
      }
    }
  });
  if (error) std::rethrow_exception(error);
}

const Tensor* opt(const std::optional<Tensor>& t) { return t.has_value() ? &*t : nullptr; }
const Tensor* nonempty(const Tensor& t) { return t.numel() > 0 ? &t : nullptr; }
// torch.ops.nf12 takes a dense exp_base as an int, bound as inline bytes.
using Base = ExpBase<TorchEncoder>;

// ---------------------------------------------------------------------------
// Validation (mirrors csrc/torch/ops.cpp with MPS in place of CUDA).

void check_mps(const Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), "nf12: ", name, " must be an MPS tensor, got ", t.device());
}

void check_shape(int64_t n, int64_t k, const char* what) {
  TORCH_CHECK(n > 0 && n % 16 == 0, "nf12: ", what, " needs N (out_features) to be a positive multiple of 16, got ",
              n);
  TORCH_CHECK(k > 0 && k % 64 == 0, "nf12: ", what, " needs K (in_features) to be a positive multiple of 64, got ", k);
}

bool aligned(const Tensor& t, int64_t bytes) {
  return (t.storage_offset() * t.element_size()) % bytes == 0;
}

// Rows with unit column stride, an even leading dimension and 4-byte alignment
// (kernels read bf16 pairs); copies only when needed.
Tensor pair_rows(const Tensor& t) {
  auto ok = [](const Tensor& v) {
    return v.stride(1) == 1 && (v.size(0) <= 1 || (v.stride(0) % 2 == 0 && v.stride(0) >= v.size(1))) &&
           aligned(v, 4);
  };
  if (ok(t)) return t;
  Tensor c = t.contiguous();
  return ok(c) ? c : c.clone();
}

int64_t row_stride(const Tensor& t) { return t.size(0) <= 1 ? t.size(1) : t.stride(0); }

Tensor check_weight_matrix(const Tensor& w, const char* op) {
  check_mps(w, "weight");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.dim() == 2, "nf12.", op, ": weight must be a bfloat16 [N, K] tensor");
  check_shape(w.size(0), w.size(1), op);
  return pair_rows(w);
}

struct WeightView {
  Tensor packed, escapes;
  int64_t n, k, num_escapes;
  uint32_t exp_base;
};

WeightView weight_view(const Tensor& packed, const Tensor& escapes, int64_t exp_base) {
  check_mps(packed, "packed");
  check_mps(escapes, "escapes");
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.dim() == 2, "nf12: packed must be a uint8 [N, K*3/2] tensor");
  TORCH_CHECK(packed.size(1) % 3 == 0, "nf12: packed row length must be K*3/2, got ", packed.size(1));
  TORCH_CHECK(packed.is_contiguous() && aligned(packed, 16), "nf12: packed must be contiguous and 16-byte aligned");
  TORCH_CHECK(escapes.scalar_type() == at::kShort && escapes.dim() == 2 && escapes.size(1) == 8,
              "nf12: escapes must be an int16 [E, 8] tensor");
  TORCH_CHECK(escapes.is_contiguous() && aligned(escapes, 16), "nf12: escapes must be contiguous and 16-byte aligned");
  TORCH_CHECK(exp_base >= 0 && exp_base <= kMaxExpBase, "nf12: exp_base must be in [0, 120], got ", exp_base);
  TORCH_CHECK(escapes.size(0) <= INT64_C(0xffffffff), "nf12: too many escapes");
  const int64_t n = packed.size(0), k = packed.size(1) / 3 * 2;
  check_shape(n, k, "packed weight");
  return {packed, escapes, n, k, escapes.size(0), static_cast<uint32_t>(exp_base)};
}

// ---------------------------------------------------------------------------
// Codec.

// Chooses the exponent base with the fewest escaped groups. Synchronizes to
// read the histograms.
std::pair<uint32_t, uint64_t> analyze(const Tensor& w, int64_t n, int64_t k) {
  const uint64_t groups = uint64_t(n) * uint64_t(k) / 8;
  TORCH_CHECK(groups <= 0xffffffffull, "nf12.encode: more than 2^32 - 1 groups");
  Tensor hist = at::empty({kHistWords}, w.options().dtype(at::kInt));
  run([&](TorchEncoder& e) { launch_encode_analyze(e, w, uint64_t(row_stride(w)), n, k, hist); });
  const Tensor h = hist.cpu();
  return best_exp_base(static_cast<const uint32_t*>(h.data_ptr()), groups);
}

// Encodes `w` (validated, pair rows) into `packed_out` ([N, K*3/2] uint8 view)
// and returns its escapes.
Tensor encode_into(const Tensor& w, int64_t n, int64_t k, uint32_t exp_base, uint64_t expected, Tensor packed_out) {
  const uint64_t ld = row_stride(w);
  Tensor counts = at::empty({int64_t(super_atoms(n, k))}, w.options().dtype(at::kInt));
  run([&](TorchEncoder& e) { launch_encode_count(e, w, ld, n, k, exp_base, counts); });
  const Tensor inclusive = at::cumsum(counts, 0, at::kLong);
  const Tensor offsets = inclusive - counts;
  const int64_t total = inclusive[-1].item<int64_t>();
  TORCH_CHECK(uint64_t(total) == expected, "nf12.encode: escape count mismatch (", total, " vs ", expected, ")");
  Tensor escapes = at::empty({total, 8}, w.options().dtype(at::kShort));
  run([&](TorchEncoder& e) {
    launch_encode_write(e, w, ld, n, k, exp_base, offsets, packed_out, total > 0 ? &escapes : nullptr);
  });
  return escapes;
}

std::tuple<Tensor, Tensor, int64_t> encode(const Tensor& w_in) {
  const Tensor w = check_weight_matrix(w_in, "encode");
  const int64_t n = w.size(0), k = w.size(1);
  const auto [exp_base, num_escapes] = analyze(w, n, k);
  Tensor packed = at::empty({n, k * 3 / 2}, w.options().dtype(at::kByte));
  Tensor escapes = encode_into(w, n, k, exp_base, num_escapes, packed);
  return {packed, escapes, int64_t(exp_base)};
}

Tensor decode(const Tensor& packed, const Tensor& escapes, int64_t exp_base) {
  const WeightView w = weight_view(packed, escapes, exp_base);
  Tensor out = at::empty({w.n, w.k}, packed.options().dtype(at::kBFloat16));
  run([&](TorchEncoder& e) {
    launch_decode(e, w.packed, nonempty(w.escapes), nullptr, Base{nullptr, w.exp_base}, 1, w.n, w.k, out, w.k);
  });
  return out;
}

Tensor pack_bf16t(const Tensor& w_in) {
  const Tensor w = check_weight_matrix(w_in, "pack_bf16t");
  const int64_t n = w.size(0), k = w.size(1);
  Tensor out = at::empty({n, 2 * k}, w.options().dtype(at::kByte));
  run([&](TorchEncoder& e) { launch_pack_bf16t(e, w, uint64_t(row_stride(w)), n, k, out); });
  return out;
}

Tensor embedding(const Tensor& ids_in, const Tensor& packed, const Tensor& escapes, int64_t exp_base) {
  const WeightView w = weight_view(packed, escapes, exp_base);
  check_mps(ids_in, "ids");
  const auto id_type = ids_in.scalar_type();
  TORCH_CHECK(id_type == at::kInt || id_type == at::kLong, "nf12.embedding: ids must be an int32 or int64 tensor");
  std::vector<int64_t> sizes(ids_in.sizes().begin(), ids_in.sizes().end());
  sizes.push_back(w.k);
  Tensor out = at::empty(sizes, packed.options().dtype(at::kBFloat16));
  const int64_t num_ids = ids_in.numel();
  if (num_ids == 0) return out;
  TORCH_CHECK(num_ids <= UINT32_MAX, "nf12.embedding: too many ids");
  const Tensor ids = ids_in.contiguous();
  run([&](TorchEncoder& e) {
    launch_embedding(e, w.packed, nonempty(w.escapes), Base{nullptr, w.exp_base}, ids, id_type == at::kLong, uint64_t(num_ids), w.n,
                     w.k, out);
  });
  return out;
}

// ---------------------------------------------------------------------------
// Linear (SPEC.md sections 1 and 6).

// Decode-then-GEMM (SPEC.md 6.5): row slices of BF16 are decoded and multiplied
// by torch's MPS matmul, adding bias before the single rounding.
void decode_gemm(const WeightView& w, const Tensor& x2, const std::optional<Tensor>& bias, Tensor& y2) {
  const int64_t rows = decode_slice_rows(w.n, w.k);
  Tensor buf = at::empty({rows, w.k}, x2.options());
  for (int64_t n0 = 0; n0 < w.n; n0 += rows) {
    const int64_t r = std::min(rows, w.n - n0);
    Tensor wd = r == rows ? buf : buf.narrow(0, 0, r);
    const Tensor slice = w.packed.narrow(0, n0, r);
    run([&](TorchEncoder& e) {
      launch_decode(e, slice, nonempty(w.escapes), nullptr, Base{nullptr, w.exp_base}, 1, r, w.k, wd, w.k);
    });
    // This matmul is part of the op, not the caller's code: autocast must not
    // retype it (an fp16 autocast region would overflow the BF16 result).
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::autocast_dispatch_keyset);
    if (r == w.n) {  // one slice: write the (contiguous) output directly
      if (bias.has_value()) {
        at::addmm_out(y2, *bias, x2, wd.t());
      } else {
        at::mm_out(y2, x2, wd.t());
      }
    } else {
      Tensor ys = bias.has_value() ? at::addmm(bias->narrow(0, n0, r), x2, wd.t()) : at::mm(x2, wd.t());
      y2.narrow(1, n0, r).copy_(ys);
    }
  }
}

// One fused GEMM tile avoids writing and rereading the decoded weight
// when output width offers enough threadgroups to fill the GPU.
Route dense_linear_route(int64_t m, int64_t n, int64_t k, bool nf12) {
  static const bool tensor = supports_tensor_linear(mtl_device());
  const bool fused_prefill = m > 16 && m <= kGemmBM && n >= 4096 && k >= 2048 && k <= 16384;
  // For smaller projections, MPS gains most when TensorOps replaces the
  // existing fused prefill tile. Its decode-GEMM is competitive at larger M.
  const bool prefer_tensor = tensor && n >= 4096 && (n >= 8192 || k > 4096 || fused_prefill);
  return linear_route(m, n, k, nf12, prefer_tensor, fused_prefill);
}

// `nf12_view` is null for bf16t weights.
Tensor run_linear(const Tensor& x, const Tensor& packed, int64_t n, int64_t k, const std::optional<Tensor>& bias_in,
                  const char* op, const WeightView* nf12_view) {
  check_mps(x, "x");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "nf12.", op, ": x must be bfloat16");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) == k, "nf12.", op, ": x must have shape [..., ", k, "]");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n;
  const int64_t m = x.numel() / k;
  Tensor y = at::empty(out_sizes, x.options());
  if (m == 0) return y;
  TORCH_CHECK(m <= INT32_MAX, "nf12.", op, ": too many rows");
  Tensor x2 = pair_rows(x.reshape({m, k}));
  std::optional<Tensor> bias;
  if (bias_in.has_value()) {
    check_mps(*bias_in, "bias");
    TORCH_CHECK(bias_in->scalar_type() == at::kBFloat16 && bias_in->dim() == 1 && bias_in->size(0) == n, "nf12.", op,
                ": bias must be a bfloat16 [", n, "] tensor");
    bias = bias_in->contiguous();
  }
  Tensor y2 = y.view({m, n});
  const bool nf12 = nf12_view != nullptr;
  const Route route = dense_linear_route(m, n, k, nf12);
  if (route == Route::kTensor && row_stride(x2) > INT32_MAX) x2 = x2.contiguous();
  if (route == Route::kDecode) {
    decode_gemm(*nf12_view, x2, bias, y2);
  } else {
    run([&](TorchEncoder& e) {
      launch_linear(e, route, x2, uint64_t(row_stride(x2)), packed, nf12, nf12 ? nonempty(nf12_view->escapes) : nullptr,
                    Base{nullptr, nf12 ? nf12_view->exp_base : 0u}, opt(bias), y2, m, n, k);
    });
  }
  return y;
}

// Slice height of the decode-then-GEMM route for nf12.linear at (m, n, k), or 0
// for the fused kernels (the MPS counterpart of nf12_linear_decode_gemm_rows).
int64_t decode_gemm_rows(int64_t m, int64_t n, int64_t k) {
  check_shape(n, k, "decode_gemm_rows");
  if (m <= 0 || dense_linear_route(m, n, k, true) != Route::kDecode) return 0;
  return decode_slice_rows(n, k);
}

Tensor linear(const Tensor& x, const Tensor& packed, const Tensor& escapes, int64_t exp_base,
              const std::optional<Tensor>& bias, bool /* static_weights */) {
  const WeightView w = weight_view(packed, escapes, exp_base);
  return run_linear(x, w.packed, w.n, w.k, bias, "linear", &w);
}

Tensor linear_bf16t(const Tensor& x, const Tensor& w_bf16t, const std::optional<Tensor>& bias) {
  check_mps(w_bf16t, "w_bf16t");
  TORCH_CHECK(w_bf16t.scalar_type() == at::kByte && w_bf16t.dim() == 2 && w_bf16t.size(1) % 2 == 0,
              "nf12.linear_bf16t: w_bf16t must be a uint8 [N, 2*K] tensor from pack_bf16t");
  TORCH_CHECK(w_bf16t.is_contiguous() && aligned(w_bf16t, 16),
              "nf12.linear_bf16t: w_bf16t must be contiguous and 16-byte aligned");
  const int64_t n = w_bf16t.size(0), k = w_bf16t.size(1) / 2;
  check_shape(n, k, "linear_bf16t");
  return run_linear(x, w_bf16t, n, k, bias, "linear_bf16t", nullptr);
}

// ---------------------------------------------------------------------------
// Mixture of experts (SPEC.md section 9).

// A contiguous 1-D tensor of `dtype` on MPS (copied if strided).
Tensor index_vector(const Tensor& t, at::ScalarType dtype, const char* op, const char* name, const char* type) {
  check_mps(t, name);
  TORCH_CHECK(t.scalar_type() == dtype && t.dim() == 1, "nf12.", op, ": ", name, " must be a 1-D ", type, " tensor");
  return t.contiguous();
}

struct MoeWeight {
  Tensor packed;
  std::optional<Tensor> escapes, escape_offsets, exp_base;  // NF12 only
  int64_t experts, n, k;
  bool nf12() const { return escapes.has_value(); }
  MoeWeightArgs<TorchEncoder> args() const {
    return {packed, nf12(), nf12() ? nonempty(*escapes) : nullptr, opt(escape_offsets), opt(exp_base), experts, n, k};
  }
};

MoeWeight moe_weight_view(const Tensor& packed, const Tensor& escapes, const Tensor& escape_offsets,
                          const Tensor& exp_base) {
  check_mps(packed, "packed");
  check_mps(escapes, "escapes");
  check_mps(escape_offsets, "escape_offsets");
  check_mps(exp_base, "exp_base");
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.dim() == 3 && packed.size(0) > 0,
              "nf12: MoE packed must be a uint8 [E, N, K*3/2] tensor with E >= 1");
  TORCH_CHECK(packed.size(2) % 3 == 0, "nf12: packed row length must be K*3/2, got ", packed.size(2));
  TORCH_CHECK(packed.is_contiguous() && aligned(packed, 16), "nf12: packed must be contiguous and 16-byte aligned");
  TORCH_CHECK(escapes.scalar_type() == at::kShort && escapes.dim() == 2 && escapes.size(1) == 8,
              "nf12: escapes must be an int16 [S, 8] tensor");
  TORCH_CHECK(escapes.is_contiguous() && aligned(escapes, 16), "nf12: escapes must be contiguous and 16-byte aligned");
  const int64_t e = packed.size(0);
  TORCH_CHECK(escape_offsets.scalar_type() == at::kLong && escape_offsets.dim() == 1 &&
                  escape_offsets.size(0) == e + 1 && escape_offsets.is_contiguous(),
              "nf12: escape_offsets must be a contiguous int64 [E + 1] tensor");
  TORCH_CHECK(exp_base.scalar_type() == at::kByte && exp_base.dim() == 1 && exp_base.size(0) == e &&
                  exp_base.is_contiguous(),
              "nf12: exp_base must be a contiguous uint8 [E] tensor");
  const int64_t n = packed.size(1), k = packed.size(2) / 3 * 2;
  check_shape(n, k, "MoE packed weight");
  return {packed, escapes, escape_offsets, exp_base, e, n, k};
}

MoeWeight moe_bf16t_view(const Tensor& packed) {
  check_mps(packed, "w_bf16t");
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.dim() == 3 && packed.size(0) > 0 && packed.size(2) % 2 == 0,
              "nf12.moe_linear_bf16t: w_bf16t must be a uint8 [E, N, 2*K] tensor from pack_bf16t");
  TORCH_CHECK(packed.is_contiguous() && aligned(packed, 16),
              "nf12.moe_linear_bf16t: w_bf16t must be contiguous and 16-byte aligned");
  const int64_t n = packed.size(1), k = packed.size(2) / 2;
  check_shape(n, k, "moe_linear_bf16t");
  return {packed, std::nullopt, std::nullopt, std::nullopt, packed.size(0), n, k};
}

std::tuple<Tensor, Tensor> moe_align(const Tensor& topk_ids_in, int64_t num_experts) {
  check_mps(topk_ids_in, "topk_ids");
  TORCH_CHECK(topk_ids_in.scalar_type() == at::kInt && topk_ids_in.dim() == 2 && topk_ids_in.size(1) > 0,
              "nf12.moe_align: topk_ids must be an int32 [T, topk] tensor with topk >= 1");
  TORCH_CHECK(num_experts >= 1 && num_experts <= INT32_MAX, "nf12.moe_align: num_experts must be in [1, 2^31 - 1]");
  const Tensor ids = topk_ids_in.contiguous();
  const int64_t t = ids.size(0), topk = ids.size(1);
  TORCH_CHECK(t <= INT32_MAX / topk, "nf12.moe_align: more than 2^31 - 1 assignments");
  if (num_experts > kMoeMaxExperts) {
    // Beyond the kernels' threadgroup tables: a stable device sort, no host sync.
    const Tensor flat = ids.view({-1});
    const Tensor keys = at::where((flat >= 0) & (flat < num_experts), flat.to(at::kLong),
                                  at::scalar_tensor(num_experts, flat.options().dtype(at::kLong)));
    const auto [sorted_keys, order] = at::sort(keys, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    const Tensor bounds = at::arange(num_experts + 1, flat.options().dtype(at::kLong));
    return {order.to(at::kInt), at::searchsorted(sorted_keys, bounds)};
  }
  const uint32_t a = uint32_t(t * topk), e = uint32_t(num_experts);
  Tensor sorted = at::empty({int64_t(a)}, ids.options());
  Tensor offsets = at::empty({num_experts + 1}, ids.options().dtype(at::kLong));
  Tensor bases = at::empty({static_cast<int64_t>(align_scratch_words(a, e))}, ids.options());
  run([&](TorchEncoder& g) { launch_moe_align(g, ids, a, e, bases, sorted, offsets); });
  return {sorted, offsets};
}

Tensor run_moe_linear(const Tensor& x, const MoeWeight& w, const Tensor& expert_offsets,
                      const std::optional<Tensor>& gather, int64_t gather_div, const std::optional<Tensor>& scatter,
                      const std::optional<Tensor>& row_scale, bool zero_fill, const char* op) {
  check_mps(x, "x");
  TORCH_CHECK(w.experts <= kMoeMaxExperts, "nf12.", op, ": at most ", kMoeMaxExperts, " experts on MPS");
  TORCH_CHECK(gather_div >= 1 && gather_div <= UINT32_MAX, "nf12.", op, ": gather_div must be >= 1, got ", gather_div);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == w.k, "nf12.", op,
              ": x must be a bfloat16 [rows, ", w.k, "] tensor");
  const Tensor offsets = index_vector(expert_offsets, at::kLong, op, "expert_offsets", "int64");
  TORCH_CHECK(offsets.size(0) == w.experts + 1, "nf12.", op, ": expert_offsets must have E + 1 = ", w.experts + 1,
              " entries, got ", offsets.size(0));
  std::optional<Tensor> g, s, r;
  if (gather.has_value()) g = index_vector(*gather, at::kInt, op, "gather", "int32");
  if (scatter.has_value()) s = index_vector(*scatter, at::kInt, op, "scatter", "int32");
  TORCH_CHECK(!g || !s || g->size(0) == s->size(0), "nf12.", op, ": gather and scatter must have the same length");
  const int64_t a = g ? g->size(0) : s ? s->size(0) : x.size(0);
  TORCH_CHECK(a <= INT32_MAX, "nf12.", op, ": more than 2^31 - 1 assignments");
  if (row_scale.has_value()) {
    r = index_vector(*row_scale, at::kFloat, op, "row_scale", "float32");
    TORCH_CHECK(r->size(0) == a, "nf12.", op, ": row_scale must have ", a, " entries, got ", r->size(0));
  }
  Tensor y = zero_fill ? at::zeros({a, w.n}, x.options()) : at::empty({a, w.n}, x.options());
  if (a == 0) return y;
  TORCH_CHECK(x.size(0) > 0, "nf12.", op, ": x has no rows");
  const Tensor x2 = pair_rows(x);
  const MoeRoute route = moe_route(a, w.experts);
  Tensor schedule = at::empty({moe_schedule_ints(a, w.experts, route.bm)}, x.options().dtype(at::kInt));
  run([&](TorchEncoder& e) {
    launch_moe_schedule(e, offsets, uint32_t(w.experts), uint32_t(a), route.bm, schedule);
    e.barrier();
    // at::zeros already cleared y.
    launch_moe_linear(e, route, x2, uint64_t(row_stride(x2)), w.args(), opt(g), uint32_t(gather_div), opt(s), opt(r),
                      schedule, y, a, /*zero_fill=*/false);
  });
  return y;
}

Tensor moe_linear(const Tensor& x, const Tensor& packed, const Tensor& escapes, const Tensor& escape_offsets,
                  const Tensor& exp_base, const Tensor& expert_offsets, const std::optional<Tensor>& gather,
                  int64_t gather_div, const std::optional<Tensor>& scatter, const std::optional<Tensor>& row_scale,
                  bool zero_fill) {
  const MoeWeight w = moe_weight_view(packed, escapes, escape_offsets, exp_base);
  return run_moe_linear(x, w, expert_offsets, gather, gather_div, scatter, row_scale, zero_fill, "moe_linear");
}

Tensor moe_linear_bf16t(const Tensor& x, const Tensor& w_bf16t, const Tensor& expert_offsets,
                        const std::optional<Tensor>& gather, int64_t gather_div, const std::optional<Tensor>& scatter,
                        const std::optional<Tensor>& row_scale, bool zero_fill) {
  const MoeWeight w = moe_bf16t_view(w_bf16t);
  return run_moe_linear(x, w, expert_offsets, gather, gather_div, scatter, row_scale, zero_fill, "moe_linear_bf16t");
}

Tensor moe_decode(const Tensor& packed, const Tensor& escapes, const Tensor& escape_offsets, const Tensor& exp_base) {
  const MoeWeight w = moe_weight_view(packed, escapes, escape_offsets, exp_base);
  Tensor out = at::empty({w.experts, w.n, w.k}, packed.options().dtype(at::kBFloat16));
  run([&](TorchEncoder& e) {
    launch_decode(e, w.packed, nonempty(*w.escapes), &*w.escape_offsets, Base{&*w.exp_base, 0}, w.experts, w.n, w.k, out,
                  w.k);
  });
  return out;
}

// Per-expert analysis synchronizes, as on CUDA.
std::tuple<Tensor, Tensor, Tensor, Tensor> encode_moe(const Tensor& w_in) {
  check_mps(w_in, "weight");
  TORCH_CHECK(w_in.scalar_type() == at::kBFloat16 && w_in.dim() == 3 && w_in.size(0) > 0,
              "nf12.encode_moe: weight must be a bfloat16 [E, N, K] tensor with E >= 1");
  const int64_t e = w_in.size(0), n = w_in.size(1), k = w_in.size(2);
  check_shape(n, k, "encode_moe");
  Tensor packed = at::empty({e, n, k * 3 / 2}, w_in.options().dtype(at::kByte));
  std::vector<Tensor> escapes;
  std::vector<int64_t> offsets(e + 1, 0);
  std::vector<uint8_t> bases(e);
  for (int64_t i = 0; i < e; ++i) {
    const Tensor we = pair_rows(w_in.select(0, i));
    const auto [base, count] = analyze(we, n, k);
    escapes.push_back(encode_into(we, n, k, base, count, packed.select(0, i)));
    bases[i] = uint8_t(base);
    offsets[i + 1] = offsets[i] + int64_t(count);
  }
  const auto cpu = at::TensorOptions().device(at::kCPU);
  Tensor escape_offsets = at::from_blob(offsets.data(), {e + 1}, cpu.dtype(at::kLong)).to(w_in.device());
  Tensor exp_base = at::from_blob(bases.data(), {e}, cpu.dtype(at::kByte)).to(w_in.device());
  return {packed, at::cat(escapes, 0), escape_offsets, exp_base};
}

Tensor moe_combine(const Tensor& y_in, int64_t topk) {
  check_mps(y_in, "y");
  TORCH_CHECK(y_in.scalar_type() == at::kBFloat16 && y_in.dim() == 2,
              "nf12.moe_combine: y must be a bfloat16 [T * topk, N] tensor");
  TORCH_CHECK(topk >= 1 && y_in.size(0) % topk == 0, "nf12.moe_combine: rows (", y_in.size(0),
              ") must be a multiple of topk (", topk, ")");
  const int64_t n = y_in.size(1), t = y_in.size(0) / topk;
  TORCH_CHECK(n % 8 == 0, "nf12.moe_combine: N (", n, ") must be a multiple of 8");
  Tensor out = at::empty({t, n}, y_in.options());
  if (t == 0 || n == 0) return out;
  TORCH_CHECK(t <= UINT32_MAX && n <= UINT32_MAX, "nf12.moe_combine: too many rows or columns");
  const Tensor y = y_in.stride(1) == 1 ? y_in : y_in.contiguous();
  const uint64_t ldy = uint64_t(y.size(0) == 1 ? n : y.stride(0));
  run([&](TorchEncoder& e) { launch_moe_combine(e, y, ldy, uint32_t(topk), t, n, out); });
  return out;
}

Tensor moe_silu_mul(const Tensor& h_in) {
  check_mps(h_in, "h");
  TORCH_CHECK(h_in.scalar_type() == at::kBFloat16 && h_in.dim() == 2,
              "nf12.moe_silu_mul: h must be a bfloat16 [rows, 2 * I] tensor");
  const int64_t inter = h_in.size(1) / 2;
  TORCH_CHECK(h_in.size(1) == 2 * inter && inter > 0 && inter % 8 == 0, "nf12.moe_silu_mul: h width (",
              h_in.size(1), ") must be 2 * I with I a positive multiple of 8");
  const int64_t rows = h_in.size(0);
  Tensor out = at::empty({rows, inter}, h_in.options());
  if (rows == 0) return out;
  TORCH_CHECK(rows <= UINT32_MAX, "nf12.moe_silu_mul: too many rows");
  const Tensor h = h_in.stride(1) == 1 ? h_in : h_in.contiguous();
  const uint64_t ldh = uint64_t(h.size(0) == 1 ? 2 * inter : h.stride(0));
  run([&](TorchEncoder& e) { launch_moe_silu_mul(e, h, ldh, rows, inter, out); });
  return out;
}

MoeWeight fused_weight_view(const Tensor& packed, const std::optional<Tensor>& escapes,
                            const std::optional<Tensor>& escape_offsets, const std::optional<Tensor>& exp_base) {
  TORCH_CHECK(escapes.has_value() == escape_offsets.has_value() && escapes.has_value() == exp_base.has_value(),
              "nf12._fused_moe: escapes, escape_offsets and exp_base must all be present or all be None");
  if (!escapes) return moe_bf16t_view(packed);
  return moe_weight_view(packed, *escapes, *escape_offsets, *exp_base);
}

// align -> w13 (gathered into sorted order) -> SiLU-mul -> w2 (scattered to
// assignment order, scaled by router weights) -> combine.
Tensor fused_moe(const Tensor& x, const Tensor& packed13, const std::optional<Tensor>& escapes13,
                 const std::optional<Tensor>& escape_offsets13, const std::optional<Tensor>& exp_base13,
                 const Tensor& packed2, const std::optional<Tensor>& escapes2,
                 const std::optional<Tensor>& escape_offsets2, const std::optional<Tensor>& exp_base2,
                 const Tensor& topk_weights, const Tensor& topk_ids, bool /* zero_fill */) {
  // As on CUDA, zero_fill stays in the schema for existing callers but y is
  // always zeroed: the combine reads every top-k slot, and an assignment
  // dropped by moe_align (id -1, with or without an expert_map) is never
  // written, so its router weight (even NaN/Inf) never reaches the output.
  const MoeWeight w13 = fused_weight_view(packed13, escapes13, escape_offsets13, exp_base13);
  const MoeWeight w2 = fused_weight_view(packed2, escapes2, escape_offsets2, exp_base2);
  TORCH_CHECK(w13.experts == w2.experts && w13.n == 2 * w2.k,
              "nf12._fused_moe: weights need the same E and N13 == 2 * K2");
  check_mps(x, "x");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == w13.k,
              "nf12._fused_moe: x must be a bfloat16 [T, ", w13.k, "] tensor");
  check_mps(topk_ids, "topk_ids");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt && topk_ids.dim() == 2 && topk_ids.size(1) > 0 &&
                  topk_ids.size(0) == x.size(0),
              "nf12._fused_moe: topk_ids must be an int32 [T, topk] tensor with topk >= 1");
  check_mps(topk_weights, "topk_weights");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat && topk_weights.dim() == 2 &&
                  topk_weights.size(0) == topk_ids.size(0) && topk_weights.size(1) == topk_ids.size(1),
              "nf12._fused_moe: topk_weights must be a float32 [T, topk] tensor");
  const int64_t t = x.size(0), topk = topk_ids.size(1);
  TORCH_CHECK(t <= INT32_MAX / topk, "nf12._fused_moe: more than 2^31 - 1 assignments");
  if (t == 0) return at::empty({0, w2.n}, x.options());
  auto [sorted, offsets] = moe_align(topk_ids, w13.experts);
  Tensor h = run_moe_linear(x, w13, offsets, sorted, topk, std::nullopt, std::nullopt, false, "_fused_moe");
  h = moe_silu_mul(h);
  const Tensor row_scale = topk_weights.contiguous().view({t * topk});
  const Tensor y = run_moe_linear(h, w2, offsets, std::nullopt, 1, sorted, row_scale, true, "_fused_moe");
  return moe_combine(y, topk);
}

}  // namespace
}  // namespace nf12_mps

TORCH_LIBRARY(nf12, m) {
  m.def("encode(Tensor w) -> (Tensor packed, Tensor escapes, int exp_base)");
  m.def("decode(Tensor packed, Tensor escapes, int exp_base) -> Tensor");
  m.def("linear(Tensor x, Tensor packed, Tensor escapes, int exp_base, Tensor? bias=None, *, bool static_weights=False) -> Tensor");
  m.def("pack_bf16t(Tensor w) -> Tensor");
  m.def("linear_bf16t(Tensor x, Tensor w_bf16t, Tensor? bias=None) -> Tensor");
  m.def("embedding(Tensor ids, Tensor packed, Tensor escapes, int exp_base) -> Tensor");
  m.def("encode_moe(Tensor w) -> (Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base)");
  m.def("moe_align(Tensor topk_ids, int num_experts) -> (Tensor sorted, Tensor expert_offsets)");
  m.def("moe_decode(Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base) -> Tensor");
  m.def(
      "moe_linear(Tensor x, Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base, "
      "Tensor expert_offsets, Tensor? gather=None, int gather_div=1, Tensor? scatter=None, Tensor? row_scale=None, "
      "bool zero_fill=False) -> Tensor");
  m.def(
      "moe_linear_bf16t(Tensor x, Tensor w_bf16t, Tensor expert_offsets, Tensor? gather=None, int gather_div=1, "
      "Tensor? scatter=None, Tensor? row_scale=None, bool zero_fill=False) -> Tensor");
  m.def("moe_combine(Tensor y, int topk) -> Tensor");
  m.def("moe_silu_mul(Tensor h) -> Tensor");
  m.def(
      "_fused_moe(Tensor x, Tensor packed13, Tensor? escapes13, Tensor? escape_offsets13, Tensor? exp_base13, "
      "Tensor packed2, Tensor? escapes2, Tensor? escape_offsets2, Tensor? exp_base2, "
      "Tensor topk_weights, Tensor topk_ids, bool zero_fill=False) -> Tensor");
  m.def("_mps_decode_gemm_rows(int m, int n, int k) -> int");
}

TORCH_LIBRARY_IMPL(nf12, MPS, m) {
  m.impl("encode", &nf12_mps::encode);
  m.impl("decode", &nf12_mps::decode);
  m.impl("pack_bf16t", &nf12_mps::pack_bf16t);
  m.impl("embedding", &nf12_mps::embedding);
  m.impl("linear", &nf12_mps::linear);
  m.impl("linear_bf16t", &nf12_mps::linear_bf16t);
  m.impl("encode_moe", &nf12_mps::encode_moe);
  m.impl("moe_align", &nf12_mps::moe_align);
  m.impl("moe_decode", &nf12_mps::moe_decode);
  m.impl("moe_linear", &nf12_mps::moe_linear);
  m.impl("moe_linear_bf16t", &nf12_mps::moe_linear_bf16t);
  m.impl("moe_combine", &nf12_mps::moe_combine);
  m.impl("moe_silu_mul", &nf12_mps::moe_silu_mul);
  m.impl("_fused_moe", &nf12_mps::fused_moe);
}

TORCH_LIBRARY_IMPL(nf12, CompositeExplicitAutograd, m) { m.impl("_mps_decode_gemm_rows", &nf12_mps::decode_gemm_rows); }
