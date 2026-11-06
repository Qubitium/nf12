// SPDX-License-Identifier: Apache-2.0
// C ABI over the codec entry points so Python tests can drive them through
// ctypes with torch-owned device pointers, independent of the library build.
// All calls use the legacy default stream and return a cudaError_t.
#include "internal.h"

extern "C" {

size_t codec_encode_workspace_bytes(int64_t n, int64_t k) { return nf12::detail::encode_workspace_bytes(n, k); }

int codec_encode_analyze(const void* w, int64_t n, int64_t k, int64_t ldw, void* ws, size_t ws_bytes,
                         uint8_t* exp_base, uint32_t* num_escapes) {
  return nf12::detail::encode_analyze(static_cast<const __nv_bfloat16*>(w), n, k, ldw, ws, ws_bytes, exp_base,
                                      num_escapes, nullptr);
}

int codec_encode(const void* w, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base, void* packed, void* escapes,
                 uint32_t capacity, void* ws, size_t ws_bytes) {
  return nf12::detail::encode(static_cast<const __nv_bfloat16*>(w), n, k, ldw, exp_base,
                              static_cast<uint8_t*>(packed), static_cast<uint4*>(escapes), capacity, ws, ws_bytes,
                              nullptr);
}

int codec_decode(const void* packed, const void* escapes, uint32_t exp_base, int64_t n, int64_t k, void* out,
                 int64_t ldo) {
  return nf12::detail::decode(static_cast<const uint8_t*>(packed), static_cast<const uint4*>(escapes), exp_base, n,
                              k, static_cast<__nv_bfloat16*>(out), ldo, nullptr);
}

int codec_pack_bf16t(const void* w, int64_t n, int64_t k, int64_t ldw, void* out) {
  return nf12::detail::pack_bf16t(static_cast<const __nv_bfloat16*>(w), n, k, ldw, static_cast<uint8_t*>(out),
                                  nullptr);
}

}  // extern "C"
