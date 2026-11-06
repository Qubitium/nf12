/* SPDX-License-Identifier: Apache-2.0
 * Plain C99 consumer of include/nf12/nf12.h, built by tests/api/test_c_consumer.py
 * with -Werror=implicit-function-declaration so a function missing from the
 * header fails to compile. Calls every public function once on a 16 x 64
 * matrix and checks the results. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include <nf12/nf12.h>

#define N 16
#define K 64
#define M 8

static int failures = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      failures++;                                                       \
    }                                                                   \
  } while (0)

#define CHECK_NF12(call)                                                                       \
  do {                                                                                         \
    nf12_status_t s_ = (call);                                                                 \
    if (s_ != NF12_OK) {                                                                       \
      fprintf(stderr, "%s:%d: %s -> %s (%s)\n", __FILE__, __LINE__, #call, nf12_status_string(s_), \
              cudaGetErrorString(nf12_last_cuda_error()));                                    \
      exit(1);                                                                                 \
    }                                                                                          \
  } while (0)

#define CHECK_CUDA(call)                                                                              \
  do {                                                                                                \
    cudaError_t e_ = (call);                                                                          \
    if (e_ != cudaSuccess) {                                                                          \
      fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_));        \
      exit(1);                                                                                        \
    }                                                                                                 \
  } while (0)

static float bf16_to_float(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

/* Small deterministic BF16 values in roughly [-0.5, 0.5). */
static uint16_t lcg_bf16(uint32_t* state) {
  *state = *state * 1664525u + 1013904223u;
  float f = ((float)(*state >> 8) / (float)(1u << 24)) - 0.5f;
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return (uint16_t)(u >> 16);
}

static void* dev_alloc(size_t bytes) {
  void* p = NULL;
  CHECK_CUDA(cudaMalloc(&p, bytes > 0 ? bytes : 16));
  return p;
}

int main(void) {
  uint16_t w[N * K], x[M * K], bias[N], y[M * N], y_twin[M * N], w_back[N * K];
  uint32_t state = 12345u;
  int device = 0, i, j, c;
  cudaStream_t stream;

  for (i = 0; i < N * K; i++) w[i] = lcg_bf16(&state);
  /* +0.0 everywhere in atom 0 (rows 0..15, cols 0..15): every group there hits
   * the M == 0 sentinel, so the escape path is exercised (32 escapes). */
  for (i = 0; i < N; i++)
    for (j = 0; j < 16; j++) w[i * K + j] = 0;
  for (i = 0; i < M * K; i++) x[i] = lcg_bf16(&state);
  for (i = 0; i < N; i++) bias[i] = lcg_bf16(&state);

  CHECK(NF12_VERSION_MAJOR == 1);
  CHECK(strcmp(nf12_status_string(NF12_OK), "NF12_OK") == 0);
  CHECK(nf12_status_string(NF12_ERROR_INVALID_ARGUMENT) != NULL);
  CHECK(nf12_packed_bytes(N, K) == 1536);
  CHECK(nf12_bf16t_bytes(N, K) == 2048);
  CHECK(nf12_packed_bytes(N, K + 1) == 0);

  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaStreamCreate(&stream));

  void* d_w = dev_alloc(sizeof w);
  void* d_x = dev_alloc(sizeof x);
  void* d_bias = dev_alloc(sizeof bias);
  void* d_y = dev_alloc(sizeof y);
  void* d_w_back = dev_alloc(sizeof w_back);
  CHECK_CUDA(cudaMemcpy(d_w, w, sizeof w, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x, x, sizeof x, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_bias, bias, sizeof bias, cudaMemcpyHostToDevice));

  size_t enc_ws_bytes = nf12_encode_workspace_bytes(N, K);
  CHECK(enc_ws_bytes > 0);
  void* d_enc_ws = dev_alloc(enc_ws_bytes);
  uint8_t exp_base = 255;
  uint32_t num_escapes = 0;
  CHECK_NF12(nf12_encode_analyze(d_w, N, K, K, d_enc_ws, enc_ws_bytes, &exp_base, &num_escapes, stream));
  CHECK(exp_base <= 120);
  CHECK(num_escapes >= 32);
  void* d_packed = dev_alloc(nf12_packed_bytes(N, K));
  void* d_escapes = dev_alloc((size_t)num_escapes * 16);
  CHECK_NF12(nf12_encode(d_w, N, K, K, exp_base, d_packed, d_escapes, num_escapes, d_enc_ws, enc_ws_bytes, stream));

  nf12_weight_t weight;
  weight.packed = d_packed;
  weight.escapes = d_escapes;
  weight.num_escapes = num_escapes;
  weight.n = N;
  weight.k = K;
  weight.exp_base = exp_base;
  CHECK_NF12(nf12_decode(&weight, d_w_back, K, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(w_back, d_w_back, sizeof w_back, cudaMemcpyDeviceToHost));
  CHECK(memcmp(w, w_back, sizeof w) == 0);

  size_t lin_ws_bytes = nf12_linear_workspace_bytes(M, N, K, device);
  void* d_lin_ws = lin_ws_bytes > 0 ? dev_alloc(lin_ws_bytes) : NULL;
  CHECK_NF12(nf12_linear(d_x, M, K, &weight, d_bias, d_y, N, d_lin_ws, lin_ws_bytes, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(y, d_y, sizeof y, cudaMemcpyDeviceToHost));

  void* d_twin = dev_alloc(nf12_bf16t_bytes(N, K));
  CHECK_NF12(nf12_pack_bf16t(d_w, N, K, K, d_twin, stream));
  CHECK_CUDA(cudaMemsetAsync(d_y, 0xff, sizeof y, stream));
  CHECK_NF12(nf12_linear_bf16t(d_x, M, K, d_twin, N, K, d_bias, d_y, N, d_lin_ws, lin_ws_bytes, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(y_twin, d_y, sizeof y_twin, cudaMemcpyDeviceToHost));
  CHECK(memcmp(y, y_twin, sizeof y) == 0); /* SPEC.md 6.2: NF12 == bf16t bitwise */
  {
    /* Decode-then-GEMM routing query: a slice height (multiple of 16, at most N) or 0. */
    const int64_t rows = nf12_linear_decode_gemm_rows(1 << 14, N, K, device);
    CHECK(rows == 0 || (rows % 16 == 0 && rows <= N));
    CHECK(nf12_linear_decode_gemm_rows(1 << 14, N + 1, K, device) == 0);
  }

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      float ref = bf16_to_float(bias[j]);
      for (c = 0; c < K; c++) ref += bf16_to_float(x[i * K + c]) * bf16_to_float(w[j * K + c]);
      float got = bf16_to_float(y[i * N + j]);
      if (fabsf(got - ref) > 1e-2f + 1e-2f * fabsf(ref)) {
        fprintf(stderr, "y[%d][%d] = %g, reference %g\n", i, j, got, ref);
        failures++;
      }
    }
  }

  /* Validation path: a misaligned x is rejected without a launch. */
  CHECK(nf12_linear((const char*)d_x + 2, M, K, &weight, NULL, d_y, N, d_lin_ws, lin_ws_bytes, stream) ==
        NF12_ERROR_INVALID_ARGUMENT);
  CHECK(nf12_last_cuda_error() == cudaSuccess);

  /* Out-of-range embedding ids give +0.0 rows. Output padding past K is untouched. */
  {
    enum { NI = 6, LDO = K + 8 };
    const int32_t ids32[NI] = {0, 15, -1, N, 3, 3};
    const int64_t ids64[NI] = {15, 0, (int64_t)1 << 32, -5, 7, 8};
    static uint16_t out[NI * LDO];
    void* d_ids32 = dev_alloc(sizeof ids32);
    void* d_ids64 = dev_alloc(sizeof ids64);
    void* d_out = dev_alloc(sizeof out);
    int pass;
    CHECK_CUDA(cudaMemcpy(d_ids32, ids32, sizeof ids32, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ids64, ids64, sizeof ids64, cudaMemcpyHostToDevice));
    for (pass = 0; pass < 2; pass++) {
      int bad = 0;
      CHECK_CUDA(cudaMemsetAsync(d_out, 0xff, sizeof out, stream));
      if (pass == 0) {
        CHECK_NF12(nf12_embedding(&weight, d_ids32, NI, d_out, LDO, stream));
      } else {
        CHECK_NF12(nf12_embedding_i64(&weight, d_ids64, NI, d_out, LDO, stream));
      }
      CHECK_CUDA(cudaStreamSynchronize(stream));
      CHECK_CUDA(cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost));
      for (i = 0; i < NI; i++) {
        const int64_t id = pass == 0 ? ids32[i] : ids64[i];
        for (c = 0; c < LDO; c++) {
          const uint16_t want = c >= K ? 0xffff : (id >= 0 && id < N) ? w[id * K + c] : 0;
          bad += out[i * LDO + c] != want;
        }
      }
      CHECK(bad == 0);
    }
    CHECK(nf12_embedding(&weight, d_ids32, NI, d_out, K + 4, stream) == NF12_ERROR_INVALID_ARGUMENT); /* ldo % 8 */
    CHECK(nf12_embedding_i64(&weight, NULL, 0, NULL, K, stream) == NF12_OK);                          /* no ids */
    CHECK(nf12_last_cuda_error() == cudaSuccess);
    cudaFree(d_ids32);
    cudaFree(d_ids64);
    cudaFree(d_out);
  }

  /* Expert 0 = w, expert 1 = w with rows reversed. */
  {
    enum { E = 2, T = 3, TOPK = 2, A = T * TOPK };
    static uint16_t w1[N * K], y_moe[A * N], y_moe_twin[A * N], out[T * N];
    const int32_t ids[A] = {0, 1, 1, 0, 1, -1}; /* -1: dropped (expert on another rank) */
    const float scale[A] = {0.25f, 0.75f, 0.5f, 0.5f, 1.0f, 3.0f};
    int32_t sorted[A];
    int64_t offsets[E + 1], esc_off[E + 1];
    uint8_t bases[E];
    uint32_t counts[E];

    for (i = 0; i < N; i++) memcpy(&w1[i * K], &w[(N - 1 - i) * K], K * sizeof(uint16_t));
    void* d_w1 = dev_alloc(sizeof w1);
    CHECK_CUDA(cudaMemcpy(d_w1, w1, sizeof w1, cudaMemcpyHostToDevice));
    const void* d_experts[E] = {d_w, d_w1};

    /* Each expert is a standalone NF12 matrix (SPEC.md 9.1), stacked expert-major. */
    uint8_t* d_moe_packed = dev_alloc(E * nf12_packed_bytes(N, K));
    for (c = 0; c < E; c++) {
      CHECK_NF12(nf12_encode_analyze(d_experts[c], N, K, K, d_enc_ws, enc_ws_bytes, &bases[c], &counts[c], stream));
    }
    esc_off[0] = 0;
    for (c = 0; c < E; c++) esc_off[c + 1] = esc_off[c] + counts[c];
    uint8_t* d_moe_escapes = dev_alloc((size_t)esc_off[E] * 16);
    for (c = 0; c < E; c++) {
      CHECK_NF12(nf12_encode(d_experts[c], N, K, K, bases[c], d_moe_packed + c * nf12_packed_bytes(N, K),
                             d_moe_escapes + esc_off[c] * 16, counts[c], d_enc_ws, enc_ws_bytes, stream));
    }
    int64_t* d_esc_off = dev_alloc(sizeof esc_off);
    uint8_t* d_bases = dev_alloc(sizeof bases);
    CHECK_CUDA(cudaMemcpy(d_esc_off, esc_off, sizeof esc_off, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bases, bases, sizeof bases, cudaMemcpyHostToDevice));
    nf12_moe_weight_t moe;
    moe.packed = d_moe_packed;
    moe.escapes = d_moe_escapes;
    moe.escape_offsets = d_esc_off;
    moe.exp_base = d_bases;
    moe.num_experts = E;
    moe.n = N;
    moe.k = K;

    /* Route: stable sort of assignment ids by expert, -1 dropped. */
    int32_t* d_ids = dev_alloc(sizeof ids);
    int32_t* d_sorted = dev_alloc(sizeof sorted);
    int64_t* d_offsets = dev_alloc(sizeof offsets);
    float* d_scale = dev_alloc(sizeof scale);
    CHECK_CUDA(cudaMemcpy(d_ids, ids, sizeof ids, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale, scale, sizeof scale, cudaMemcpyHostToDevice));
    size_t align_ws_bytes = nf12_moe_align_workspace_bytes(T, TOPK, E);
    void* d_align_ws = align_ws_bytes > 0 ? dev_alloc(align_ws_bytes) : NULL;
    CHECK_NF12(nf12_moe_align(d_ids, T, TOPK, E, d_sorted, d_offsets, d_align_ws, align_ws_bytes, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(sorted, d_sorted, sizeof sorted, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(offsets, d_offsets, sizeof offsets, cudaMemcpyDeviceToHost));
    CHECK(offsets[0] == 0 && offsets[1] == 2 && offsets[2] == 5);
    CHECK(sorted[0] == 0 && sorted[1] == 3 && sorted[2] == 1 && sorted[3] == 2 && sorted[4] == 4);

    /* y[a] = scale[a] * x[a / TOPK] . W_e^T in assignment order: gather tokens, scatter to ids.
     * Rows of dropped assignments are not written, so y starts zeroed. */
    size_t moe_ws_bytes = nf12_moe_linear_workspace_bytes(A, E, N, K, device);
    void* d_moe_ws = moe_ws_bytes > 0 ? dev_alloc(moe_ws_bytes) : NULL;
    void* d_y_moe = dev_alloc(sizeof y_moe);
    CHECK_CUDA(cudaMemsetAsync(d_y_moe, 0, sizeof y_moe, stream));
    CHECK_NF12(nf12_moe_linear(d_x, K, d_sorted, TOPK, d_sorted, d_offsets, A, &moe, d_scale, d_y_moe, N, d_moe_ws,
                               moe_ws_bytes, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(y_moe, d_y_moe, sizeof y_moe, cudaMemcpyDeviceToHost));

    /* bf16t twin: N % 16 == 0, so packing the stacked [E*N, K] matrix gives the [E, N, 2K] layout. */
    void* d_stacked = dev_alloc(E * sizeof w);
    CHECK_CUDA(cudaMemcpy(d_stacked, w, sizeof w, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((char*)d_stacked + sizeof w, w1, sizeof w1, cudaMemcpyHostToDevice));
    void* d_moe_twin = dev_alloc(E * nf12_bf16t_bytes(N, K));
    CHECK_NF12(nf12_pack_bf16t(d_stacked, E * N, K, K, d_moe_twin, stream));
    CHECK_CUDA(cudaMemsetAsync(d_y_moe, 0, sizeof y_moe, stream));
    CHECK_NF12(nf12_moe_linear_bf16t(d_x, K, d_sorted, TOPK, d_sorted, d_offsets, A, d_moe_twin, E, N, K, d_scale,
                                     d_y_moe, N, d_moe_ws, moe_ws_bytes, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(y_moe_twin, d_y_moe, sizeof y_moe_twin, cudaMemcpyDeviceToHost));
    CHECK(memcmp(y_moe, y_moe_twin, sizeof y_moe) == 0); /* NF12 == bf16t bitwise */

    /* Decoded experts (rows 0..N) match the inputs; group ends as moe_linear reads the offsets. */
    {
      static uint16_t decoded[E * N * K];
      int32_t ends[E];
      uint16_t* d_decoded = dev_alloc(sizeof decoded);
      int32_t* d_ends = dev_alloc(sizeof ends);
      CHECK_NF12(nf12_moe_decode(&moe, d_offsets, A, 0, N, d_decoded, K, stream));
      CHECK_NF12(nf12_moe_grouped_operands(NULL, K, NULL, 1, d_offsets, A, E, K, NULL, K, d_ends, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
      CHECK_CUDA(cudaMemcpy(decoded, d_decoded, sizeof decoded, cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(ends, d_ends, sizeof ends, cudaMemcpyDeviceToHost));
      CHECK(memcmp(decoded, w, sizeof w) == 0 && memcmp(decoded + N * K, w1, sizeof w1) == 0);
      CHECK(ends[0] == 2 && ends[1] == 5);
      CHECK(nf12_moe_linear_decode_gemm_rows(A, E, N, K, device) % 16 == 0);
      cudaFree(d_decoded);
      cudaFree(d_ends);
    }

    /* out[t] = sum_j y[t * TOPK + j]. */
    void* d_out = dev_alloc(sizeof out);
    CHECK_NF12(nf12_moe_combine(d_y_moe, N, T, TOPK, N, d_out, N, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost));
    for (i = 0; i < T; i++) {
      for (j = 0; j < N; j++) {
        float ref = 0.0f;
        int a;
        for (a = i * TOPK; a < (i + 1) * TOPK; a++) {
          const uint16_t* we = ids[a] == 0 ? w : ids[a] == 1 ? w1 : NULL;
          float dot = 0.0f;
          if (we == NULL) continue;
          for (c = 0; c < K; c++) dot += bf16_to_float(x[i * K + c]) * bf16_to_float(we[j * K + c]);
          CHECK(fabsf(bf16_to_float(y_moe[a * N + j]) - scale[a] * dot) <= 1e-2f + 1e-2f * fabsf(scale[a] * dot));
          ref += scale[a] * dot;
        }
        if (fabsf(bf16_to_float(out[i * N + j]) - ref) > 2e-2f + 2e-2f * fabsf(ref)) {
          fprintf(stderr, "moe out[%d][%d] = %g, reference %g\n", i, j, bf16_to_float(out[i * N + j]), ref);
          failures++;
        }
      }
    }
    /* The dropped assignment's row stays zero. */
    for (j = 0; j < N; j++) CHECK(y_moe[5 * N + j] == 0);
    CHECK(nf12_moe_linear(d_x, K, d_sorted, 0, d_sorted, d_offsets, A, &moe, d_scale, d_y_moe, N, d_moe_ws,
                          moe_ws_bytes, stream) == NF12_ERROR_INVALID_ARGUMENT); /* gather_div < 1 */
    CHECK(nf12_last_cuda_error() == cudaSuccess);

    cudaFree(d_w1);
    cudaFree(d_moe_packed);
    cudaFree(d_moe_escapes);
    cudaFree(d_esc_off);
    cudaFree(d_bases);
    cudaFree(d_ids);
    cudaFree(d_sorted);
    cudaFree(d_offsets);
    cudaFree(d_scale);
    cudaFree(d_align_ws);
    cudaFree(d_moe_ws);
    cudaFree(d_y_moe);
    cudaFree(d_stacked);
    cudaFree(d_moe_twin);
    cudaFree(d_out);
  }

  cudaFree(d_w);
  cudaFree(d_x);
  cudaFree(d_bias);
  cudaFree(d_y);
  cudaFree(d_w_back);
  cudaFree(d_enc_ws);
  cudaFree(d_packed);
  cudaFree(d_escapes);
  cudaFree(d_lin_ws);
  cudaFree(d_twin);
  cudaStreamDestroy(stream);

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("c_consumer: OK (exp_base=%u, escapes=%u, linear workspace=%zu)\n", exp_base, num_escapes, lin_ws_bytes);
  return 0;
}
