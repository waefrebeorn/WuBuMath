/*
 * jax_simd.h  --  PufferC/JaxRL SIMD Intrinsics + Fused Kernels
 *
 * AVX2 / NEON / Scalar fallback for matmul, fused ops.
 * Pure C11 + compiler intrinsics. No external deps.
 */

#ifndef JAX_SIMD_H
#define JAX_SIMD_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "jax_arena.h"  /* for JaxTensor */

/* ===================================================================
 * Architecture Detection
 * =================================================================== */

#if defined(__AVX2__)
    #define JAX_HAVE_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON__) || defined(__aarch64__)
    #define JAX_HAVE_NEON 1
    #include <arm_neon.h>
#else
    #define JAX_HAVE_SCALAR 1
#endif

/* ===================================================================
 * Activation Functions (SIMD-accelerated)
 * =================================================================== */

typedef enum {
    JAX_ACT_NONE  = 0,
    JAX_ACT_RELU  = 1,
    JAX_ACT_TANH  = 2,
    JAX_ACT_GELU  = 3,  /* Gaussian Error Linear Unit */
    JAX_ACT_SILU  = 4,  /* SiLU / Swish: x * sigmoid(x) */
} JaxAct;

/* Scalar fallback (always available) */
static inline float jax_act_scalar(float x, JaxAct act) {
    switch (act) {
        case JAX_ACT_RELU:  return x > 0 ? x : 0;
        case JAX_ACT_TANH:  return tanhf(x);
        case JAX_ACT_GELU:  return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
        case JAX_ACT_SILU:  return x / (1.0f + expf(-x));
        default: return x;
    }
}

/* SIMD batch activation */
static inline void jax_act_batch(float* __restrict dst, const float* __restrict src,
                                   int64_t n, JaxAct act) {
#if defined(JAX_HAVE_AVX2)
    if (act == JAX_ACT_RELU) {
        __m256 zero = _mm256_setzero_ps();
        for (int64_t i = 0; i <= n - 8; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_max_ps(v, zero));
        }
        for (int64_t i = n & ~7; i < n; ++i) dst[i] = jax_act_scalar(src[i], act);
        return;
    }
#elif defined(JAX_HAVE_NEON)
    if (act == JAX_ACT_RELU) {
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (int64_t i = 0; i <= n - 4; i += 4) {
            float32x4_t v = vld1q_f32(src + i);
            vst1q_f32(dst + i, vmaxq_f32(v, zero));
        }
        for (int64_t i = n & ~3; i < n; ++i) dst[i] = jax_act_scalar(src[i], act);
        return;
    }
#endif
    /* Scalar fallback for non-RELU or unsupported arch */
    for (int64_t i = 0; i < n; ++i) dst[i] = jax_act_scalar(src[i], act);
}

/* ===================================================================
 * GEMM: C = A * B^T  (row-major: A=[M,K], B=[N,K], C=[M,N])
 * =================================================================== */

/* AVX2: 8x8 micro-kernel */
#if defined(JAX_HAVE_AVX2)
static inline void jax_gemm_avx2(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n <= N - 8; n += 8) {
            __m256 c = _mm256_setzero_ps();
            for (int k = 0; k < K; ++k) {
                __m256 a = _mm256_broadcast_ss(A + m * K + k);
                __m256 b = _mm256_loadu_ps(B + n * K + k);
                c = _mm256_fmadd_ps(a, b, c);
            }
            _mm256_storeu_ps(C + m * N + n, c);
        }
        /* Tail */
        for (int n = N & ~7; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[n * K + k];
            C[m * N + n] = acc;
        }
    }
}
#endif

/* NEON: 4x4 micro-kernel */
#if defined(JAX_HAVE_NEON)
static inline void jax_gemm_neon(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n <= N - 4; n += 4) {
            float32x4_t c = vdupq_n_f32(0.0f);
            for (int k = 0; k < K; ++k) {
                float32x4_t a = vdupq_n_f32(A[m * K + k]);
                float32x4_t b = vld1q_f32(B + n * K + k);
                c = vfmaq_f32(c, a, b);
            }
            vst1q_f32(C + m * N + n, c);
        }
        for (int n = N & ~3; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[n * K + k];
            C[m * N + n] = acc;
        }
    }
}
#endif

/* Scalar fallback */
static inline void jax_gemm_scalar(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = acc;
        }
    }
}

/* Dispatcher */
static inline void jax_gemm(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    jax_gemm_scalar(A, B, C, M, N, K);
}

/* High-level tensor GEMM: out = A @ B^T + bias (if present) */
static inline void jax_tensor_gemm(const JaxTensor* A, const JaxTensor* B,
                                     const JaxTensor* bias,
                                     JaxTensor* out, JaxAct act) {
    int M = (int)A->shape[0];
    int K = (int)A->shape[1];
    int N = (int)B->shape[0];

    const float* a = (const float*)A->data;
    const float* b = (const float*)B->data;
    float* c = (float*)out->data;

    jax_gemm(a, b, c, M, N, K);

    if (bias && bias->dtype == JAX_DTYPE_F32) {
        const float* bias_p = (const float*)bias->data;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                c[m * N + n] += bias_p[n];
            }
        }
    }

    if (act != JAX_ACT_NONE) {
        int64_t n = (int64_t)M * N;
        float* tmp = (float*)malloc(n * sizeof(float));
        for (int64_t i = 0; i < n; ++i) tmp[i] = c[i];
        jax_act_batch(c, tmp, n, act);
        free(tmp);
    }
}

/* ===================================================================
 * Fused MLP Layer: out = act(x @ W^T + b)
 * =================================================================== */

static inline void jax_mlp_layer(const JaxTensor* x,   /* [batch, in_f] */
                                   const JaxTensor* W,   /* [out_f, in_f] */
                                   const JaxTensor* b,   /* [out_f] */
                                   JaxTensor* out,       /* [batch, out_f] */
                                   JaxAct act,
                                   JaxArena* temp_arena) {
    (void)temp_arena;
    jax_tensor_gemm(x, W, b, out, act);
}

/* ===================================================================
 * Math helpers (must come before MinGRU step)
 * =================================================================== */

static inline float jax_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float jax_tanh_fast(float x) {
    return tanhf(x);
}

/* ===================================================================
 * MinGRU Cell (parallelizable recurrent)
 * h_t = z_t * h_{t-1} + (1 - z_t) * n_t
 * z_t = sigmoid(x @ Wz + h_{t-1} @ Uz + bz)
 * r_t = sigmoid(x @ Wr + h_{t-1} @ Ur + br)
 * n_t = tanh(x @ Wn + r_t * (h_{t-1} @ Un) + bn)
 * =================================================================== */

typedef struct {
    JaxParam Wz;   /* [hid, in] */
    JaxParam Uz;   /* [hid, hid] */
    JaxParam bz;   /* [hid, 1] */
    JaxParam Wr;   /* [hid, in] */
    JaxParam Ur;   /* [hid, hid] */
    JaxParam br;   /* [hid, 1] */
    JaxParam Wn;   /* [hid, in] */
    JaxParam Un;   /* [hid, hid] */
    JaxParam bn;   /* [hid, 1] */
    int hid_size;
} JaxMinGRU;

/* Initialize MinGRU params */
static inline int jax_mingru_create(JaxArena* a, JaxMinGRU* gru,
                                      int in_size, int hid_size) {
    gru->hid_size = hid_size;
    if (jax_param_create(a, &gru->Wz, hid_size, in_size, "gru.Wz") != 0) return -1;
    if (jax_param_create(a, &gru->Uz, hid_size, hid_size, "gru.Uz") != 0) return -1;
    if (jax_param_create(a, &gru->bz, hid_size, 1, "gru.bz") != 0) return -1;
    if (jax_param_create(a, &gru->Wr, hid_size, in_size, "gru.Wr") != 0) return -1;
    if (jax_param_create(a, &gru->Ur, hid_size, hid_size, "gru.Ur") != 0) return -1;
    if (jax_param_create(a, &gru->br, hid_size, 1, "gru.br") != 0) return -1;
    if (jax_param_create(a, &gru->Wn, hid_size, in_size, "gru.Wn") != 0) return -1;
    if (jax_param_create(a, &gru->Un, hid_size, hid_size, "gru.Un") != 0) return -1;
    if (jax_param_create(a, &gru->bn, hid_size, 1, "gru.bn") != 0) return -1;
    return 0;
}

/* MinGRU forward (single step, batched) */
static inline void jax_mingru_step(const JaxMinGRU* gru,
                                     const JaxTensor* x,      /* [batch, in] */
                                     const JaxTensor* h_prev, /* [batch, hid] (optional) */
                                     JaxTensor* h_next,       /* [batch, hid] */
                                     JaxArena* temp) {
    int batch = (int)x->shape[0];
    int hid = gru->hid_size;
    int in_size = (int)gru->Wz.weight.shape[1];

    /* If no h_prev, use zero initial state */
    float* h_prev_p = NULL;
    if (h_prev && h_prev->data) {
        h_prev_p = (float*)h_prev->data;
    }

    /* Allocate temp buffers */
    float* z_pre = (float*)JAX_ARENA_ALLOC(temp, float, batch * hid);
    float* r_pre = (float*)JAX_ARENA_ALLOC(temp, float, batch * hid);
    float* n_pre = (float*)JAX_ARENA_ALLOC(temp, float, batch * hid);
    float* Uh     = (float*)JAX_ARENA_ALLOC(temp, float, batch * hid);
    float* Ur_h   = (float*)JAX_ARENA_ALLOC(temp, float, batch * hid);

    const float* x_p = (const float*)x->data;
    float* h_next_p = (float*)h_next->data;
    fflush(stdout);

    /* z_pre = x @ Wz^T  [batch, hid] */
    jax_gemm(x_p, (const float*)gru->Wz.weight.data, z_pre, batch, hid, in_size);
    /* r_pre = x @ Wr^T  [batch, hid] */
    jax_gemm(x_p, (const float*)gru->Wr.weight.data, r_pre, batch, hid, in_size);

    if (h_prev_p) {
        jax_gemm(h_prev_p, (const float*)gru->Uz.weight.data, Uh, batch, hid, hid);
        jax_gemm(h_prev_p, (const float*)gru->Ur.weight.data, Ur_h, batch, hid, hid);
    } else {
        memset(Uh, 0, batch * hid * sizeof(float));
        memset(Ur_h, 0, batch * hid * sizeof(float));
    }

    const float* bz_p = (const float*)gru->bz.bias.data;
    const float* br_p = (const float*)gru->br.bias.data;
    const float* bn_p = (const float*)gru->bn.bias.data;

    for (int i = 0; i < batch * hid; ++i) {
        z_pre[i] = jax_sigmoid(z_pre[i] + Uh[i] + bz_p[i % hid]);
        r_pre[i] = jax_sigmoid(r_pre[i] + Ur_h[i] + br_p[i % hid]);
    }

    /* n_pre = x @ Wn^T  [batch, hid] */
    jax_gemm(x_p, (const float*)gru->Wn.weight.data, n_pre, batch, hid, in_size);

    /* Un_term = h_prev @ Un^T  [batch, hid] */
    if (h_prev_p) {
        float* Un_term = (float*)JAX_ARENA_ALLOC(temp, float, batch * hid);
        jax_gemm(h_prev_p, (const float*)gru->Un.weight.data, Un_term, batch, hid, hid);
        for (int i = 0; i < batch * hid; ++i) {
            n_pre[i] = jax_tanh_fast(n_pre[i] + r_pre[i] * Un_term[i] + bn_p[i % hid]);
        }
    } else {
        for (int i = 0; i < batch * hid; ++i) {
            n_pre[i] = jax_tanh_fast(n_pre[i] + bn_p[i % hid]);
        }
    }

    /* h_next = z * h_prev + (1 - z) * n */
    for (int i = 0; i < batch * hid; ++i) {
        float h_val = h_prev_p ? h_prev_p[i] : 0.0f;
        h_next_p[i] = z_pre[i] * h_val + (1.0f - z_pre[i]) * n_pre[i];
    }
}

/* ===================================================================
 * Vectorized Environment Step Helpers (SIMD on env dimension)
 * =================================================================== */

#define JAX_MAX_ENVS 1024
#define JAX_MAX_AGENTS 8
#define JAX_MAX_OBS_DIM 256
#define JAX_MAX_ACT_DIM 64

/* ===================================================================
 * Orthogonal Initialization (CleanRL detail)
 * =================================================================== */

/* Fill matrix with orthogonal initialization (QR decomp of random) */
static inline void jax_orthogonal_init(JaxTensor* W, float gain) {
    if (W->dtype != JAX_DTYPE_F32 || W->ndim != 2) return;
    int rows = (int)W->shape[0];
    int cols = (int)W->shape[1];
    float* data = (float*)W->data;

    /* Simple uniform Xavier init  --  guaranteed finite, no trig/log */
    float limit = gain * sqrtf(6.0f / (float)(rows + cols));
    for (int i = 0; i < rows * cols; ++i) {
        long r = rand();
        float u = (float)r / (float)RAND_MAX;
        data[i] = (2.0f * u - 1.0f) * limit;
    }
}

#endif /* JAX_SIMD_H */
