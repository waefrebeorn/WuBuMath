/*
 * jax_simd.c -- Non-inline SIMD implementations (dispatcher, etc.)
 */

#include "jax_simd.h"

/* Batch GEMM for sequence processing: [batch, M, K] @ [batch, K, N] -> [batch, M, N] */
void jax_batch_gemm(const float* A, const float* B, float* C, int batch, int M, int N, int K) {
    for (int b = 0; b < batch; ++b) {
        jax_gemm(A + b * M * K, B + b * K * N, C + b * M * N, M, N, K);
    }
}

/* Fused bias add + activation */
void jax_bias_act(float* out, const float* bias, int M, int N, JaxAct act) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            out[m * N + n] += bias[n];
        }
    }
    if (act != JAX_ACT_NONE) {
        int64_t n = (int64_t)M * N;
        float* tmp = (float*)malloc(n * sizeof(float));
        for (int64_t i = 0; i < n; ++i) tmp[i] = out[i];
        jax_act_batch(out, tmp, n, act);
        free(tmp);
    }
}

/* Softmax (stable, batched) */
void jax_softmax(float* out, const float* in, int batch, int dim) {
    for (int b = 0; b < batch; ++b) {
        const float* x = in + b * dim;
        float* y = out + b * dim;
        
        /* Find max for numerical stability */
        float max_val = x[0];
        for (int i = 1; i < dim; ++i) if (x[i] > max_val) max_val = x[i];
        
        /* Exp and sum */
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            y[i] = expf(x[i] - max_val);
            sum += y[i];
        }
        
        /* Normalize */
        for (int i = 0; i < dim; ++i) y[i] /= sum;
    }
}

/* LogSoftmax (stable, batched) */
void jax_log_softmax(float* out, const float* in, int batch, int dim) {
    for (int b = 0; b < batch; ++b) {
        const float* x = in + b * dim;
        float* y = out + b * dim;
        
        float max_val = x[0];
        for (int i = 1; i < dim; ++i) if (x[i] > max_val) max_val = x[i];
        
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) sum += expf(x[i] - max_val);
        float log_sum = logf(sum);
        
        for (int i = 0; i < dim; ++i) y[i] = x[i] - max_val - log_sum;
    }
}

/* Cross-entropy loss */
float jax_cross_entropy(const float* logits, const int* targets, int batch, int num_classes) {
    float loss = 0.0f;
    for (int b = 0; b < batch; ++b) {
        const float* x = logits + b * num_classes;
        int t = targets[b];
        
        float max_val = x[0];
        for (int i = 1; i < num_classes; ++i) if (x[i] > max_val) max_val = x[i];
        
        float sum = 0.0f;
        for (int i = 0; i < num_classes; ++i) sum += expf(x[i] - max_val);
        
        loss -= (x[t] - max_val) - logf(sum);
    }
    return loss / batch;
}

/* LayerNorm */
void jax_layer_norm(float* out, const float* in, const float* weight, const float* bias,
                     int batch, int dim, float eps) {
    for (int b = 0; b < batch; ++b) {
        const float* x = in + b * dim;
        float* y = out + b * dim;
        
        /* Mean */
        float mean = 0.0f;
        for (int i = 0; i < dim; ++i) mean += x[i];
        mean /= dim;
        
        /* Variance */
        float var = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= dim;
        
        /* Normalize and scale */
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; ++i) {
            y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

/* RMSNorm (simpler, used in modern LLMs) */
void jax_rms_norm(float* out, const float* in, const float* weight,
                   int batch, int dim, float eps) {
    for (int b = 0; b < batch; ++b) {
        const float* x = in + b * dim;
        float* y = out + b * dim;
        
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
        float rms = sqrtf(sum_sq / dim + eps);
        
        for (int i = 0; i < dim; ++i) {
            y[i] = x[i] / rms * weight[i];
        }
    }
}
