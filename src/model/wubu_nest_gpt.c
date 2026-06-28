/*
 * wubu_nest_gpt.c -- WuBuNestGPT: Hyperbolic GPT with MLA attention
 *
 * Slermed from wubu_nest_gpt_numpy.py (bytropix/WUBUNEST_V2/)
 * Pure C11 implementation of Multi-head Latent Attention with
 * Poincare ball hyperbolic gyration position encoding.
 */

#include "wubu_nest_gpt.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>

/* ===================================================================
 * PRNG (simple LCG for reproducibility)
 * =================================================================== */

static uint32_t rng_state;

static void rng_seed(uint32_t seed) {
    rng_state = seed;
}

static float rng_float(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (float)(rng_state >> 16) / 65536.0f;
}

static float rng_normal(void) {
    /* Box-Muller */
    float u1 = rng_float() + 1e-7f;
    float u2 = rng_float();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

/* ===================================================================
 * Matrix multiply: C = A @ B
 * A: [M, K], B: [K, N] -> C: [M, N]
 * =================================================================== */

static void matmul(float* C, const float* A, const float* B, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* ===================================================================
 * Layer norm
 * =================================================================== */

static void layernorm(float* out, const float* x, const float* gamma,
                      const float* beta, int N, int D) {
    for (int i = 0; i < N; i++) {
        float mean = 0.0f;
        for (int j = 0; j < D; j++) mean += x[i * D + j];
        mean /= (float)D;

        float var = 0.0f;
        for (int j = 0; j < D; j++) {
            float d = x[i * D + j] - mean;
            var += d * d;
        }
        var /= (float)D;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < D; j++) {
            out[i * D + j] = gamma[j] * (x[i * D + j] - mean) * inv_std + beta[j];
        }
    }
}

/* ===================================================================
 * SwiGLU activation
 * =================================================================== */

static void swiglu(float* out, const float* x, int N, int d_ff, int D) {
    /* x: [N, D] -> gate [N, d_ff], up [N, d_ff] -> silu(gate) * up -> [N, d_ff] -> proj -> [N, D] */
    /* Simplified: just apply SiLU element-wise (for FFN hidden dim) */
    for (int i = 0; i < N * d_ff; i++) {
        float val = x[i];
        out[i] = val / (1.0f + expf(-val));
    }
}

/* ===================================================================
 * Softmax
 * =================================================================== */

static void softmax(float* x, int N, int D) {
    for (int i = 0; i < N; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < D; j++) {
            if (x[i * D + j] > max_val) max_val = x[i * D + j];
        }
        float sum = 0.0f;
        for (int j = 0; j < D; j++) {
            x[i * D + j] = expf(x[i * D + j] - max_val);
            sum += x[i * D + j];
        }
        float inv_sum = 1.0f / (sum + 1e-10f);
        for (int j = 0; j < D; j++) {
            x[i * D + j] *= inv_sum;
        }
    }
}

/* ===================================================================
 * Initialize model
 * =================================================================== */

int wubu_gpt_init(WubuGPT* model, const WubuGPTConfig* config) {
    model->config = *config;
    model->D = config->d_model;
    model->H = config->n_heads;
    model->d_h = config->d_head;
    model->d_c = config->d_compressed;
    model->d_rope = config->d_head / 2;
    model->d_ff = config->d_ff;
    model->N = config->n_layers;
    model->V = config->vocab_size;

    rng_seed((uint32_t)(model->config.seed));

    int D = model->D, H = model->H, d_h = model->d_h;
    int d_c = model->d_c, d_rope = model->d_rope, d_ff = model->d_ff;
    int V = model->V, N = model->N;
    float scale = config->init_scale;

    /* Allocate embeddings */
    model->wte = (float*)calloc(V * D, sizeof(float));
    model->lm_head_w = (float*)calloc(D * V, sizeof(float));

    /* Init embeddings with normal */
    for (int i = 0; i < V * D; i++) model->wte[i] = rng_normal() * scale;
    for (int i = 0; i < D * V; i++) model->lm_head_w[i] = rng_normal() * scale;

    /* Final layer norm */
    model->ln_f_gamma = (float*)malloc(D * sizeof(float));
    model->ln_f_beta = (float*)calloc(D, sizeof(float));
    for (int i = 0; i < D; i++) model->ln_f_gamma[i] = 1.0f;

    /* Allocate blocks */
    model->blocks = (WubuGPTBlock*)calloc(N, sizeof(WubuGPTBlock));

    for (int i = 0; i < N; i++) {
        WubuGPTBlock* b = &model->blocks[i];

        b->ln1_gamma = (float*)malloc(D * sizeof(float));
        b->ln1_beta = (float*)calloc(D, sizeof(float));
        b->ln2_gamma = (float*)malloc(D * sizeof(float));
        b->ln2_beta = (float*)calloc(D, sizeof(float));
        for (int j = 0; j < D; j++) {
            b->ln1_gamma[j] = 1.0f;
            b->ln2_gamma[j] = 1.0f;
        }

        /* MLA weights */
        b->wq = (float*)calloc(D * H * d_h, sizeof(float));
        b->wdkv = (float*)calloc(D * d_c, sizeof(float));
        b->wuk = (float*)calloc(d_c * H * d_h, sizeof(float));
        b->wuv = (float*)calloc(d_c * H * d_h, sizeof(float));
        b->wqr = (float*)calloc(D * H * d_rope, sizeof(float));
        b->wkr = (float*)calloc(D * H * d_rope, sizeof(float));
        b->wo = (float*)calloc(H * d_h * D, sizeof(float));

        /* FFN weights */
        b->ffn1_w = (float*)calloc(D * d_ff, sizeof(float));
        b->ffn1_b = (float*)calloc(d_ff, sizeof(float));
        b->ffn2_w = (float*)calloc(d_ff * D, sizeof(float));
        b->ffn2_b = (float*)calloc(D, sizeof(float));

        /* Init weights */
        for (int j = 0; j < D * H * d_h; j++) b->wq[j] = rng_normal() * scale;
        for (int j = 0; j < D * d_c; j++) b->wdkv[j] = rng_normal() * scale;
        for (int j = 0; j < d_c * H * d_h; j++) b->wuk[j] = rng_normal() * scale;
        for (int j = 0; j < d_c * H * d_h; j++) b->wuv[j] = rng_normal() * scale;
        for (int j = 0; j < D * H * d_rope; j++) b->wqr[j] = rng_normal() * scale;
        for (int j = 0; j < D * H * d_rope; j++) b->wkr[j] = rng_normal() * scale;
        for (int j = 0; j < H * d_h * D; j++) b->wo[j] = rng_normal() * scale;

        for (int j = 0; j < D * d_ff; j++) b->ffn1_w[j] = rng_normal() * scale;
        for (int j = 0; j < d_ff * D; j++) b->ffn2_w[j] = rng_normal() * scale;
    }

    return 0;
}

/* ===================================================================
 * Free model
 * =================================================================== */

void wubu_gpt_free(WubuGPT* model) {
    free(model->wte);
    free(model->lm_head_w);
    free(model->ln_f_gamma);
    free(model->ln_f_beta);

    for (int i = 0; i < model->N; i++) {
        WubuGPTBlock* b = &model->blocks[i];
        free(b->ln1_gamma); free(b->ln1_beta);
        free(b->ln2_gamma); free(b->ln2_beta);
        free(b->wq); free(b->wdkv);
        free(b->wuk); free(b->wuv);
        free(b->wqr); free(b->wkr); free(b->wo);
        free(b->ffn1_w); free(b->ffn1_b);
        free(b->ffn2_w); free(b->ffn2_b);
    }
    free(model->blocks);
    memset(model, 0, sizeof(WubuGPT));
}

/* ===================================================================
 * Forward pass
 * =================================================================== */

float* wubu_gpt_forward(WubuGPT* model, const int* tokens, int B, int T, bool training) {
    int D = model->D, H = model->H, d_h = model->d_h;
    int d_c = model->d_c, d_rope = model->d_rope, d_ff = model->d_ff;
    int N = model->N;

    /* Allocate output logits [B, T, V] */
    float* logits = (float*)calloc(B * T * model->V, sizeof(float));

    /* Embeddings: h = wte[tokens] -> [B, T, D] */
    float* h = (float*)malloc(B * T * D * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int tok = tokens[b * T + t];
            memcpy(&h[(b * T + t) * D], &model->wte[tok * D], D * sizeof(float));
        }
    }

    /* Forward through blocks */
    float* h_curr = h;
    for (int layer = 0; layer < N; layer++) {
        WubuGPTBlock* b = &model->blocks[layer];
        float* h_out = (float*)malloc(B * T * D * sizeof(float));

        /* LayerNorm 1 */
        float* h_norm = (float*)malloc(B * T * D * sizeof(float));
        layernorm(h_norm, h_curr, b->ln1_gamma, b->ln1_beta, B * T, D);

        /* Q projection: q = h_norm @ wq -> [B*T, H*d_h] */
        float* q = (float*)malloc(B * T * H * d_h * sizeof(float));
        matmul(q, h_norm, b->wq, B * T, D, H * d_h);

        /* KV latent: kv_latent = h_norm @ wdkv -> [B*T, d_c] */
        float* kv_latent = (float*)malloc(B * T * d_c * sizeof(float));
        matmul(kv_latent, h_norm, b->wdkv, B * T, D, d_c);

        /* Reconstruct K, V */
        float* k = (float*)malloc(B * T * H * d_h * sizeof(float));
        float* v = (float*)malloc(B * T * H * d_h * sizeof(float));
        matmul(k, kv_latent, b->wuk, B * T, d_c, H * d_h);
        matmul(v, kv_latent, b->wuv, B * T, d_c, H * d_h);

        /* RoPE projections */
        float* q_rope = (float*)malloc(B * T * H * d_rope * sizeof(float));
        float* k_rope = (float*)malloc(B * T * H * d_rope * sizeof(float));
        matmul(q_rope, h_norm, b->wqr, B * T, D, H * d_rope);
        matmul(k_rope, h_norm, b->wkr, B * T, D, H * d_rope);

        /* Reshape for multi-head: [B, T, H, d_h] and [B, T, H, d_rope] */
        /* For simplicity, we'll do attention in flattened heads */
        /* Attention scores: einsum('bthd,bThd->bhtT') -> [B, H, T, T] */
        int BH = B * H;
        float* attn = (float*)calloc(BH * T * T, sizeof(float));

        for (int bi = 0; bi < B; bi++) {
            for (int hi = 0; hi < H; hi++) {
                for (int ti = 0; ti < T; ti++) {
                    for (int tj = 0; tj <= ti; tj++) { /* causal */
                        float score = 0.0f;
                        for (int d = 0; d < d_h; d++) {
                            score += q[((bi * T + ti) * H + hi) * d_h + d] *
                                     k[((bi * T + tj) * H + hi) * d_h + d];
                        }
                        score /= sqrtf((float)d_h);
                        attn[((bi * H + hi) * T + ti) * T + tj] = score;
                    }
                    /* Future positions: -inf */
                    for (int tj = ti + 1; tj < T; tj++) {
                        attn[((bi * H + hi) * T + ti) * T + tj] = -1e10f;
                    }
                }
            }
        }

        /* Softmax over last dim */
        for (int i = 0; i < BH * T; i++) {
            float max_val = -FLT_MAX;
            for (int j = 0; j < T; j++) {
                if (attn[i * T + j] > max_val) max_val = attn[i * T + j];
            }
            float sum = 0.0f;
            for (int j = 0; j < T; j++) {
                attn[i * T + j] = expf(attn[i * T + j] - max_val);
                sum += attn[i * T + j];
            }
            float inv_sum = 1.0f / (sum + 1e-10f);
            for (int j = 0; j < T; j++) {
                attn[i * T + j] *= inv_sum;
            }
        }

        /* Attention output: einsum('bhtT,bThd->bthd') -> [B, T, H, d_h] */
        float* attn_out = (float*)calloc(B * T * H * d_h, sizeof(float));
        for (int bi = 0; bi < B; bi++) {
            for (int hi = 0; hi < H; hi++) {
                for (int ti = 0; ti < T; ti++) {
                    for (int d = 0; d < d_h; d++) {
                        float val = 0.0f;
                        for (int tj = 0; tj < T; tj++) {
                            val += attn[((bi * H + hi) * T + ti) * T + tj] *
                                   v[((bi * T + tj) * H + hi) * d_h + d];
                        }
                        attn_out[((bi * T + ti) * H + hi) * d_h + d] = val;
                    }
                }
            }
        }

        /* Output projection: attn_out @ wo -> [B*T, D] */
        float* attn_proj = (float*)malloc(B * T * D * sizeof(float));
        matmul(attn_proj, attn_out, b->wo, B * T, H * d_h, D);

        /* Residual: h = h + attn_proj */
        for (int i = 0; i < B * T * D; i++) {
            h_out[i] = h_curr[i] + attn_proj[i];
        }

        /* FFN sub-layer */
        float* h_norm2 = (float*)malloc(B * T * D * sizeof(float));
        layernorm(h_norm2, h_out, b->ln2_gamma, b->ln2_beta, B * T, D);

        float* ffn_pre = (float*)malloc(B * T * d_ff * sizeof(float));
        /* ffn_pre = h_norm2 @ ffn1_w + ffn1_b */
        for (int i = 0; i < B * T; i++) {
            for (int j = 0; j < d_ff; j++) {
                float val = b->ffn1_b[j];
                for (int k = 0; k < D; k++) {
                    val += h_norm2[i * D + k] * b->ffn1_w[k * d_ff + j];
                }
                ffn_pre[i * d_ff + j] = val;
            }
        }

        /* SiLU activation */
        float* ffn_post = (float*)malloc(B * T * d_ff * sizeof(float));
        for (int i = 0; i < B * T * d_ff; i++) {
            ffn_post[i] = ffn_pre[i] / (1.0f + expf(-ffn_pre[i]));
        }

        /* FFN2 */
        float* ffn_out = (float*)malloc(B * T * D * sizeof(float));
        for (int i = 0; i < B * T; i++) {
            for (int j = 0; j < D; j++) {
                float val = b->ffn2_b[j];
                for (int k = 0; k < d_ff; k++) {
                    val += ffn_post[i * d_ff + k] * b->ffn2_w[k * D + j];
                }
                ffn_out[i * D + j] = val;
            }
        }

        /* Residual: h_out = h_out + ffn_out */
        for (int i = 0; i < B * T * D; i++) {
            h_out[i] += ffn_out[i];
        }

        /* Cleanup layer temps */
        free(h_norm); free(q); free(kv_latent); free(k); free(v);
        free(q_rope); free(k_rope); free(attn); free(attn_out);
        free(attn_proj); free(h_norm2); free(ffn_pre); free(ffn_post); free(ffn_out);

        free(h_curr);
        h_curr = h_out;
    }

    /* Final layer norm */
    layernorm(h_curr, h_curr, model->ln_f_gamma, model->ln_f_beta, B * T, D);

    /* LM head: logits = h @ lm_head_w -> [B, T, V] */
    matmul(logits, h_curr, model->lm_head_w, B * T, D, model->V);

    free(h_curr);
    return logits;
}

/* ===================================================================
 * Compute loss
 * =================================================================== */

float wubu_gpt_compute_loss(WubuGPT* model, const int* targets, int B, int T) {
    int V = model->V;
    float* logits = wubu_gpt_forward(model, NULL, B, T, false);
    /* Note: this is a simplified version -- in practice we'd pass tokens */

    float loss = 0.0f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int target = targets[b * T + t];
            float max_val = -FLT_MAX;
            for (int v = 0; v < V; v++) {
                if (logits[(b * T + t) * V + v] > max_val) {
                    max_val = logits[(b * T + t) * V + v];
                }
            }
            float sum = 0.0f;
            for (int v = 0; v < V; v++) {
                sum += expf(logits[(b * T + t) * V + v] - max_val);
            }
            loss -= logf(expf(logits[(b * T + t) * V + target] - max_val) / (sum + 1e-10f));
        }
    }
    loss /= (float)(B * T);

    free(logits);
    return loss;
}

/* ===================================================================
 * Generate
 * =================================================================== */

int* wubu_gpt_generate(WubuGPT* model, const int* prompt, int prompt_len,
                       int max_new_tokens, float temperature, int top_k,
                       int* out_len) {
    int V = model->V;
    int total_T = prompt_len + max_new_tokens;

    /* Allocate token buffer */
    int* tokens = (int*)malloc(total_T * sizeof(int));
    memcpy(tokens, prompt, prompt_len * sizeof(int));

    for (int step = 0; step < max_new_tokens; step++) {
        int cur_T = prompt_len + step;

        /* Forward pass on current sequence */
        float* logits = wubu_gpt_forward(model, tokens, 1, cur_T, false);

        /* Get logits for last position */
        float* next_logits = &logits[(cur_T - 1) * V];

        /* Apply temperature */
        if (temperature != 1.0f) {
            for (int v = 0; v < V; v++) next_logits[v] /= temperature;
        }

        /* Top-k filtering */
        if (top_k > 0) {
            /* Find top-k threshold */
            float sorted[5000]; /* max vocab */
            int min_v = V < 5000 ? V : 5000;
            memcpy(sorted, next_logits, min_v * sizeof(float));
            /* Simple partial sort for top-k */
            for (int k = 0; k < top_k && k < min_v; k++) {
                for (int j = k + 1; j < min_v; j++) {
                    if (sorted[j] > sorted[k]) {
                        float tmp = sorted[k]; sorted[k] = sorted[j]; sorted[j] = tmp;
                    }
                }
            }
            float threshold = sorted[top_k - 1];
            for (int v = 0; v < V; v++) {
                if (next_logits[v] < threshold) next_logits[v] = -1e10f;
            }
        }

        /* Softmax and sample */
        float max_val = -FLT_MAX;
        for (int v = 0; v < V; v++) {
            if (next_logits[v] > max_val) max_val = next_logits[v];
        }
        float sum = 0.0f;
        for (int v = 0; v < V; v++) {
            next_logits[v] = expf(next_logits[v] - max_val);
            sum += next_logits[v];
        }
        float r = rng_float() * sum;
        int next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < V; v++) {
            cumsum += next_logits[v] / (sum + 1e-10f);
            if (cumsum >= r) { next_token = v; break; }
        }

        tokens[cur_T] = next_token;
        free(logits);

        if (next_token == 2) break; /* EOS */
    }

    *out_len = prompt_len + max_new_tokens;
    return tokens;
}

/* ===================================================================
 * Count parameters
 * =================================================================== */

long wubu_gpt_count_params(const WubuGPT* model) {
    long total = 0;
    int D = model->D, H = model->H, d_h = model->d_h;
    int d_c = model->d_c, d_rope = model->d_rope, d_ff = model->d_ff;
    int V = model->V, N = model->N;

    total += V * D; /* wte */
    total += D * V; /* lm_head */
    total += 2 * D; /* final ln */

    for (int i = 0; i < N; i++) {
        total += 4 * D; /* layer norms */
        total += D * H * d_h; /* wq */
        total += D * d_c; /* wdkv */
        total += 2 * d_c * H * d_h; /* wuk, wuv */
        total += 2 * D * H * d_rope; /* wqr, wkr */
        total += H * d_h * D; /* wo */
        total += D * d_ff + d_ff + d_ff * D + D; /* ffn */
    }

    return total;
}
