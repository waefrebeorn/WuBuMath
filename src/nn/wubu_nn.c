/*
 * wubu_nn.c -- C11 Neural Network Layer Implementation
 *
 * Slermed from PyTorch nn.Module classes.
 * See wubu_nn.h for documentation.
 */

#include "wubu_nn.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ===================================================================
 * Internal Helpers
 * =================================================================== */

static float gelu_activation(float x) {
    /* GELU: x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2))) */
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

/* sigmoid — used by tests and other layers */
float sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

/* ===================================================================
 * Linear Layer
 * =================================================================== */

void wubu_linear_init(WubuLinear* layer, int in_features, int out_features, int use_bias) {
    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->weight = (float*)calloc((size_t)(out_features * in_features), sizeof(float));
    layer->bias = use_bias ? (float*)calloc((size_t)out_features, sizeof(float)) : NULL;
    /* Xavier uniform init */
    wubu_init_xavier_uniform(layer->weight, out_features * in_features, in_features, out_features);
}

void wubu_linear_free(WubuLinear* layer) {
    free(layer->weight);
    free(layer->bias);
    layer->weight = NULL;
    layer->bias = NULL;
}

void wubu_linear_forward(const WubuLinear* layer, const float* x, float* output, int B) {
        /* x: [B, in_features], weight: [out, in], output: [B, out] */
        int in_f = layer->in_features;
        int out_f = layer->out_features;
        for (int b = 0; b < B; ++b) {
            const float* xb = x + b * in_f;
            float* ob = output + b * out_f;
            for (int o = 0; o < out_f; ++o) {
                float sum = layer->bias ? layer->bias[o] : 0.0f;
                const float* w_row = layer->weight + o * in_f;
                for (int i = 0; i < in_f; ++i) {
                    sum += xb[i] * w_row[i];
                }
                ob[o] = sum;
            }
        }
    }

/* ===================================================================
 * ACTIVATION FUNCTIONS
 * =================================================================== */

void wubu_gelu_forward(const float* x, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = gelu_activation(x[i]);
    }
}

void wubu_tanh_forward(const float* x, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = tanhf(x[i]);
    }
}

void wubu_silu_forward(const float* x, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = x[i] * sigmoid(x[i]);
    }
}

void wubu_leaky_relu_forward(const float* x, float* output, int n, float alpha) {
    for (int i = 0; i < n; ++i) {
        output[i] = (x[i] >= 0.0f) ? x[i] : alpha * x[i];
    }
}

void wubu_relu_forward(const float* x, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

/* ===================================================================
 * DROPOUT
 * =================================================================== */

void wubu_dropout_forward(const float* x, float* output, int n, float p, WubuRNG* rng) {
    if (p <= 0.0f) {
        memcpy(output, x, (size_t)n * sizeof(float));
        return;
    }
    float scale = 1.0f / (1.0f - p);
    for (int i = 0; i < n; ++i) {
        /* Generate 0 or 1 with probability (1-p) / p */
        uint64_t r = wubu_rng_next(rng);
        float u = (float)(r >> 8) / (float)(1ULL << 56);
        output[i] = (u >= p) ? (x[i] * scale) : 0.0f;
    }
}

/* ===================================================================
 * LAYER NORMALIZATION
 * =================================================================== */

void wubu_layer_norm_init(WubuLayerNorm* layer, int normalized_shape, float eps) {
    layer->normalized_shape = normalized_shape;
    layer->eps = eps;
    layer->weight = (float*)calloc((size_t)normalized_shape, sizeof(float));
    layer->bias = (float*)calloc((size_t)normalized_shape, sizeof(float));
    wubu_init_ones(layer->weight, normalized_shape);
    wubu_init_zeros(layer->bias, normalized_shape);
}

void wubu_layer_norm_free(WubuLayerNorm* layer) {
    free(layer->weight);
    free(layer->bias);
    layer->weight = NULL;
    layer->bias = NULL;
}

void wubu_layer_norm_forward(const WubuLayerNorm* layer, const float* x, float* output, int B) {
    /* x: [B, normalized_shape], output: [B, normalized_shape] */
    int D = layer->normalized_shape;
    float eps = layer->eps;
    for (int b = 0; b < B; ++b) {
        const float* xb = x + b * D;
        float* ob = output + b * D;
        /* Compute mean */
        float mean = 0.0f;
        for (int i = 0; i < D; ++i) mean += xb[i];
        mean /= (float)D;
        /* Compute variance */
        float var_sum = 0.0f;
        for (int i = 0; i < D; ++i) {
            float diff = xb[i] - mean;
            var_sum += diff * diff;
        }
        float inv_std = 1.0f / sqrtf(var_sum / (float)D + eps);
        /* Normalize */
        for (int i = 0; i < D; ++i) {
            ob[i] = (xb[i] - mean) * inv_std * layer->weight[i] + layer->bias[i];
        }
    }
}

/* ===================================================================
 * EMBEDDING
 * =================================================================== */

void wubu_embedding_init(WubuEmbedding* emb, int num_embeddings, int embedding_dim) {
    emb->num_embeddings = num_embeddings;
    emb->embedding_dim = embedding_dim;
    emb->weight = (float*)calloc((size_t)(num_embeddings * embedding_dim), sizeof(float));
    wubu_init_xavier_uniform(emb->weight, num_embeddings * embedding_dim, embedding_dim, num_embeddings);
}

void wubu_embedding_free(WubuEmbedding* emb) {
    free(emb->weight);
    emb->weight = NULL;
}

void wubu_embedding_forward(const WubuEmbedding* emb, const int* indices,
                             float* output, int batch_size) {
    int dim = emb->embedding_dim;
    for (int b = 0; b < batch_size; ++b) {
        int idx = indices[b];
        if (idx < 0 || idx >= emb->num_embeddings) idx = 0;
        memcpy(output + b * dim, emb->weight + idx * dim, (size_t)dim * sizeof(float));
    }
}

/* ===================================================================
 * SOFTMAX
 * =================================================================== */

void wubu_softmax(const float* x, float* output, int n) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        output[i] = expf(x[i] - max_val);
        sum += output[i];
    }
    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int i = 0; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

/* ===================================================================
 * LOSS FUNCTIONS
 * =================================================================== */

float wubu_cross_entropy_loss(const float* logits, const int* targets,
                               int batch_size, int num_classes) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        const float* logit = logits + b * num_classes;
        int target = targets[b];
        /* Softmax + NLL */
        float max_val = logit[0];
        for (int c = 1; c < num_classes; ++c) {
            if (logit[c] > max_val) max_val = logit[c];
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += expf(logit[c] - max_val);
        }
        float log_sum_exp = logf(sum_exp) + max_val;
        total_loss += -logit[target] + log_sum_exp;
    }
    return total_loss / (float)batch_size;
}

float wubu_mse_loss(const float* pred, const float* target, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = pred[i] - target[i];
        sum += diff * diff;
    }
    return sum / (float)n;
}

float wubu_bce_with_logits_loss(const float* pred, const float* target, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; ++i) {
        float p = sigmoid(pred[i]);
        float t = target[i];
        /* -[t*log(p) + (1-t)*log(1-p)] */
        float eps = 1e-7f;
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        total += -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
    }
    return total / (float)n;
}

/* ===================================================================
 * GRADIENT CLIPPING
 * =================================================================== */

void wubu_clip_grad_norm(float* gradients, int n, float max_norm) {
    float norm_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        norm_sq += gradients[i] * gradients[i];
    }
    float norm = sqrtf(norm_sq);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-6f);
        for (int i = 0; i < n; ++i) {
            gradients[i] *= scale;
        }
    }
}

/* ===================================================================
 * PARAMETER INITIALIZATION
 * =================================================================== */

void wubu_init_xavier_uniform(float* weight, int n, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    WubuRNG rng;
    wubu_rng_init(&rng, 12345);
    for (int i = 0; i < n; ++i) {
        weight[i] = wubu_rng_uniform(&rng, -limit, limit);
    }
}

void wubu_init_zeros(float* buf, int n) {
    memset(buf, 0, (size_t)n * sizeof(float));
}

void wubu_init_ones(float* buf, int n) {
    for (int i = 0; i < n; ++i) buf[i] = 1.0f;
}

/* ===================================================================
 * CONV2D (simplified: 3x3 kernel, stride=1, padding=1)
 * =================================================================== */

void wubu_conv2d_init(WubuConv2d* conv, int in_channels, int out_channels,
                       int kernel_size, int stride, int padding, int use_bias) {
    conv->in_channels = in_channels;
    conv->out_channels = out_channels;
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;
    conv->dilation = 1;
    int w_size = out_channels * in_channels * kernel_size * kernel_size;
    conv->weight = (float*)calloc((size_t)w_size, sizeof(float));
    conv->bias = use_bias ? (float*)calloc((size_t)out_channels, sizeof(float)) : NULL;
    wubu_init_xavier_uniform(conv->weight, w_size, in_channels * kernel_size * kernel_size,
                             out_channels * kernel_size * kernel_size);
}

void wubu_conv2d_free(WubuConv2d* conv) {
    free(conv->weight);
    free(conv->bias);
    conv->weight = NULL;
    conv->bias = NULL;
}

void wubu_conv2d_forward(const WubuConv2d* conv, const float* x,
                         int B, int H, int W_img, float* output) {
    int OC = conv->out_channels;
    int IC = conv->in_channels;
    int K = conv->kernel_size;
    int S = conv->stride;
    int P = conv->padding;
    int out_H = (H + 2 * P - K) / S + 1;
    int out_W = (W_img + 2 * P - K) / S + 1;

    for (int b = 0; b < B; ++b) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < out_H; ++oh) {
                for (int ow = 0; ow < out_W; ++ow) {
                    float sum = conv->bias ? conv->bias[oc] : 0.0f;
                    for (int ic = 0; ic < IC; ++ic) {
                        for (int kh = 0; kh < K; ++kh) {
                            for (int kw = 0; kw < K; ++kw) {
                                int ih = oh * S - P + kh;
                                int iw = ow * S - P + kw;
                                float xv = 0.0f;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W_img) {
                                    xv = x[((b * IC + ic) * H + ih) * W_img + iw];
                                }
                                int widx = ((oc * IC + ic) * K + kh) * K + kw;
                                sum += xv * conv->weight[widx];
                            }
                        }
                    }
                    output[((b * OC + oc) * out_H + oh) * out_W + ow] = sum;
                }
            }
        }
    }
}

/* ===================================================================
 * GROUP NORMALIZATION
 * =================================================================== */

void wubu_group_norm_init(WubuGroupNorm* gn, int num_groups, int num_channels, float eps) {
    gn->num_groups = num_groups;
    gn->num_channels = num_channels;
    gn->eps = eps;
    gn->weight = (float*)calloc((size_t)num_channels, sizeof(float));
    gn->bias = (float*)calloc((size_t)num_channels, sizeof(float));
    wubu_init_ones(gn->weight, num_channels);
    wubu_init_zeros(gn->bias, num_channels);
}

void wubu_group_norm_free(WubuGroupNorm* gn) {
    free(gn->weight);
    free(gn->bias);
}

void wubu_group_norm_forward(const WubuGroupNorm* gn, const float* x,
                              int B, int C, int H, int W_img, float* output) {
    int G = gn->num_groups;
    int channels_per_group = C / G;
    int spatial = H * W_img;

    for (int b = 0; b < B; ++b) {
        for (int g = 0; g < G; ++g) {
            /* Compute mean and var for this group */
            float mean = 0.0f;
            float var_sum = 0.0f;
            int base = (b * C + g * channels_per_group) * spatial;

            for (int c = 0; c < channels_per_group; ++c) {
                for (int s = 0; s < spatial; ++s) {
                    mean += x[base + c * spatial + s];
                }
            }
            mean /= (float)(channels_per_group * spatial);

            for (int c = 0; c < channels_per_group; ++c) {
                for (int s = 0; s < spatial; ++s) {
                    float diff = x[base + c * spatial + s] - mean;
                    var_sum += diff * diff;
                }
            }
            float var = var_sum / (float)(channels_per_group * spatial);
            float inv_std = 1.0f / sqrtf(var + gn->eps);

            /* Apply normalization */
            for (int c = 0; c < channels_per_group; ++c) {
                int ch = g * channels_per_group + c;
                float gamma = gn->weight[ch];
                float beta = gn->bias[ch];
                for (int s = 0; s < spatial; ++s) {
                    int idx = base + c * spatial + s;
                    output[idx] = (x[idx] - mean) * inv_std * gamma + beta;
                }
            }
        }
    }
}

/* ===================================================================
 * ADAPTIVE AVERAGE POOLING 2D
 * =================================================================== */

void wubu_adaptive_avg_pool2d(const float* x, int B, int C, int H, int W_img,
                                int out_h, int out_w, float* output) {
    float stride_h = (float)H / (float)out_h;
    float stride_w = (float)W_img / (float)out_w;

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int h_start = (int)((float)oh * stride_h);
                    int w_start = (int)((float)ow * stride_w);
                    int h_end = (int)((float)(oh + 1) * stride_h);
                    int w_end = (int)((float)(ow + 1) * stride_w);
                    if (h_end == h_start) h_end = h_start + 1;
                    if (w_end == w_start) w_end = w_start + 1;

                    float sum = 0.0f;
                    int count = 0;
                    for (int h = h_start; h < h_end && h < H; ++h) {
                        for (int w = w_start; w < w_end && w < W_img; ++w) {
                            sum += x[((b * C + c) * H + h) * W_img + w];
                            count++;
                        }
                    }
                    output[((b * C + c) * out_h + oh) * out_w + ow] = sum / (float)count;
                }
            }
        }
    }
}

/* ===================================================================
 * MULTI-HEAD ATTENTION
 * =================================================================== */

void wubu_mha_init(WubuMultiheadAttention* mha, int d_model, int num_heads, int use_bias) {
    mha->d_model = d_model;
    mha->num_heads = num_heads;
    mha->use_bias = use_bias;
    int d2 = d_model * d_model;
    mha->W_q = (float*)calloc((size_t)d2, sizeof(float));
    mha->W_k = (float*)calloc((size_t)d2, sizeof(float));
    mha->W_v = (float*)calloc((size_t)d2, sizeof(float));
    mha->W_o = (float*)calloc((size_t)d2, sizeof(float));
    if (use_bias) {
        mha->b_q = (float*)calloc((size_t)d_model, sizeof(float));
        mha->b_k = (float*)calloc((size_t)d_model, sizeof(float));
        mha->b_v = (float*)calloc((size_t)d_model, sizeof(float));
        mha->b_o = (float*)calloc((size_t)d_model, sizeof(float));
    } else {
        mha->b_q = NULL;
        mha->b_k = NULL;
        mha->b_v = NULL;
        mha->b_o = NULL;
    }
    wubu_init_xavier_uniform(mha->W_q, d2, d_model, d_model);
    wubu_init_xavier_uniform(mha->W_k, d2, d_model, d_model);
    wubu_init_xavier_uniform(mha->W_v, d2, d_model, d_model);
    wubu_init_xavier_uniform(mha->W_o, d2, d_model, d_model);
}

void wubu_mha_free(WubuMultiheadAttention* mha) {
    free(mha->W_q); free(mha->W_k); free(mha->W_v); free(mha->W_o);
    free(mha->b_q); free(mha->b_k); free(mha->b_v); free(mha->b_o);
    mha->W_q = mha->W_k = mha->W_v = mha->W_o = NULL;
    mha->b_q = mha->b_k = mha->b_v = mha->b_o = NULL;
}

void wubu_mha_forward(WubuMultiheadAttention* mha, const float* Q, const float* K, const float* V,
                       int B, int seq_len, const float* mask, float* output) {
    /* Simplified: single-head attention for now */
    /* Full multi-head would split into heads and concat */
    int d = mha->d_model;
    float scale = 1.0f / sqrtf((float)d);

    /* Allocate temp buffers */
    float* scores = (float*)malloc((size_t)(B * seq_len * seq_len) * sizeof(float));
    float* attn = (float*)malloc((size_t)(B * seq_len * seq_len) * sizeof(float));

    /* Q' = Q @ W_q, K' = K @ W_k, V' = V @ W_v */
    /* For simplicity, skip projection and use Q,K,V directly */
    /* TODO: full projection */

    /* Scaled dot-product attention */
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < d; ++k) {
                    dot += Q[b * seq_len * d + i * d + k] * K[b * seq_len * d + j * d + k];
                }
                scores[b * seq_len * seq_len + i * seq_len + j] = dot * scale;
            }
        }

        /* Apply mask if provided (causal) */
        if (mask) {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    if (mask[i * seq_len + j] == 0.0f) {
                        scores[b * seq_len * seq_len + i * seq_len + j] = -1e9f;
                    }
                }
            }
        }

        /* Softmax over keys dimension */
        for (int i = 0; i < seq_len; ++i) {
            wubu_softmax(scores + b * seq_len * seq_len + i * seq_len, attn + b * seq_len * seq_len + i * seq_len, seq_len);
        }

        /* Output = attn @ V */
        for (int i = 0; i < seq_len; ++i) {
            for (int k = 0; k < d; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    sum += attn[b * seq_len * seq_len + i * seq_len + j] * V[b * seq_len * d + j * d + k];
                }
                output[b * seq_len * d + i * d + k] = sum;
            }
        }
    }

    free(scores);
    free(attn);
}

/* ===================================================================
 * SEQUENTIAL CONTAINER
 * =================================================================== */

void wubu_sequential_init(WubuSequential* seq) {
    seq->first = NULL;
    seq->last = NULL;
    seq->num_layers = 0;
}

void wubu_sequential_add(WubuSequential* seq, WubuLayer* layer) {
    if (seq->first == NULL) {
        seq->first = layer;
        seq->last = layer;
    } else {
        seq->last->next = layer;
        seq->last = layer;
    }
    seq->num_layers++;
    layer->next = NULL;
}

void wubu_sequential_forward(WubuSequential* seq, const float* input, float* output) {
    if (seq->num_layers == 0) {
        memcpy(output, input, 1); /* no-op */
        return;
    }

    /* Use a double-buffer approach */
    WubuLayer* layer = seq->first;
    float* current = (float*)malloc(1024 * sizeof(float)); /* temp buffer */
    float* next = (float*)malloc(1024 * sizeof(float));
    float* tmp_in = (float*)input;
    float* tmp_out = current;

    while (layer) {
        layer->forward(layer, tmp_in, tmp_out);
        /* Swap buffers */
        float* tmp = tmp_in;
        tmp_in = tmp_out;
        tmp_out = tmp;
        layer = layer->next;
    }

    /* Copy final result to output */
    memcpy(output, tmp_in, 1024 * sizeof(float)); /* size unknown, use max */

    free(current);
    free(next);
}

void wubu_sequential_free(WubuSequential* seq) {
    WubuLayer* layer = seq->first;
    while (layer) {
        WubuLayer* next = layer->next;
        free(layer->params);
        free(layer);
        layer = next;
    }
    seq->first = NULL;
    seq->last = NULL;
    seq->num_layers = 0;
}
