/*
 * jax_opt.c -- JAX-slermed: Adam + Muon optimizers
 *
 * Slermed from WuBuOS/bear_opt patterns.
 */

#include "jax_nn.h"
#include "jax_arena.h"
#include "jax_simd.h"
#include <math.h>
#include <string.h>

/* ===================================================================
 * Adam Optimizer
 * =================================================================== */

void jax_adam_step(JaxParam* p, float lr, float beta1, float beta2, float eps) {
    if (!p || !p->grad.data || !p->weight.data) return;
    
    p->step++;
    float bc1 = 1.0f - powf(beta1, p->step);
    float bc2 = 1.0f - powf(beta2, p->step);
    
    int64_t n = jax_tensor_numel(&p->weight);
    float* w = (float*)p->weight.data;
    float* g = (float*)p->grad.data;
    float* m = (float*)p->mom.data;
    float* v = (float*)p->var.data;
    
    for (int64_t i = 0; i < n; ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void jax_adam_step_layer(JaxLayer* l, float lr, float beta1, float beta2, float eps) {
    jax_adam_step(l->param, lr, beta1, beta2, eps);
}

/* ===================================================================
 * Muon Optimizer (Newton-style orthogonal update)
 * =================================================================== */

void jax_muon_step(JaxParam* p, float lr, float momentum, float nesterov) {
    if (!p || !p->grad.data || !p->weight.data) return;
    
    int64_t n = jax_tensor_numel(&p->weight);
    float* w = (float*)p->weight.data;
    float* g = (float*)p->grad.data;
    float* vel = (float*)p->vel.data;
    
    /* Update velocity */
    for (int64_t i = 0; i < n; ++i) {
        vel[i] = momentum * vel[i] + g[i];
    }
    
    /* Nesterov: grad + momentum * vel */
    float* update = (float*)malloc(n * sizeof(float));
    if (!update) return;
    for (int64_t i = 0; i < n; ++i) {
        update[i] = g[i] + momentum * vel[i];
    }
    
    /* For Muon we'd normally do SVD/Newton-Schulz on the 2D weight matrix.
     * For now, simple SGD-style update with momentum. */
    for (int64_t i = 0; i < n; ++i) {
        w[i] -= lr * update[i];
    }
    
    free(update);
}

/* ===================================================================
 * SGD (simple, no momentum)
 * =================================================================== */

void jax_sgd_step(JaxParam* p, float lr) {
    if (!p || !p->grad.data || !p->weight.data) return;
    
    int64_t n = jax_tensor_numel(&p->weight);
    float* w = (float*)p->weight.data;
    float* g = (float*)p->grad.data;
    
    for (int64_t i = 0; i < n; ++i) {
        w[i] -= lr * g[i];
    }
}

/* ===================================================================
 * Gradient clipping
 * =================================================================== */

void jax_clip_grad_norm(JaxPolicyNet* net, float max_norm) {
    if (!net || !net->layers) return;
    
    float total_norm_sq = 0.0f;
    int total_params = 0;
    
    for (int i = 0; i < net->num_layers; ++i) {
        JaxParam* p = net->layers[i].param;
        if (!p || !p->grad.data) continue;
        int64_t n = jax_tensor_numel(&p->grad);
        float* g = (float*)p->grad.data;
        for (int64_t j = 0; j < n; ++j) total_norm_sq += g[j] * g[j];
        total_params += n;
    }
    
    float norm = sqrtf(total_norm_sq);
    if (norm <= max_norm) return;
    
    float scale = max_norm / norm;
    for (int i = 0; i < net->num_layers; ++i) {
        JaxParam* p = net->layers[i].param;
        if (!p || !p->grad.data) continue;
        int64_t n = jax_tensor_numel(&p->grad);
        float* g = (float*)p->grad.data;
        for (int64_t j = 0; j < n; ++j) g[j] *= scale;
    }
}

void jax_clip_grad_norm_value(JaxValueNet* vnet, float max_norm) {
    if (!vnet || !vnet->layers) return;
    
    float total_norm_sq = 0.0f;
    for (int i = 0; i < vnet->num_layers; ++i) {
        JaxParam* p = vnet->layers[i].param;
        if (!p || !p->grad.data) continue;
        int64_t n = jax_tensor_numel(&p->grad);
        float* g = (float*)p->grad.data;
        for (int64_t j = 0; j < n; ++j) total_norm_sq += g[j] * g[j];
    }
    
    float norm = sqrtf(total_norm_sq);
    if (norm <= max_norm) return;
    
    float scale = max_norm / norm;
    for (int i = 0; i < vnet->num_layers; ++i) {
        JaxParam* p = vnet->layers[i].param;
        if (!p || !p->grad.data) continue;
        int64_t n = jax_tensor_numel(&p->grad);
        float* g = (float*)p->grad.data;
        for (int64_t j = 0; j < n; ++j) g[j] *= scale;
    }
}

/* ===================================================================
 * Weight decay (L2 regularization)
 * =================================================================== */

void jax_weight_decay(JaxPolicyNet* net, float decay) {
    if (!net || !net->layers) return;
    for (int i = 0; i < net->num_layers; ++i) {
        JaxParam* p = net->layers[i].param;
        if (!p || !p->grad.data || !p->weight.data) continue;
        int64_t n = jax_tensor_numel(&p->weight);
        float* w = (float*)p->weight.data;
        float* g = (float*)p->grad.data;
        for (int64_t j = 0; j < n; ++j) g[j] += decay * w[j];
    }
}

void jax_weight_decay_value(JaxValueNet* vnet, float decay) {
    if (!vnet || !vnet->layers) return;
    for (int i = 0; i < vnet->num_layers; ++i) {
        JaxParam* p = vnet->layers[i].param;
        if (!p || !p->grad.data || !p->weight.data) continue;
        int64_t n = jax_tensor_numel(&p->weight);
        float* w = (float*)p->weight.data;
        float* g = (float*)p->grad.data;
        for (int64_t j = 0; j < n; ++j) g[j] += decay * w[j];
    }
}
