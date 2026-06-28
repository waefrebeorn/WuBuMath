/*
 * wubu_riemannian_sgd.c -- Riemannian Enhanced SGD Optimizer
 *
 * Slermed from WuBuSpecTrans_v0.2.0_TotalStrategy.py (bytropix/draftPY/)
 * Faithful C11 implementation of RiemannianEnhancedSGD
 */

#include "wubu_riemannian_sgd.h"
#include "wubu_hyperbolic.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===================================================================
 * Initialize
 * =================================================================== */

void wubu_sgd_init(WubuSGD* opt, const WubuSGDConfig* config, WubuManifoldBinding manifold, int param_count) {
    opt->config = *config;
    opt->manifold = manifold;
    opt->current_lr = config->learning_rate;
    opt->current_momentum = config->momentum_factor;
    opt->step_count = 0;
    opt->param_state.N = param_count;
    opt->param_state.initialized = false;

    if (param_count > 0) {
        opt->param_state.momentum = (float*)calloc(param_count, sizeof(float));
        opt->param_state.initialized = true;
    } else {
        opt->param_state.momentum = NULL;
    }
}

/* ===================================================================
 * Free
 * =================================================================== */

void wubu_sgd_free(WubuSGD* opt) {
    if (opt->param_state.momentum) {
        free(opt->param_state.momentum);
        opt->param_state.momentum = NULL;
    }
    opt->param_state.initialized = false;
}

/* ===================================================================
 * Clip gradient to max norm
 * =================================================================== */

static void clip_gradient(float* grad_out, const float* grad_in, int N, float max_norm) {
    if (max_norm <= 0.0f || !isfinite(max_norm)) {
        memcpy(grad_out, grad_in, N * sizeof(float));
        return;
    }

    float norm_sq = 0.0f;
    for (int i = 0; i < N; i++) norm_sq += grad_in[i] * grad_in[i];
    float norm = sqrtf(norm_sq);

    if (norm > max_norm) {
        float scale = max_norm / (norm + WUBU_SGD_EPS);
        for (int i = 0; i < N; i++) grad_out[i] = grad_in[i] * scale;
    } else {
        memcpy(grad_out, grad_in, N * sizeof(float));
    }
}

/* ===================================================================
 * Check all finite
 * =================================================================== */

static bool all_finite(const float* x, int N) {
    for (int i = 0; i < N; i++) {
        if (!isfinite(x[i])) return false;
    }
    return true;
}

/* ===================================================================
 * Euclidean SGD step
 * =================================================================== */

void wubu_sgd_step_euclidean(WubuSGD* opt, float* param, const float* grad, int N) {
    if (!opt->param_state.initialized || opt->param_state.N != N) {
        /* Reallocate if size mismatch */
        if (opt->param_state.momentum) free(opt->param_state.momentum);
        opt->param_state.momentum = (float*)calloc(N, sizeof(float));
        opt->param_state.N = N;
        opt->param_state.initialized = true;
    }

    float lr = opt->current_lr;
    float mom = opt->current_momentum;
    float wd = opt->config.weight_decay;
    float max_gn = opt->config.max_grad_norm;

    /* Clip gradient */
    float clipped[512];
    bool heap_alloc = (N > 512);
    float* g;
    if (heap_alloc) g = (float*)malloc(N * sizeof(float));
    else g = clipped;

    clip_gradient(g, grad, N, max_gn);

    /* Add weight decay */
    if (wd != 0.0f) {
        for (int i = 0; i < N; i++) g[i] += wd * param[i];
    }

    /* Momentum update */
    float* buf = opt->param_state.momentum;
    if (mom != 0.0f) {
        for (int i = 0; i < N; i++) buf[i] = mom * buf[i] + g[i];
    } else {
        memcpy(buf, g, N * sizeof(float));
    }

    /* Check momentum finite */
    if (!all_finite(buf, N)) {
        memset(buf, 0, N * sizeof(float));
    }

    /* Parameter update: p = p - lr * buf */
    for (int i = 0; i < N; i++) param[i] -= lr * buf[i];

    /* Final stability check */
    if (!all_finite(param, N)) {
        for (int i = 0; i < N; i++) {
            if (!isfinite(param[i])) param[i] = 0.0f;
            if (param[i] > WUBU_SGD_EUCLIDEAN_CLAMP) param[i] = WUBU_SGD_EUCLIDEAN_CLAMP;
            if (param[i] < -WUBU_SGD_EUCLIDEAN_CLAMP) param[i] = -WUBU_SGD_EUCLIDEAN_CLAMP;
        }
        memset(buf, 0, N * sizeof(float));
    }

    opt->step_count++;
    if (heap_alloc) free(g);
}

/* ===================================================================
 * Hyperbolic SGD step (Poincare ball)
 * =================================================================== */

void wubu_sgd_step_hyperbolic(WubuSGD* opt, float* param, const float* grad, int N) {
    if (!opt->param_state.initialized || opt->param_state.N != N) {
        if (opt->param_state.momentum) free(opt->param_state.momentum);
        opt->param_state.momentum = (float*)calloc(N, sizeof(float));
        opt->param_state.N = N;
        opt->param_state.initialized = true;
    }

    float lr = opt->current_lr;
    float mom = opt->current_momentum;
    float wd = opt->config.weight_decay;
    float max_gn = opt->config.max_grad_norm;
    float c = opt->manifold.c;

    /* 1. Project parameter onto manifold */
    float p_proj[512];
    bool heap = (N > 512);
    float* pp;
    if (heap) pp = (float*)malloc(N * sizeof(float));
    else pp = p_proj;
    wubu_poincare_clip(pp, param, N, c);

    /* 2. Clip gradient */
    float g_clipped[512];
    float* gc;
    if (heap) gc = (float*)malloc(N * sizeof(float));
    else gc = g_clipped;
    clip_gradient(gc, grad, N, max_gn);

    /* 3. Weight decay */
    if (wd != 0.0f) {
        for (int i = 0; i < N; i++) gc[i] += wd * pp[i];
    }

    /* 4. Convert to Riemannian gradient */
    float rgrad[512];
    float* rg;
    if (heap) rg = (float*)malloc(N * sizeof(float));
    else rg = rgrad;
    wubu_egrad2rgrad(rg, pp, gc, N, c);

    /* Check finite */
    if (!all_finite(rg, N)) {
        memset(opt->param_state.momentum, 0, N * sizeof(float));
        if (heap) { free(pp); free(gc); free(rg); }
        return;
    }

    /* 5. Momentum update */
    float* buf = opt->param_state.momentum;
    if (mom != 0.0f) {
        for (int i = 0; i < N; i++) buf[i] = mom * buf[i] + rg[i];
    } else {
        memcpy(buf, rg, N * sizeof(float));
    }

    /* Check momentum finite */
    if (!all_finite(buf, N)) {
        memset(buf, 0, N * sizeof(float));
        if (heap) { free(pp); free(gc); free(rg); }
        return;
    }

    /* 6. Scale by -lr */
    for (int i = 0; i < N; i++) buf[i] = -lr * buf[i];

    /* 7. Exponential map: new_p = expmap(p_proj, tangent_vector) */
    /*    Using mobius_add(p_proj, tangent_vector) as retraction */
    float new_p[512];
    float* np;
    if (heap) np = (float*)malloc(N * sizeof(float));
    else np = new_p;
    wubu_mobius_add(np, pp, buf, N, c);

    /* 8. Project result */
    wubu_poincare_clip(np, np, N, c);

    /* 9. Check result finite + copy back */
    if (!all_finite(np, N)) {
        /* Fallback: reset to origin */
        float origin[512];
        for (int i = 0; i < N; i++) origin[i] = 0.0f;
        wubu_expmap(np, origin, N, c);
        memset(buf, 0, N * sizeof(float));
    }

    memcpy(param, np, N * sizeof(float));
    opt->step_count++;

    if (heap) { free(pp); free(gc); free(rg); free(np); }
}

/* ===================================================================
 * Utility functions
 * =================================================================== */

void wubu_sgd_zero_momentum(WubuSGD* opt) {
    if (opt->param_state.initialized && opt->param_state.momentum) {
        memset(opt->param_state.momentum, 0, opt->param_state.N * sizeof(float));
    }
}

void wubu_sgd_set_lr(WubuSGD* opt, float lr) {
    if (lr > 1e-8f && lr < 1.0f) {
        opt->current_lr = lr;
    }
}

void wubu_sgd_set_momentum(WubuSGD* opt, float momentum) {
    if (momentum >= 0.0f && momentum < 1.0f) {
        opt->current_momentum = momentum;
    }
}
