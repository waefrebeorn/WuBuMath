/*
 * wubu_flow_matching.c -- Flow Matching on Poincaré Ball for Hamilton Quaternion Point Clouds
 *
 * Implements the core math:
 *   1. Geodesic interpolation on Poincaré ball
 *   2. Target velocity field (tangent to geodesic)
 *   3. Velocity network (MLP with time positional encoding)
 *   4. Flow matching loss + training
 *   5. ODE inference for intermediate frame generation
 */

#include "wubu_flow_matching.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>

/* ===================================================================
 * PRNG (local to avoid dependency)
 * =================================================================== */

static uint32_t fm_rng_state;

static void fm_rng_seed(uint32_t seed) {
    fm_rng_state = seed;
}

static float fm_rng_float(void) {
    fm_rng_state = fm_rng_state * 1103515245u + 12345u;
    return (float)(fm_rng_state >> 16) / 65536.0f;
}

static float fm_rng_normal(void) {
    float u1 = fm_rng_float() + 1e-7f;
    float u2 = fm_rng_float();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/* ===================================================================
 * Positional Encoding for Time
 * =================================================================== */

static void positional_encode_t(float* pe, float t, int num_freqs) {
    for (int k = 0; k < num_freqs; k++) {
        float freq = powf(2.0f, (float)k) * 3.14159265f;
        pe[k] = sinf(t * freq);
        pe[num_freqs + k] = cosf(t * freq);
    }
}

/* ===================================================================
 * Initialize Velocity Network
 * =================================================================== */

int wubu_flow_init(WubuFlowMatching* model, const WubuFlowConfig* config, float c) {
    model->config = *config;
    model->c = c;
    model->step_count = 0;

    WubuVelocityNet* net = &model->velocity_net;
    net->latent_dim = config->latent_dim;
    net->hidden_dim = config->hidden_dim;
    net->num_freqs = config->num_freqs;
    net->input_dim = config->latent_dim + 2 * config->num_freqs;
    net->initialized = true;

    int in_dim = net->input_dim;
    int h_dim = net->hidden_dim;
    int out_dim = net->latent_dim;

    net->w1 = (float*)calloc((size_t)(h_dim * in_dim), sizeof(float));
    net->b1 = (float*)calloc((size_t)h_dim, sizeof(float));
    net->w2 = (float*)calloc((size_t)(h_dim * h_dim), sizeof(float));
    net->b2 = (float*)calloc((size_t)h_dim, sizeof(float));
    net->w_out = (float*)calloc((size_t)(out_dim * h_dim), sizeof(float));
    net->b_out = (float*)calloc((size_t)out_dim, sizeof(float));

    /* Xavier initialization */
    fm_rng_seed(42);
    float limit1 = sqrtf(6.0f / (float)(in_dim + h_dim));
    float limit2 = sqrtf(6.0f / (float)(h_dim + h_dim));
    float limit_out = sqrtf(6.0f / (float)(h_dim + out_dim));

    for (int i = 0; i < h_dim * in_dim; i++)
        net->w1[i] = (fm_rng_float() * 2.0f - 1.0f) * limit1;
    for (int i = 0; i < h_dim * h_dim; i++)
        net->w2[i] = (fm_rng_float() * 2.0f - 1.0f) * limit2;
    for (int i = 0; i < out_dim * h_dim; i++)
        net->w_out[i] = (fm_rng_float() * 2.0f - 1.0f) * limit_out;

    return 0;
}

void wubu_flow_free(WubuFlowMatching* model) {
    WubuVelocityNet* net = &model->velocity_net;
    free(net->w1); net->w1 = NULL;
    free(net->b1); net->b1 = NULL;
    free(net->w2); net->w2 = NULL;
    free(net->b2); net->b2 = NULL;
    free(net->w_out); net->w_out = NULL;
    free(net->b_out); net->b_out = NULL;
    net->initialized = false;
}

/* ===================================================================
 * Geodesic Interpolation on Poincaré Ball
 * μ_t = exp_{x_0}(t · log_{x_0}(x_1))
 * =================================================================== */

void wubu_flow_geodesic_interpolate(float* mu_t, const float* x_0, const float* x_1,
                                     float t, int N, int D, float c) {
    for (int i = 0; i < N; i++) {
        const float* p0 = x_0 + i * D;
        const float* p1 = x_1 + i * D;
        float* out = mu_t + i * D;

        /* Boundary conditions */
        if (t <= 0.0f) {
            memcpy(out, p0, D * sizeof(float));
            continue;
        }
        if (t >= 1.0f) {
            memcpy(out, p1, D * sizeof(float));
            continue;
        }

        /* Compute log_{x_0}(x_1) — tangent vector at x_0 pointing to x_1 */
        float tangent[64]; /* max dim */
        wubu_logmap(tangent, p1, D, c);

        /* Scale by t */
        for (int d = 0; d < D; d++) tangent[d] *= t;

        /* Exponential map from x_0 along scaled tangent:
         * exp_{x_0}(v) = x_0 (+)_c expmap_0(v)
         * (Mobius addition of base point with Euclidean expmap result) */
        float exp_v[64];
        wubu_expmap(exp_v, tangent, D, c);
        wubu_mobius_add(out, p0, exp_v, D, c);
    }
}

/* ===================================================================
 * Target Velocity: d/dt μ_t = parallel transport of log_{x_0}(x_1) to μ_t
 * For Poincaré ball, this is approximately log_{μ_t}(x_1) / (1-t) for t < 1
 * More precisely: v_target(t) = log_{μ_t}(x_1) * (1-t) + log_{μ_t}(x_0) * t
 * Simplified: v_target = log_{μ_t}(x_1) for the conditional flow
 * =================================================================== */

void wubu_flow_target_velocity(float* v_target, const float* x_0, const float* x_1,
                                float t, int N, int D, float c) {
    /* Compute μ_t = geodesic interpolation at time t */
    float* mu_t = (float*)calloc((size_t)(N * D), sizeof(float));
    wubu_flow_geodesic_interpolate(mu_t, x_0, x_1, t, N, D, c);

    /* Target velocity: d/dt μ_t
     * For conditional flow: v_t = x_1 - x_0 (straight line path)
     */
    for (int i = 0; i < N; i++) {
        /* Euclidean approximation: v = x_1 - x_0 */
        for (int d = 0; d < D; d++) {
            v_target[i * D + d] = x_1[i * D + d] - x_0[i * D + d];
        }
    }

    free(mu_t);
}

/* ===================================================================
 * Velocity Network Forward Pass
 * Input: [PE(t), x_tangent] → Output: predicted tangent vector
 * =================================================================== */

void wubu_flow_predict_velocity(WubuFlowMatching* model, float* v_pred,
                                 const float* x, float t, int N, int D) {
    WubuVelocityNet* net = &model->velocity_net;
    if (!net->initialized) return;

    int in_dim = net->input_dim;
    int h_dim = net->hidden_dim;

    float pe[128]; /* max 2*num_freqs */
    positional_encode_t(pe, t, net->num_freqs);

    float hidden1[256];
    float hidden2[256];

    for (int i = 0; i < N; i++) {
        /* Build input: [PE(t), x[i]] */
        float input[320]; /* max input_dim */
        memcpy(input, pe, 2 * net->num_freqs * sizeof(float));
        memcpy(input + 2 * net->num_freqs, x + i * D, D * sizeof(float));

        /* Layer 1: hidden1 = silu(w1 @ input + b1) */
        for (int j = 0; j < h_dim; j++) {
            float val = net->b1[j];
            for (int k = 0; k < in_dim; k++) {
                val += net->w1[j * in_dim + k] * input[k];
            }
            hidden1[j] = val / (1.0f + expf(-val)); /* SiLU */
        }

        /* Layer 2: hidden2 = silu(w2 @ hidden1 + b2) */
        for (int j = 0; j < h_dim; j++) {
            float val = net->b2[j];
            for (int k = 0; k < h_dim; k++) {
                val += net->w2[j * h_dim + k] * hidden1[k];
            }
            hidden2[j] = val / (1.0f + expf(-val)); /* SiLU */
        }

        /* Output: v = w_out @ hidden2 + b_out */
        for (int d = 0; d < D; d++) {
            float val = net->b_out[d];
            for (int k = 0; k < h_dim; k++) {
                val += net->w_out[d * h_dim + k] * hidden2[k];
            }
            v_pred[i * D + d] = val;
        }
    }
}

/* ===================================================================
 * Flow Matching Loss
 * =================================================================== */

float wubu_flow_compute_loss(WubuFlowMatching* model,
                              const float* x_0, const float* x_1,
                              int N, int D, float t) {
    /* Compute geodesic interpolation μ_t */
    float* mu_t = (float*)malloc((size_t)(N * D) * sizeof(float));
    wubu_flow_geodesic_interpolate(mu_t, x_0, x_1, t, N, D, model->c);

    /* Compute target velocity */
    float* v_target = (float*)malloc((size_t)(N * D) * sizeof(float));
    wubu_flow_target_velocity(v_target, x_0, x_1, t, N, D, model->c);

    /* Predict velocity */
    float* v_pred = (float*)malloc((size_t)(N * D) * sizeof(float));
    wubu_flow_predict_velocity(model, v_pred, mu_t, t, N, D);

    /* MSE loss */
    float loss = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float diff = v_pred[i] - v_target[i];
        loss += diff * diff;
    }
    loss /= (float)(N * D);

    free(mu_t);
    free(v_target);
    free(v_pred);
    return loss;
}

/* ===================================================================
 * Training Step — Simplified SGD on velocity network
 * =================================================================== */

float wubu_flow_train_step(WubuFlowMatching* model,
                            const float* key_latents, int num_keyframes,
                            int points_per_frame) {
    int D = model->config.latent_dim;
    int N = points_per_frame;

    /* Sample a random pair of consecutive key frames */
    const float* x_0 = NULL;
    const float* x_1 = NULL;
    if (num_keyframes < 2) {
        /* Not enough frames — use the same frame for x0 and x1 (degenerate case) */
        x_0 = key_latents;
        x_1 = key_latents;
    } else {
        int idx0 = (int)(fm_rng_float() * (float)(num_keyframes - 1));
        int idx1 = idx0 + 1;
        x_0 = key_latents + idx0 * N * D;
        x_1 = key_latents + idx1 * N * D;
    }

    /* Sample random time t ∈ [0.01, 0.99] to avoid boundary NaN */
    float t = 0.01f + fm_rng_float() * 0.98f;

    /* Compute loss */
    float loss = wubu_flow_compute_loss(model, x_0, x_1, N, D, t);

    /* Simplified gradient update — perturb weights slightly in direction of loss reduction
     * (Full backprop would require storing activations; for now use finite differences
     *  on a small subset of weights as a proof of concept) */
    WubuVelocityNet* net = &model->velocity_net;
    float lr = model->config.learning_rate;
    float eps = 1e-4f;

    /* Update a small random subset of output weights via finite difference */
    int num_updates = 20;
    for (int u = 0; u < num_updates; u++) {
        int wi = (int)(fm_rng_float() * (float)(D * net->hidden_dim));

        /* Perturb +eps */
        net->w_out[wi] += eps;
        float loss_plus = wubu_flow_compute_loss(model, x_0, x_1, N, D, t);

        /* Perturb -eps (from original) */
        net->w_out[wi] -= 2.0f * eps;
        float loss_minus = wubu_flow_compute_loss(model, x_0, x_1, N, D, t);

        /* Restore and apply gradient */
        net->w_out[wi] += eps; /* back to original */
        float grad = (loss_plus - loss_minus) / (2.0f * eps);
        net->w_out[wi] -= lr * grad;
    }

    model->step_count++;
    return loss;
}

/* ===================================================================
 * Inference — Generate Intermediate Frames via ODE Solve
 * Uses Euler integration: x_{k+1} = exp_{x_k}(h · v_θ(t_k, x_k))
 * =================================================================== */

float* wubu_flow_generate_intermediate(WubuFlowMatching* model,
                                        const float* x_0, const float* x_1,
                                        int N, int D,
                                        int num_intermediate) {
    float* output = (float*)calloc((size_t)(num_intermediate * N * D), sizeof(float));
    if (!output) return NULL;

    int steps = model->config.ode_steps;
    float h = 1.0f / (float)steps;

    float* current = (float*)malloc((size_t)(N * D) * sizeof(float));
    memcpy(current, x_0, (size_t)(N * D) * sizeof(float));

    float* velocity = (float*)malloc((size_t)(N * D) * sizeof(float));

    int output_idx = 0;

    for (int step = 0; step < steps; step++) {
        float t = (float)step * h;

        /* Predict velocity at current position */
        wubu_flow_predict_velocity(model, velocity, current, t, N, D);

        /* Euler step: x_{k+1} = exp_{x_k}(h * v) */
        for (int i = 0; i < N; i++) {
            float step_vec[64];
            for (int d = 0; d < D; d++) {
                step_vec[d] = h * velocity[i * D + d];
            }
            float new_pos[64];
            wubu_expmap(new_pos, step_vec, D, model->c);
            /* Mobius add: new = x_k (+) exp(h*v) */
            float result[64];
            wubu_mobius_add(result, current + i * D, new_pos, D, model->c);
            memcpy(current + i * D, result, D * sizeof(float));
        }

        /* Store intermediate frame at evenly spaced intervals */
        if ((step + 1) % (steps / num_intermediate) == 0 && output_idx < num_intermediate) {
            memcpy(output + output_idx * N * D, current, (size_t)(N * D) * sizeof(float));
            output_idx++;
        }
    }

    free(current);
    free(velocity);
    return output;
}
