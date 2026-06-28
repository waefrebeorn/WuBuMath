/*
 * wubu_flow_matching.h -- Flow Matching on Poincaré Ball for Hamilton Quaternion Point Clouds
 *
 * Slermed from first principles: flow matching for temporal coherence between
 * key-frame Hamilton latent spaces.
 *
 * Architecture:
 *   Key frames: RGB → Hamilton quaternion latent (Poincaré ball points)
 *   Motion: Flow matching learns velocity field v_t(x) on tangent space T_x M
 *   Interpolation: probability path μ_t = (1-t)·x_0 + t·x_1 on manifold (geodesic)
 *   Training: regress velocity field to match geodesic tangent vectors
 *   Inference: ODE solve from x_0 to generate intermediate frames
 *
 * Math:
 *   - x_0, x_1: quaternion latent points on Poincaré ball (from key frames)
 *   - Geodesic interpolation: μ_t = exp_{x_0}(t · log_{x_0}(x_1))
 *   - Target velocity: d/dt μ_t = parallel transport of log_{x_0}(x_1) along geodesic
 *   - Learned velocity: v_θ(t, x) — MLP on tangent space at x
 *   - Loss: E[||v_θ(t, μ_t) - d/dt μ_t||²]
 *   - Inference: dx/dt = v_θ(t, x), x(0) ~ N(0,σ²I), solve ODE to x(1)
 */

#ifndef WUBU_FLOW_MATCHING_H
#define WUBU_FLOW_MATCHING_H

#include "wubu_hyperbolic.h"
#include "wubu_quaternion.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Flow Matching Configuration
 * =================================================================== */

typedef struct {
    int latent_dim;       /* dimension of each latent point (4 for quaternion) */
    int hidden_dim;       /* velocity network hidden dimension */
    int num_layers;       /* velocity network depth */
    int num_freqs;        /* positional encoding frequencies for time t */
    float sigma_min;      /* minimum noise for probability path */
    float sigma_max;      /* maximum noise for probability path */
    float learning_rate;  /* optimizer LR */
    int batch_size;       /* training batch size */
    int ode_steps;        /* number of ODE integration steps for inference */
} WubuFlowConfig;

/* ===================================================================
 * Velocity Network — MLP on Tangent Space
 * =================================================================== */

typedef struct {
    /* Weights */
    float* w1;            /* [hidden_dim, latent_dim + 2*num_freqs] — input: [PE(t), x_tangent] */
    float* b1;            /* [hidden_dim] */
    float* w2;            /* [hidden_dim, hidden_dim] */
    float* b2;            /* [hidden_dim] */
    float* w_out;         /* [latent_dim, hidden_dim] — output: tangent vector */
    float* b_out;         /* [latent_dim] */

    /* Config */
    int latent_dim;
    int hidden_dim;
    int num_freqs;
    int input_dim;        /* latent_dim + 2*num_freqs */
    bool initialized;
} WubuVelocityNet;

/* ===================================================================
 * Flow Matching Model
 * =================================================================== */

typedef struct {
    WubuVelocityNet velocity_net;
    WubuFlowConfig config;
    float c;              /* Poincaré ball curvature */
    int step_count;
} WubuFlowMatching;

/* ===================================================================
 * Initialize / Free
 * =================================================================== */

int wubu_flow_init(WubuFlowMatching* model, const WubuFlowConfig* config, float c);
void wubu_flow_free(WubuFlowMatching* model);

/* ===================================================================
 * Probability Path — Geodesic Interpolation on Poincaré Ball
 * =================================================================== */

/* Compute geodesic interpolation: μ_t = exp_{x_0}(t · log_{x_0}(x_1)) */
void wubu_flow_geodesic_interpolate(float* mu_t, const float* x_0, const float* x_1,
                                     float t, int N, int D, float c);

/* Compute target velocity: d/dt μ_t at time t */
void wubu_flow_target_velocity(float* v_target, const float* x_0, const float* x_1,
                                float t, int N, int D, float c);

/* ===================================================================
 * Velocity Network Forward
 * =================================================================== */

/* Predict velocity at time t, position x (on tangent space at x) */
void wubu_flow_predict_velocity(WubuFlowMatching* model, float* v_pred,
                                 const float* x, float t, int N, int D);

/* ===================================================================
 * Training
 * =================================================================== */

/* Single training step: sample pairs (x_0, x_1) from key frame latents,
 * compute flow matching loss, update velocity network via SGD */
float wubu_flow_train_step(WubuFlowMatching* model,
                            const float* key_latents, int num_keyframes,
                            int points_per_frame);

/* ===================================================================
 * Inference — Generate Intermediate Frames
 * =================================================================== */

/* Generate intermediate frame between two key frames using ODE solve */
/* x_0: source latent [N, D], x_1: target latent [N, D] */
/* num_intermediate: number of frames to generate between key frames */
/* Output: [num_intermediate, N, D] generated latents */
float* wubu_flow_generate_intermediate(WubuFlowMatching* model,
                                        const float* x_0, const float* x_1,
                                        int N, int D,
                                        int num_intermediate);

/* ===================================================================
 * Loss Computation
 * =================================================================== */

/* Flow matching loss: E[||v_θ(t, μ_t) - d/dt μ_t||²] */
float wubu_flow_compute_loss(WubuFlowMatching* model,
                              const float* x_0, const float* x_1,
                              int N, int D, float t);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_FLOW_MATCHING_H */
