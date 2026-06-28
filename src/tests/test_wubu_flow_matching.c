/*
 * test_wubu_flow_matching.c -- Tests for flow matching on Poincaré ball
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "wubu_flow_matching.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define RUN_TEST(name) do { \
    printf("  %-55s ", #name "..."); fflush(stdout); \
    name(); printf("PASS\n"); tests_passed++; \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (fabsf((a)-(b)) > (tol)) { \
        printf("FAIL: %s=%g expected %g (tol=%g)\n", #a, (a), (b), (tol)); \
        tests_failed++; return; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("FAIL: %s\n", #cond); \
        tests_failed++; return; \
    } \
} while(0)

/* ===================================================================
 * Geodesic interpolation tests
 * =================================================================== */

static void test_geodesic_endpoints(void) {
    /* At t=0, μ_0 = x_0; at t=1, μ_1 = x_1 */
    float x0[] = {0.1f, 0.2f, 0.0f, 0.0f};
    float x1[] = {0.3f, -0.1f, 0.2f, 0.0f};
    float mu[4];

    wubu_flow_geodesic_interpolate(mu, x0, x1, 0.0f, 1, 4, 1.0f);
    ASSERT_NEAR(mu[0], x0[0], 1e-3f);
    ASSERT_NEAR(mu[1], x0[1], 1e-3f);

    wubu_flow_geodesic_interpolate(mu, x0, x1, 1.0f, 1, 4, 1.0f);
    ASSERT_NEAR(mu[0], x1[0], 1e-3f);
    ASSERT_NEAR(mu[1], x1[1], 1e-3f);
}

static void test_geodesic_midpoint(void) {
    /* At t=0.5, μ should be between x_0 and x_1 */
    float x0[] = {0.1f, 0.0f, 0.0f, 0.0f};
    float x1[] = {0.5f, 0.0f, 0.0f, 0.0f};
    float mu[4];

    wubu_flow_geodesic_interpolate(mu, x0, x1, 0.5f, 1, 4, 1.0f);
    /* Midpoint should be between 0.1 and 0.5 */
    ASSERT_TRUE(mu[0] > 0.1f && mu[0] < 0.5f);
}

static void test_geodesic_stays_on_manifold(void) {
    /* All interpolated points should be inside Poincaré ball */
    float x0[] = {0.3f, 0.2f, 0.1f, 0.0f};
    float x1[] = {-0.2f, 0.4f, -0.1f, 0.0f};
    float mu[4];

    for (int step = 0; step <= 10; step++) {
        float t = (float)step / 10.0f;
        wubu_flow_geodesic_interpolate(mu, x0, x1, t, 1, 4, 1.0f);
        float norm = sqrtf(mu[0]*mu[0] + mu[1]*mu[1] + mu[2]*mu[2] + mu[3]*mu[3]);
        ASSERT_TRUE(norm < 1.0f / sqrtf(1.0f) + 0.01f);
    }
}

/* ===================================================================
 * Velocity network tests
 * =================================================================== */

static void test_velocity_net_init(void) {
    WubuFlowConfig config = {
        .latent_dim = 4,
        .hidden_dim = 32,
        .num_layers = 2,
        .num_freqs = 8,
        .sigma_min = 0.001f,
        .sigma_max = 1.0f,
        .learning_rate = 1e-3f,
        .batch_size = 16,
        .ode_steps = 50
    };
    WubuFlowMatching model;
    int rc = wubu_flow_init(&model, &config, 1.0f);
    ASSERT_TRUE(rc == 0);
    ASSERT_TRUE(model.velocity_net.initialized);

    wubu_flow_free(&model);
}

static void test_velocity_prediction_finite(void) {
    WubuFlowConfig config = {
        .latent_dim = 4, .hidden_dim = 32, .num_layers = 2,
        .num_freqs = 8, .sigma_min = 0.001f, .sigma_max = 1.0f,
        .learning_rate = 1e-3f, .batch_size = 16, .ode_steps = 50
    };
    WubuFlowMatching model;
    wubu_flow_init(&model, &config, 1.0f);

    float x[] = {0.1f, 0.2f, 0.0f, 0.0f};
    float v_pred[4];
    wubu_flow_predict_velocity(&model, v_pred, x, 0.5f, 1, 4);

    for (int d = 0; d < 4; d++) {
        ASSERT_TRUE(isfinite(v_pred[d]));
    }

    wubu_flow_free(&model);
}

/* ===================================================================
 * Flow matching loss test
 * =================================================================== */

static void test_flow_loss_positive(void) {
    WubuFlowConfig config = {
        .latent_dim = 4, .hidden_dim = 32, .num_layers = 2,
        .num_freqs = 8, .sigma_min = 0.001f, .sigma_max = 1.0f,
        .learning_rate = 1e-3f, .batch_size = 16, .ode_steps = 50
    };
    WubuFlowMatching model;
    wubu_flow_init(&model, &config, 1.0f);

    float x0[] = {0.1f, 0.0f, 0.0f, 0.0f};
    float x1[] = {0.5f, 0.1f, 0.0f, 0.0f};

    float loss = wubu_flow_compute_loss(&model, x0, x1, 1, 4, 0.5f);
    ASSERT_TRUE(loss >= 0.0f);
    ASSERT_TRUE(isfinite(loss));

    wubu_flow_free(&model);
}

/* ===================================================================
 * Training step test
 * =================================================================== */

static void test_training_step(void) {
    WubuFlowConfig config = {
        .latent_dim = 4, .hidden_dim = 32, .num_layers = 2,
        .num_freqs = 8, .sigma_min = 0.001f, .sigma_max = 1.0f,
        .learning_rate = 1e-4f, .batch_size = 16, .ode_steps = 50
    };
    WubuFlowMatching model;
    wubu_flow_init(&model, &config, 1.0f);

    /* Create 4 key frames with 8 points each */
    int num_frames = 4;
    int N = 8;
    int D = 4;
    float key_latents[4 * 8 * 4];
    for (int f = 0; f < num_frames; f++) {
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < D; d++) {
                key_latents[(f * N + i) * D + d] = ((float)(f + i + d) / 20.0f) - 0.3f;
            }
        }
    }

    float loss = wubu_flow_train_step(&model, key_latents, num_frames, N);
    ASSERT_TRUE(isfinite(loss));
    ASSERT_TRUE(loss >= 0.0f);

    wubu_flow_free(&model);
}

/* ===================================================================
 * Inference test
 * =================================================================== */

static void test_inference_generates_output(void) {
    WubuFlowConfig config = {
        .latent_dim = 4, .hidden_dim = 32, .num_layers = 2,
        .num_freqs = 8, .sigma_min = 0.001f, .sigma_max = 1.0f,
        .learning_rate = 1e-4f, .batch_size = 16, .ode_steps = 20
    };
    WubuFlowMatching model;
    wubu_flow_init(&model, &config, 1.0f);

    int N = 4, D = 4;
    float x0[] = {0.1f, 0.0f, 0.0f, 0.0f,
                  0.2f, 0.1f, 0.0f, 0.0f,
                  0.0f, 0.3f, 0.1f, 0.0f,
                  0.1f, 0.1f, 0.2f, 0.0f};
    float x1[] = {0.4f, 0.1f, 0.0f, 0.0f,
                  0.5f, 0.2f, 0.1f, 0.0f,
                  0.3f, 0.4f, 0.2f, 0.0f,
                  0.4f, 0.3f, 0.3f, 0.1f};

    float* intermediate = wubu_flow_generate_intermediate(&model, x0, x1, N, D, 3);
    ASSERT_TRUE(intermediate != NULL);

    /* Check all finite */
    for (int i = 0; i < 3 * N * D; i++) {
        ASSERT_TRUE(isfinite(intermediate[i]));
    }

    free(intermediate);
    wubu_flow_free(&model);
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("=== WuBuMath Flow Matching Tests ===\n\n");

    RUN_TEST(test_geodesic_endpoints);
    RUN_TEST(test_geodesic_midpoint);
    RUN_TEST(test_geodesic_stays_on_manifold);
    RUN_TEST(test_velocity_net_init);
    RUN_TEST(test_velocity_prediction_finite);
    RUN_TEST(test_flow_loss_positive);
    RUN_TEST(test_training_step);
    RUN_TEST(test_inference_generates_output);

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
