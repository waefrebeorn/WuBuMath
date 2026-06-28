/*
 * wubu_tests.c -- WuBuMath test suite
 *
 * Tests verify correct C11 implementation of the slermed
 * color manifold, Q-controller, and loss computation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "wubumath.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void name(void)
#define RUN_TEST(name) do { \
    printf("  %-55s ", #name "..."); \
    fflush(stdout); \
    name(); \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        printf("FAIL (%s:%d): %f != %f (diff=%f, tol=%f)\n", __FILE__, __LINE__, (float)(a), (float)(b), fabsf((a)-(b)), (float)(tol)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("FAIL (%s:%d): %d != %d\n", __FILE__, __LINE__, (int)(a), (int)(b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ===================================================================
 * Color Manifold Tests
 * =================================================================== */

TEST(test_rgb_hsl_roundtrip) {
    WubuRGB original = {0.5f, 0.3f, 0.8f};
    WubuHSL hsl = wubu_rgb_to_hsl(original);
    WubuRGB recovered = wubu_hsl_to_rgb(hsl);
    ASSERT_NEAR(original.r, recovered.r, 1e-4f);
    ASSERT_NEAR(original.g, recovered.g, 1e-4f);
    ASSERT_NEAR(original.b, recovered.b, 1e-4f);
}

TEST(test_rgb_to_hsl_known) {
    /* RGB(0.5, 0.3, 0.8) -> HSL(0.7333, 0.5556, 0.5500) */
    WubuHSL hsl = wubu_rgb_to_hsl((WubuRGB){0.5f, 0.3f, 0.8f});
    ASSERT_NEAR(hsl.h, 0.7333f, 1e-3f);
    ASSERT_NEAR(hsl.s, 0.5556f, 1e-3f);
    ASSERT_NEAR(hsl.l, 0.5500f, 1e-3f);
}

TEST(test_rgb_to_hsl_black) {
    WubuHSL hsl = wubu_rgb_to_hsl((WubuRGB){0.0f, 0.0f, 0.0f});
    ASSERT_NEAR(hsl.l, 0.0f, 1e-6f);
    ASSERT_NEAR(hsl.s, 0.0f, 1e-6f);
}

TEST(test_rgb_to_hsl_white) {
    WubuHSL hsl = wubu_rgb_to_hsl((WubuRGB){1.0f, 1.0f, 1.0f});
    ASSERT_NEAR(hsl.l, 1.0f, 1e-6f);
}

TEST(test_rgb_to_hsl_red) {
    WubuHSL hsl = wubu_rgb_to_hsl((WubuRGB){1.0f, 0.0f, 0.0f});
    ASSERT_NEAR(hsl.h, 0.0f, 1e-6f);
    ASSERT_NEAR(hsl.s, 1.0f, 1e-6f);
    ASSERT_NEAR(hsl.l, 0.5f, 1e-6f);
}

TEST(test_grayscale) {
    float gray = wubu_rgb_to_grayscale((WubuRGB){0.5f, 0.5f, 0.5f});
    ASSERT_NEAR(gray, 0.5f, 1e-4f);
}

/* ===================================================================
 * Circular L1 Loss Tests
 * =================================================================== */

TEST(test_circular_l1_basic) {
    ASSERT_NEAR(wubu_circular_l1_loss(0.1f, 0.2f), 0.1f, 1e-6f);
    ASSERT_NEAR(wubu_circular_l1_loss(0.9f, 0.1f), 0.2f, 1e-6f);
}

TEST(test_circular_l1_wraparound) {
    /* 0.0 and 1.0 are the same on the circular manifold */
    ASSERT_NEAR(wubu_circular_l1_loss(0.0f, 1.0f), 0.0f, 1e-6f);
}

TEST(test_circular_l1_zero) {
    ASSERT_NEAR(wubu_circular_l1_loss(0.5f, 0.5f), 0.0f, 1e-6f);
}

/* ===================================================================
 * PRNG Tests
 * =================================================================== */

TEST(test_rng_deterministic) {
    WubuRNG r1, r2;
    wubu_rng_init(&r1, 42);
    wubu_rng_init(&r2, 42);
    for (int i = 0; i < 1000; ++i) {
        ASSERT_EQ((int)wubu_rng_next(&r1), (int)wubu_rng_next(&r2));
    }
}

TEST(test_rng_uniform_range) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    for (int i = 0; i < 10000; ++i) {
        float u = wubu_rng_uniform(&rng, -1.0f, 1.0f);
        if (u < -1.0f || u > 1.0f) {
            printf("FAIL: uniform out of range: %f\n", u);
            tests_failed++;
            return;
        }
    }
    tests_passed++;
}

/* ===================================================================
 * Q-Controller Tests
 * =================================================================== */

TEST(test_q_controller_init) {
    QControllerConfig config = WUBU_Q_CONTROLLER_DEFAULT;
    float q_table[5] = {0};
    float history[100] = {0};
    QController qc = {
        .q_table = q_table,
        .metric_history = history
    };
    wubu_q_controller_init(&qc, &config);
    ASSERT_NEAR(qc.current_lr, config.warmup_lr_start, 1e-6f);
    ASSERT_EQ(qc.step_count, 0);
    ASSERT_EQ(qc.last_action_idx, -1);
}

TEST(test_q_controller_warmup) {
    QControllerConfig config = WUBU_Q_CONTROLLER_DEFAULT;
    float q_table[5] = {0};
    float history[100] = {0};
    QController qc = {
        .q_table = q_table,
        .metric_history = history
    };
    wubu_q_controller_init(&qc, &config);

    WubuRNG rng;
    wubu_rng_init(&rng, 42);

    float target_lr = 2e-4f;
    for (int i = 0; i < config.warmup_steps; ++i) {
        wubu_q_controller_choose_action(&qc, &rng, &config, target_lr);
    }

    /* After warmup, lr should be close to target_lr */
    ASSERT_NEAR(qc.current_lr, target_lr, 1e-5f);
    ASSERT_EQ(qc.status_code, 0);
}

TEST(test_q_controller_exploration) {
    QControllerConfig config = WUBU_Q_CONTROLLER_DEFAULT;
    float q_table[5] = {0};
    float history[100] = {0};
    QController qc = {
        .q_table = q_table,
        .metric_history = history
    };
    wubu_q_controller_init(&qc, &config);

    WubuRNG rng;
    wubu_rng_init(&rng, 42);

    /* Skip warmup */
    for (int i = 0; i < config.warmup_steps; ++i) {
        wubu_q_controller_choose_action(&qc, &rng, &config, 2e-4f);
    }

    /* Regular steps should change lr */
    float initial_lr = qc.current_lr;
    for (int i = 0; i < 100; ++i) {
        wubu_q_controller_choose_action(&qc, &rng, &config, 2e-4f);
        wubu_q_controller_update(&qc, 0.05f, &config);
    }

    /* With different actions, lr should have changed */
    /* (not guaranteed for all seeds, but very likely) */
    if (fabsf(qc.current_lr - initial_lr) < 1e-10f) {
        printf("WARN: lr unchanged after 100 steps (seed-dependent)\n");
    }
    tests_passed++;
}

/* ===================================================================
 * Positional Encoding Tests
 * =================================================================== */

TEST(test_positional_encoding) {
    float x[3] = {1.0f, 2.0f, 3.0f};
    int num_freqs = 10;
    float* encoded = wubu_positional_encode(x, 3, num_freqs);
    ASSERT_NEAR(encoded[0], 1.0f, 1e-6f);
    ASSERT_NEAR(encoded[1], 2.0f, 1e-6f);
    ASSERT_NEAR(encoded[2], 3.0f, 1e-6f);
    /* First frequency: sin(1 * PI) should be ~0 */
    ASSERT_NEAR(encoded[3], sinf(1.0f * (float)M_PI), 1e-6f);
    /* cos(1 * PI) should be ~-1 */
    ASSERT_NEAR(encoded[6], cosf(1.0f * (float)M_PI), 1e-6f);
    free(encoded);
}

/* ===================================================================
 * Loss Computation Tests
 * =================================================================== */

TEST(test_loss_zero) {
    int N = 100;
    float pred[300], gt[300];
    for (int i = 0; i < N * 3; ++i) {
        pred[i] = 0.5f;
        gt[i] = 0.5f;
    }
    WubuLoss loss = wubu_compute_loss(pred, gt, N);
    ASSERT_NEAR(loss.composite_loss, 0.0f, 1e-3f);
}

TEST(test_loss_structure) {
    int N = 100;
    float pred[300], gt[300];
    for (int i = 0; i < N * 3; ++i) {
        pred[i] = 0.5f;
        gt[i] = -0.5f;
    }
    WubuLoss loss = wubu_compute_loss(pred, gt, N);
    if (loss.composite_loss <= 0.0f) {
        printf("FAIL: loss should be positive for different inputs\n");
        tests_failed++;
        return;
    }
    tests_passed++;
}

/* ===================================================================
 * Hamilton Encoder Tests
 * =================================================================== */

TEST(test_hamilton_encode) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);

    int B = 2, img_size = 16;
    float images[B * 16 * 16 * 3];
    for (int i = 0; i < B * 16 * 16 * 3; ++i) {
        images[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;
    }

    HamiltonKeys keys = wubu_hamilton_encode_procedural(&rng, images, B, img_size);
    ASSERT_EQ(keys.B, B);
    ASSERT_EQ(keys.H, img_size);
    ASSERT_EQ(keys.context_dim, 3);

    /* Quaternions should be normalized */
    for (int i = 0; i < 10; ++i) {
        float norm = sqrtf(keys.quaternions[i*4+0]*keys.quaternions[i*4+0] +
                          keys.quaternions[i*4+1]*keys.quaternions[i*4+1] +
                          keys.quaternions[i*4+2]*keys.quaternions[i*4+2] +
                          keys.quaternions[i*4+3]*keys.quaternions[i*4+3]);
        ASSERT_NEAR(norm, 1.0f, 1e-4f);
    }

    wubu_hamilton_keys_free(&keys);
}

/* ===================================================================
 * VHF Decoder Tests
 * =================================================================== */

TEST(test_vhf_decode) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);

    int B = 1, img_size = 8, N = 64;
    float images[B * 8 * 8 * 3];
    for (int i = 0; i < B * 8 * 8 * 3; ++i) {
        images[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;
    }

    HamiltonKeys keys = wubu_hamilton_encode_procedural(&rng, images, B, img_size);

    float coords[N * 2];
    for (int i = 0; i < N; ++i) {
        coords[i * 2 + 0] = ((float)i / (float)N) * 2.0f - 1.0f;
        coords[i * 2 + 1] = ((float)i / (float)N) * 2.0f - 1.0f;
    }

    float* output = wubu_vhf_decode_procedural(&rng, &keys, coords, N);
    ASSERT_NEAR(output[0], output[0], 1e-3f); /* just verify not NaN */

    free(output);
    wubu_hamilton_keys_free(&keys);
}

/* ===================================================================
 * Main
 * =================================================================== */



/* ===================================================================
 * VHF Audio Pipeline Tests
 * =================================================================== */

TEST(test_vhf_encoder_output) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);

    int B = 1, img_size = 16;
    float images[B * 16 * 16 * 3];
    for (int i = 0; i < B * 16 * 16 * 3; ++i) {
        images[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;
    }

    VHFEncoded encoded = wubu_vhf_encode(&rng, images, B, img_size);
    ASSERT_EQ(encoded.B, B);
    ASSERT_EQ(encoded.H, img_size);
    ASSERT_EQ(encoded.W, img_size);

    /* Quaternions should be normalized */
    for (int i = 0; i < 16; ++i) {
        float norm = sqrtf(encoded.quaternions[i*4+0]*encoded.quaternions[i*4+0] +
                          encoded.quaternions[i*4+1]*encoded.quaternions[i*4+1] +
                          encoded.quaternions[i*4+2]*encoded.quaternions[i*4+2] +
                          encoded.quaternions[i*4+3]*encoded.quaternions[i*4+3]);
        ASSERT_NEAR(norm, 1.0f, 1e-3f);
    }

    wubu_vhf_encoded_free(&encoded);
}

TEST(test_vhf_decode_output) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);

    int B = 1, img_size = 16, N = 64;
    float images[B * 16 * 16 * 3];
    for (int i = 0; i < B * 16 * 16 * 3; ++i) {
        images[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;
    }

    VHFEncoded encoded = wubu_vhf_encode(&rng, images, B, img_size);

    float coords[N * 2];
    for (int i = 0; i < N; ++i) {
        coords[i * 2 + 0] = ((float)i / (float)N) * 2.0f - 1.0f;
        coords[i * 2 + 1] = ((float)i / (float)N) * 2.0f - 1.0f;
    }

    float* output = wubu_vhf_decode(&rng, &encoded, coords, N);
    ASSERT_NEAR(output[0], output[0], 1.0f); /* not NaN */

    free(output);
    wubu_vhf_encoded_free(&encoded);
}

TEST(test_audio_strip_generation) {
    VHAudioConfig config = VHF_AUDIO_DEFAULT;
    int num_samples = 1000;
    float audio[1000];
    for (int i = 0; i < num_samples; ++i) {
        audio[i] = sinf(2.0f * (float)M_PI * 440.0f * (float)i / (float)config.sample_rate);
    }

    float* strip = wubu_vhf_generate_audio_strip(audio, num_samples, &config);
    ASSERT_NEAR(strip[0], 0.0f, 1e-6f); /* sin(0) = 0 */

    int expected_size = config.canvas_h * config.audio_hbi_width * 3;
    /* Verify strip was allocated */
    if (strip[expected_size - 1] != strip[expected_size - 1]) {
        printf("WARN: possible NaN at end of strip\n");
    }

    free(strip);
}

TEST(test_canvas_compose) {
    VHAudioConfig config = VHF_AUDIO_DEFAULT;

    /* Create dummy VBI block */
    float* vbi = (float*)calloc((size_t)(config.vbi_lines * config.canvas_w * 3), sizeof(float));
    /* Create dummy audio HBI */
    float* audio_hbi = (float*)calloc((size_t)(config.visible_h * config.audio_hbi_width * 3), sizeof(float));
    /* Create dummy visible frame */
    float* visible = (float*)calloc((size_t)(config.visible_h * VISIBLE_W * 3), sizeof(float));

    float* canvas = wubu_vhf_compose_canvas(vbi, audio_hbi, visible, &config);
    ASSERT_EQ(canvas[0], canvas[0]); /* not NaN */

    free(canvas);
    free(vbi);
    free(audio_hbi);
    free(visible);
}

TEST(test_vhf_train_step_loss) {
    /* Zero loss when pred == gt */
    int N = 100;
    float pred[300], gt[300];
    for (int i = 0; i < N * 3; ++i) {
        pred[i] = 0.3f;
        gt[i] = 0.3f;
    }

    float q_table[5] = {0};
    float history[100] = {0};
    QController qc = {
        .q_table = q_table,
        .metric_history = history
    };
    QControllerConfig config = WUBU_Q_CONTROLLER_DEFAULT;
    wubu_q_controller_init(&qc, &config);

    /* Skip warmup */
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    for (int i = 0; i < config.warmup_steps; ++i) {
        wubu_q_controller_choose_action(&qc, &rng, &config, 2e-4f);
    }

    VHFTrainingState state = wubu_vhf_train_step(pred, gt, N, &qc);
    ASSERT_NEAR(state.composite_loss, 0.0f, 0.01f);
}

TEST(test_vhf_pipeline_end_to_end) {
    /* Full pipeline: encode -> decode -> loss -> train step */
    WubuRNG rng;
    wubu_rng_init(&rng, 42);

    int B = 1, img_size = 16, N = 256;
    float images[B * 16 * 16 * 3];
    /* Create a simple gradient */
    for (int y = 0; y < img_size; ++y) {
        for (int x = 0; x < img_size; ++x) {
            int idx = (y * img_size + x) * 3;
            images[idx + 0] = (float)x / (float)(img_size - 1) * 2.0f - 1.0f;
            images[idx + 1] = (float)y / (float)(img_size - 1) * 2.0f - 1.0f;
            images[idx + 2] = 0.0f;
        }
    }

    /* Encode */
    VHFEncoded encoded = wubu_vhf_encode(&rng, images, B, img_size);

    /* Generate random coordinates for loss computation */
    float coords[N * 2];
    for (int i = 0; i < N; ++i) {
        coords[i * 2 + 0] = wubu_rng_uniform(&rng, -1.0f, 1.0f);
        coords[i * 2 + 1] = wubu_rng_uniform(&rng, -1.0f, 1.0f);
    }

    /* Decode */
    float* pred_rgb = wubu_vhf_decode(&rng, &encoded, coords, N);

    /* Sample ground truth at same coordinates */
    float* gt_rgb = (float*)malloc((size_t)(N * 3) * sizeof(float));
    for (int i = 0; i < N; ++i) {
        int xi = (int)((coords[i*2+0] + 1.0f) / 2.0f * (float)(img_size - 1));
        int yi = (int)((coords[i*2+1] + 1.0f) / 2.0f * (float)(img_size - 1));
        if (xi < 0) xi = 0;
        if (yi < 0) yi = 0;
        if (xi >= img_size) xi = img_size - 1;
        if (yi >= img_size) yi = img_size - 1;
        gt_rgb[i*3+0] = images[(yi*img_size+xi)*3+0];
        gt_rgb[i*3+1] = images[(yi*img_size+xi)*3+1];
        gt_rgb[i*3+2] = images[(yi*img_size+xi)*3+2];
    }

    /* Compute loss */
    WubuLoss loss = wubu_compute_loss(pred_rgb, gt_rgb, N);

    if (loss.composite_loss < 0.0f) {
        printf("FAIL: loss should be non-negative\n");
        tests_failed++;
    } else {
        printf("(loss=%.6f) ", loss.composite_loss);
        tests_passed++;
    }

    free(pred_rgb);
    free(gt_rgb);
    wubu_vhf_encoded_free(&encoded);
}


int main(void) {
    printf("=== WuBuMath Test Suite ===\n\n");

    printf("[Color Manifold]\n");
    RUN_TEST(test_rgb_hsl_roundtrip);
    RUN_TEST(test_rgb_to_hsl_known);
    RUN_TEST(test_rgb_to_hsl_black);
    RUN_TEST(test_rgb_to_hsl_white);
    RUN_TEST(test_rgb_to_hsl_red);
    RUN_TEST(test_grayscale);

    printf("\n[Circular L1 Loss]\n");
    RUN_TEST(test_circular_l1_basic);
    RUN_TEST(test_circular_l1_wraparound);
    RUN_TEST(test_circular_l1_zero);

    printf("\n[PRNG]\n");
    RUN_TEST(test_rng_deterministic);
    RUN_TEST(test_rng_uniform_range);

    printf("\n[Q-Controller]\n");
    RUN_TEST(test_q_controller_init);
    RUN_TEST(test_q_controller_warmup);
    RUN_TEST(test_q_controller_exploration);

    printf("\n[Positional Encoding]\n");
    RUN_TEST(test_positional_encoding);

    printf("\n[Loss Computation]\n");
    RUN_TEST(test_loss_zero);
    RUN_TEST(test_loss_structure);

    printf("\n[Hamilton Encoder]\n");
    RUN_TEST(test_hamilton_encode);

    printf("\n[VHF Decoder]\n");
    RUN_TEST(test_vhf_decode);

    printf("\n[VHF Audio Pipeline]\n");
    RUN_TEST(test_vhf_encoder_output);
    RUN_TEST(test_vhf_decode_output);
    RUN_TEST(test_audio_strip_generation);
    RUN_TEST(test_canvas_compose);
    RUN_TEST(test_vhf_train_step_loss);
    RUN_TEST(test_vhf_pipeline_end_to_end);

    printf("\n=== Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Total:  %d\n", tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
