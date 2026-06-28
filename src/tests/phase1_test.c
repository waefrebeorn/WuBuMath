/*
 * phase1_test.c -- Phase 1 Symmetric Encoder test suite
 *
 * Tests verify correct C11 implementation of:
 *   - Complex mask generation (ellipse, rect, boolean ops)
 *   - Synthetic RGBA texture batch
 *   - Color pattern generation (32 frames)
 *   - Moving shape patterns (animated)
 *   - VHF tone pipeline (8 tones)
 *   - FiLM layer
 *   - Full pipeline (encode -> observe -> decode)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
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

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("FAIL (%s:%d): assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ===================================================================
 * Complex Mask Generation Tests
 * =================================================================== */

TEST(test_ellipse_mask_dimensions) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int H = 32, W = 32;
    float* mask = wubu_phase1_generate_ellipse_mask(&rng, H, W);
    ASSERT_TRUE(mask != NULL);
    /* Check mask is in [0,1] range */
    for (int i = 0; i < H * W; ++i) {
        if (mask[i] < -0.01f || mask[i] > 1.01f) {
            printf("FAIL: mask[%d] = %f out of range\n", i, mask[i]);
            tests_failed++;
            free(mask);
            return;
        }
    }
    free(mask);
    tests_passed++;
}

TEST(test_ellipse_mask_nonzero) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int H = 32, W = 32;
    float* mask = wubu_phase1_generate_ellipse_mask(&rng, H, W);
    ASSERT_TRUE(mask != NULL);
    /* Ellipse mask should have some non-zero values */
    int nonzero = 0;
    for (int i = 0; i < H * W; ++i) {
        if (mask[i] > 0.01f) nonzero++;
    }
    ASSERT_TRUE(nonzero > 0);
    free(mask);
    tests_passed++;
}

TEST(test_rect_mask_dimensions) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int H = 32, W = 32;
    float* mask = wubu_phase1_generate_rect_mask(&rng, H, W);
    ASSERT_TRUE(mask != NULL);
    /* Check mask is in [0,1] range */
    for (int i = 0; i < H * W; ++i) {
        if (mask[i] < -0.01f || mask[i] > 1.01f) {
            printf("FAIL: mask[%d] = %f out of range\n", i, mask[i]);
            tests_failed++;
            free(mask);
            return;
        }
    }
    free(mask);
    tests_passed++;
}

TEST(test_rect_mask_nonzero) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int H = 32, W = 32;
    float* mask = wubu_phase1_generate_rect_mask(&rng, H, W);
    ASSERT_TRUE(mask != NULL);
    int nonzero = 0;
    for (int i = 0; i < H * W; ++i) {
        if (mask[i] > 0.01f) nonzero++;
    }
    ASSERT_TRUE(nonzero > 0);
    free(mask);
    tests_passed++;
}

TEST(test_complex_mask_dimensions) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int H = 32, W = 32;
    float* mask = wubu_phase1_generate_complex_mask(&rng, H, W, 3, 7);
    ASSERT_TRUE(mask != NULL);
    /* After blur, values should be in [0,1] */
    for (int i = 0; i < H * W; ++i) {
        if (mask[i] < -0.01f || mask[i] > 1.01f) {
            printf("FAIL: complex mask[%d] = %f out of range\n", i, mask[i]);
            tests_failed++;
            free(mask);
            return;
        }
    }
    free(mask);
    tests_passed++;
}

TEST(test_complex_mask_deterministic) {
    WubuRNG rng1, rng2;
    wubu_rng_init(&rng1, 42);
    wubu_rng_init(&rng2, 42);
    int H = 32, W = 32;
    float* m1 = wubu_phase1_generate_complex_mask(&rng1, H, W, 3, 7);
    float* m2 = wubu_phase1_generate_complex_mask(&rng2, H, W, 3, 7);
    ASSERT_TRUE(m1 != NULL && m2 != NULL);
    for (int i = 0; i < H * W; ++i) {
        ASSERT_NEAR(m1[i], m2[i], 1e-6f);
    }
    free(m1);
    free(m2);
}

/* ===================================================================
 * Synthetic RGBA Texture Batch Tests
 * =================================================================== */

TEST(test_rgba_texture_batch_dimensions) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int B = 2, H = 16, W = 16;
    float fg[B * H * W * 3];
    float bg[B * H * W * 3];
    for (int i = 0; i < B * H * W * 3; ++i) {
        fg[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;
        bg[i] = (float)((i + 50) % 100) / 100.0f * 2.0f - 1.0f;
    }
    Phase1TextureBatch batch = wubu_phase1_create_synthetic_rgba_texture_batch(
        &rng, fg, bg, B, H, W);
    ASSERT_TRUE(batch.data != NULL);
    ASSERT_EQ(batch.B, B);
    ASSERT_EQ(batch.H, H);
    ASSERT_EQ(batch.W, W);
    /* Check values are in reasonable range */
    for (int i = 0; i < B * H * W * 4; ++i) {
        if (batch.data[i] < -1.5f || batch.data[i] > 1.5f) {
            printf("FAIL: texture data[%d] = %f out of range\n", i, batch.data[i]);
            tests_failed++;
            wubu_phase1_texture_batch_free(&batch);
            return;
        }
    }
    wubu_phase1_texture_batch_free(&batch);
    tests_passed++;
}

TEST(test_rgba_texture_batch_not_nan) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int B = 1, H = 16, W = 16;
    float fg[B * H * W * 3];
    float bg[B * H * W * 3];
    for (int i = 0; i < B * H * W * 3; ++i) {
        fg[i] = 0.5f;
        bg[i] = -0.5f;
    }
    Phase1TextureBatch batch = wubu_phase1_create_synthetic_rgba_texture_batch(
        &rng, fg, bg, B, H, W);
    ASSERT_TRUE(batch.data != NULL);
    for (int i = 0; i < B * H * W * 4; ++i) {
        if (batch.data[i] != batch.data[i]) { /* NaN check */
            printf("FAIL: NaN detected at index %d\n", i);
            tests_failed++;
            wubu_phase1_texture_batch_free(&batch);
            return;
        }
    }
    wubu_phase1_texture_batch_free(&batch);
    tests_passed++;
}

/* ===================================================================
 * Color Pattern Generation Tests
 * =================================================================== */

TEST(test_color_patterns_32_frames) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int H = 16, W = 16;
    Phase1ColorPatterns patterns = wubu_phase1_generate_color_patterns(&rng, H, W);
    ASSERT_TRUE(patterns.frames != NULL);
    ASSERT_EQ(patterns.num_frames, 32);
    ASSERT_EQ(patterns.H, H);
    ASSERT_EQ(patterns.W, W);
    /* Check values in [-1,1] */
    for (int i = 0; i < 32 * H * W * 3; ++i) {
        if (patterns.frames[i] < -1.01f || patterns.frames[i] > 1.01f) {
            printf("FAIL: pattern[%d] = %f out of [-1,1]\n", i, patterns.frames[i]);
            tests_failed++;
            wubu_phase1_color_patterns_free(&patterns);
            return;
        }
    }
    wubu_phase1_color_patterns_free(&patterns);
    tests_passed++;
}

TEST(test_color_patterns_different_frames) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int H = 16, W = 16;
    Phase1ColorPatterns patterns = wubu_phase1_generate_color_patterns(&rng, H, W);
    ASSERT_TRUE(patterns.frames != NULL);
    /* Frame 0 and frame 16 should be different (half rotation) */
    int different = 0;
    for (int i = 0; i < H * W * 3; ++i) {
        if (fabsf(patterns.frames[i] - patterns.frames[H * W * 3 + i]) > 0.01f) {
            different++;
        }
    }
    ASSERT_TRUE(different > 0);
    wubu_phase1_color_patterns_free(&patterns);
    tests_passed++;
}

/* ===================================================================
 * Moving Shape Patterns Tests
 * =================================================================== */

TEST(test_moving_shapes_dimensions) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int H = 16, W = 16, num_frames = 8;
    Phase1MovingShapes shapes = wubu_phase1_generate_moving_shapes(&rng, H, W, num_frames);
    ASSERT_TRUE(shapes.frames != NULL);
    ASSERT_EQ(shapes.num_frames, num_frames);
    ASSERT_EQ(shapes.H, H);
    ASSERT_EQ(shapes.W, W);
    /* Check values in [-1,1] */
    for (int i = 0; i < num_frames * H * W * 3; ++i) {
        if (shapes.frames[i] < -1.01f || shapes.frames[i] > 1.01f) {
            printf("FAIL: shape[%d] = %f out of [-1,1]\n", i, shapes.frames[i]);
            tests_failed++;
            wubu_phase1_moving_shapes_free(&shapes);
            return;
        }
    }
    wubu_phase1_moving_shapes_free(&shapes);
    tests_passed++;
}

TEST(test_moving_shapes_animation) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int H = 16, W = 16, num_frames = 8;
    Phase1MovingShapes shapes = wubu_phase1_generate_moving_shapes(&rng, H, W, num_frames);
    ASSERT_TRUE(shapes.frames != NULL);
    /* Consecutive frames should differ (animation) */
    int different = 0;
    for (int f = 1; f < num_frames; ++f) {
        for (int i = 0; i < H * W * 3; ++i) {
            if (fabsf(shapes.frames[f * H * W * 3 + i] - shapes.frames[(f-1) * H * W * 3 + i]) > 0.01f) {
                different++;
            }
        }
    }
    ASSERT_TRUE(different > 0);
    wubu_phase1_moving_shapes_free(&shapes);
    tests_passed++;
}

/* ===================================================================
 * VHF Tone Pipeline Tests
 * =================================================================== */

TEST(test_vhf_tones_count) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    Phase1VHFTones tones = wubu_phase1_generate_vhf_tones(&rng, 44100, 0.5f);
    ASSERT_TRUE(tones.samples != NULL);
    ASSERT_EQ(tones.sample_rate, 44100);
    ASSERT_EQ(tones.num_samples, (int)(44100 * 0.5f));
    wubu_phase1_vhf_tones_free(&tones);
    tests_passed++;
}

TEST(test_vhf_tones_in_range) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    Phase1VHFTones tones = wubu_phase1_generate_vhf_tones(&rng, 44100, 0.1f);
    ASSERT_TRUE(tones.samples != NULL);
    for (int i = 0; i < tones.num_samples; ++i) {
        if (tones.samples[i] < -1.01f || tones.samples[i] > 1.01f) {
            printf("FAIL: tone[%d] = %f out of [-1,1]\n", i, tones.samples[i]);
            tests_failed++;
            wubu_phase1_vhf_tones_free(&tones);
            return;
        }
    }
    wubu_phase1_vhf_tones_free(&tones);
    tests_passed++;
}

TEST(test_vhf_tones_not_silent) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    Phase1VHFTones tones = wubu_phase1_generate_vhf_tones(&rng, 44100, 0.1f);
    ASSERT_TRUE(tones.samples != NULL);
    float max_amp = 0.0f;
    for (int i = 0; i < tones.num_samples; ++i) {
        float a = fabsf(tones.samples[i]);
        if (a > max_amp) max_amp = a;
    }
    ASSERT_TRUE(max_amp > 0.01f);
    wubu_phase1_vhf_tones_free(&tones);
    tests_passed++;
}

/* ===================================================================
 * FiLM Layer Tests
 * =================================================================== */

TEST(test_film_layer_shape_preserves) {
    int N = 4, D = 8, context_dim = 4;
    float x[32], context[4];
    for (int i = 0; i < N * D; ++i) x[i] = 0.5f;
    for (int i = 0; i < context_dim; ++i) context[i] = 0.1f;

    wubu_phase1_film_layer(x, context, N, D, context_dim);

    /* No NaN */
    for (int i = 0; i < N * D; ++i) {
        if (x[i] != x[i]) {
            printf("FAIL: NaN after FiLM at %d\n", i);
            tests_failed++;
            return;
        }
    }
    tests_passed++;
}

TEST(test_film_layer_zero_context_identity) {
    int N = 4, D = 8, context_dim = 4;
    float x[32], x_orig[32], context[4];
    for (int i = 0; i < N * D; ++i) { x[i] = 0.5f; x_orig[i] = 0.5f; }
    for (int i = 0; i < context_dim; ++i) context[i] = 0.0f;

    wubu_phase1_film_layer(x, context, N, D, context_dim);

    /* With zero context, gamma=tanh(0)=0, beta=0 => x*(0+1)+0 = x (identity) */
    for (int i = 0; i < N * D; ++i) {
        ASSERT_NEAR(x[i], x_orig[i], 1e-5f);
    }
}

/* ===================================================================
 * Path Modulator Tests
 * =================================================================== */

TEST(test_path_modulate_output) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int B = 1, img_size = 32;
    float images[B * img_size * img_size * 3];
    for (int i = 0; i < B * img_size * img_size * 3; ++i) {
        images[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;
    }

    Phase1Config config = PHASE1_DEFAULT;
    Phase1PathOutput path_out = wubu_phase1_path_modulate(&rng, images, B, img_size, &config);
    ASSERT_TRUE(path_out.path_params != NULL);
    ASSERT_TRUE(path_out.context != NULL);
    ASSERT_EQ(path_out.B, B);
    ASSERT_EQ(path_out.H, config.latent_grid_size);
    ASSERT_EQ(path_out.W, config.latent_grid_size);
    ASSERT_EQ(path_out.context_dim, config.d_model);

    /* Path params should be in reasonable range */
    int L = path_out.H * path_out.W;
    for (int i = 0; i < B * L * 3; ++i) {
        if (path_out.path_params[i] < -10.0f || path_out.path_params[i] > 10.0f) {
            printf("FAIL: path_params[%d] = %f out of range\n", i, path_out.path_params[i]);
            tests_failed++;
            wubu_phase1_path_output_free(&path_out);
            return;
        }
    }
    wubu_phase1_path_output_free(&path_out);
    tests_passed++;
}

/* ===================================================================
 * Topological Observer Tests
 * =================================================================== */

TEST(test_topological_observe_output) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int B = 1, H = 8, W = 8;
    int L = H * W;
    float path_params[B * L * 3];
    for (int i = 0; i < B * L * 3; ++i) {
        path_params[i] = (float)(i % 100) / 100.0f;
    }

    Phase1Config config = PHASE1_DEFAULT;
    float* features = wubu_phase1_topological_observe(path_params, B, H, W, &config);
    ASSERT_TRUE(features != NULL);

    /* Features should not be all zeros */
    int nonzero = 0;
    for (int i = 0; i < B * L * config.latent_dim; ++i) {
        if (features[i] != 0.0f) nonzero++;
    }
    ASSERT_TRUE(nonzero > 0);
    free(features);
    tests_passed++;
}

/* ===================================================================
 * Full Pipeline Tests
 * =================================================================== */

TEST(test_pipeline_output) {
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    int B = 1, img_size = 16, N = 64;
    float images[B * img_size * img_size * 3];
    for (int i = 0; i < B * img_size * img_size * 3; ++i) {
        images[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;
    }

    Phase1Config config = PHASE1_DEFAULT;
    Phase1DecodeOutput result = wubu_phase1_pipeline(&rng, images, B, img_size, N, &config);
    ASSERT_TRUE(result.pixels_rgba_struct != NULL);
    ASSERT_EQ(result.B, B);
    ASSERT_EQ(result.N, N);

    /* Check output is not NaN */
    for (int i = 0; i < B * N * 4; ++i) {
        if (result.pixels_rgba_struct[i] != result.pixels_rgba_struct[i]) {
            printf("FAIL: NaN in pipeline output at %d\n", i);
            tests_failed++;
            wubu_phase1_decode_output_free(&result);
            return;
        }
    }
    wubu_phase1_decode_output_free(&result);
    tests_passed++;
}

TEST(test_pipeline_deterministic) {
    WubuRNG rng1, rng2;
    wubu_rng_init(&rng1, 42);
    wubu_rng_init(&rng2, 42);
    int B = 1, img_size = 16, N = 32;
    float images[B * img_size * img_size * 3];
    for (int i = 0; i < B * img_size * img_size * 3; ++i) {
        images[i] = 0.3f;
    }

    Phase1Config config = PHASE1_DEFAULT;
    Phase1DecodeOutput r1 = wubu_phase1_pipeline(&rng1, images, B, img_size, N, &config);
    Phase1DecodeOutput r2 = wubu_phase1_pipeline(&rng2, images, B, img_size, N, &config);
    ASSERT_TRUE(r1.pixels_rgba_struct != NULL && r2.pixels_rgba_struct != NULL);

    for (int i = 0; i < B * N * 4; ++i) {
        ASSERT_NEAR(r1.pixels_rgba_struct[i], r2.pixels_rgba_struct[i], 1e-5f);
    }
    wubu_phase1_decode_output_free(&r1);
    wubu_phase1_decode_output_free(&r2);
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("=== Phase 1 Symmetric Encoder Tests ===\n\n");

    printf("[Complex Mask Generation]\n");
    RUN_TEST(test_ellipse_mask_dimensions);
    RUN_TEST(test_ellipse_mask_nonzero);
    RUN_TEST(test_rect_mask_dimensions);
    RUN_TEST(test_rect_mask_nonzero);
    RUN_TEST(test_complex_mask_dimensions);
    RUN_TEST(test_complex_mask_deterministic);

    printf("\n[Synthetic RGBA Texture Batch]\n");
    RUN_TEST(test_rgba_texture_batch_dimensions);
    RUN_TEST(test_rgba_texture_batch_not_nan);

    printf("\n[Color Pattern Generation]\n");
    RUN_TEST(test_color_patterns_32_frames);
    RUN_TEST(test_color_patterns_different_frames);

    printf("\n[Moving Shape Patterns]\n");
    RUN_TEST(test_moving_shapes_dimensions);
    RUN_TEST(test_moving_shapes_animation);

    printf("\n[VHF Tone Pipeline]\n");
    RUN_TEST(test_vhf_tones_count);
    RUN_TEST(test_vhf_tones_in_range);
    RUN_TEST(test_vhf_tones_not_silent);

    printf("\n[FiLM Layer]\n");
    RUN_TEST(test_film_layer_shape_preserves);
    RUN_TEST(test_film_layer_zero_context_identity);

    printf("\n[Path Modulator]\n");
    RUN_TEST(test_path_modulate_output);

    printf("\n[Topological Observer]\n");
    RUN_TEST(test_topological_observe_output);

    printf("\n[Full Pipeline]\n");
    RUN_TEST(test_pipeline_output);
    RUN_TEST(test_pipeline_deterministic);

    printf("\n=== Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Total:  %d\n", tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
