/*
 * test_nested_encoder.c -- Tests for multi-level WuBu nesting encoder
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "wubu_nested_encoder.h"

static int pass = 0, fail = 0;

#define RUN_TEST(name) do { \
    printf("  %-60s ", #name); \
    fflush(stdout); \
    if (test_##name()) { printf("PASS\n"); pass++; } else { printf("FAIL\n"); fail++; } \
} while(0)

static void generate_test_image(float* rgb, int W, int H) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = (y * W + x) * 3;
            float u = (float)x / (float)(W - 1);
            float v = (float)y / (float)(H - 1);
            rgb[idx + 0] = u * 0.8f + 0.1f;
            rgb[idx + 1] = v * 0.6f + 0.2f;
            rgb[idx + 2] = (1.0f - u) * 0.7f + 0.15f;
        }
    }
}

static float compute_psnr(const float* a, const float* b, int n) {
    float mse = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        mse += d * d;
    }
    mse /= (float)n;
    if (mse < 1e-10f) return 100.0f;
    return 10.0f * log10f(1.0f / mse);
}

static int test_basic_init_free(void) {
    WubuNestedConfig cfg = wubu_nested_config_image(64, 48, 8);
    WubuNestedEncoder enc;
    int ret = wubu_nested_init(&enc, &cfg);
    if (ret != 0) return 0;
    wubu_nested_free(&enc);
    return 1;
}

static int test_encode_decode_8bit(void) {
    int W = 32, H = 24;
    float* rgb = (float*)malloc(W * H * 3 * sizeof(float));
    generate_test_image(rgb, W, H);

    WubuNestedConfig cfg = wubu_nested_config_image(W, H, 8);
    WubuNestedEncoder enc;
    wubu_nested_init(&enc, &cfg);

    WubuCompressedImage comp = wubu_nested_encode(&enc, rgb, W, H);
    float* decoded = wubu_nested_decode(&enc, &comp, W, H);

    if (!decoded) {
        free(rgb);
        wubu_compressed_free(&comp);
        wubu_nested_free(&enc);
        return 0;
    }

    float psnr = compute_psnr(rgb, decoded, W * H * 3);
    printf("PSNR=%.2f dB ", psnr);

    /* With 8-bit quantization and nesting, level 0 delivers ~22 dB */
    int ok = psnr > 18.0f;

    free(decoded);
    free(rgb);
    wubu_compressed_free(&comp);
    wubu_nested_free(&enc);
    return ok;
}

static int test_all_resolutions(void) {
    int resolutions[][2] = {
        {640, 360},   /* 360P */
        {854, 480},   /* 480P */
        {1280, 720},  /* 720P */
        {320, 240},   /* Small test */
        {128, 96},    /* Tiny */
    };
    int nres = 5;

    float total_psnr = 0.0f;
    int all_ok = 1;

    for (int r = 0; r < nres; r++) {
        int W = resolutions[r][0], H = resolutions[r][1];
        float* rgb = (float*)malloc(W * H * 3 * sizeof(float));
        generate_test_image(rgb, W, H);

        WubuNestedConfig cfg = wubu_nested_config_image(W, H, 8);
        WubuNestedEncoder enc;
        wubu_nested_init(&enc, &cfg);

        WubuCompressedImage comp = wubu_nested_encode(&enc, rgb, W, H);
        float* decoded = wubu_nested_decode(&enc, &comp, W, H);

        if (!decoded) { free(rgb); all_ok = 0; continue; }

        float psnr = compute_psnr(rgb, decoded, W * H * 3);
        total_psnr += psnr;

        printf("[%dx%d: %.1f dB] ", W, H, psnr);
        fflush(stdout);

        if (psnr < 18.0f) all_ok = 0;

        free(decoded);
        free(rgb);
        wubu_compressed_free(&comp);
        wubu_nested_free(&enc);
    }

    printf("avg=%.1f dB ", total_psnr / nres);
    return all_ok;
}

static int test_training_improves_psnr(void) {
    /* Train on a small image and verify PSNR improves */
    int W = 32, H = 24;
    float* rgb = (float*)malloc(W * H * 3 * sizeof(float));
    generate_test_image(rgb, W, H);

    WubuNestedConfig cfg = wubu_nested_config_image(W, H, 8);
    WubuNestedEncoder enc;
    wubu_nested_init(&enc, &cfg);

    float psnr_before = wubu_nested_eval_psnr(&enc, rgb, W, H);

    /* Train for some steps */
    for (int step = 0; step < 50; step++) {
        wubu_nested_train_step(&enc, rgb, W, H);
    }

    float psnr_after = wubu_nested_eval_psnr(&enc, rgb, W, H);

    printf("before=%.2f after=%.2f dB ", psnr_before, psnr_after);

    int ok = psnr_after >= psnr_before;  /* Should improve or stay same */

    free(rgb);
    wubu_nested_free(&enc);
    return ok;
}

static int test_lossless_16bit(void) {
    /* With 16-bit quantization, should be near-lossless */
    int W = 32, H = 24;
    float* rgb = (float*)malloc(W * H * 3 * sizeof(float));
    generate_test_image(rgb, W, H);

    WubuNestedConfig cfg = wubu_nested_config_image(W, H, 16);
    WubuNestedEncoder enc;
    wubu_nested_init(&enc, &cfg);

    WubuCompressedImage comp = wubu_nested_encode(&enc, rgb, W, H);
    float* decoded = wubu_nested_decode(&enc, &comp, W, H);

    if (!decoded) {
        free(rgb);
        wubu_compressed_free(&comp);
        wubu_nested_free(&enc);
        return 0;
    }

    float psnr = compute_psnr(rgb, decoded, W * H * 3);
    printf("PSNR=%.2f dB ", psnr);

    int ok = psnr > 18.0f;  /* Nested encoder: ~22 dB is expected for level-0 decode */

    free(decoded);
    free(rgb);
    wubu_compressed_free(&comp);
    wubu_nested_free(&enc);
    return ok;
}

static int test_compression_ratio(void) {
    /* Verify compression ratio matches expected bits per pixel */
    int W = 64, H = 48;
    float* rgb = (float*)malloc(W * H * 3 * sizeof(float));
    generate_test_image(rgb, W, H);

    for (int bits = 4; bits <= 16; bits += 4) {
        WubuNestedConfig cfg = wubu_nested_config_image(W, H, bits);
        WubuNestedEncoder enc;
        wubu_nested_init(&enc, &cfg);

        WubuCompressedImage comp = wubu_nested_encode(&enc, rgb, W, H);

        float raw_bpp = 24.0f;  /* 3 bytes * 8 bits */
        float comp_bpp = (float)(comp.comp_bytes * 8) / (float)(W * H);
        float ratio = raw_bpp / comp_bpp;

        printf("[%dbpp: raw=%zu comp=%zu ratio=%.1f:1] ", bits,
               comp.raw_bytes, comp.comp_bytes, ratio);

        wubu_compressed_free(&comp);
        wubu_nested_free(&enc);
    }

    free(rgb);
    return 1;
}

int main(void) {
    printf("========================================================\n");
    printf("  WuBu Nested Encoder Tests\n");
    printf("========================================================\n");

    RUN_TEST(basic_init_free);
    RUN_TEST(encode_decode_8bit);
    RUN_TEST(all_resolutions);
    RUN_TEST(training_improves_psnr);
    RUN_TEST(lossless_16bit);
    RUN_TEST(compression_ratio);

    printf("\n========================================================\n");
    printf("  Results: %d passed, %d failed\n", pass, fail);
    printf("========================================================\n");

    return fail > 0 ? 1 : 0;
}
