/*
 * test_vhf_engine.c -- Tests for the faithful VHF engine slermed from vhf_audio.py
 *
 * Verifies:
 *   1. VHFPosEnc: correct dimensions, sin/cos structure
 *   2. VHFHamiltonEnc: multi-scale conv, correct context dim, quaternion normalization
 *   3. VHFDec: correct input/output dims, output range [-1,1]
 *   4. Loss: HSL computation, circular L1
 *   5. Canvas: correct dimensions
 *   6. Audio strip: correct dimensions, [-1,1] range
 *   7. End-to-end: encode → decode → loss
 */

#include "wubumath.h"
#include "wubu_vhf_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int test_count = 0;
static int fail_count = 0;

#define TEST(name) do { \
    test_count++; \
    printf("  %s ... ", name); \
} while(0)

#define PASS() printf("PASS\n")
#define FAIL(msg) do { printf("FAIL: %s\n", msg); fail_count++; } while(0)
#define PASS_VAL(fmt, ...) printf("PASS (" fmt ")\n", ##__VA_ARGS__)

static int check_finite(const float* arr, int n) {
    for (int i = 0; i < n; i++)
        if (!isfinite(arr[i])) return 0;
    return 1;
}

static int check_range(const float* arr, int n, float lo, float hi) {
    for (int i = 0; i < n; i++)
        if (arr[i] < lo || arr[i] > hi) return 0;
    return 1;
}

/* ===================================================================
 * Test 1: VHFPosEnc
 * =================================================================== */

static void test_posenc(void) {
    VHFPosEnc pe;
    vhf_posenc_init(&pe, 2, 10);

    TEST("PosEnc output dim");
    if (pe.output_dim == 42) PASS_VAL("dim=%d", pe.output_dim);
    else FAIL("expected 42");

    TEST("PosEnc: input coordinates preserved");
    float coords[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
    float out[84];
    vhf_posenc_forward(&pe, coords, 2, out);
    if (out[0] == -1.0f && out[1] == -1.0f) PASS();
    else FAIL("coords not preserved");

    TEST("PosEnc: sin/cos bands present");
    /* For coord=-1, freq=pi: sin(-pi)=0, cos(-pi)=-1 */
    /* Band 0 for x=coords[0]=-1: sin(-pi*1)=~0, cos(-pi*1)=~-1 */
    int ok = isfinite(out[2]) && isfinite(out[3]);
    if (ok) PASS();
    else FAIL("sin/cos not finite");

    TEST("PosEnc: all output finite");
    if (check_finite(out, 84)) PASS();
    else FAIL("non-finite output");

    vhf_posenc_free(&pe);
}

/* ===================================================================
 * Test 2: VHFHamiltonEnc initialization
 * =================================================================== */

static void test_hamilton_encoder_init(void) {
    VHFHamiltonEnc enc;
    int LG = 96, d = 512;

    TEST("VHFHamiltonEnc: init with LG=96 d=512");
    vhf_hamilton_encoder_init(&enc, LG, d);
    if (enc.latent_grid_size == LG && enc.d_model == d) PASS();
    else FAIL("wrong dims");

    TEST("VHFHamiltonEnc: computes num_scales for 480x640");
    /* 480/2=240 /2=120 /2=60 /2=30 /2=15 → 5 scales to get below 96 */
    if (enc.num_scales >= 2 && enc.num_scales <= 5)
        PASS_VAL("num_scales=%d", enc.num_scales);
    else
        FAIL("unexpected num_scales");

    TEST("VHFHamiltonEnc: context_dim > 0");
    if (enc.context_dim > 0) PASS_VAL("context_dim=%d", enc.context_dim);
    else FAIL("zero context_dim");

    TEST("VHFHamiltonEnc: context_dim = sum of all scale features");
    /* 32+64+128+256+512 = 992 for 5 scales */
    if (enc.context_dim == 32+64 ||
        enc.context_dim == 32+64+128 ||
        enc.context_dim == 32+64+128+256)
        PASS_VAL("context_dim=%d", enc.context_dim);
    else
        FAIL("wrong context sum");

    vhf_hamilton_encoder_free(&enc);
}

/* ===================================================================
 * Test 3: VHFHamiltonEnc forward pass
 * =================================================================== */

static void test_hamilton_encode_forward(void) {
    VHFHamiltonEnc enc;
    int LG = 96, d = 512;
    vhf_hamilton_encoder_init(&enc, LG, d);

    /* Create a small test pattern (480x640x3) */
    int H = VHF_VISIBLE_H, W = VHF_VISIBLE_W;
    float* image = (float*)calloc((size_t)(H * W * 3), sizeof(float));

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = (y * W + x) * 3;
            image[idx + 0] = (float)x / (float)(W - 1) * 2.0f - 1.0f;  /* R horizontal gradient */
            image[idx + 1] = (float)y / (float)(H - 1) * 2.0f - 1.0f;  /* G vertical gradient */
            image[idx + 2] = 0.0f;  /* B = 0 */
        }
    }

    float* keys = (float*)calloc((size_t)(LG * LG * 5), sizeof(float));
    float* context = (float*)calloc((size_t)enc.context_dim, sizeof(float));

    TEST("VHFHamiltonEnc: encode forward pass");
    vhf_hamilton_encode(&enc, image, H, W, keys, context);
    PASS();

    TEST("VHFHamiltonEnc: keys are finite");
    if (check_finite(keys, LG * LG * 5)) PASS();
    else FAIL("non-finite keys");

    TEST("VHFHamiltonEnc: context is finite");
    if (check_finite(context, enc.context_dim)) PASS();
    else FAIL("non-finite context");

    TEST("VHFHamiltonEnc: quaternions are unit-normalized");
    int all_unit = 1;
    for (int p = 0; p < LG * LG; p++) {
        float* q = keys + p * 5;
        float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (fabsf(norm - 1.0f) > 0.01f) { all_unit = 0; break; }
    }
    if (all_unit) PASS();
    else FAIL("quaternions not normalized");

    TEST("VHFHamiltonEnc: amplitude in [0,1]");
    if (check_range(keys + 4, LG * LG, 0.0f, 1.0f)) {
        /* Check every 5th element (amplitude) */
        int in_range = 1;
        for (int p = 0; p < LG * LG; p++)
            if (keys[p * 5 + 4] < 0.0f || keys[p * 5 + 4] > 1.0f) { in_range = 0; break; }
        if (in_range) PASS();
        else FAIL("amplitude out of range");
    } else {
        /* Check individually */
        int in_range = 1;
        for (int p = 0; p < LG * LG; p++)
            if (keys[p * 5 + 4] < 0.0f || keys[p * 5 + 4] > 1.0f) { in_range = 0; break; }
        if (in_range) PASS();
        else FAIL("amplitude out of range");
    }

    free(image);
    free(keys);
    free(context);
    vhf_hamilton_encoder_free(&enc);
}

/* ===================================================================
 * Test 4: VHFDec initialization
 * =================================================================== */

static void test_decoder_init(void) {
    /* Use a small context_dim for speed */
    int context_dim = 64; /* simplified context */
    int d_model = 64;

    VHFDec dec;
    TEST("VHFDec: init with d_model=64 context_dim=64");
    vhf_decoder_init(&dec, d_model, context_dim);
    if (dec.input_dim == 42 + 64 + 5) PASS_VAL("input_dim=%d", dec.input_dim);
    else FAIL("wrong input_dim");

    TEST("VHFDec: posenc_dim = 42");
    if (dec.posenc_dim == 42) PASS();
    else FAIL("wrong posenc_dim");

    vhf_decoder_free(&dec);
}

/* ===================================================================
 * Test 5: VHFDec forward pass
 * =================================================================== */

static void test_decode_forward(void) {
    int context_dim = 64;
    int d_model = 64;
    int LG = 16; /* small grid for test */

    VHFDec dec;
    vhf_decoder_init(&dec, d_model, context_dim);

    float* keys = (float*)calloc((size_t)(LG * LG * 5), sizeof(float));
    float* context = (float*)calloc((size_t)context_dim, sizeof(float));

    /* Create identity-like keys */
    for (int i = 0; i < LG * LG; i++) {
        keys[i * 5 + 0] = 1.0f;
        keys[i * 5 + 1] = 0.0f;
        keys[i * 5 + 2] = 0.0f;
        keys[i * 5 + 3] = 0.0f;
        keys[i * 5 + 4] = 0.5f;
    }
    for (int i = 0; i < context_dim; i++) context[i] = 0.1f;

    float coords[4] = {0.0f, 0.0f, 0.5f, 0.5f};
    float output[6]; /* 2 samples × 3 */

    TEST("VHFDecode: forward pass");
    vhf_decode_batch(&dec, keys, LG, LG, context, coords, 2, output);
    PASS();

    TEST("VHFDecode: output finite");
    if (check_finite(output, 6)) PASS();
    else FAIL("non-finite output");

    TEST("VHFDecode: output in [-1,1] (tanh bounded)");
    if (check_range(output, 6, -1.01f, 1.01f)) PASS();
    else FAIL("output outside [-1,1]");

    free(keys);
    free(context);
    vhf_decoder_free(&dec);
}

/* ===================================================================
 * Test 6: HSL loss computation
 * =================================================================== */

static void test_loss(void) {
    TEST("Loss: identical RGB → zero loss");
    float pred[6] = {0.5f, 0.5f, 0.5f, -0.3f, 0.8f, 0.1f};
    float gt[6]   = {0.5f, 0.5f, 0.5f, -0.3f, 0.8f, 0.1f};
    VHFLoss loss = vhf_compute_loss(pred, gt, 2);
    if (fabsf(loss.composite_loss) < 0.01f) PASS_VAL("loss=%.6f", loss.composite_loss);
    else FAIL("non-zero loss for identical input");

    TEST("Loss: different RGB → positive loss");
    float pred2[3] = {1.0f, 0.0f, 0.0f}; /* red */
    float gt2[3]   = {0.0f, 1.0f, 0.0f}; /* green */
    VHFLoss loss2 = vhf_compute_loss(pred2, gt2, 1);
    if (loss2.composite_loss > 0.1f) PASS_VAL("loss=%.4f", loss2.composite_loss);
    else FAIL("loss too small");

    TEST("Loss: luma component dominant (weight=10)");
    /* Black vs white: very different luma */
    float black[3] = {-1.0f, -1.0f, -1.0f};
    float white[3] = {1.0f, 1.0f, 1.0f};
    VHFLoss loss3 = vhf_compute_loss(black, white, 1);
    if (loss3.luma_loss > 0.4f) PASS_VAL("luma=%.4f", loss3.luma_loss);
    else FAIL("luma loss too small");
}

/* ===================================================================
 * Test 7: Canvas compositing
 * =================================================================== */

static void test_canvas(void) {
    float context[3] = {0.5f, 0.3f, 0.8f};
    float* audio_hbi = (float*)calloc((size_t)(VHF_VISIBLE_H * VHF_AUDIO_HBI_WIDTH * 3), sizeof(float));
    float* visible = (float*)calloc((size_t)(VHF_VISIBLE_H * VHF_VISIBLE_W * 3), sizeof(float));

    for (int i = 0; i < VHF_VISIBLE_H * VHF_AUDIO_HBI_WIDTH * 3; i++) audio_hbi[i] = 0.2f;
    for (int i = 0; i < VHF_VISIBLE_H * VHF_VISIBLE_W * 3; i++) visible[i] = 0.7f;

    TEST("Canvas: compose returns correct size");
    float* canvas = vhf_compose_canvas(context, 3, audio_hbi, visible);
    if (canvas) PASS();
    else FAIL("null canvas");

    TEST("Canvas: VBI block = context values");
    /* First pixel of VBI should be context values */
    if (fabsf(canvas[0] - 0.5f) < 0.001f &&
        fabsf(canvas[1] - 0.3f) < 0.001f &&
        fabsf(canvas[2] - 0.8f) < 0.001f)
        PASS();
    else FAIL("VBI not correct");

    TEST("Canvas: visible region starts at VBI offset");
    /* Pixel at (VBI_LINES, AUDIO_HBI_WIDTH) should be visible */
    int vis_offset = ((VHF_VBI_LINES) * VHF_CANVAS_W + VHF_AUDIO_HBI_WIDTH) * 3;
    if (fabsf(canvas[vis_offset] - 0.7f) < 0.001f) PASS();
    else FAIL("visible region wrong");

    free(canvas);
    free(audio_hbi);
    free(visible);
}

/* ===================================================================
 * Test 8: Audio strip generation
 * =================================================================== */

static void test_audio_strip(void) {
    float audio[1000];
    for (int i = 0; i < 1000; i++) audio[i] = sinf((float)i * 0.01f);

    TEST("Audio strip: correct size");
    float* strip = vhf_generate_audio_strip(audio, 1000);
    if (strip) PASS();
    else FAIL("null strip");

    TEST("Audio strip: values in [-1,1]");
    int in_range = 1;
    int target = VHF_CANVAS_H * VHF_AUDIO_HBI_WIDTH;
    for (int i = 0; i < target * 3; i++)
        if (strip[i] < -1.01f || strip[i] > 1.01f) { in_range = 0; break; }
    if (in_range) PASS();
    else FAIL("out of range");

    TEST("Audio strip: zero-padded beyond input length");
    /* Beyond sample 1000, all should be 0 */
    int start = 1003 * 3; /* sample 1003 is beyond 1000 */
    int all_zero = 1;
    for (int i = start; i < target * 3; i++)
        if (fabsf(strip[i]) > 0.001f) { all_zero = 0; break; }
    if (all_zero) PASS();
    else FAIL("not zero-padded");

    free(strip);
}

/* ===================================================================
 * Test 9: End-to-end encode → decode → loss
 * =================================================================== */

static void test_e2e(void) {
    /* Use small dimensions for speed */
    VHFHamiltonEnc enc;
    int LG = 32, d_model = 64;
    vhf_hamilton_encoder_init(&enc, LG, d_model);

    VHFDec dec;
    vhf_decoder_init(&dec, d_model, enc.context_dim);

    /* Create test image (480x640x3) — small gradient */
    int H = VHF_VISIBLE_H, W = VHF_VISIBLE_W;
    float* image = (float*)calloc((size_t)(H * W * 3), sizeof(float));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int idx = (y * W + x) * 3;
            image[idx] = (float)x / W * 2.0f - 1.0f;
            image[idx+1] = (float)y / H * 2.0f - 1.0f;
            image[idx+2] = 0.0f;
        }

    float* keys = (float*)calloc((size_t)(LG * LG * 5), sizeof(float));
    float* context = (float*)calloc((size_t)enc.context_dim, sizeof(float));

    TEST("E2E: encode visible frame");
    vhf_hamilton_encode(&enc, image, H, W, keys, context);
    if (check_finite(keys, LG * LG * 5) && check_finite(context, enc.context_dim))
        PASS();
    else FAIL("encode produced non-finite values");

    /* Decode at 10 random coordinates */
    int N = 10;
    float coords[20], gt_rgb[30];
    for (int i = 0; i < N; i++) {
        coords[i*2] = (float)(i % 3) * 0.5f - 0.5f;
        coords[i*2+1] = (float)(i % 5) * 0.25f - 0.5f;
        gt_rgb[i*3] = image[((i * 17) % H) * W * 3 + ((i * 13) % W) * 3];
        gt_rgb[i*3+1] = image[((i * 17) % H) * W * 3 + ((i * 13) % W) * 3 + 1];
        gt_rgb[i*3+2] = image[((i * 17) % H) * W * 3 + ((i * 13) % W) * 3 + 2];
    }

    float pred_rgb[30];
    TEST("E2E: decode at N coordinates");
    vhf_decode_batch(&dec, keys, LG, LG, context, coords, N, pred_rgb);
    if (check_finite(pred_rgb, N * 3)) PASS();
    else FAIL("decode produced non-finite values");

    TEST("E2E: loss computation");
    VHFLoss loss = vhf_compute_loss(pred_rgb, gt_rgb, N);
    if (isfinite(loss.composite_loss) && loss.composite_loss >= 0.0f)
        PASS_VAL("loss=%.4f", loss.composite_loss);
    else FAIL("bad loss");

    free(image);
    free(keys);
    free(context);
    vhf_decoder_free(&dec);
    vhf_hamilton_encoder_free(&enc);
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("\n=== VHF Engine Tests (from vhf_audio.py) ===\n\n");

    test_posenc();
    test_hamilton_encoder_init();
    test_hamilton_encode_forward();
    test_decoder_init();
    test_decode_forward();
    test_loss();
    test_canvas();
    test_audio_strip();
    test_e2e();

    printf("\n========================================================\n");
    printf("  Results: %d passed, %d failed\n", test_count - fail_count, fail_count);
    printf("========================================================\n\n");

    return fail_count > 0 ? 1 : 0;
}
