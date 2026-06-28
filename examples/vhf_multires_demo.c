/*
 * vhf_multires_demo.c -- Multi-resolution VHF pipeline proof
 * Generates proof outputs at 360P, 480P, 720P, 1080P, 1440P, and 4K
 * showing encode → decode → canvas composite → training at each resolution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_canvas.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* HSL to RGB helper */
static void hsl_to_rgb(float h, float s, float l, float* r, float* g, float* b) {
    if (s < 1e-8f) { *r = *g = *b = l; return; }
    float c_val = (1.0f - fabsf(2.0f * l - 1.0f)) * s;
    float hh = h * 6.0f;
    float x_val = c_val * (1.0f - fabsf(fmodf(hh, 2.0f) - 1.0f));
    float m = l - c_val / 2.0f;
    float r1, g1, b1;
    if (hh < 1)      { r1 = c_val; g1 = x_val; b1 = 0; }
    else if (hh < 2) { r1 = x_val; g1 = c_val; b1 = 0; }
    else if (hh < 3) { r1 = 0; g1 = c_val; b1 = x_val; }
    else if (hh < 4) { r1 = 0; g1 = x_val; b1 = c_val; }
    else if (hh < 5) { r1 = x_val; g1 = 0; b1 = c_val; }
    else             { r1 = c_val; g1 = 0; b1 = x_val; }
    *r = r1 + m; *g = g1 + m; *b = b1 + m;
}

/* Generate test image with pattern */
static void generate_test_image(float* img, int H, int W, int frame_idx, int total_frames) {
    float hue_base = (float)frame_idx / (float)total_frames;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float u = (float)x / (float)(W - 1);
            float v = (float)y / (float)(H - 1);

            float cx = u - 0.5f;
            float cy = v - 0.5f;
            float dist = sqrtf(cx * cx + cy * cy) * 2.0f;
            float angle = atan2f(cy, cx) / (2.0f * M_PI) + 0.5f;

            float h = fmodf(hue_base + angle * 0.3f + dist * 0.1f, 1.0f);
            float s = fminf(dist * 1.5f, 1.0f);
            float l = 0.5f - dist * 0.3f;

            float r, g, b;
            hsl_to_rgb(h, s, fmaxf(0.1f, fminf(0.9f, l)), &r, &g, &b);

            int idx = (y * W + x) * 3;
            img[idx + 0] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }
    }
}

/* Save PPM */
static void save_ppm(const char* path, const float* rgb, int H, int W, int channels, int swap_rb) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot create %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int p = 0; p < H * W; p++) {
        unsigned char bytes[3];
        float r = rgb[p * channels + 0];
        float g = rgb[p * channels + 1];
        float b = rgb[p * channels + 2];
        if (r < 0) r = 0; if (r > 1) r = 1;
        if (g < 0) g = 0; if (g > 1) g = 1;
        if (b < 0) b = 0; if (b > 1) b = 1;
        if (swap_rb) {
            bytes[0] = (unsigned char)(b * 255);
            bytes[2] = (unsigned char)(r * 255);
        } else {
            bytes[0] = (unsigned char)(r * 255);
            bytes[2] = (unsigned char)(b * 255);
        }
        bytes[1] = (unsigned char)(g * 255);
        fwrite(bytes, 1, 3, f);
    }
    fclose(f);
}

/* Downsample image for thumbnail proof */
static float* downsample(const float* img, int src_h, int src_w, int dst_h, int dst_w) {
    float* out = (float*)malloc(dst_h * dst_w * 3 * sizeof(float));
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            int sy = y * src_h / dst_h;
            int sx = x * src_w / dst_w;
            for (int c = 0; c < 3; c++) {
                out[(y * dst_w + x) * 3 + c] = img[(sy * src_w + sx) * 3 + c];
            }
        }
    }
    return out;
}

int main(void) {
    printf("=== WuBu VHF Multi-Resolution Pipeline ===\n\n");

    WubuResolution resolutions[] = {
        WUBU_RES_360P, WUBU_RES_480P, WUBU_RES_720P,
        WUBU_RES_1080P, WUBU_RES_1440P, WUBU_RES_4K
    };
    const char* res_names[] = {"360P", "480P", "720P", "1080P", "1440P", "4K"};
    int num_res = 6;

    WubuRNG rng;
    wubu_rng_init(&rng, 42);

    /* Q-controller for training */
    float q_table[5] = {0};
    float history[100] = {0};
    QController qc = { .q_table = q_table, .metric_history = history };
    wubu_q_controller_init(&qc, &WUBU_Q_CONTROLLER_DEFAULT);

    /* For the proof, we'll use a smaller internal resolution for speed
     * but demonstrate the full pipeline at each target canvas size.
     * The latent grid is the visible resolution. */

    for (int ri = 0; ri < num_res; ri++) {
        const WubuCanvasConfig* cfg = wubu_canvas_get_config(resolutions[ri]);
        printf("[%s] Visible: %dx%d, Canvas: %dx%d, Total: %d pixels\n",
               res_names[ri], cfg->visible_w, cfg->visible_h,
               cfg->canvas_w, cfg->canvas_h, cfg->total_pixels);
        fflush(stdout);

        int vw = cfg->visible_w;
        int vh = cfg->visible_h;

        /* Generate test image at this resolution */
        float* test_img = (float*)malloc(vw * vh * 3 * sizeof(float));
        generate_test_image(test_img, vh, vw, ri, num_res);

        /* Encode: RGB → Hamilton latent */
        WubuLatent latent = wubu_hamilton_encode(&rng, test_img, 1, vh, vw);
        printf("  Encoded: %d quaternions (%d floats)\n",
               vh * vw, vh * vw * 4);
        fflush(stdout);

        /* Decode: latent → RGB */
        int N = vh * vw;
        float* coords = (float*)malloc(N * 2 * sizeof(float));
        for (int y = 0; y < vh; y++)
            for (int x = 0; x < vw; x++) {
                coords[(y * vw + x) * 2 + 0] = (float)x / (float)(vw - 1) * 2.0f - 1.0f;
                coords[(y * vw + x) * 2 + 1] = (float)y / (float)(vh - 1) * 2.0f - 1.0f;
            }

        float* reconstructed = wubu_hamilton_decode(&rng, &latent, coords, N);
        printf("  Decoded: %d pixels\n", N);
        fflush(stdout);

        /* Generate audio HBI strip */
        int audio_samples = cfg->canvas_h * cfg->hbi_width;
        float* audio = (float*)malloc(audio_samples * sizeof(float));
        float freqs[8] = {440.0f, 554.37f, 659.25f, 880.0f, 1108.73f, 1318.51f, 1760.0f, 2217.46f};
        for (int t = 0; t < 8; t++) {
            for (int s = 0; s < cfg->hbi_width; s++) {
                int idx = t * cfg->hbi_width + s;
                if (idx < audio_samples) {
                    float env = 1.0f;
                    if (s < 2) env = (float)s / 2.0f;
                    if (s > cfg->hbi_width - 3) env = (float)(cfg->hbi_width - s) / 2.0f;
                    audio[idx] = 0.3f * env * sinf(2.0f * M_PI * freqs[t] * (float)s / 44100.0f);
                }
            }
        }
        float* hbi_strip = wubu_audio_make_hbi_strip(audio, audio_samples, cfg);
        printf("  Audio HBI: %d x %d x 3\n", cfg->visible_h, cfg->hbi_width);
        fflush(stdout);

        /* Compose canvas */
        float* vbi = (float*)calloc(cfg->vbi_lines * cfg->canvas_w * 3, sizeof(float));
        for (int y = 0; y < cfg->vbi_lines; y++)
            for (int x = 0; x < cfg->canvas_w; x++) {
                int idx = (y * cfg->canvas_w + x) * 3;
                vbi[idx + 0] = 0.05f;
                vbi[idx + 1] = 0.1f;
                vbi[idx + 2] = 0.2f;
            }

        /* Map reconstructed [-1,1] to [0,1] for canvas */
        float* visible = (float*)malloc(vh * vw * 3 * sizeof(float));
        for (int i = 0; i < N * 3; i++) {
            visible[i] = reconstructed[i] * 0.5f + 0.5f;
            if (visible[i] < 0) visible[i] = 0;
            if (visible[i] > 1) visible[i] = 1;
        }

        float* canvas = wubu_compose_canvas_res(vbi, hbi_strip, visible, cfg);
        printf("  Canvas: %d x %d x 3\n", cfg->canvas_w, cfg->canvas_h);
        fflush(stdout);

        /* Training step */
        float* gt = (float*)malloc(N * 3 * sizeof(float));
        for (int i = 0; i < N * 3; i++) {
            gt[i] = test_img[i] * 0.5f + 0.5f;  /* map to [0,1] */
        }
        WubuTrainState state = wubu_train_step(visible, gt, 1, vh, vw, &qc);
        printf("  Training: loss=%.4f q_status=%d\n", state.composite_loss, state.q_status);
        fflush(stdout);

        /* Save proof output — downsample for manageable file size */
        char path[256];
        int thumb_h = 180;
        int thumb_w = thumb_h * cfg->canvas_w / cfg->canvas_h;

        float* thumb = downsample(canvas, cfg->canvas_h, cfg->canvas_w, thumb_h, thumb_w);
        snprintf(path, sizeof(path), "output/proof_%s_canvas.ppm", res_names[ri]);
        save_ppm(path, thumb, thumb_h, thumb_w, 3, 0);
        printf("  -> %s (%dx%d)\n", path, thumb_w, thumb_h);
        fflush(stdout);

        free(thumb);
        free(test_img);
        free(reconstructed);
        free(coords);
        free(audio);
        free(hbi_strip);
        free(vbi);
        free(visible);
        free(gt);
        free(canvas);
        wubu_latent_free(&latent);
    }

    wubu_q_controller_free(&qc);

    printf("\n=== Multi-Resolution Pipeline Complete ===\n");
    printf("All 6 resolutions processed through full VHF pipeline.\n");
    return 0;
}
