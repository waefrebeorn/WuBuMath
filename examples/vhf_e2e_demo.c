/*
 * vhf_e2e_demo.c -- End-to-end VHF pipeline demonstration
 *
 * Runs the complete WuBu VHF pipeline:
 *   1. Hamilton Encoder: RGB images → quaternion latent space
 *   2. VHF Decoder: latent → RGB reconstruction
 *   3. Canvas compositing: VBI + audio HBI + visible → 656×525 canvas
 *   4. Training step: loss computation + Q-controller update
 *   5. Audio strip generation
 *
 * Outputs viewable PPM images of the canvas and intermediate frames.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubumath.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Generate a synthetic RGB image for testing [B, H, W, 3] in [-1,1] */
static void generate_test_image(float* img, int B, int H, int W, WubuRNG* rng) {
    for (int b = 0; b < B; b++) {
        float hue_base = (float)b / (float)B;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float u = (float)x / (float)(W - 1);
                float v = (float)y / (float)(H - 1);

                float cx = u - 0.5f;
                float cy = v - 0.5f;
                float dist = sqrtf(cx * cx + cy * cy) * 2.0f;
                float angle = atan2f(cy, cx) / (2.0f * M_PI) + 0.5f;

                float h = fmodf(hue_base + angle * 0.3f, 1.0f);
                float s = fminf(dist * 1.5f, 1.0f);
                float l = 0.5f - dist * 0.3f;

                /* HSL to RGB */
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

                int idx = ((b * H + y) * W + x) * 3;
                img[idx + 0] = r1 + m;
                img[idx + 1] = g1 + m;
                img[idx + 2] = b1 + m;
            }
        }
    }
}

/* Save a single RGB image as PPM */
static void save_ppm(const char* path, const float* rgb, int H, int W, int channels) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot create %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int p = 0; p < H * W; p++) {
        unsigned char rgb_bytes[3];
        for (int c = 0; c < 3; c++) {
            float val = rgb[p * channels + c];
            /* Convert from [-1,1] to [0,255] */
            if (val < -1.0f) val = -1.0f;
            if (val > 1.0f) val = 1.0f;
            rgb_bytes[c] = (unsigned char)((val * 0.5f + 0.5f) * 255.0f);
        }
        fwrite(rgb_bytes, 1, 3, f);
    }
    fclose(f);
}

/* Save canvas (656×525) as PPM */
static void save_canvas_ppm(const char* path, const float* canvas, int H, int W) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot create %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int p = 0; p < H * W; p++) {
        unsigned char rgb[3];
        for (int c = 0; c < 3; c++) {
            float val = canvas[p * 3 + c];
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            rgb[c] = (unsigned char)(val * 255.0f);
        }
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== WuBu VHF End-to-End Pipeline ===\n\n");

    WubuRNG rng;
    wubu_rng_init(&rng, 42);

    int B = 2;
    int img_size = 32; /* smaller for demo speed */

    /* Step 1: Generate test images */
    printf("Step 1: Generating %d test images (%dx%d)...\n", B, img_size, img_size);
    float* images = (float*)malloc(B * img_size * img_size * 3 * sizeof(float));
    generate_test_image(images, B, img_size, img_size, &rng);
    save_ppm("output/vhf_input_image.ppm", images, img_size, img_size, 3);
    printf("  -> output/vhf_input_image.ppm OK\n");

    /* Step 2: Hamilton Encoder: RGB -> quaternion latent space */
    printf("\nStep 2: Hamilton Encoder (RGB -> quaternion latent)...\n");
    VHFEncoded encoded = wubu_vhf_encode(&rng, images, B, img_size);
    printf("  Encoded: B=%d, H=%d, W=%d\n", encoded.B, encoded.H, encoded.W);
    printf("  Quaternions: %d floats, Amplitude: %d floats, Context: %d floats\n",
           encoded.B * encoded.H * encoded.W * 4,
           encoded.B * encoded.H * encoded.W * 1,
           encoded.B * 3);
    fflush(stdout);

    /* Step 3: VHF Decoder: latent -> RGB reconstruction */
    printf("\nStep 3: VHF Decoder (quaternion latent -> RGB)...\n");
    int N_pixels = img_size * img_size;
    fflush(stdout);
    float* coords = (float*)malloc(N_pixels * 2 * sizeof(float));
    for (int y = 0; y < img_size; y++) {
        for (int x = 0; x < img_size; x++) {
            coords[(y * img_size + x) * 2 + 0] = (float)x / (float)(img_size - 1);
            coords[(y * img_size + x) * 2 + 1] = (float)y / (float)(img_size - 1);
        }
    }
    float* reconstructed = wubu_vhf_decode(&rng, &encoded, coords, N_pixels);
    printf("  Reconstructed: %d pixels\n", N_pixels);
    save_ppm("output/vhf_reconstructed.ppm", reconstructed, img_size, img_size, 3);
    printf("  -> output/vhf_reconstructed.ppm OK\n");
    fflush(stdout);

    /* Step 4: Generate audio HBI strip */
    printf("\nStep 4: Audio HBI strip generation...\n");
    fflush(stdout);
    float audio[CANVAS_H * AUDIO_HBI_WIDTH];
    float frequencies[8] = {440.0f, 554.37f, 659.25f, 880.0f, 1108.73f, 1318.51f, 1760.0f, 2217.46f};
    for (int t = 0; t < 8; t++) {
        for (int s = 0; s < AUDIO_HBI_WIDTH; s++) {
            int sample_idx = t * AUDIO_HBI_WIDTH + s;
            if (sample_idx < CANVAS_H * AUDIO_HBI_WIDTH) {
                float time = (float)s / (float)AUDIO_HBI_WIDTH;
                float env = 1.0f;
                if (s < 2) env = (float)s / 2.0f;
                if (s > AUDIO_HBI_WIDTH - 3) env = (float)(AUDIO_HBI_WIDTH - s) / 2.0f;
                audio[sample_idx] = 0.3f * env * sinf(2.0f * M_PI * frequencies[t] * time);
            }
        }
    }
    float* audio_strip = wubu_vhf_generate_audio_strip(audio, CANVAS_H * AUDIO_HBI_WIDTH, &VHF_AUDIO_DEFAULT);
    printf("  Audio strip: %d x %d x 3 (%d floats)\n", VISIBLE_H, AUDIO_HBI_WIDTH, VISIBLE_H * AUDIO_HBI_WIDTH * 3);
    fflush(stdout);

    /* Step 5: Canvas compositing: VBI + audio HBI + visible */
    printf("\nStep 5: Canvas compositing (VBI + audio HBI + visible)...\n");
    fflush(stdout);
    float* vbi_block = (float*)calloc(VBI_LINES * CANVAS_W * 3, sizeof(float));
    /* Fill VBI with dark blue pattern */
    for (int y = 0; y < VBI_LINES; y++) {
        for (int x = 0; x < CANVAS_W; x++) {
            int idx = (y * CANVAS_W + x) * 3;
            vbi_block[idx + 0] = 0.05f;
            vbi_block[idx + 1] = 0.1f;
            vbi_block[idx + 2] = 0.2f;
        }
    }

    /* Use reconstructed image as visible portion (scaled to VISIBLE_H x VISIBLE_W) */
    float* visible = (float*)calloc(VISIBLE_H * VISIBLE_W * 3, sizeof(float));
    /* Simple nearest-neighbor upscale from img_size to VISIBLE_W x VISIBLE_H */
    for (int y = 0; y < VISIBLE_H; y++) {
        for (int x = 0; x < VISIBLE_W; x++) {
            int src_y = (int)((float)y / (float)VISIBLE_H * img_size);
            int src_x = (int)((float)x / (float)VISIBLE_W * img_size);
            if (src_y >= img_size) src_y = img_size - 1;
            if (src_x >= img_size) src_x = img_size - 1;
            for (int c = 0; c < 3; c++) {
                /* Get from first batch of reconstructed */
                float val = reconstructed[(src_y * img_size + src_x) * 3 + c];
                /* Map from [-1,1] to [0,1] */
                visible[(y * VISIBLE_W + x) * 3 + c] = val * 0.5f + 0.5f;
            }
        }
    }

    float* canvas = wubu_vhf_compose_canvas(vbi_block, audio_strip, visible, &VHF_AUDIO_DEFAULT);
    printf("  Canvas: %d x %d x 3\n", CANVAS_H, CANVAS_W);
    save_canvas_ppm("output/vhf_canvas.ppm", canvas, CANVAS_H, CANVAS_W);
    printf("  -> output/vhf_canvas.ppm OK (%d x %d)\n", CANVAS_W, CANVAS_H);
    fflush(stdout);

    /* Step 6: Training step (loss + Q-controller) */
    printf("\nStep 6: Training step (loss computation)...\n");
    fflush(stdout);
    QControllerConfig qc_config = WUBU_Q_CONTROLLER_DEFAULT;
    float q_table[5] = {0};
    float history[100] = {0};
    QController qc = {
        .q_table = q_table,
        .metric_history = history
    };
    wubu_q_controller_init(&qc, &qc_config);

    /* Use reconstructed as prediction, original as ground truth (scaled) */
    float* pred_rgb = (float*)malloc(N_pixels * 3 * sizeof(float));
    float* gt_rgb = (float*)malloc(N_pixels * 3 * sizeof(float));
    for (int i = 0; i < N_pixels * 3; i++) {
        pred_rgb[i] = reconstructed[i] * 0.5f + 0.5f; /* [-1,1] -> [0,1] */
        gt_rgb[i] = images[i] * 0.5f + 0.5f;
    }

    VHFTrainingState state = wubu_vhf_train_step(pred_rgb, gt_rgb, N_pixels, &qc);
    printf("  Loss: composite=%.4f luma=%.4f phase=%.4f sat=%.4f\n",
           state.composite_loss, state.luma_loss, state.phase_loss, state.sat_loss);
    printf("  Q-status: %d, step: %d\n", state.q_status, state.step_count);
    fflush(stdout);

    /* Step 7: Run multiple training steps to show convergence */
    printf("\nStep 7: Running 50 training steps...\n");
    float losses[50];
    for (int step = 0; step < 50; step++) {
        /* Slightly modify pred to simulate training progress */
        float improvement = 1.0f - (float)step / 60.0f;
        for (int i = 0; i < N_pixels * 3; i++) {
            pred_rgb[i] = gt_rgb[i] + (pred_rgb[i] - gt_rgb[i]) * improvement;
        }
        state = wubu_vhf_train_step(pred_rgb, gt_rgb, N_pixels, &qc);
        losses[step] = state.composite_loss;
        if (step % 10 == 0) {
            printf("  Step %3d: loss=%.4f (q_status=%d)\n", step, state.composite_loss, state.q_status);
        }
    }

    /* Save loss curve as text for visualization */
    printf("\n  Loss progression: ");
    for (int i = 0; i < 50; i += 5) {
        printf("%.3f ", losses[i]);
    }
    printf("%.3f\n", losses[49]);

    /* Cleanup */
    free(images);
    free(reconstructed);
    free(coords);
    free(audio_strip);
    free(vbi_block);
    free(visible);
    free(canvas);
    free(pred_rgb);
    free(gt_rgb);
    wubu_vhf_encoded_free(&encoded);
    /* wubu_q_controller_free not needed — q_table/metric_history are stack-allocated */

    printf("\n=== VHF Pipeline Complete ===\n");
    printf("Output files:\n");
    printf("  output/vhf_input_image.ppm    - Input RGB images\n");
    printf("  output/vhf_reconstructed.ppm  - Hamilton decoded reconstruction\n");
    printf("  output/vhf_canvas.ppm         - Full 656x525 composited canvas\n");
    return 0;
}
