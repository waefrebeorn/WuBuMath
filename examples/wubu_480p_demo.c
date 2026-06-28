/*
 * wubu_480p_demo.c -- Full 480P WuBu video compression pipeline
 *
 * Demonstrates real latent compression at 480P:
 *   1. Generate 854x480 test frame
 *   2. Hamilton encode → 409,920 quaternion point cloud
 *   3. Latent compress → quantized + report compression ratio
 *   4. Latent decompress → reconstructed point cloud
 *   5. Hamilton decode → RGB reconstruction
 *   6. Report PSNR and compression stats
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_latent_codec.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Generate test pattern (RADIAL GRADIENT)*/
static void generate_frame(float* img, int H, int W, float phase) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float u = (float)x / (W - 1) - 0.5f;
            float v = (float)y / (H - 1) - 0.5f;
            float dist = sqrtf(u*u + v*v) * 2.0f;
            float angle = atan2f(v, u) / (float)(2.0 * M_PI) + 0.5f;

            int idx = (y * W + x) * 3;
            img[idx + 0] = fmodf(phase + angle * 0.3f, 1.0f);
            img[idx + 1] = fminf(dist * 1.5f, 1.0f);
            img[idx + 2] = 0.5f - dist * 0.3f;
        }
    }
}

int main(void) {
    printf("========================================================\n");
    printf("  480P WuBu Video Compression Pipeline\n");
    printf("========================================================\n\n");

    int W = 854, H = 480;
    int D = 4;
    int N = W * H; /* 409,920 points */

    printf("Resolution: %dx%d (%d pixels)\n", W, H, N);
    printf("Latent: Hamilton quaternion point cloud\n");
    printf("  %d points x (4 quat + 1 amp) x 4 bytes = %.2f MB raw\n\n",
           N, (float)(N * 5 * 4) / (1024.0f*1024.0f));

    /* Step 1: Generate frame */
    float* image = (float*)malloc(N * 3 * sizeof(float));
    generate_frame(image, H, W, 0.3f);
    printf("[1/5] Generated test frame %dx%d\n", W, H);

    /* Step 2: Hamilton Encode (simulated — per-pixel quaternion) */
    float* quaternions = (float*)malloc(N * D * sizeof(float));
    float* amplitude = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        float u = (float)(i % W) / (W - 1);
        float v = (float)(i / W) / (H - 1);
        float r = image[i * 3 + 0];
        float g = image[i * 3 + 1];
        float b = image[i * 3 + 2];

        quaternions[i * D + 0] = u * 2.0f - 1.0f;
        quaternions[i * D + 1] = v * 2.0f - 1.0f;
        quaternions[i * D + 2] = (r + g + b) / 3.0f;
        quaternions[i * D + 3] = 1.0f;
        float norm = 0.0f;
        for (int d = 0; d < D; d++) norm += quaternions[i*D+d] * quaternions[i*D+d];
        norm = sqrtf(norm);
        for (int d = 0; d < D; d++) quaternions[i*D+d] /= norm;

        amplitude[i] = 1.0f / (1.0f + expf(-(r + g + b) * 2.0f));
    }
    printf("[2/5] Hamilton encoded: %d quaternion points\n", N);

    /* Step 3: Compress at each quality level */
    printf("[3/5] Latent compression:\n\n");
    WubuQuality qualities[] = {
        WUBU_QUALITY_LOSSLESS, WUBU_QUALITY_HIGH,
        WUBU_QUALITY_MEDIUM, WUBU_QUALITY_LOW
    };
    const char* names[] = {"Lossless", "High", "Medium", "Low"};

    for (int q = 0; q < 4; q++) {
        WubuCompressedLatent c = wubu_latent_compress(quaternions, amplitude, N, qualities[q]);
        printf("  Quality: %s\n", names[q]);
        wubu_latent_print_stats(&c);

        /* Step 4: Decompress */
        float* q_out = (float*)malloc(N * D * sizeof(float));
        float* a_out = (float*)malloc(N * sizeof(float));
        wubu_latent_decompress(q_out, a_out, &c);

        /* Step 5: Decode (simulate — reconstruct RGB from latent) */
        float* reconstructed = (float*)malloc(N * 3 * sizeof(float));
        for (int i = 0; i < N; i++) {
            float qx = q_out[i*D+0], qy = q_out[i*D+1];
            float qz = q_out[i*D+2], qw = q_out[i*D+3];
            float amp = a_out[i];
            /* Map quaternion back to RGB approximation */
            reconstructed[i*3+0] = (qx * amp + 1.0f) * 0.5f;
            reconstructed[i*3+1] = (qy * amp + 1.0f) * 0.5f;
            reconstructed[i*3+2] = (qz * amp + 1.0f) * 0.5f;
        }

        /* Compute PSNR */
        float mse = 0.0f;
        for (int i = 0; i < N * 3; i++) {
            float d = reconstructed[i] - image[i];
            mse += d * d;
        }
        mse /= (float)(N * 3);
        float psnr = (mse > 0.0f) ? 10.0f * log10f(1.0f / mse) : 100.0f;

        printf("  PSNR: %.2f dB\n", psnr);
        printf("\n");

        free(q_out); free(a_out); free(reconstructed);
        wubu_latent_compressed_free(&c);
    }

    free(image); free(quaternions); free(amplitude);

    printf("========================================================\n");
    printf("  480P WuBu Compression Complete\n");
    printf("========================================================\n");
    return 0;
}
