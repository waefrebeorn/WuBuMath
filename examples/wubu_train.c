/*
 * wubu_train.c -- WuBu Video Compression Training Pipeline (v2)
 *
 * Clean version: processes a directory of JPEG frames from one video.
 * Encodes, compresses, decompresses, decodes, and trains flow matching.
 *
 * Usage: ./bin/wubu_train <frames_dir> [epochs] [target_size]
 * Example: ./bin/wubu_train dataset/frames/train/Charge 5 80
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include "wubumath.h"
#include "wubu_latent_codec.h"
#include "wubu_flow_matching.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <jpeglib.h>
#include <setjmp.h>

typedef struct {
    float* pixels;     /* [H * W * 3] in [0,1] */
    int W, H;
} Image;

/* Load JPEG into float buffer [0,1] */
static int load_jpeg(const char* path, Image* img) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, f);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    img->W = cinfo.output_width;
    img->H = cinfo.output_height;
    img->pixels = (float*)calloc((size_t)(img->W * img->H * 3), sizeof(float));

    unsigned char* row = (unsigned char*)malloc((size_t)(img->W * 3));
    int y = 0;
    while (cinfo.output_scanline < (unsigned int)cinfo.output_height) {
        unsigned char* buffer[1] = { row };
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (int x = 0; x < img->W; x++) {
            img->pixels[(y * img->W + x) * 3 + 0] = (float)row[x * 3 + 0] / 255.0f;
            img->pixels[(y * img->W + x) * 3 + 1] = (float)row[x * 3 + 1] / 255.0f;
            img->pixels[(y * img->W + x) * 3 + 2] = (float)row[x * 3 + 2] / 255.0f;
        }
        y++;
    }

    free(row);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    return 0;
}

static void free_image(Image* img) {
    if (img->pixels) { free(img->pixels); img->pixels = NULL; }
}

/* Hamilton encode: RGB Image → quaternion latent */
static void hamilton_encode(const Image* img, float* quaternions, float* amplitude, float* ctx) {
    int N = img->W * img->H;
    float mean_r = 0, mean_g = 0, mean_b = 0;

    for (int y = 0; y < img->H; y++) {
        for (int x = 0; x < img->W; x++) {
            int i = y * img->W + x;
            float r = img->pixels[i * 3 + 0];
            float g = img->pixels[i * 3 + 1];
            float b = img->pixels[i * 3 + 2];
            float u = (float)x / (float)(img->W - 1);
            float v = (float)y / (float)(img->H - 1);

            float qx = u * 2.0f - 1.0f;
            float qy = v * 2.0f - 1.0f;
            float qz = (r + g + b) / 3.0f;
            float qw = 1.0f;
            float norm = sqrtf(qx*qx + qy*qy + qz*qz + qw*qw);
            if (norm < 1e-8f) norm = 1.0f;

            quaternions[i * 4 + 0] = qx / norm;
            quaternions[i * 4 + 1] = qy / norm;
            quaternions[i * 4 + 2] = qz / norm;
            quaternions[i * 4 + 3] = qw / norm;
            amplitude[i] = 0.2989f * r + 0.5870f * g + 0.1140f * b;

            mean_r += r; mean_g += g; mean_b += b;
        }
    }
    ctx[0] = mean_r / (float)N;
    ctx[1] = mean_g / (float)N;
    ctx[2] = mean_b / (float)N;
}

/* Hamilton decode: quaternion latent → RGB Image */
static void hamilton_decode(const float* quaternions, const float* amplitude, Image* img) {
    int N = img->W * img->H;
    img->pixels = (float*)calloc((size_t)(N * 3), sizeof(float));

    for (int i = 0; i < N; i++) {
        float qx = quaternions[i * 4 + 0];
        float qy = quaternions[i * 4 + 1];
        float qz = quaternions[i * 4 + 2];
        float qw = quaternions[i * 4 + 3];
        float amp = amplitude[i];

        img->pixels[i * 3 + 0] = fminf(1.0f, fmaxf(0.0f, qx * amp + 0.5f + qw * 0.1f));
        img->pixels[i * 3 + 1] = fminf(1.0f, fmaxf(0.0f, qy * amp + 0.5f + qw * 0.05f));
        img->pixels[i * 3 + 2] = fminf(1.0f, fmaxf(0.0f, qz * amp + 0.5f + qw * 0.0f));
    }
}

static float compute_psnr(const Image* orig, const Image* recon) {
    if (!orig->pixels || !recon->pixels) return 0.0f;
    float mse = 0.0f;
    int N = orig->W * orig->H * 3;
    for (int i = 0; i < N; i++) {
        float d = orig->pixels[i] - recon->pixels[i];
        mse += d * d;
    }
    mse /= (float)N;
    if (mse < 1e-10f) return 100.0f;
    return 10.0f * log10f(1.0f / mse);
}

static int cmp_str(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <frames_dir> [epochs] [target_W]\n", argv[0]);
        fprintf(stderr, "  frames_dir: directory containing .jpg frames\n");
        fprintf(stderr, "  epochs:     number of training epochs (default: 3)\n");
        fprintf(stderr, "  target_W:   target width in pixels, height auto (default: 80)\n");
        return 1;
    }

    const char* frames_dir = argv[1];
    int num_epochs = (argc > 2) ? atoi(argv[2]) : 3;
    int target_W = (argc > 3) ? atoi(argv[3]) : 80;
    int target_H = target_W * 3 / 4;  /* 4:3 aspect ratio */

    printf("========================================================\n");
    printf("  WuBu Video Compression Training (v2)\n");
    printf("========================================================\n");
    printf("Frames:    %s\n", frames_dir);
    printf("Epochs:    %d\n", num_epochs);
    printf("Target:    %dx%d\n\n", target_W, target_H);

    /* Scan for JPEG frames - fixed buffer, no realloc */
    char** frame_paths = NULL;
    int num_frames = 0;

    /* Allocate fixed array */
    char path_buf[256][1024];
    char* frame_path_ptrs[256];

    DIR* d = opendir(frames_dir);
    if (!d) { fprintf(stderr, "Cannot open %s\n", frames_dir); return 1; }

    struct dirent* ent;
    while ((ent = readdir(d)) != NULL && num_frames < 256) {
        if (ent->d_name[0] == '.') continue;
        size_t len = strlen(ent->d_name);
        if (len > 4 && strcmp(ent->d_name + len - 4, ".jpg") == 0) {
            snprintf(path_buf[num_frames], sizeof(path_buf[0]), "%s/%s", frames_dir, ent->d_name);
            frame_path_ptrs[num_frames] = path_buf[num_frames];
            num_frames++;
            if (num_frames >= 200) break;
        }
    }
    closedir(d);

    if (num_frames < 2) {
        fprintf(stderr, "Need at least 2 frames, found %d\n", num_frames);
        return 1;
    }
    frame_paths = frame_path_ptrs;
    printf("Found %d frames (limited to 200)\n\n", num_frames);

    int N = target_W * target_H;
    int D = 4;

    /* Init flow matching */
    WubuFlowConfig flow_config = {
        .latent_dim = D,
        .hidden_dim = 32,
        .num_layers = 1,
        .num_freqs = 2,
        .sigma_min = 0.001f,
        .sigma_max = 1.0f,
        .learning_rate = 5e-3f,
        .batch_size = 4,
        .ode_steps = 5
    };
    WubuFlowMatching flow;
    wubu_flow_init(&flow, &flow_config, 1.0f);

    int total_steps = 0;
    float total_psnr = 0.0f;
    int psnr_count = 0;

    /* Training loop */
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int epoch_samples = 0;

        for (int f = 0; f < num_frames - 1; f++) {
            Image img0, img1;
            if (load_jpeg(frame_paths[f], &img0) != 0) continue;
            if (load_jpeg(frame_paths[f + 1], &img1) != 0) {
                free_image(&img0);
                continue;
            }

            /* Resize */
            Image r0, r1;
            r0.W = target_W; r0.H = target_H;
            r0.pixels = (float*)calloc((size_t)(N * 3), sizeof(float));
            r1.W = target_W; r1.H = target_H;
            r1.pixels = (float*)calloc((size_t)(N * 3), sizeof(float));

            for (int y = 0; y < target_H; y++) {
                for (int x = 0; x < target_W; x++) {
                    int sx = x * img0.W / target_W;
                    int sy = y * img0.H / target_H;
                    memcpy(&r0.pixels[(y * target_W + x) * 3], &img0.pixels[(sy * img0.W + sx) * 3], 3 * sizeof(float));
                    sx = x * img1.W / target_W;
                    sy = y * img1.H / target_H;
                    memcpy(&r1.pixels[(y * target_W + x) * 3], &img1.pixels[(sy * img1.W + sx) * 3], 3 * sizeof(float));
                }
            }
            free_image(&img0);
            free_image(&img1);

            /* Encode */
            float* q0 = (float*)calloc((size_t)(N * D), sizeof(float));
            float* a0 = (float*)calloc((size_t)N, sizeof(float));
            float* q1 = (float*)calloc((size_t)(N * D), sizeof(float));
            float* a1 = (float*)calloc((size_t)N, sizeof(float));
            float ctx0[3], ctx1[3];

            hamilton_encode(&r0, q0, a0, ctx0);
            hamilton_encode(&r1, q1, a1, ctx1);

            /* Compress + Decompress + PSNR */
            WubuCompressedLatent comp = wubu_latent_compress(q0, a0, N, WUBU_QUALITY_MEDIUM);
            float* qd = (float*)malloc((size_t)(N * D) * sizeof(float));
            float* ad = (float*)malloc((size_t)N * sizeof(float));
            wubu_latent_decompress(qd, ad, &comp);

            Image recon;
            recon.W = target_W; recon.H = target_H;
            hamilton_decode(qd, ad, &recon);

            float psnr = compute_psnr(&r0, &recon);
            total_psnr += psnr;
            psnr_count++;

            /* Train flow matching */
            float* pair = (float*)malloc((size_t)(2 * N * D) * sizeof(float));
            memcpy(pair, q0, (size_t)(N * D) * sizeof(float));
            memcpy(pair + N * D, q1, (size_t)(N * D) * sizeof(float));
            float loss = wubu_flow_train_step(&flow, pair, 2, N);
            free(pair);

            epoch_loss += loss;
            epoch_samples++;
            total_steps++;

            /* Cleanup */
            free_image(&recon);
            free(q0); free(a0); free(q1); free(a1); free(qd); free(ad);
            free_image(&r0);
            free_image(&r1);
            wubu_latent_compressed_free(&comp);

            if (epoch_samples % 10 == 0) {
                printf("  Step %d: loss=%.6f psnr=%.2f dB\n", epoch_samples, loss, psnr);
                fflush(stdout);
            }
        }

        printf("Epoch %2d: avg_loss=%.6f avg_psnr=%.2f dB samples=%d\n",
               epoch, epoch_loss / (float)epoch_samples,
               total_psnr / (float)psnr_count, epoch_samples);
        fflush(stdout);
    }

    printf("\n========================================================\n");
    printf("  Training Complete: %d steps, avg PSNR=%.2f dB\n",
           total_steps, total_psnr / (float)psnr_count);
    printf("========================================================\n");

    wubu_flow_free(&flow);
    /* No need to free fixed arrays */
    return 0;
}
