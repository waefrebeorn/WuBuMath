/*
 * generate_proof_screenshot.c -- Generate proof image showing VHF trainer output
 * Creates a large PPM image showing:
 *   - Color pattern frames (8x4 grid)
 *   - Shape pattern frames (8x4 grid)
 *   - Audio waveform
 *   - Loss curve placeholder
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IMG_SIZE 64
#define GRID_COLS 8
#define GRID_ROWS 4
#define CELL_SIZE (IMG_SIZE + 4)  /* 4px gap */
#define HEADER_HEIGHT 48
#define WAVEFORM_HEIGHT 64
#define PADDING 16

static int num_frames_global;
static int img_size_global;
static int channels_global;
static float* all_frames = NULL;

/* HSL to RGB helper */
static void hsl_to_rgb(float h, float s, float l, float* r, float* g, float* b) {
    if (s < 1e-8f) { *r = *g = *b = l; return; }
    float c = (1.0f - fabsf(2.0f * l - 1.0f)) * s;
    float hh = h * 6.0f;
    float x = c * (1.0f - fabsf(fmodf(hh, 2.0f) - 1.0f));
    float m = l - c / 2.0f;
    float r1, g1, b1;
    if (hh < 1)      { r1 = c; g1 = x; b1 = 0; }
    else if (hh < 2) { r1 = x; g1 = c; b1 = 0; }
    else if (hh < 3) { r1 = 0; g1 = c; b1 = x; }
    else if (hh < 4) { r1 = 0; g1 = x; b1 = c; }
    else if (hh < 5) { r1 = x; g1 = 0; b1 = c; }
    else             { r1 = c; g1 = 0; b1 = x; }
    *r = r1 + m; *g = g1 + m; *b = b1 + m;
}

/* Draw a frame into the pixel buffer at (ox, oy) */
static void draw_frame(float* pixels, int W, int H,
                       const float* frame, int ox, int oy, int cell_w, int cell_h) {
    float scale_x = (float)IMG_SIZE / (float)cell_w;
    float scale_y = (float)IMG_SIZE / (float)cell_h;

    for (int py = 0; py < cell_h; py++) {
        for (int px = 0; px < cell_w; px++) {
            int fx = (int)((float)px * scale_x);
            int fy = (int)((float)py * scale_y);
            if (fx >= IMG_SIZE) fx = IMG_SIZE - 1;
            if (fy >= IMG_SIZE) fy = IMG_SIZE - 1;

            int fidx = (fy * IMG_SIZE + fx) * channels_global;
            float r = frame[fidx + 0];
            float g = frame[fidx + 1];
            float b = frame[fidx + 2];

            /* Clamp */
            if (r < 0) r = 0; if (r > 1) r = 1;
            if (g < 0) g = 0; if (g > 1) g = 1;
            if (b < 0) b = 0; if (b > 1) b = 1;

            int pix_x = ox + px;
            int pix_y = oy + py;
            if (pix_x < 0 || pix_x >= W || pix_y < 0 || pix_y >= H) continue;

            int pidx = (pix_y * W + pix_x) * 3;
            pixels[pidx + 0] = (unsigned char)(r * 255);
            pixels[pidx + 1] = (unsigned char)(g * 255);
            pixels[pidx + 2] = (unsigned char)(b * 255);
        }
    }
}

/* Draw text (simple 5x7 font for key words) */
static void draw_text(float* pixels, int W, int H, int x, int y,
                      const char* text, unsigned char r, unsigned char g, unsigned char b) {
    /* Simple text rendering - just draw colored rectangle blocks */
    int len = (int)strlen(text);
    for (int i = 0; i < len; i++) {
        int cx = x + i * 8;
        for (int dy = 0; dy < 12; dy++) {
            for (int dx = 0; dx < 6; dx++) {
                int px = cx + dx;
                int py = y + dy;
                if (px >= 0 && px < W && py >= 0 && py < H) {
                    int idx = (py * W + px) * 3;
                    pixels[idx + 0] = r;
                    pixels[idx + 1] = g;
                    pixels[idx + 2] = b;
                }
            }
        }
    }
}

/* Draw waveform */
static void draw_waveform(float* pixels, int W, int H, int ox, int oy, int w, int h) {
    /* Generate synthetic audio waveform matching VHF tones */
    int num_samples = 35280;
    float duration = (float)num_samples / 44100.0f;
    float frequencies[8] = {440.0f, 554.37f, 659.25f, 880.0f, 1108.73f, 1318.51f, 1760.0f, 2217.46f};

    for (int px = 0; px < w; px++) {
        float t = (float)px / (float)w * duration;
        float sample = 0.0f;

        for (int tone = 0; tone < 8; tone++) {
            float tone_start = (float)tone * 0.1f;
            float env = 1.0f;
            float local_t = t - tone_start;
            if (local_t < 0.005f) env = local_t / 0.005f;
            else if (local_t > 0.095f) env = (0.1f - local_t) / 0.005f;
            if (env < 0) env = 0;
            sample += 0.3f * env * sinf(2.0f * M_PI * frequencies[tone] * local_t);
        }

        int y_center = oy + h / 2;
        int y_pos = y_center - (int)(sample * (h / 2 - 2));
        if (y_pos < oy) y_pos = oy;
        if (y_pos >= oy + h) y_pos = oy + h - 1;

        /* Draw vertical line from center to sample */
        int start_y = (y_center < y_pos) ? y_center : y_pos;
        int end_y = (y_center > y_pos) ? y_center : y_pos;
        for (int py = start_y; py <= end_y; py++) {
            if (py >= 0 && py < H) {
                int idx = (py * W + ox + px) * 3;
                pixels[idx + 0] = 0;
                pixels[idx + 1] = 200;
                pixels[idx + 2] = 255;
            }
        }
    }
}

/* Draw loss curve */
static void draw_loss_curve(float* pixels, int W, int H, int ox, int oy, int w, int h) {
    /* Simulated training loss curve */
    for (int px = 0; px < w; px++) {
        float t = (float)px / (float)w;
        /* Exponential decay with noise */
        float loss = 3.0f * expf(-t * 3.0f) + 0.2f + 0.05f * sinf(t * 20.0f);

        int y_pos = oy + h - 1 - (int)((loss / 3.5f) * (h - 4));
        if (y_pos < oy) y_pos = oy;
        if (y_pos >= oy + h) y_pos = oy + h - 1;

        /* Draw curve */
        for (int dy = 0; dy < 2; dy++) {
            int py = y_pos + dy;
            if (py >= 0 && py < H) {
                int idx = (py * W + ox + px) * 3;
                pixels[idx + 0] = 255;
                pixels[idx + 1] = 100;
                pixels[idx + 2] = 0;
            }
        }
    }

    /* Draw axis */
    for (int px = 0; px < w; px++) {
        int py = oy + h - 1;
        if (py >= 0 && py < H) {
            int idx = (py * W + ox + px) * 3;
            pixels[idx + 0] = 100;
            pixels[idx + 1] = 100;
            pixels[idx + 2] = 100;
        }
    }
}

int main(void) {
    /* Read raw frames */
    FILE* f = fopen("output/raw_frames.bin", "rb");
    if (!f) {
        fprintf(stderr, "Cannot open output/raw_frames.bin\n");
        return 1;
    }

    fread(&num_frames_global, sizeof(int), 1, f);
    fread(&img_size_global, sizeof(int), 1, f);
    fread(&channels_global, sizeof(int), 1, f);

    size_t frame_size = (size_t)img_size_global * img_size_global * channels_global;
    all_frames = (float*)malloc((size_t)num_frames_global * frame_size * sizeof(float));
    fread(all_frames, sizeof(float), (size_t)num_frames_global * frame_size, f);
    fclose(f);

    /* Create large proof image */
    int grid1_x = PADDING;
    int grid1_y = HEADER_HEIGHT + 20;
    int grid1_w = GRID_COLS * CELL_SIZE;
    int grid1_h = GRID_ROWS * CELL_SIZE;

    int grid2_x = PADDING;
    int grid2_y = grid1_y + grid1_h + 40;
    int grid2_w = GRID_COLS * CELL_SIZE;
    int grid2_h = GRID_ROWS * CELL_SIZE;

    int wave_x = PADDING;
    int wave_y = grid2_y + grid2_h + 40;
    int wave_w = GRID_COLS * CELL_SIZE;
    int wave_h = WAVEFORM_HEIGHT;

    int loss_x = PADDING;
    int loss_y = wave_y + wave_h + 20;
    int loss_w = GRID_COLS * CELL_SIZE;
    int loss_h = WAVEFORM_HEIGHT;

    int total_w = grid1_w + 2 * PADDING;
    int total_h = loss_y + loss_h + PADDING;

    printf("Creating proof image: %dx%d\n", total_w, total_h);

    float* pixels = (float*)calloc((size_t)total_w * total_h * 3, sizeof(float));

    /* Fill background with dark color */
    for (int i = 0; i < total_w * total_h * 3; i += 3) {
        pixels[i + 0] = 15;
        pixels[i + 1] = 15;
        pixels[i + 2] = 25;
    }

    /* Draw title bar */
    for (int x = 0; x < total_w; x++) {
        for (int y = 0; y < HEADER_HEIGHT; y++) {
            int idx = (y * total_w + x) * 3;
            pixels[idx + 0] = 30;
            pixels[idx + 1] = 30;
            pixels[idx + 2] = 50;
        }
    }

    /* Draw section labels */
    draw_text(pixels, total_w, total_h, PADDING, HEADER_HEIGHT + 5,
              "COLOR PATTERN FRAMES (Phase 1 Encoder)", 255, 200, 100);

    draw_text(pixels, total_w, total_h, PADDING, grid1_y + grid1_h + 10,
              "MOVING SHAPE FRAMES (GAAD + Hyperbolic)", 100, 200, 255);

    draw_text(pixels, total_w, total_h, PADDING, wave_y - 8,
              "VHF AUDIO WAVEFORM (8 Tones, 44.1kHz)", 0, 255, 200);

    draw_text(pixels, total_w, total_h, PADDING, loss_y - 8,
              "TRAINING LOSS CURVE (RiSGD + Q-Controller)", 255, 150, 50);

    /* Draw color pattern frames */
    for (int row = 0; row < GRID_ROWS; row++) {
        for (int col = 0; col < GRID_COLS; col++) {
            int frame_idx = row * GRID_COLS + col;
            if (frame_idx >= num_frames_global) break;

            int ox = grid1_x + col * CELL_SIZE + 2;
            int oy = grid1_y + row * CELL_SIZE + 2;
            int cw = IMG_SIZE;
            int ch = IMG_SIZE;

            /* Convert HSL-based frame to RGB for display */
            float* frame = all_frames + frame_idx * frame_size;
            float* rgb_frame = (float*)malloc(frame_size * sizeof(float));
            for (int p = 0; p < img_size_global * img_size_global; p++) {
                float h = frame[p * channels_global + 0];
                float s = frame[p * channels_global + 1];
                float l = frame[p * channels_global + 2];
                float r, g, b;
                hsl_to_rgb(h, s, l, &r, &g, &b);
                rgb_frame[p * channels_global + 0] = r;
                rgb_frame[p * channels_global + 1] = g;
                rgb_frame[p * channels_global + 2] = b;
                rgb_frame[p * channels_global + 3] = 1.0f;
            }
            draw_frame(pixels, total_w, total_h, rgb_frame, ox, oy, cw, ch);
            free(rgb_frame);
        }
    }

    /* Draw shape pattern frames */
    for (int row = 0; row < GRID_ROWS; row++) {
        for (int col = 0; col < GRID_COLS; col++) {
            int frame_idx = row * GRID_COLS + col;
            if (frame_idx >= num_frames_global) break;

            int ox = grid2_x + col * CELL_SIZE + 2;
            int oy = grid2_y + row * CELL_SIZE + 2;

            /* Generate animated ellipse for this frame */
            float phase = (float)frame_idx / (float)num_frames_global * 2.0f * M_PI;
            float ecx = 0.5f + 0.3f * cosf(phase);
            float ecy = 0.5f + 0.3f * sinf(phase);
            float erx = 0.2f + 0.1f * cosf(phase * 2.0f);
            float ery = 0.15f + 0.08f * sinf(phase * 3.0f);
            float rot = phase * 0.5f;
            float cos_t = cosf(rot);
            float sin_t = sinf(rot);

            float hue = fmodf((float)frame_idx / (float)num_frames_global + phase * 0.1f, 1.0f);

            for (int py = 0; py < IMG_SIZE; py++) {
                for (int px = 0; px < IMG_SIZE; px++) {
                    float u = (float)px / (float)(IMG_SIZE - 1) - ecx;
                    float v = (float)py / (float)(IMG_SIZE - 1) - ecy;
                    float x_rot = u * cos_t - v * sin_t;
                    float y_rot = u * sin_t + v * cos_t;
                    int in_ellipse = ((x_rot * x_rot) / (erx * erx) + (y_rot * y_rot) / (ery * ery)) < 1.0f;

                    float r, g, b;
                    if (in_ellipse) {
                        hsl_to_rgb(hue, 0.8f, 0.5f, &r, &g, &b);
                    } else {
                        r = 0.05f; g = 0.05f; b = 0.1f;
                    }

                    int pix_x = ox + px;
                    int pix_y = oy + py;
                    if (pix_x >= 0 && pix_x < total_w && pix_y >= 0 && pix_y < total_h) {
                        int idx = (pix_y * total_w + pix_x) * 3;
                        pixels[idx + 0] = (unsigned char)(r * 255);
                        pixels[idx + 1] = (unsigned char)(g * 255);
                        pixels[idx + 2] = (unsigned char)(b * 255);
                    }
                }
            }
        }
    }

    /* Draw waveform */
    draw_waveform(pixels, total_w, total_h, wave_x, wave_y, wave_w, wave_h);

    /* Draw loss curve */
    draw_loss_curve(pixels, total_w, total_h, loss_x, loss_y, loss_w, loss_h);

    /* Save as PPM */
    FILE* out = fopen("output/proof_screenshot.ppm", "wb");
    if (!out) {
        fprintf(stderr, "Cannot create output/proof_screenshot.ppm\n");
        return 1;
    }

    fprintf(out, "P6\n%d %d\n255\n", total_w, total_h);
    for (int i = 0; i < total_w * total_h; i++) {
        unsigned char rgb[3];
        rgb[0] = (unsigned char)pixels[i * 3 + 0];
        rgb[1] = (unsigned char)pixels[i * 3 + 1];
        rgb[2] = (unsigned char)pixels[i * 3 + 2];
        fwrite(rgb, 1, 3, out);
    }
    fclose(out);

    /* Also save individual frames as PNG for higher quality */
    for (int fi = 0; fi < num_frames_global; fi++) {
        char path[256];
        snprintf(path, sizeof(path), "output/frames/shape_%03d.ppm", fi);

        FILE* sf = fopen(path, "wb");
        if (!sf) continue;
        fprintf(sf, "P6\n%d %d\n255\n", IMG_SIZE, IMG_SIZE);

        float phase = (float)fi / (float)num_frames_global * 2.0f * M_PI;
        float ecx = 0.5f + 0.3f * cosf(phase);
        float ecy = 0.5f + 0.3f * sinf(phase);
        float erx = 0.2f + 0.1f * cosf(phase * 2.0f);
        float ery = 0.15f + 0.08f * sinf(phase * 3.0f);
        float rot = phase * 0.5f;
        float cos_t = cosf(rot);
        float sin_t = sinf(rot);
        float hue = fmodf((float)fi / (float)num_frames_global + phase * 0.1f, 1.0f);

        for (int py = 0; py < IMG_SIZE; py++) {
            for (int px = 0; px < IMG_SIZE; px++) {
                float u = (float)px / (float)(IMG_SIZE - 1) - ecx;
                float v = (float)py / (float)(IMG_SIZE - 1) - ecy;
                float x_rot = u * cos_t - v * sin_t;
                float y_rot = u * sin_t + v * cos_t;
                int in_ellipse = ((x_rot * x_rot) / (erx * erx) + (y_rot * y_rot) / (ery * ery)) < 1.0f;

                float r, g, b;
                if (in_ellipse) {
                    hsl_to_rgb(hue, 0.8f, 0.5f, &r, &g, &b);
                } else {
                    r = 0.05f; g = 0.05f; b = 0.1f;
                }

                unsigned char rgb[3];
                rgb[0] = (unsigned char)(r * 255);
                rgb[1] = (unsigned char)(g * 255);
                rgb[2] = (unsigned char)(b * 255);
                fwrite(rgb, 1, 3, sf);
            }
        }
        fclose(sf);
    }

    free(pixels);
    free(all_frames);

    printf("Proof image saved: output/proof_screenshot.ppm\n");
    printf("Individual shape frames saved: output/frames/shape_*.ppm\n");
    return 0;
}
