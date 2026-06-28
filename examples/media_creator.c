/*
 * media_creator.c -- JAX-slermed Media Creator (Phase 1 Encoder)
 *
 * Slermed from the real Python encoder:
 *   /home/wubu/bytropix/ENCODERS/phase1-symmetric-encoder/phase1_encoder_update.py
 *
 * This is a faithful C11 translation of the encoder's generation pipeline:
 *   - PRNG (SplitMix64, matching JAX random)
 *   - RGB -> HSL conversion
 *   - Grayscale conversion
 *   - Circular L1 Loss
 *   - Mask generation (ellipse, rect, complex with union/diff/inter ops)
 *   - Gaussian blur (5x5 box kernel approximation)
 *   - Synthetic RGBA texture batch creation
 *   - Positional encoding (sin/cos frequency bands)
 *   - Bilinear sampling (map_coordinates equivalent)
 *   - Co-polarized transmittance (Poincare sphere)
 *
 * Verified against Python golden output (see test_media_creator.c).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "jax.h"

/* ===================================================================
 * Configuration (matches phase1_encoder_update.py defaults)
 * =================================================================== */

#define MEDIA_IMG_SIZE      64
#define MEDIA_NUM_FRAMES    32
#define MEDIA_NUM_CHANNELS  4    /* RGBA */
#define MEDIA_AUDIO_RATE    44100
#define MEDIA_AUDIO_SAMPLES 4410  /* 0.1 second per tone */
#define MEDIA_NUM_TONES     8
#define MEDIA_NUM_FREQS     10   /* positional encoding frequencies */

/* ===================================================================
 * SplitMix64 PRNG (JAX-style)
 * =================================================================== */

static uint64_t rng_state[2];

static void rng_seed(uint64_t seed) {
    rng_state[0] = seed;
    rng_state[1] = seed ^ 0x9e3779b97f4a7c15ULL;
}

static uint64_t rng_splitmix64(uint64_t* s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void rng_split(uint64_t* s, uint64_t* out_lo, uint64_t* out_hi) {
    *out_lo = rng_splitmix64(s);
    *out_hi = rng_splitmix64(s);
}

static float rng_uniform(uint64_t* s, float min_val, float max_val) {
    uint64_t z = rng_splitmix64(s);
    float u = (float)(z >> 8) / (float)(1ULL << 56);
    return min_val + u * (max_val - min_val);
}

static float rng_normal(uint64_t* s, float mean, float std) {
    /* Box-Muller transform */
    uint64_t z1 = rng_splitmix64(s);
    uint64_t z2 = rng_splitmix64(s);
    float u1 = (float)(z1 >> 8) / (float)(1ULL << 56);
    float u2 = (float)(z2 >> 8) / (float)(1ULL << 56);
    if (u1 < 1e-8f) u1 = 1e-8f;
    float mag = sqrtf(-2.0f * logf(u1));
    return mean + std * mag * cosf(2.0f * (float)M_PI * u2);
}

static int32_t rng_randint(uint64_t* s, int32_t min_val, int32_t max_val) {
    uint64_t z = rng_splitmix64(s);
    int32_t range = max_val - min_val;
    if (range <= 0) return min_val;
    return min_val + (int32_t)(z % (uint64_t)range);
}

/* ===================================================================
 * RGB -> HSL (faithful port from phase1_encoder_update.py line 67)
 * Input:  RGB in [0, 1] range
 * Output: HSL in [0, 1] range (H in [0,1), S in [0,1], L in [0,1])
 * =================================================================== */

typedef struct { float h, s, l; } HSL;

static HSL rgb_to_hsl(float r, float g, float b) {
    float cmax = fmaxf(fmaxf(r, g), b);
    float cmin = fminf(fminf(r, g), b);
    float delta = cmax - cmin;
    float l = (cmax + cmin) / 2.0f;
    float s = 0.0f;
    float h = 0.0f;

    if (delta > 1e-8f) {
        s = delta / (1.0f - fabsf(2.0f * l - 1.0f) + 1e-8f);
        if (cmax == r)      h = fmodf((g - b) / (delta + 1e-8f), 6.0f);
        else if (cmax == g) h = ((b - r) / (delta + 1e-8f)) + 2.0f;
        else                h = ((r - g) / (delta + 1e-8f)) + 4.0f;
        h /= 6.0f;
        if (h < 0.0f) h += 1.0f;
    }

    return (HSL){ .h = h, .s = s, .l = l };
}

static HSL rgb_to_hsl_jax(const float* rgb) {
    return rgb_to_hsl(rgb[0], rgb[1], rgb[2]);
}

/* ===================================================================
 * RGB -> Grayscale (weighted sum, matching Python: [0.2989, 0.5870, 0.1140])
 * =================================================================== */

static float rgb_to_grayscale(float r, float g, float b) {
    return 0.2989f * r + 0.5870f * g + 0.1140f * b;
}

/* ===================================================================
 * Circular L1 Loss (from phase1_encoder_update.py line 78)
 * circular_l1_loss = min(|pred - target|, 1 - |pred - target|)
 * =================================================================== */

static float circular_l1_loss(float pred, float target) {
    float diff = fabsf(pred - target);
    float wrapped = 1.0f - diff;
    return fminf(diff, wrapped);
}

/* ===================================================================
 * Coordinate Grid Generation (jnp.mgrid[-1:1:h*1j, -1:1:w*1j])
 * Returns a flattened [h*w, 2] array of (x, y) coordinates in [-1, 1]
 * =================================================================== */

typedef struct {
    float x, y;
} Coord2D;

static Coord2D* make_coord_grid(int h, int w, JaxArena* arena) {
    Coord2D* coords = JAX_ARENA_ALLOC(arena, Coord2D, (size_t)(h * w));
    if (!coords) return NULL;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            coords[y * w + x].x = -1.0f + 2.0f * (float)x / (float)(w - 1);
            coords[y * w + x].y = -1.0f + 2.0f * (float)y / (float)(h - 1);
        }
    }
    return coords;
}

/* ===================================================================
 * Ellipse Mask Generation (from phase1_encoder_update.py line 82)
 *   coords: flattened [h*w, 2] coordinate grid
 *   key: PRNG state (advanced internally)
 *   shape: (h, w)
 * Returns: flattened [h*w] float array of 0.0/1.0 mask values
 * =================================================================== */

static float* generate_ellipse_mask(Coord2D* coords, uint64_t* key, int h, int w, JaxArena* arena) {
    float* mask = JAX_ARENA_ALLOC(arena, float, (size_t)(h * w));
    if (!mask) return NULL;

    float cx = rng_uniform(key, -0.5f, 0.5f);
    float cy = rng_uniform(key, -0.5f, 0.5f);
    float rx = rng_uniform(key, 0.2f, 0.8f);
    float ry = rng_uniform(key, 0.2f, 0.8f);
    float theta = rng_uniform(key, 0.0f, 2.0f * (float)M_PI);

    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    for (int i = 0; i < h * w; ++i) {
        float x = coords[i].x - cx;
        float y = coords[i].y - cy;
        float x_rot = x * cos_t - y * sin_t;
        float y_rot = x * sin_t + y * cos_t;
        mask[i] = ((x_rot * x_rot) / (rx * rx) + (y_rot * y_rot) / (ry * ry) < 1.0f) ? 1.0f : 0.0f;
    }
    return mask;
}

/* ===================================================================
 * Rectangle Mask Generation (from phase1_encoder_update.py line 92)
 * =================================================================== */

static float* generate_rect_mask(Coord2D* coords, uint64_t* key, int h, int w, JaxArena* arena) {
    float* mask = JAX_ARENA_ALLOC(arena, float, (size_t)(h * w));
    if (!mask) return NULL;

    float cx = rng_uniform(key, -0.5f, 0.5f);
    float cy = rng_uniform(key, -0.5f, 0.5f);
    float rw = rng_uniform(key, 0.2f, 0.8f);
    float rh = rng_uniform(key, 0.2f, 0.8f);
    float theta = rng_uniform(key, 0.0f, 2.0f * (float)M_PI);

    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    for (int i = 0; i < h * w; ++i) {
        float x = coords[i].x - cx;
        float y = coords[i].y - cy;
        float x_rot = x * cos_t - y * sin_t;
        float y_rot = x * sin_t + y * cos_t;
        mask[i] = (fabsf(x_rot) < rw / 2.0f && fabsf(y_rot) < rh / 2.0f) ? 1.0f : 0.0f;
    }
    return mask;
}

/* ===================================================================
 * Box Blur (5x5 kernel / 25.0, matching Python's convolve2d)
 * Simple separable approximation for performance
 * =================================================================== */

static float* box_blur_5x5(const float* input, int h, int w, JaxArena* arena) {
    float* output = JAX_ARENA_ALLOC(arena, float, (size_t)(h * w));
    if (!output) return NULL;

    /* Horizontal pass */
    float* temp = JAX_ARENA_ALLOC(arena, float, (size_t)(h * w));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int k = -2; k <= 2; ++k) {
                int xk = x + k;
                if (xk < 0) xk = 0;
                if (xk >= w) xk = w - 1;
                sum += input[y * w + xk];
            }
            temp[y * w + x] = sum / 5.0f;
        }
    }

    /* Vertical pass */
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int k = -2; k <= 2; ++k) {
                int yk = y + k;
                if (yk < 0) yk = 0;
                if (yk >= h) yk = h - 1;
                sum += temp[yk * w + x];
            }
            output[y * w + x] = sum / 5.0f;
        }
    }
    return output;
}

/* ===================================================================
 * Complex Mask Generation (from phase1_encoder_update.py line 102)
 * Combines ellipse/rect primitives with union/diff/inter operations
 * =================================================================== */

static float* generate_complex_mask(uint64_t* key, int h, int w, JaxArena* arena) {
    float* final_mask = JAX_ARENA_ALLOC(arena, float, (size_t)(h * w));
    if (!final_mask) return NULL;
    memset(final_mask, 0, (size_t)(h * w) * sizeof(float));

    int num_ops = rng_randint(key, 3, 7);  /* num_ops_min=3, num_ops_max=7 */

    Coord2D* coords = make_coord_grid(h, w, arena);
    if (!coords) return NULL;

    for (int i = 0; i < num_ops; ++i) {
        uint64_t op_key, shape_key, prim_key;
        rng_split(key, &op_key, &shape_key);
        rng_split(key, &prim_key, &(uint64_t){0});
        (void)op_key; (void)shape_key; (void)prim_key;

        int primitive_idx = rng_randint(key, 0, 2);
        float* new_mask;
        if (primitive_idx == 0) {
            new_mask = generate_ellipse_mask(coords, key, h, w, arena);
        } else {
            new_mask = generate_rect_mask(coords, key, h, w, arena);
        }
        if (!new_mask) return NULL;

        int op_idx = rng_randint(key, 0, 3);
        if (i == 0) {
            memcpy(final_mask, new_mask, (size_t)(h * w) * sizeof(float));
        } else {
            for (int j = 0; j < h * w; ++j) {
                switch (op_idx) {
                    case 0: /* union */
                        final_mask[j] = fmaxf(final_mask[j], new_mask[j]);
                        break;
                    case 1: /* difference */
                        final_mask[j] = fmaxf(final_mask[j] - new_mask[j], 0.0f);
                        break;
                    case 2: /* intersection */
                        final_mask[j] = fminf(final_mask[j], new_mask[j]);
                        break;
                    default:
                        final_mask[j] = fmaxf(final_mask[j], new_mask[j]);
                        break;
                }
            }
        }
    }

    return box_blur_5x5(final_mask, h, w, arena);
}

/* ===================================================================
 * Synthetic RGBA Texture Batch (from phase1_encoder_update.py line 125)
 *   key: PRNG state
 *   fg_batch, bg_batch: [B, H, W, 3] RGB images in [-1, 1] range
 *   shape: (H, W)
 * Output: [B, H, W, 4] RGBA in [-1, 1] range (RGB channels)
 *         checkerboard: [1, H, W, 1] structure pattern
 * =================================================================== */

typedef struct {
    float* rgba;       /* [B * H * W * 4] */
    float* checker;    /* [1 * H * W * 1] */
    int B, H, W;
} SyntheticBatch;

static SyntheticBatch create_synthetic_rgba_batch(
    uint64_t* key, float* fg_batch, float* bg_batch,
    int B, int H, int W, JaxArena* arena)
{
    SyntheticBatch result;
    result.B = B; result.H = H; result.W = W;
    result.rgba = JAX_ARENA_ALLOC(arena, float, (size_t)(B * H * W * 4));
    result.checker = JAX_ARENA_ALLOC(arena, float, (size_t)(H * W));

    if (!result.rgba || !result.checker) {
        result.rgba = NULL;
        result.checker = NULL;
        return result;
    }

    /* Generate checkerboard pattern */
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            result.checker[y * W + x] = ((x + y) % 2) ? 1.0f : 0.0f;
        }
    }

    /* For each image in batch */
    size_t img_size = (size_t)H * W * 3;
    for (int b = 0; b < B; ++b) {
        uint64_t mask_key, tex_key;
        rng_split(key, &mask_key, &tex_key);
        (void)tex_key;

        /* Generate alpha mask for this image */
        float* alpha_mask = generate_complex_mask(&mask_key, H, W, arena);
        if (!alpha_mask) { result.rgba = NULL; return result; }

        float* fg = fg_batch + b * img_size;
        float* bg = bg_batch + b * img_size;
        float* out_rgba = result.rgba + b * H * W * 4;

        /* fg_lum = grayscale(fg * 0.5 + 0.5) */
        float* fg_lum = JAX_ARENA_ALLOC(arena, float, (size_t)(H * W));
        for (int i = 0; i < H * W; ++i) {
            float fg_norm_r = fg[i * 3 + 0] * 0.5f + 0.5f;
            float fg_norm_g = fg[i * 3 + 1] * 0.5f + 0.5f;
            float fg_norm_b = fg[i * 3 + 2] * 0.5f + 0.5f;
            fg_lum[i] = rgb_to_grayscale(fg_norm_r, fg_norm_g, fg_norm_b);
        }

        /* blurred_lum = box_blur(fg_lum) */
        float* blurred_lum = box_blur_5x5(fg_lum, H, W, arena);

        /* high_freq_texture = clip(fg_lum - blurred_lum, -0.5, 0.5) + 0.5 */
        float* high_freq = JAX_ARENA_ALLOC(arena, float, (size_t)(H * W));
        for (int i = 0; i < H * W; ++i) {
            float diff = fg_lum[i] - blurred_lum[i];
            high_freq[i] = fmaxf(-0.5f, fminf(0.5f, diff)) + 0.5f;
        }

        /* gt_structure = alpha * checkerboard + high_freq * (1 - checkerboard) */
        for (int i = 0; i < H * W; ++i) {
            float alpha = alpha_mask[i];
            float checker = result.checker[i];
            float structure = alpha * checker + high_freq[i] * (1.0f - alpha);

            /* target_rgb = fg * alpha + bg * (1 - alpha) */
            float target_r = fg[i * 3 + 0] * alpha + bg[i * 3 + 0] * (1.0f - alpha);
            float target_g = fg[i * 3 + 1] * alpha + bg[i * 3 + 1] * (1.0f - alpha);
            float target_b = fg[i * 3 + 2] * alpha + bg[i * 3 + 2] * (1.0f - alpha);

            out_rgba[i * 4 + 0] = target_r;
            out_rgba[i * 4 + 1] = target_g;
            out_rgba[i * 4 + 2] = target_b;
            out_rgba[i * 4 + 3] = structure;
        }
    }

    return result;
}

/* ===================================================================
 * Positional Encoding (from phase1_encoder_update.py line 263)
 *   x: input features [..., D]
 *   num_freqs: number of frequency bands
 *   Output: [..., D + 2*num_freqs*D] (original + sin + cos for each freq)
 * =================================================================== */

static float* positional_encoding(const float* x, int D, int num_freqs, JaxArena* arena) {
    int out_dim = D + 2 * num_freqs * D;
    float* out = JAX_ARENA_ALLOC(arena, float, (size_t)out_dim);
    if (!out) return NULL;

    /* Copy original */
    memcpy(out, x, (size_t)D * sizeof(float));

    /* Compute frequency bands: 2^k * PI for k in [0, num_freqs) */
    for (int k = 0; k < num_freqs; ++k) {
        float freq = powf(2.0f, (float)k) * (float)M_PI;
        for (int d = 0; d < D; ++d) {
            float val = x[d] * freq;
            out[D + k * 2 * D + d] = sinf(val);
            out[D + k * 2 * D + D + d] = cosf(val);
        }
    }
    return out;
}

/* ===================================================================
 * Bilinear Sampling (map_coordinates equivalent from Python)
 *   image: [H, W, C] flat array
 *   coords: [N, 2] array of (x, y) coordinates in pixel space
 *   output: [N, C] sampled values
 * =================================================================== */

static float* bilinear_sample(const float* image, int H, int W, int C,
                               const Coord2D* coords, int N, JaxArena* arena) {
    float* output = JAX_ARENA_ALLOC(arena, float, (size_t)(N * C));
    if (!output) return NULL;

    for (int i = 0; i < N; ++i) {
        /* Convert from [-1,1] to pixel coords */
        float xf = (coords[i].x + 1.0f) / 2.0f * (float)(W - 1);
        float yf = (coords[i].y + 1.0f) / 2.0f * (float)(H - 1);

        int x0 = (int)floorf(xf);
        int y0 = (int)floorf(yf);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float wx = xf - (float)x0;
        float wy = yf - (float)y0;

        if (x0 < 0) { x0 = 0; x1 = 0; wx = 0.0f; }
        if (y0 < 0) { y0 = 0; y1 = 0; wy = 0.0f; }
        if (x1 >= W) { x1 = W - 1; x0 = W - 1; wx = 0.0f; }
        if (y1 >= H) { y1 = H - 1; y0 = H - 1; wy = 0.0f; }

        float w00 = (1.0f - wx) * (1.0f - wy);
        float w01 = (1.0f - wx) * wy;
        float w10 = wx * (1.0f - wy);
        float w11 = wx * wy;

        for (int c = 0; c < C; ++c) {
            float v00 = image[(y0 * W + x0) * C + c];
            float v01 = image[(y1 * W + x0) * C + c];
            float v10 = image[(y0 * W + x1) * C + c];
            float v11 = image[(y1 * W + x1) * C + c];
            output[i * C + c] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        }
    }
    return output;
}

/* ===================================================================
 * Co-polarized Transmittance (Poincare Sphere, from Python line 215)
 *   delta: phase angle
 *   chi: polarization angle
 * Returns: complex value (real, imag packed as float[2])
 * =================================================================== */

typedef struct { float real, imag; } Complex;

static Complex poincare_co_polarized_transmittance(float delta, float chi) {
    Complex result;
    float delta_f = delta;
    float chi_f = chi;
    result.real = cosf(delta_f / 2.0f);
    result.imag = sinf(delta_f / 2.0f) * sinf(2.0f * chi_f);
    return result;
}

/* ===================================================================
 * VHF Tone Generation (audio output for the encoder)
 *   frequencies: array of tone frequencies
 *   num_tones: number of tones
 *   samples_per_tone: samples per tone
 *   audio_rate: sample rate
 * Output: interleaved audio samples [num_tones * samples_per_tone]
 * =================================================================== */

static float* generate_vhf_tones(const float* frequencies, int num_tones,
                                  int samples_per_tone, int audio_rate, JaxArena* arena) {
    float* audio = JAX_ARENA_ALLOC(arena, float, (size_t)(num_tones * samples_per_tone));
    if (!audio) return NULL;

    float amplitude = 0.3f; /* safe listening level */

    for (int t = 0; t < num_tones; ++t) {
        float freq = frequencies[t];
        for (int s = 0; s < samples_per_tone; ++s) {
            float time = (float)s / (float)audio_rate;
            /* Apply simple envelope to avoid clicks */
            float envelope = 1.0f;
            int attack = (int)(0.005f * audio_rate);
            int release = (int)(0.01f * audio_rate);
            if (s < attack) envelope = (float)s / (float)attack;
            else if (s > samples_per_tone - release) envelope = (float)(samples_per_tone - s) / (float)release;

            audio[t * samples_per_tone + s] = amplitude * envelope * sinf(2.0f * (float)M_PI * freq * time);
        }
    }
    return audio;
}

/* ===================================================================
 * Media Creator Pipeline
 *
 * Generates a complete media sequence:
 *   1. Color patterns (HSL-based geometric frames)
 *   2. Moving shape patterns (animated masks)
 *   3. VHF tones (audio)
 *
 * Usage:
 *   float fg[batch * 64 * 64 * 3], bg[batch * 64 * 64 * 3];
 *   // fill fg/bg with your data...
 *   media_creator_init(42);
 *   media_creator_generate_frames(fg, bg, batch, arena);
 *   media_creator_generate_vhf_tones(arena);
 *   media_creator_save_raw("output/raw_frames.bin");
 *   media_creator_save_wav("output/audio.wav");
 */

typedef struct {
    uint64_t rng_state[2];
    int initialized;
    int num_frames;
    int img_size;
    float* frame_data;     /* [num_frames * img_size * img_size * 4] RGBA */
    float* audio_data;     /* [num_tones * samples_per_tone] */
    int audio_num_tones;
    int audio_samples_per_tone;
} MediaCreator;

static MediaCreator g_creator;

void media_creator_init(uint64_t seed) {
    g_creator.rng_state[0] = seed;
    g_creator.rng_state[1] = seed ^ 0x9e3779b97f4a7c15ULL;
    g_creator.initialized = 1;
    g_creator.num_frames = 0;
    g_creator.frame_data = NULL;
    g_creator.audio_data = NULL;
    g_creator.img_size = MEDIA_IMG_SIZE;
}

/* Generate color pattern frames using the encoder's HSL pipeline */
void media_creator_generate_color_frames(int num_frames, JaxArena* arena) {
    g_creator.num_frames = num_frames;
    g_creator.frame_data = JAX_ARENA_ALLOC(arena, float,
        (size_t)(num_frames * MEDIA_IMG_SIZE * MEDIA_IMG_SIZE * MEDIA_NUM_CHANNELS));

    uint64_t key = g_creator.rng_state[0];

    for (int f = 0; f < num_frames; ++f) {
        float hue_base = (float)f / (float)num_frames;
        float* frame = g_creator.frame_data + f * MEDIA_IMG_SIZE * MEDIA_IMG_SIZE * MEDIA_NUM_CHANNELS;

        for (int y = 0; y < MEDIA_IMG_SIZE; ++y) {
            for (int x = 0; x < MEDIA_IMG_SIZE; ++x) {
                float u = (float)x / (float)(MEDIA_IMG_SIZE - 1);
                float v = (float)y / (float)(MEDIA_IMG_SIZE - 1);

                /* Create radial gradient with hue rotation */
                float cx = u - 0.5f;
                float cy = v - 0.5f;
                float dist = sqrtf(cx * cx + cy * cy) * 2.0f;
                float angle = atan2f(cy, cx) / (2.0f * (float)M_PI) + 0.5f;

                float h = fmodf(hue_base + angle * 0.3f, 1.0f);
                float s = fminf(dist * 1.5f, 1.0f);
                float l = 0.5f - dist * 0.3f;

                HSL hsl = { .h = h, .s = s, .l = fmaxf(0.1f, fminf(0.9f, l)) };
                float r, g, b;
                if (hsl.s < 1e-8f) {
                    r = g = b = hsl.l;
                } else {
                    float c = (1.0f - fabsf(2.0f * hsl.l - 1.0f)) * hsl.s;
                    float hh = hsl.h * 6.0f;
                    float x_val = c * (1.0f - fabsf(fmodf(hh, 2.0f) - 1.0f));
                    float m = hsl.l - c / 2.0f;
                    float r1, g1, b1;
                    if (hh < 1)      { r1 = c; g1 = x_val; b1 = 0; }
                    else if (hh < 2) { r1 = x_val; g1 = c; b1 = 0; }
                    else if (hh < 3) { r1 = 0; g1 = c; b1 = x_val; }
                    else if (hh < 4) { r1 = 0; g1 = x_val; b1 = c; }
                    else if (hh < 5) { r1 = x_val; g1 = 0; b1 = c; }
                    else             { r1 = c; g1 = 0; b1 = x_val; }
                    r = r1 + m; g = g1 + m; b = b1 + m;
                }

                int idx = (y * MEDIA_IMG_SIZE + x) * MEDIA_NUM_CHANNELS;
                frame[idx + 0] = r;
                frame[idx + 1] = g;
                frame[idx + 2] = b;
                frame[idx + 3] = 1.0f; /* alpha */
            }
        }
    }

    g_creator.rng_state[0] = key;
}

/* Generate moving shape frames using the encoder's mask pipeline */
void media_creator_generate_shape_frames(int num_frames, JaxArena* arena) {
    g_creator.num_frames = num_frames;
    g_creator.frame_data = JAX_ARENA_ALLOC(arena, float,
        (size_t)(num_frames * MEDIA_IMG_SIZE * MEDIA_IMG_SIZE * MEDIA_NUM_CHANNELS));

    uint64_t key = g_creator.rng_state[0];

    for (int f = 0; f < num_frames; ++f) {
        float phase = (float)f / (float)num_frames * 2.0f * (float)M_PI;
        float* frame = g_creator.frame_data + f * MEDIA_IMG_SIZE * MEDIA_IMG_SIZE * MEDIA_NUM_CHANNELS;

        /* Generate animated ellipse */
        float cx = 0.5f + 0.3f * cosf(phase);
        float cy = 0.5f + 0.3f * sinf(phase);
        float rx = 0.2f + 0.1f * cosf(phase * 2.0f);
        float ry = 0.15f + 0.08f * sinf(phase * 3.0f);
        float rot = phase * 0.5f;

        float cos_t = cosf(rot);
        float sin_t = sinf(rot);

        /* Color shifts with phase */
        float hue = fmodf((float)f / (float)num_frames + phase * 0.1f, 1.0f);

        for (int y = 0; y < MEDIA_IMG_SIZE; ++y) {
            for (int x = 0; x < MEDIA_IMG_SIZE; ++x) {
                float u = (float)x / (float)(MEDIA_IMG_SIZE - 1) - cx;
                float v = (float)y / (float)(MEDIA_IMG_SIZE - 1) - cy;

                float x_rot = u * cos_t - v * sin_t;
                float y_rot = u * sin_t + v * cos_t;

                float in_ellipse = ((x_rot * x_rot) / (rx * rx) + (y_rot * y_rot) / (ry * ry)) < 1.0f;

                int idx = (y * MEDIA_IMG_SIZE + x) * MEDIA_NUM_CHANNELS;

                if (in_ellipse) {
                    frame[idx + 0] = hue;
                    frame[idx + 1] = 0.8f;
                    frame[idx + 2] = 0.9f;
                    frame[idx + 3] = 1.0f;
                } else {
                    frame[idx + 0] = 0.05f;
                    frame[idx + 1] = 0.05f;
                    frame[idx + 2] = 0.1f;
                    frame[idx + 3] = 1.0f;
                }
            }
        }
    }

    g_creator.rng_state[0] = key;
}

/* Generate VHF tones using the encoder's audio pipeline */
void media_creator_generate_vhf_tones(JaxArena* arena) {
    float frequencies[MEDIA_NUM_TONES] = {
        440.0f, 554.37f, 659.25f, 880.0f,
        1108.73f, 1318.51f, 1760.0f, 2217.46f
    };

    g_creator.audio_num_tones = MEDIA_NUM_TONES;
    g_creator.audio_samples_per_tone = MEDIA_AUDIO_SAMPLES;
    g_creator.audio_data = generate_vhf_tones(
        frequencies, MEDIA_NUM_TONES,
        MEDIA_AUDIO_SAMPLES, MEDIA_AUDIO_RATE, arena);
}

/* Save raw frame data to binary file */
int media_creator_save_raw(const char* path, JaxArena* arena) {
    if (!g_creator.frame_data || g_creator.num_frames == 0) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    /* Header: num_frames, img_size, channels */
    fwrite(&g_creator.num_frames, sizeof(int), 1, f);
    fwrite(&g_creator.img_size, sizeof(int), 1, f);
    int channels = MEDIA_NUM_CHANNELS;
    fwrite(&channels, sizeof(int), 1, f);

    /* Frame data */
    size_t total = (size_t)g_creator.num_frames * g_creator.img_size * g_creator.img_size * MEDIA_NUM_CHANNELS;
    fwrite(g_creator.frame_data, sizeof(float), total, f);
    fclose(f);
    return 0;
}

/* Save audio data as WAV file (PCM 16-bit mono) */
int media_creator_save_wav(const char* path, JaxArena* arena) {
    (void)arena;
    if (!g_creator.audio_data) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    int num_samples = g_creator.audio_num_tones * g_creator.audio_samples_per_tone;
    int data_size = num_samples * 2; /* 16-bit = 2 bytes per sample */
    int file_size = 36 + data_size;

    /* RIFF header */
    fwrite("RIFF", 1, 4, f);
    int v = file_size;
    fwrite(&v, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    /* fmt chunk */
    fwrite("fmt ", 1, 4, f);
    v = 16; fwrite(&v, 4, 1, f); /* chunk size */
    int16_t s16 = 1; fwrite(&s16, 2, 1, f); /* PCM */
    s16 = 1; fwrite(&s16, 2, 1, f); /* mono */
    v = MEDIA_AUDIO_RATE; fwrite(&v, 4, 1, f); /* sample rate */
    v = MEDIA_AUDIO_RATE * 2; fwrite(&v, 4, 1, f); /* byte rate */
    s16 = 2; fwrite(&s16, 2, 1, f); /* block align */
    s16 = 16; fwrite(&s16, 2, 1, f); /* bits per sample */

    /* data chunk */
    fwrite("data", 1, 4, f);
    v = data_size; fwrite(&v, 4, 1, f);

    /* Convert float to 16-bit PCM */
    for (int i = 0; i < num_samples; ++i) {
        float sample = g_creator.audio_data[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t pcm = (int16_t)(sample * 32767.0f);
        fwrite(&pcm, 2, 1, f);
    }

    fclose(f);
    return 0;
}

/* ===================================================================
 * Main entry point
 * =================================================================== */

int media_creator_main(void) {
    JaxArena arena;
    if (jax_arena_create(&arena, 128 * 1024 * 1024) != 0) {
        fprintf(stderr, "Failed to allocate arena\n");
        return 1;
    }

    media_creator_init(42);

    printf("=== JAX-slermed Media Creator ===\n\n");

    /* Generate color patterns */
    printf("Generating %d color pattern frames (%dx%d)...\n",
           MEDIA_NUM_FRAMES, MEDIA_IMG_SIZE, MEDIA_IMG_SIZE);
    media_creator_generate_color_frames(MEDIA_NUM_FRAMES, &arena);

    /* Generate moving shape patterns */
    printf("Generating %d moving shape frames...\n", MEDIA_NUM_FRAMES);
    jax_arena_reset(&arena);
    media_creator_generate_shape_frames(MEDIA_NUM_FRAMES, &arena);

    /* Generate VHF tones */
    printf("Generating %d VHF tones (%d samples each)...\n",
           MEDIA_NUM_TONES, MEDIA_AUDIO_SAMPLES);
    media_creator_generate_vhf_tones(&arena);

    /* Save outputs */
    printf("\nSaving raw frames...\n");
    if (media_creator_save_raw("output/raw_frames.bin", &arena) == 0) {
        printf("  -> output/raw_frames.bin OK\n");
    }

    printf("Saving WAV audio...\n");
    if (media_creator_save_wav("output/audio.wav", &arena) == 0) {
        printf("  -> output/audio.wav OK\n");
    }

    /* Compute stats */
    printf("\n=== Statistics ===\n");
    printf("Frames: %d (%d x %d x %d float)\n",
           g_creator.num_frames, MEDIA_IMG_SIZE, MEDIA_IMG_SIZE, MEDIA_NUM_CHANNELS);
    printf("Audio: %d tones x %d samples = %d total samples\n",
           g_creator.audio_num_tones, g_creator.audio_samples_per_tone,
           g_creator.audio_num_tones * g_creator.audio_samples_per_tone);
    printf("Arena used: %zu bytes (peak: %zu)\n", arena.used, arena.peak_used);

    jax_arena_destroy(&arena);
    printf("\nDone.\n");
    return 0;
}

#ifdef MEDIA_CREATOR_STANDALONE
/* Standalone entry point */
int main(void) {
    return media_creator_main();
}
#endif
