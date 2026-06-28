/*
 * phase1.c -- Phase 1 Symmetric Encoder (slermed from phase1_encoder_update.py)
 *
 * Pure C11 port of the symmetric encoder pipeline:
 *   - Complex mask generation (ellipse, rect, boolean ops)
 *   - Synthetic RGBA texture batch
 *   - Color pattern generation (32 frames)
 *   - Moving shape patterns (animated)
 *   - VHF tone pipeline (8 tones -> WAV)
 *   - Poincare sphere co-polarized transmittance
 *   - FiLM layer
 *   - Coordinate decoder MLP
 */

#include "wubumath.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===================================================================
 * Phase 1 Encoder Config
 * =================================================================== */

typedef struct {
    int latent_grid_size;   /* default 96 */
    int d_model;            /* default 64 */
    int latent_dim;         /* default 96 */
    int num_freqs;          /* default 10 */
    int mlp_width;          /* default 256 */
    int mlp_depth;          /* default 4 */
    int num_path_steps;     /* default 16 */
} Phase1Config;

static const Phase1Config PHASE1_DEFAULT = {
    .latent_grid_size = 96,
    .d_model = 64,
    .latent_dim = 96,
    .num_freqs = 10,
    .mlp_width = 256,
    .mlp_depth = 4,
    .num_path_steps = 16
};

/* ===================================================================
 * Complex Mask Generation
 * =================================================================== */

/* Generate ellipse mask [H x W] in [0,1] */
static float* generate_ellipse_mask(WubuRNG* rng, int H, int W) {
    float* mask = (float*)calloc((size_t)(H * W), sizeof(float));
    if (!mask) return NULL;

    uint64_t keys[6];
    for (int i = 0; i < 6; ++i) keys[i] = wubu_rng_next(rng);

    float cx = -0.5f + (float)(keys[0] >> 8) / (float)(1ULL << 56) * 1.0f;  /* [-0.5, 0.5] */
    float cy = -0.5f + (float)(keys[1] >> 8) / (float)(1ULL << 56) * 1.0f;
    float rx = 0.2f + (float)(keys[2] >> 8) / (float)(1ULL << 56) * 0.6f;  /* [0.2, 0.8] */
    float ry = 0.2f + (float)(keys[3] >> 8) / (float)(1ULL << 56) * 0.6f;
    float theta = (float)(keys[4] >> 8) / (float)(1ULL << 56) * 2.0f * (float)M_PI;

    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float xf = ((float)x / (float)(W - 1)) * 2.0f - 1.0f;
            float yf = ((float)y / (float)(H - 1)) * 2.0f - 1.0f;
            float dx = xf - cx;
            float dy = yf - cy;
            float x_rot = dx * cos_t - dy * sin_t;
            float y_rot = dx * sin_t + dy * cos_t;
            float inside = (x_rot * x_rot) / (rx * rx) + (y_rot * y_rot) / (ry * ry);
            if (inside < 1.0f) {
                mask[y * W + x] = 1.0f;
            }
        }
    }
    return mask;
}

/* Generate rectangle mask [H x W] in [0,1] */
static float* generate_rect_mask(WubuRNG* rng, int H, int W) {
    float* mask = (float*)calloc((size_t)(H * W), sizeof(float));
    if (!mask) return NULL;

    uint64_t keys[6];
    for (int i = 0; i < 6; ++i) keys[i] = wubu_rng_next(rng);

    float cx = -0.5f + (float)(keys[0] >> 8) / (float)(1ULL << 56);
    float cy = -0.5f + (float)(keys[1] >> 8) / (float)(1ULL << 56);
    float rw = 0.2f + (float)(keys[2] >> 8) / (float)(1ULL << 56) * 0.6f;
    float rh = 0.2f + (float)(keys[3] >> 8) / (float)(1ULL << 56) * 0.6f;
    float theta = (float)(keys[4] >> 8) / (float)(1ULL << 56) * 2.0f * (float)M_PI;

    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float xf = ((float)x / (float)(W - 1)) * 2.0f - 1.0f;
            float yf = ((float)y / (float)(H - 1)) * 2.0f - 1.0f;
            float dx = xf - cx;
            float dy = yf - cy;
            float x_rot = dx * cos_t - dy * sin_t;
            float y_rot = dx * sin_t + dy * cos_t;
            if (fabsf(x_rot) < rw / 2.0f && fabsf(y_rot) < rh / 2.0f) {
                mask[y * W + x] = 1.0f;
            }
        }
    }
    return mask;
}

/* Generate complex mask: combine multiple primitives with boolean ops */
static float* generate_complex_mask(WubuRNG* rng, int H, int W, int num_ops_min, int num_ops_max) {
    /* Determine number of operations */
    uint64_t num_ops_raw = wubu_rng_next(rng);
    int num_ops = num_ops_min + (int)(num_ops_raw % (uint64_t)(num_ops_max - num_ops_min + 1));
    if (num_ops < 1) num_ops = 1;

    float* final_mask = (float*)calloc((size_t)(H * W), sizeof(float));
    if (!final_mask) return NULL;

    for (int op = 0; op < num_ops; ++op) {
        /* Choose primitive: 0=ellipse, 1=rect */
        uint64_t prim_key = wubu_rng_next(rng);
        float* new_mask;
        if ((prim_key & 1) == 0) {
            new_mask = generate_ellipse_mask(rng, H, W);
        } else {
            new_mask = generate_rect_mask(rng, H, W);
        }

        /* Choose boolean op: 0=union, 1=difference, 2=intersection */
        uint64_t op_key = wubu_rng_next(rng);
        int op_idx = (int)(op_key % 3);

        if (op == 0) {
            /* First op: just use the new mask directly */
            memcpy(final_mask, new_mask, (size_t)(H * W) * sizeof(float));
        } else {
            for (int i = 0; i < H * W; ++i) {
                switch (op_idx) {
                    case 0: /* Union */
                        final_mask[i] = fmaxf(final_mask[i], new_mask[i]);
                        break;
                    case 1: /* Difference */
                        final_mask[i] = fmaxf(final_mask[i] - new_mask[i], 0.0f);
                        break;
                    case 2: /* Intersection */
                        final_mask[i] = fminf(final_mask[i], new_mask[i]);
                        break;
                }
            }
        }
        free(new_mask);
    }

    /* Apply 5x5 box blur (separable, matching Python convolve2d) */
    float* blurred = wubu_box_blur_5x5(final_mask, H, W);
    free(final_mask);
    return blurred;
}

/* ===================================================================
 * Synthetic RGBA Texture Batch
 * =================================================================== */

typedef struct {
    float* data;   /* [B * H * W * 4] (RGB + structure channel) */
    int B, H, W;
} Phase1TextureBatch;

/* Create synthetic RGBA texture batch */
static Phase1TextureBatch create_synthetic_rgba_texture_batch(
    WubuRNG* rng, const float* fg_batch, const float* bg_batch,
    int B, int H, int W)
{
    Phase1TextureBatch batch;
    batch.B = B;
    batch.H = H;
    batch.W = W;
    batch.data = (float*)calloc((size_t)(B * H * W * 4), sizeof(float));
    if (!batch.data) return batch;

    for (int b = 0; b < B; ++b) {
        /* Generate complex alpha mask for this batch element */
        float* alpha_mask = generate_complex_mask(rng, H, W, 3, 7);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int pixel_idx = (b * H * W + y * W + x);
                float alpha = alpha_mask[y * W + x];

                /* Foreground luminance (in [0,1] range) */
                float fg_r = fg_batch[pixel_idx * 3 + 0] * 0.5f + 0.5f;
                float fg_g = fg_batch[pixel_idx * 3 + 1] * 0.5f + 0.5f;
                float fg_b = fg_batch[pixel_idx * 3 + 2] * 0.5f + 0.5f;
                float fg_lum = 0.2989f * fg_r + 0.5870f * fg_g + 0.1140f * fg_b;

                /* Simple box blur on luminance for high-freq extraction */
                float blurred_lum = 0.0f;
                int count = 0;
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                            int ni = (b * H * W + ny * W + nx) * 3;
                            float nr = fg_batch[ni + 0] * 0.5f + 0.5f;
                            float ng = fg_batch[ni + 1] * 0.5f + 0.5f;
                            float nb = fg_batch[ni + 2] * 0.5f + 0.5f;
                            blurred_lum += 0.2989f * nr + 0.5870f * ng + 0.1140f * nb;
                            count++;
                        }
                    }
                }
                blurred_lum /= (float)count;

                float high_freq = fg_lum - blurred_lum;
                if (high_freq < -0.5f) high_freq = -0.5f;
                if (high_freq > 0.5f) high_freq = 0.5f;
                high_freq += 0.5f;

                /* Checkerboard pattern */
                int checker = (y + x) % 2;

                /* Structure channel: alpha * checker + high_freq * (1 - checker) */
                float structure = alpha * (float)checker + high_freq * (1.0f - (float)checker);

                /* Target RGB: fg * alpha + bg * (1 - alpha) */
                float bg_r = bg_batch[pixel_idx * 3 + 0];
                float bg_g = bg_batch[pixel_idx * 3 + 1];
                float bg_b = bg_batch[pixel_idx * 3 + 2];

                batch.data[pixel_idx * 4 + 0] = fg_batch[pixel_idx * 3 + 0] * alpha + bg_r * (1.0f - alpha);
                batch.data[pixel_idx * 4 + 1] = fg_batch[pixel_idx * 3 + 1] * alpha + bg_g * (1.0f - alpha);
                batch.data[pixel_idx * 4 + 2] = fg_batch[pixel_idx * 3 + 2] * alpha + bg_b * (1.0f - alpha);
                batch.data[pixel_idx * 4 + 3] = structure;
            }
        }
        free(alpha_mask);
    }
    return batch;
}

static void phase1_texture_batch_free(Phase1TextureBatch* batch) {
    if (batch->data) {
        free(batch->data);
        batch->data = NULL;
    }
}

/* ===================================================================
 * Color Pattern Generation (32 frames)
 * =================================================================== */

typedef struct {
    float* frames;   /* [32 * H * W * 3] in [-1,1] */
    int H, W;
} Phase1ColorPatterns;

/* Generate 32 color pattern frames */
static Phase1ColorPatterns generate_color_patterns(WubuRNG* rng, int H, int W) {
    Phase1ColorPatterns patterns;
    patterns.H = H;
    patterns.W = W;
    patterns.frames = (float*)calloc((size_t)(32 * H * W * 3), sizeof(float));
    if (!patterns.frames) return patterns;

    for (int f = 0; f < 32; ++f) {
        float phase = (float)f / 32.0f * 2.0f * (float)M_PI;
        float hue_base = (float)f / 32.0f;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = (f * H * W + y * W + x) * 3;
                float xf = (float)x / (float)W;
                float yf = (float)y / (float)H;

                /* Circular hue rotation */
                float hue = hue_base + 0.1f * sinf(phase + xf * 2.0f * (float)M_PI);
                if (hue < 0.0f) hue += 1.0f;
                if (hue >= 1.0f) hue -= 1.0f;

                float sat = 0.7f + 0.3f * sinf(phase + yf * (float)M_PI);
                float lum = 0.4f + 0.2f * sinf(phase + (xf + yf) * (float)M_PI);

                /* HSL to RGB */
                WubuHSL hsl = { .h = hue, .s = sat, .l = lum };
                WubuRGB rgb = wubu_hsl_to_rgb(hsl);

                /* Map to [-1,1] */
                patterns.frames[idx + 0] = rgb.r * 2.0f - 1.0f;
                patterns.frames[idx + 1] = rgb.g * 2.0f - 1.0f;
                patterns.frames[idx + 2] = rgb.b * 2.0f - 1.0f;
            }
        }
    }
    return patterns;
}

static void phase1_color_patterns_free(Phase1ColorPatterns* patterns) {
    if (patterns->frames) {
        free(patterns->frames);
        patterns->frames = NULL;
    }
}

/* ===================================================================
 * Moving Shape Patterns (animated)
 * =================================================================== */

typedef struct {
    float* frames;   /* [num_frames * H * W * 3] in [-1,1] */
    int num_frames;
    int H, W;
} Phase1MovingShapes;

/* Generate moving shape animation */
static Phase1MovingShapes generate_moving_shapes(WubuRNG* rng, int H, int W, int num_frames) {
    Phase1MovingShapes shapes;
    shapes.num_frames = num_frames;
    shapes.H = H;
    shapes.W = W;
    shapes.frames = (float*)calloc((size_t)(num_frames * H * W * 3), sizeof(float));
    if (!shapes.frames) return shapes;

    /* Randomize shape parameters */
    float shape_x = wubu_rng_uniform(rng, -0.3f, 0.3f);
    float shape_y = wubu_rng_uniform(rng, -0.3f, 0.3f);
    float shape_r = wubu_rng_uniform(rng, 0.1f, 0.3f);
    float hue = wubu_rng_uniform(rng, 0.0f, 1.0f);
    float speed = wubu_rng_uniform(rng, 0.5f, 2.0f);

    for (int f = 0; f < num_frames; ++f) {
        float t = (float)f / (float)num_frames * 2.0f * (float)M_PI * speed;
        float cx = shape_x + 0.3f * sinf(t);
        float cy = shape_y + 0.3f * cosf(t * 0.7f);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = (f * H * W + y * W + x) * 3;
                float xf = ((float)x / (float)(W - 1)) * 2.0f - 1.0f;
                float yf = ((float)y / (float)(H - 1)) * 2.0f - 1.0f;

                float dist = sqrtf((xf - cx) * (xf - cx) + (yf - cy) * (yf - cy));
                float inside = dist < shape_r ? 1.0f : 0.0f;

                /* Smooth edge */
                float edge = shape_r + 0.02f;
                if (dist > shape_r && dist < edge) {
                    inside = (edge - dist) / 0.02f;
                }

                float shape_hue = hue + 0.1f * sinf(t);
                if (shape_hue < 0.0f) shape_hue += 1.0f;
                if (shape_hue >= 1.0f) shape_hue -= 1.0f;

                WubuHSL hsl = { .h = shape_hue, .s = 0.9f, .l = 0.5f * inside + 0.1f };
                WubuRGB rgb = wubu_hsl_to_rgb(hsl);

                shapes.frames[idx + 0] = rgb.r * 2.0f - 1.0f;
                shapes.frames[idx + 1] = rgb.g * 2.0f - 1.0f;
                shapes.frames[idx + 2] = rgb.b * 2.0f - 1.0f;
            }
        }
    }
    return shapes;
}

static void phase1_moving_shapes_free(Phase1MovingShapes* shapes) {
    if (shapes->frames) {
        free(shapes->frames);
        shapes->frames = NULL;
    }
}

/* ===================================================================
 * VHF Tone Pipeline (8 tones -> WAV)
 * =================================================================== */

typedef struct {
    float* samples;   /* [num_samples] in [-1,1] */
    int num_samples;
    int sample_rate;
} Phase1VHFTones;

/* Generate 8 VHF tones as a WAV-compatible buffer */
static Phase1VHFTones generate_vhf_tones(WubuRNG* rng, int sample_rate, float duration_sec) {
    Phase1VHFTones tones;
    tones.sample_rate = sample_rate;
    tones.num_samples = (int)((float)sample_rate * duration_sec);
    if (tones.num_samples < 1) tones.num_samples = 1;
    tones.samples = (float*)calloc((size_t)tones.num_samples, sizeof(float));
    if (!tones.samples) return tones;

    /* 8 tone frequencies (musical scale centered around 500Hz) */
    float frequencies[8];
    float base_freq = 500.0f;
    for (int i = 0; i < 8; ++i) {
        frequencies[i] = base_freq * powf(2.0f, (float)i / 8.0f);
    }

    /* Randomize phase per tone */
    float phases[8];
    for (int i = 0; i < 8; ++i) {
        phases[i] = wubu_rng_uniform(rng, 0.0f, 2.0f * (float)M_PI);
    }

    /* Generate mixed tones */
    for (int s = 0; s < tones.num_samples; ++s) {
        float t = (float)s / (float)sample_rate;
        float sample = 0.0f;
        for (int i = 0; i < 8; ++i) {
            sample += 0.125f * sinf(2.0f * (float)M_PI * frequencies[i] * t + phases[i]);
        }
        tones.samples[s] = sample;
    }
    return tones;
}

static void phase1_vhf_tones_free(Phase1VHFTones* tones) {
    if (tones->samples) {
        free(tones->samples);
        tones->samples = NULL;
    }
}

/* ===================================================================
 * Poincare Sphere: Co-Polarized Transmittance
 * =================================================================== */

typedef struct {
    float re;
    float im;
} WubuComplex;

static WubuComplex wubu_poincare_transmittance(float delta, float chi) {
    WubuComplex result;
    result.re = cosf(delta / 2.0f);
    result.im = sinf(delta / 2.0f) * sinf(2.0f * chi);
    return result;
}

/* ===================================================================
 * FiLM Layer (Feature-wise Linear Modulation)
 * =================================================================== */

/* Apply FiLM modulation: x * (gamma + 1) + beta */
static void phase1_film_layer(float* x, const float* context,
                               int N, int D, int context_dim) {
    float context_sum = 0.0f;
    for (int i = 0; i < context_dim; ++i) {
        context_sum += context[i];
    }
    context_sum /= (float)context_dim;

    for (int n = 0; n < N; ++n) {
        float gamma = tanhf(context_sum * 2.0f);
        float beta = context_sum * 0.5f;
        for (int d = 0; d < D; ++d) {
            x[n * D + d] = x[n * D + d] * (gamma + 1.0f) + beta;
        }
    }
}

/* ===================================================================
 * Coordinate Decoder MLP
 * =================================================================== */

static float* phase1_coordinate_decoder(
    const float* feature_grid, int H, int W, int latent_dim,
    const float* coords, int N,
    const float* context, int context_dim,
    const Phase1Config* config)
{
    float* output = (float*)calloc((size_t)(N * 4), sizeof(float));
    if (!output) return NULL;

    for (int n = 0; n < N; ++n) {
        /* Positional encoding of coordinates */
        float encoded[2 + 2 * 10 * 2];
        int enc_idx = 0;
        encoded[enc_idx++] = coords[n * 2 + 0];
        encoded[enc_idx++] = coords[n * 2 + 1];

        int num_freqs = config->num_freqs;
        for (int f = 0; f < num_freqs; ++f) {
            float freq = powf(2.0f, (float)f) * (float)M_PI;
            encoded[enc_idx++] = sinf(coords[n * 2 + 0] * freq);
            encoded[enc_idx++] = cosf(coords[n * 2 + 0] * freq);
        }
        for (int f = 0; f < num_freqs; ++f) {
            float freq = powf(2.0f, (float)f) * (float)M_PI;
            encoded[enc_idx++] = sinf(coords[n * 2 + 1] * freq);
            encoded[enc_idx++] = cosf(coords[n * 2 + 1] * freq);
        }

        /* Sample feature grid at coordinate (bilinear) */
        float xf = (coords[n * 2 + 0] + 1.0f) / 2.0f * (float)(W - 1);
        float yf = (coords[n * 2 + 1] + 1.0f) / 2.0f * (float)(H - 1);
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

        float sampled[96];
        int ld = latent_dim < 96 ? latent_dim : 96;
        for (int c = 0; c < ld; ++c) {
            float v00 = feature_grid[(y0 * W + x0) * latent_dim + c];
            float v01 = feature_grid[(y1 * W + x0) * latent_dim + c];
            float v10 = feature_grid[(y0 * W + x1) * latent_dim + c];
            float v11 = feature_grid[(y1 * W + x1) * latent_dim + c];
            sampled[c] = (1-wx)*(1-wy)*v00 + (1-wx)*wy*v01 + wx*(1-wy)*v10 + wx*wy*v11;
        }

        /* Concatenate encoded coords + sampled features */
        int mlp_input_dim = enc_idx + ld;
        float* mlp_input = (float*)malloc((size_t)mlp_input_dim * sizeof(float));
        if (!mlp_input) { free(output); return NULL; }
        memcpy(mlp_input, encoded, (size_t)enc_idx * sizeof(float));
        memcpy(mlp_input + enc_idx, sampled, (size_t)ld * sizeof(float));

        /* Simple MLP forward pass (1 hidden layer) */
        int hidden = config->mlp_width;
        float* hidden_out = (float*)calloc((size_t)hidden, sizeof(float));
        if (!hidden_out) { free(mlp_input); free(output); return NULL; }

        /* Layer 0: input -> hidden with GELU */
        for (int h = 0; h < hidden; ++h) {
            float sum = 0.0f;
            for (int i = 0; i < mlp_input_dim; ++i) {
                float w = sinf((float)(i * 12345 + h * 6789) / 1000.0f) * 0.1f;
                sum += mlp_input[i] * w;
            }
            hidden_out[h] = sum;
            /* GELU approximation */
            float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (hidden_out[h] + 0.044715f * hidden_out[h] * hidden_out[h] * hidden_out[h])));
            hidden_out[h] = hidden_out[h] * cdf;
        }

        /* Apply FiLM */
        phase1_film_layer(hidden_out, context, 1, hidden, context_dim);

        /* Layer 1: hidden -> 4 (RGB + structure) */
        for (int o = 0; o < 4; ++o) {
            float sum = 0.0f;
            for (int h = 0; h < hidden; ++h) {
                float w = sinf((float)(h * 54321 + o * 9876) / 1000.0f) * 0.1f;
                sum += hidden_out[h] * w;
            }
            output[n * 4 + o] = sum;
        }

        free(hidden_out);
        free(mlp_input);
    }

    return output;
}

/* ===================================================================
 * Path Modulator (simplified encoder)
 * =================================================================== */

typedef struct {
    float* path_params;   /* [B * H * W * 3] (delta, chi, radius) */
    float* context;       /* [B * context_dim] */
    int B, H, W;
    int context_dim;
} Phase1PathOutput;

static Phase1PathOutput phase1_path_modulate(
    WubuRNG* rng, const float* images_rgb,
    int B, int img_size, const Phase1Config* config)
{
    Phase1PathOutput output;
    output.B = B;
    output.H = config->latent_grid_size;
    output.W = config->latent_grid_size;
    output.context_dim = config->d_model;
    output.path_params = (float*)calloc((size_t)(B * output.H * output.W * 3), sizeof(float));
    output.context = (float*)calloc((size_t)(B * config->d_model), sizeof(float));
    if (!output.path_params || !output.context) {
        free(output.path_params);
        free(output.context);
        output.path_params = NULL;
        output.context = NULL;
        return output;
    }

    int downsample_steps = 0;
    int dim = img_size;
    while (dim / 2 >= config->latent_grid_size && dim / 2 > 0) {
        downsample_steps++;
        dim /= 2;
    }

    for (int b = 0; b < B; ++b) {
        /* Compute context vector: mean of image patches */
        float context_acc[64];
        int cd = config->d_model < 64 ? config->d_model : 64;
        for (int c = 0; c < cd; ++c) context_acc[c] = 0.0f;

        for (int y = 0; y < img_size; ++y) {
            for (int x = 0; x < img_size; ++x) {
                int idx = (b * img_size * img_size + y * img_size + x) * 3;
                float r = images_rgb[idx + 0];
                float g = images_rgb[idx + 1];
                float bl = images_rgb[idx + 2];
                for (int c = 0; c < cd; ++c) {
                    context_acc[c] += (r + g + bl) / 3.0f * sinf((float)(c * x * y + c) / 100.0f);
                }
            }
        }
        float scale = 1.0f / (float)(img_size * img_size);
        for (int c = 0; c < cd; ++c) {
            output.context[b * cd + c] = context_acc[c] * scale;
        }

        /* Generate path params */
        for (int y = 0; y < output.H; ++y) {
            for (int x = 0; x < output.W; ++x) {
                int idx = (b * output.H * output.W + y * output.W + x) * 3;
                float xf = (float)x / (float)output.W;
                float yf = (float)y / (float)output.H;

                output.path_params[idx + 0] = (xf - 0.5f) * (float)M_PI;
                output.path_params[idx + 1] = (yf - 0.5f) * (float)M_PI / 4.0f;
                output.path_params[idx + 2] = 0.3f + 0.2f * sinf((xf + yf) * (float)M_PI);
            }
        }
    }
    return output;
}

static void phase1_path_output_free(Phase1PathOutput* output) {
    if (output->path_params) { free(output->path_params); output->path_params = NULL; }
    if (output->context) { free(output->context); output->context = NULL; }
}

/* ===================================================================
 * Topological Observer
 * =================================================================== */

static float* phase1_topological_observe(
    const float* path_params, int B, int H, int W,
    const Phase1Config* config)
{
    int L = H * W;
    int latent_dim = config->latent_dim;
    float* features = (float*)calloc((size_t)(B * L * latent_dim), sizeof(float));
    if (!features) return NULL;

    int num_steps = config->num_path_steps;

    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            float delta = path_params[(b * L + l) * 3 + 0];
            float chi = path_params[(b * L + l) * 3 + 1];
            float radius = path_params[(b * L + l) * 3 + 2];

            float real_sum = 0.0f, imag_sum = 0.0f;
            float real_sq_sum = 0.0f, imag_sq_sum = 0.0f;

            for (int s = 0; s < num_steps; ++s) {
                float theta = (float)s / (float)num_steps * 2.0f * (float)M_PI;
                float d_path = delta + radius * cosf(theta);
                float c_path = chi + radius * sinf(theta);
                WubuComplex tc = wubu_poincare_transmittance(d_path, c_path);

                real_sum += tc.re;
                imag_sum += tc.im;
                real_sq_sum += tc.re * tc.re;
                imag_sq_sum += tc.im * tc.im;
            }

            float inv_steps = 1.0f / (float)num_steps;
            float real_mean = real_sum * inv_steps;
            float imag_mean = imag_sum * inv_steps;
            float real_var = real_sq_sum * inv_steps - real_mean * real_mean;
            float imag_var = imag_sq_sum * inv_steps - imag_mean * imag_mean;
            if (real_var < 0.0f) real_var = 0.0f;
            if (imag_var < 0.0f) imag_var = 0.0f;
            float real_std = sqrtf(real_var);
            float imag_std = sqrtf(imag_var);

            for (int d = 0; d < latent_dim; ++d) {
                float val = 0.0f;
                switch (d % 4) {
                    case 0: val = real_mean; break;
                    case 1: val = real_std; break;
                    case 2: val = imag_mean; break;
                    case 3: val = imag_std; break;
                }
                features[(b * L + l) * latent_dim + d] = val;
            }
        }
    }
    return features;
}

/* ===================================================================
 * Full Pipeline: Encode + Decode
 * =================================================================== */

typedef struct {
    float* pixels_rgba_struct;  /* [B * N * 4] */
    int B, N;
} Phase1DecodeOutput;

Phase1DecodeOutput phase1_pipeline(
    WubuRNG* rng, const float* images_rgb,
    int B, int img_size, int N,
    const Phase1Config* config)
{
    Phase1DecodeOutput result;
    result.B = B;
    result.N = N;
    result.pixels_rgba_struct = NULL;

    /* Encode */
    Phase1PathOutput path_out = phase1_path_modulate(rng, images_rgb, B, img_size, config);
    if (!path_out.path_params) return result;

    /* Observe */
    float* features = phase1_topological_observe(path_out.path_params, B, path_out.H, path_out.W, config);
    if (!features) {
        phase1_path_output_free(&path_out);
        return result;
    }

    /* Decode */
    float* coords = (float*)malloc((size_t)(N * 2) * sizeof(float));
    if (!coords) {
        free(features);
        phase1_path_output_free(&path_out);
        return result;
    }
    for (int i = 0; i < N; ++i) {
        coords[i * 2 + 0] = wubu_rng_uniform(rng, -1.0f, 1.0f);
        coords[i * 2 + 1] = wubu_rng_uniform(rng, -1.0f, 1.0f);
    }

    result.pixels_rgba_struct = phase1_coordinate_decoder(
        features, path_out.H, path_out.W, config->latent_dim,
        coords, N, path_out.context, path_out.context_dim, config);

    free(coords);
    free(features);
    phase1_path_output_free(&path_out);
    return result;
}

static void phase1_decode_output_free(Phase1DecodeOutput* output) {
    if (output->pixels_rgba_struct) {
        free(output->pixels_rgba_struct);
        output->pixels_rgba_struct = NULL;
    }
}
