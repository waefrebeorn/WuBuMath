/*
 * train_vhf.c -- Full training loop for VHF decoder with backprop
 *
 * Strategy:
 * 1. Generate synthetic training data
 * 2. Encode with initialized HamiltonEncoder (random weights, frozen)
 * 3. Train VHFDecoder: forward per-pixel → MSE loss → backward → Adam
 * 4. Log loss/PSNR every epoch
 *
 * Build: make bin/train_vhf
 * Run:   bin/train_vhf
 */

#include "wubumath.h"
#include "wubu_vhf_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Config
 * ============================================================================ */

#define IMG_H      64
#define IMG_W      64
#define NUM_TRAIN  32
#define NUM_EPOCHS 1000
#define BATCH_SIZE 8
#define LG_SIZE    16
#define D_MODEL    32
#define NUM_PIXELS  64
#define LR_INIT    5e-4f

/* ============================================================================
 * Adam optimizer
 * ============================================================================ */

typedef struct {
    int num_params;
    float* m;
    float* v;
    int step;
} AdamState;

static void adam_init(AdamState* s, int n) {
    s->num_params = n;
    s->m = (float*)calloc((size_t)n, sizeof(float));
    s->v = (float*)calloc((size_t)n, sizeof(float));
    s->step = 0;
}

static void adam_free(AdamState* s) {
    free(s->m); free(s->v);
}

static void adam_update(AdamState* s, float* p, const float* g, int n, float lr) {
    s->step++;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, (float)s->step);
    float bc2 = 1.0f - powf(b2, (float)s->step);
    for (int i = 0; i < n; i++) {
        s->m[i] = b1 * s->m[i] + (1.0f - b1) * g[i];
        s->v[i] = b2 * s->v[i] + (1.0f - b2) * g[i] * g[i];
        p[i] -= lr * (s->m[i] / bc1) / (sqrtf(s->v[i] / bc2) + eps);
    }
}

/* ============================================================================
 * Generate synthetic training data
 * ============================================================================ */

static void generate_sample(float* img, int id) {
    int pattern = id % 8;
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int idx = (y * IMG_W + x) * 3;
            float u = (float)x / (float)(IMG_W - 1);
            float v = (float)y / (float)(IMG_H - 1);
            float n = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.03f;
            switch (pattern) {
                case 0: img[idx]=u*2-1+n; img[idx+1]=0+n; img[idx+2]=(1-u)*2-1+n; break;
                case 1: img[idx]=0+n; img[idx+1]=v*2-1+n; img[idx+2]=(1-v)*2-1+n; break;
                case 2: img[idx]=((x/8+y/8)%2)?0.8f:-0.8f+n; img[idx+1]=((x/8)%2)?0.6f:-0.6f+n; img[idx+2]=u*v*2-1+n; break;
                case 3: {
                    float r=sqrtf((u-0.5f)*(u-0.5f)+(v-0.5f)*(v-0.5f))*2;
                    img[idx]=fminf(1,r)*2-1+n; img[idx+1]=fminf(1,1-r)*2-1+n; img[idx+2]=0.5f*sinf(r*6.28f)+n; break;
                }
                case 4: img[idx]=sinf(u*12.56f)*0.8f+n; img[idx+1]=cosf(v*12.56f)*0.8f+n; img[idx+2]=sinf((u+v)*6.28f)*0.8f+n; break;
                case 5: img[idx]=(fabsf(u-0.5f)<0.1f)?0.9f:-0.3f+n; img[idx+1]=(fabsf(v-0.5f)<0.1f)?0.9f:-0.3f+n; img[idx+2]=0+n; break;
                case 6: { float d=(u+v)*0.5f; img[idx]=sinf(d*12.56f)*0.8f+n; img[idx+1]=cosf(d*18.84f)*0.6f+n; img[idx+2]=sinf(d*6.28f)*0.4f+n; break; }
                case 7: {
                    float cx=u-0.3f,cy=v-0.7f,g1=expf(-(cx*cx+cy*cy)*50);
                    cx=u-0.7f;cy=v-0.3f;float g2=expf(-(cx*cx+cy*cy)*40);
                    img[idx]=g1*2-1+n; img[idx+1]=g2*2-1+n; img[idx+2]=(g1+g2)*0.5f*2-1+n; break;
                }
            }
            img[idx]=fminf(1,fmaxf(-1,img[idx]));
            img[idx+1]=fminf(1,fmaxf(-1,img[idx+1]));
            img[idx+2]=fminf(1,fmaxf(-1,img[idx+2]));
        }
    }
}

/* ============================================================================
 * GELU and derivative
 * ============================================================================ */

static float gelu_f(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
}

static float gelu_d(float x) {
    float c = 0.7978845608f;
    float inner = c * x * (1.0f + 0.044715f * x * x);
    float t = tanhf(inner);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * 0.044715f * x * x);
}

/* ============================================================================
 * Forward + backward through decoder for a single pixel
 * ============================================================================ */

static float decoder_fb(VHFDec* dec, const float* input,
                          const float* target,
                          float** grad_w, float** grad_b,
                          float* grad_ow, float* grad_ob) {
    int d = dec->d_model;
    int in_dim = dec->input_dim;

    /* Forward */
    float h0[512], h1[512], h2[512], h3[512];
    float pa[4][512]; /* pre-activation */

    for (int o = 0; o < d; o++) {
        float s = dec->mlp_b[0][o];
        for (int i = 0; i < in_dim; i++) s += dec->mlp_w[0][o * in_dim + i] * input[i];
        pa[0][o] = s; h0[o] = gelu_f(s);
    }
    for (int o = 0; o < d; o++) {
        float s = dec->mlp_b[1][o];
        for (int i = 0; i < d; i++) s += dec->mlp_w[1][o * d + i] * h0[i];
        pa[1][o] = s; h1[o] = gelu_f(s);
    }
    for (int o = 0; o < d; o++) {
        float s = dec->mlp_b[2][o];
        for (int i = 0; i < d; i++) s += dec->mlp_w[2][o * d + i] * h1[i];
        pa[2][o] = s; h2[o] = gelu_f(s);
    }
    for (int o = 0; o < d; o++) {
        float s = dec->mlp_b[3][o];
        for (int i = 0; i < d; i++) s += dec->mlp_w[3][o * d + i] * h2[i];
        pa[3][o] = s; h3[o] = gelu_f(s);
    }

    float raw[3], pred[3];
    for (int o = 0; o < 3; o++) {
        float s = dec->out_b[o];
        for (int i = 0; i < d; i++) s += dec->out_w[o * d + i] * h3[i];
        raw[o] = s; pred[o] = tanhf(s);
    }

    float loss = 0;
    float d_pred[3];
    for (int c = 0; c < 3; c++) {
        float diff = pred[c] - target[c];
        loss += diff * diff;
        d_pred[c] = 2.0f * diff;
    }
    loss /= 3.0f;

    /* Backward */
    float d_raw[3];
    for (int c = 0; c < 3; c++) {
        float t = tanhf(raw[c]);
        d_raw[c] = d_pred[c] * (1.0f - t * t);
    }

    float d_h3[512] = {0};
    for (int o = 0; o < 3; o++) {
        grad_ob[o] = d_raw[o];
        for (int i = 0; i < d; i++) {
            grad_ow[o * d + i] = d_raw[o] * h3[i];
            d_h3[i] += d_raw[o] * dec->out_w[o * d + i];
        }
    }

    float d_pre3[512], d_h2[512] = {0};
    for (int i = 0; i < d; i++) d_pre3[i] = d_h3[i] * gelu_d(pa[3][i]);
    for (int o = 0; o < d; o++) {
        grad_b[3][o] = d_pre3[o];
        for (int i = 0; i < d; i++) {
            grad_w[3][o * d + i] = d_pre3[o] * h2[i];
            d_h2[i] += d_pre3[o] * dec->mlp_w[3][o * d + i];
        }
    }

    float d_pre2[512], d_h1[512] = {0};
    for (int i = 0; i < d; i++) d_pre2[i] = d_h2[i] * gelu_d(pa[2][i]);
    for (int o = 0; o < d; o++) {
        grad_b[2][o] = d_pre2[o];
        for (int i = 0; i < d; i++) {
            grad_w[2][o * d + i] = d_pre2[o] * h1[i];
            d_h1[i] += d_pre2[o] * dec->mlp_w[2][o * d + i];
        }
    }

    float d_pre1[512], d_h0[512] = {0};
    for (int i = 0; i < d; i++) d_pre1[i] = d_h1[i] * gelu_d(pa[1][i]);
    for (int o = 0; o < d; o++) {
        grad_b[1][o] = d_pre1[o];
        for (int i = 0; i < d; i++) {
            grad_w[1][o * d + i] = d_pre1[o] * h0[i];
            d_h0[i] += d_pre1[o] * dec->mlp_w[1][o * d + i];
        }
    }

    float d_pre0[1024];
    for (int i = 0; i < d; i++) d_pre0[i] = d_h0[i] * gelu_d(pa[0][i]);
    for (int o = 0; o < d; o++) {
        grad_b[0][o] = d_pre0[o];
        for (int i = 0; i < in_dim; i++)
            grad_w[0][o * in_dim + i] = d_pre0[o] * input[i];
    }

    return loss;
}

/* ============================================================================
 * Pack/unpack decoder params and grads
 * ============================================================================ */

static int count_params(VHFDec* dec) {
    int d = dec->d_model, in = dec->input_dim;
    return (in * d + d) + 3 * (d * d + d) + (3 * d + 3);
}


static void pack_params(VHFDec* dec, float* buf) {
    int d = dec->d_model, in = dec->input_dim, idx = 0;
    memcpy(buf+idx, dec->mlp_w[0], (size_t)(in*d)*sizeof(float)); idx += in*d;
    memcpy(buf+idx, dec->mlp_b[0], (size_t)d*sizeof(float)); idx += d;
    for (int l = 1; l < 4; l++) {
        memcpy(buf+idx, dec->mlp_w[l], (size_t)(d*d)*sizeof(float)); idx += d*d;
        memcpy(buf+idx, dec->mlp_b[l], (size_t)d*sizeof(float)); idx += d;
    }
    memcpy(buf+idx, dec->out_w, (size_t)(3*d)*sizeof(float)); idx += 3*d;
    memcpy(buf+idx, dec->out_b, 3*sizeof(float)); idx += 3;
}

static void unpack_params(VHFDec* dec, const float* buf) {
    int d = dec->d_model, in = dec->input_dim, idx = 0;
    memcpy(dec->mlp_w[0], buf+idx, (size_t)(in*d)*sizeof(float)); idx += in*d;
    memcpy(dec->mlp_b[0], buf+idx, (size_t)d*sizeof(float)); idx += d;
    for (int l = 1; l < 4; l++) {
        memcpy(dec->mlp_w[l], buf+idx, (size_t)(d*d)*sizeof(float)); idx += d*d;
        memcpy(dec->mlp_b[l], buf+idx, (size_t)d*sizeof(float)); idx += d;
    }
    memcpy(dec->out_w, buf+idx, (size_t)(3*d)*sizeof(float)); idx += 3*d;
    memcpy(dec->out_b, buf+idx, 3*sizeof(float)); idx += 3;
}

static void pack_grads(float** gw, float** gb, float* gow, float* gob,
                       float* buf, int in_dim, int d) {
    int idx = 0;
    memcpy(buf+idx, gw[0], (size_t)(in_dim*d)*sizeof(float)); idx += in_dim*d;
    memcpy(buf+idx, gb[0], (size_t)d*sizeof(float)); idx += d;
    for (int l = 1; l < 4; l++) {
        memcpy(buf+idx, gw[l], (size_t)(d*d)*sizeof(float)); idx += d*d;
        memcpy(buf+idx, gb[l], (size_t)d*sizeof(float)); idx += d;
    }
    memcpy(buf+idx, gow, (size_t)(3*d)*sizeof(float)); idx += 3*d;
    memcpy(buf+idx, gob, 3*sizeof(float)); idx += 3;
}

int main(void) {
    srand(42);

    printf("=============================================\n");
    printf(" VHF Decoder Training (with backprop)\n");
    printf("=============================================\n");
    printf(" Image: %dx%d, LatentGrid: %d, D-Model: %d\n", IMG_H, IMG_W, LG_SIZE, D_MODEL);
    printf(" Train: %d samples, %d epochs, batch %d\n", NUM_TRAIN, NUM_EPOCHS, BATCH_SIZE);
    printf(" Pixels/sample: %d\n", NUM_PIXELS);
    printf("=============================================\n\n");

    /* Init encoder (frozen) */
    VHFHamiltonEnc enc;
    vhf_hamilton_encoder_init(&enc, LG_SIZE, D_MODEL);
    printf(" Encoder: %d scales, context_dim=%d\n", enc.num_scales, enc.context_dim);

    /* Init decoder */
    VHFDec dec;
    vhf_decoder_init(&dec, D_MODEL, enc.context_dim);
    printf(" Decoder: input_dim=%d, d_model=%d\n\n", dec.input_dim, dec.d_model);

    /* Generate training data */
    float** train_images = malloc((size_t)NUM_TRAIN * sizeof(float*));
    for (int i = 0; i < NUM_TRAIN; i++) {
        train_images[i] = malloc((size_t)(IMG_H * IMG_W * 3) * sizeof(float));
        generate_sample(train_images[i], i);
    }

    /* Pre-encode */
    float** train_keys = malloc((size_t)NUM_TRAIN * sizeof(float*));
    float** train_ctx = malloc((size_t)NUM_TRAIN * sizeof(float*));
    for (int i = 0; i < NUM_TRAIN; i++) {
        train_keys[i] = calloc((size_t)(LG_SIZE * LG_SIZE * 5), sizeof(float));
        train_ctx[i] = calloc((size_t)enc.context_dim, sizeof(float));
        vhf_hamilton_encode(&enc, train_images[i], IMG_H, IMG_W, train_keys[i], train_ctx[i]);
    }
    printf(" Pre-encoded %d samples\n", NUM_TRAIN);

    /* Build decoder inputs */
    int in_dim = dec.input_dim;
    float** dec_inputs = malloc((size_t)NUM_TRAIN * sizeof(float*));
    float** dec_targets = malloc((size_t)NUM_TRAIN * sizeof(float*));

    for (int i = 0; i < NUM_TRAIN; i++) {
        dec_inputs[i] = malloc((size_t)(NUM_PIXELS * in_dim) * sizeof(float));
        dec_targets[i] = malloc((size_t)(NUM_PIXELS * 3) * sizeof(float));
        for (int p = 0; p < NUM_PIXELS; p++) {
            int y = rand() % IMG_H;
            int x = rand() % IMG_W;
            float cx = 2.0f * (float)x / (float)(IMG_W - 1) - 1.0f;
            float cy = 2.0f * (float)y / (float)(IMG_H - 1) - 1.0f;

            float* inp = dec_inputs[i] + p * in_dim;
            int idx = 0;
            inp[idx++] = cx;
            inp[idx++] = cy;
            for (int f = 0; f < 10; f++) {
                float freq = powf(2.0f, (float)f) * M_PI;
                inp[idx++] = sinf(cx * freq);
                inp[idx++] = cosf(cx * freq);
                inp[idx++] = sinf(cy * freq);
                inp[idx++] = cosf(cy * freq);
            }
            memcpy(inp + idx, train_ctx[i], (size_t)enc.context_dim * sizeof(float));
            idx += enc.context_dim;
            float key[5];
            float coord[2] = {cx, cy};
            vhf_bilinear_sample(train_keys[i], LG_SIZE, LG_SIZE, 5, coord, 1, key);
            memcpy(inp + idx, key, 5 * sizeof(float));

            float* tgt = dec_targets[i] + p * 3;
            tgt[0] = train_images[i][(y * IMG_W + x) * 3 + 0];
            tgt[1] = train_images[i][(y * IMG_W + x) * 3 + 1];
            tgt[2] = train_images[i][(y * IMG_W + x) * 3 + 2];
        }
    }
    printf(" Built decoder inputs (%d pixels each)\n\n", NUM_PIXELS);

    /* Adam setup */
    int num_params = count_params(&dec);
    float* params = malloc((size_t)num_params * sizeof(float));
    pack_params(&dec, params);
    AdamState adam;
    adam_init(&adam, num_params);

    /* Gradient buffers */
    float* grad_buf = calloc((size_t)num_params, sizeof(float));
    float* gw[4], *gb[4];
    gw[0] = calloc((size_t)(in_dim * D_MODEL), sizeof(float));
    for (int l = 0; l < 4; l++) {
        gb[l] = calloc((size_t)D_MODEL, sizeof(float));
        if (l > 0) gw[l] = calloc((size_t)(D_MODEL * D_MODEL), sizeof(float));
    }
    float gow[3 * 512], gob[3];

    /* Training loop */
    FILE* log = fopen("/home/wubu/WuBuMath/output/train_vhf_log.csv", "w");
    fprintf(log, "epoch,loss,psnr\n");

    printf(" Epoch  | Loss      | PSNR (dB)\n");
    printf("-------------------------------\n");

    for (int ep = 0; ep < NUM_EPOCHS; ep++) {
        float epoch_loss = 0;
        int nb = NUM_TRAIN / BATCH_SIZE;

        int* shuf = malloc((size_t)NUM_TRAIN * sizeof(int));
        for (int i = 0; i < NUM_TRAIN; i++) shuf[i] = i;
        for (int i = NUM_TRAIN - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int t = shuf[i]; shuf[i] = shuf[j]; shuf[j] = t;
        }

        for (int b = 0; b < nb; b++) {
            memset(grad_buf, 0, (size_t)num_params * sizeof(float));
            float batch_loss = 0;

            for (int n = b * BATCH_SIZE; n < (b + 1) * BATCH_SIZE; n++) {
                int si = shuf[n];

                for (int l = 0; l < 4; l++) {
                    int ws = (l == 0) ? in_dim * D_MODEL : D_MODEL * D_MODEL;
                    memset(gw[l], 0, (size_t)ws * sizeof(float));
                    memset(gb[l], 0, (size_t)D_MODEL * sizeof(float));
                }
                memset(gow, 0, sizeof(gow));
                memset(gob, 0, sizeof(gob));

                float sample_loss = 0;
                for (int p = 0; p < NUM_PIXELS; p++) {
                    sample_loss += decoder_fb(&dec, dec_inputs[si] + p * in_dim,
                                              dec_targets[si] + p * 3,
                                              gw, gb, gow, gob);
                }
                batch_loss += sample_loss / (float)NUM_PIXELS;
                pack_grads(gw, gb, gow, gob, grad_buf, in_dim, D_MODEL);
            }

            for (int i = 0; i < num_params; i++)
                grad_buf[i] /= (float)(BATCH_SIZE * NUM_PIXELS);

            float lr = LR_INIT * (1.0f - (float)ep / (float)NUM_EPOCHS * 0.5f);
            adam_update(&adam, params, grad_buf, num_params, lr);
            unpack_params(&dec, params);

            epoch_loss += batch_loss / (float)nb;
        }
        free(shuf);

        float psnr = (epoch_loss < 1e-10f) ? 100.0f : 10.0f * log10f(1.0f / (epoch_loss + 1e-10f));

        if ((ep + 1) % 25 == 0 || ep == 0) {
            printf(" %5d  | %.6f | %8.3f\n", ep + 1, epoch_loss, psnr);
            fprintf(log, "%d,%.6f,%.3f\n", ep + 1, epoch_loss, psnr);
        }
    }

    printf("-------------------------------\n");
    printf(" Training complete.\n");
    printf(" Log: output/train_vhf_log.csv\n");

    fclose(log);
    free(params); free(grad_buf);
    adam_free(&adam);
    for (int i = 0; i < NUM_TRAIN; i++) {
        free(train_images[i]); free(train_keys[i]); free(train_ctx[i]);
        free(dec_inputs[i]); free(dec_targets[i]);
    }
    free(train_images); free(train_keys); free(train_ctx);
    free(dec_inputs); free(dec_targets);
    vhf_decoder_free(&dec);
    vhf_hamilton_encoder_free(&enc);

    return 0;
}
