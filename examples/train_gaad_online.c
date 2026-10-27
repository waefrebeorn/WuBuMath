/* train_gaad_online.c -- Train with on-the-fly data generation (memory efficient)
 *
 * Generates training data during training instead of loading all frames.
 * This allows training on unlimited data without memory constraints.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_gaad_encoder.h"
#include "wubu_riemannian_sgd.h"
#include "wubumath.h"

#define IMG_H 64
#define IMG_W 64
#define DIM 32
#define LDIM 32
#define NUM_PATCH 256
#define PATCH_PIX 16
#define BATCH 16
#define EPOCHS 500
#define STEPS_PER_EPOCH 100

static float gf(float x){return .5f*x*(1+tanhf(.7978845608f*x*(1+.044715f*x*x)));}
static float gb(float x){float c=.7978845608f,inner=c*x*(1+.044715f*x*x),t=tanhf(inner);return .5f*(1+t)+.5f*x*(1-t*t)*c*(1+3*.044715f*x*x);}

typedef struct {
    float pw[1*DIM], pb[DIM];
    float dw[NUM_PATCH*PATCH_PIX*LDIM], db[NUM_PATCH*PATCH_PIX];
    struct { int dim, nb, fh; float c; float bd[4*32], desc[32], spr; float fw1[32*97], fb1[32], fw2[32*32], fb2[32]; } levels[4];
} Model;

static Model model;
static unsigned rng_v=42;
static float rf(void){rng_v=rng_v*1103515245u+12345u;return(float)(rng_v&0x7FFF)/32768.0f;}

static void init_model(void){
    memset(&model,0,sizeof(model));
    float lim=sqrtf(6.0f/(1+DIM));
    for(int i=0;i<1*DIM;i++) model.pw[i]=(rf()*2-1)*lim;
    for(int i=0;i<NUM_PATCH*PATCH_PIX*LDIM;i++) model.dw[i]=rf()*0.001f;
    float c_base=1.0f; int nbs[4]={4,3,2,2};
    for(int l=0;l<4;l++){
        model.levels[l].dim=DIM; model.levels[l].nb=nbs[l]; model.levels[l].fh=32;
        float pp=(float)(l%4)-1.5f;
        model.levels[l].c=c_base*powf(WUBU_PHI,pp);
        if(model.levels[l].c<0.01f)model.levels[l].c=0.01f;
        model.levels[l].spr=0.1f;
        for(int i=0;i<nbs[l]*DIM;i++)model.levels[l].bd[i]=(rf()-0.5f)*0.02f;
        for(int i=0;i<DIM;i++)model.levels[l].desc[i]=(rf()-0.5f)*0.02f;
        int cd=DIM*3+1; float l1=sqrtf(6.0f/(cd+32));
        for(int i=0;i<32*cd;i++)model.levels[l].fw1[i]=(rf()*2-1)*l1;
        float l2=sqrtf(0.01f/(32+32));
        for(int i=0;i<32*32;i++)model.levels[l].fw2[i]=(rf()*2-1)*l2;
    }
}

static void level_forward(Model* m, int l, const float* v_in, float* v_out, float* ctx_out, float* hb_out, float* hba_out) {
    int nb = m->levels[l].nb, fh = m->levels[l].fh, dim = m->levels[l].dim;
    float c = m->levels[l].c;
    float v[32]; memcpy(v, v_in, dim * sizeof(float));
    float n2 = 0; for (int i = 0; i < dim; i++) n2 += v[i]*v[i];
    float mx = 0.999f/c;
    if (n2 > mx) { float s = sqrtf(mx/(n2+1e-10f)); for (int i = 0; i < dim; i++) v[i] *= s; }
    float tv[32];
    if (n2 < 1e-15f) { memcpy(tv, v, dim * sizeof(float)); }
    else { float vn = sqrtf(n2); float c_inv = 1.0f/sqrtf(c); float arg = c_inv*vn; if (arg > 0.999f) arg = 0.999f; float ah = atanhf(arg); float sc = ah/(c_inv*vn); for (int i = 0; i < dim; i++) tv[i] = v[i]*sc; }
    float bm[32]; memset(bm, 0, sizeof(bm));
    for (int p = 0; p < nb; p++) for (int d = 0; d < dim; d++) bm[d] += m->levels[l].bd[p*dim+d];
    for (int d = 0; d < dim; d++) bm[d] /= (float)nb;
    float ctx[97]; int ci = 0;
    for (int d = 0; d < dim; d++) ctx[ci++] = tv[d];
    for (int d = 0; d < dim; d++) ctx[ci++] = bm[d];
    for (int d = 0; d < dim; d++) ctx[ci++] = m->levels[l].desc[d];
    ctx[ci++] = m->levels[l].spr;
    if (ctx_out) memcpy(ctx_out, ctx, 97*sizeof(float));
    float hb[32], hba[32], fv[32];
    for (int h = 0; h < fh; h++) { float s = m->levels[l].fb1[h]; for (int c2 = 0; c2 < 97; c2++) s += m->levels[l].fw1[h*97+c2]*ctx[c2]; hb[h] = s; hba[h] = gf(s); }
    for (int o = 0; o < dim; o++) { float s = m->levels[l].fb2[o]; for (int h = 0; h < fh; h++) s += m->levels[l].fw2[o*fh+h]*hba[h]; fv[o] = s; }
    if (hb_out) memcpy(hb_out, hb, 32*sizeof(float));
    if (hba_out) memcpy(hba_out, hba, 32*sizeof(float));
    float to[32]; for (int i = 0; i < dim; i++) to[i] = tv[i] + fv[i];
    float ns = 0; for (int i = 0; i < dim; i++) ns += to[i]*to[i];
    if (ns > 100.0f) { float sc = 10.0f/sqrtf(ns); for (int i = 0; i < dim; i++) to[i] *= sc; }
    float vn2 = 0; for (int i = 0; i < dim; i++) vn2 += to[i]*to[i];
    float vn = sqrtf(vn2);
    if (vn < 1e-8f) { memcpy(v_out, to, dim * sizeof(float)); }
    else { float c_inv = 1.0f/sqrtf(c); float th = tanhf(c_inv*vn); float sc = th/(c_inv*vn); for (int i = 0; i < dim; i++) v_out[i] = to[i]*sc; }
    n2 = 0; for (int i = 0; i < dim; i++) n2 += v_out[i]*v_out[i];
    if (n2 > mx) { float s = sqrtf(mx/(n2+1e-10f)); for (int i = 0; i < dim; i++) v_out[i] *= s; }
}

static void level_inverse(Model* m, int l, const float* v_in, float* v_out, const float* ctx, const float* hba) {
    int fh = m->levels[l].fh, dim = m->levels[l].dim;
    float c = m->levels[l].c;
    float v_dec[32]; memcpy(v_dec, v_in, dim * sizeof(float));
    float n2 = 0; for (int i = 0; i < dim; i++) n2 += v_dec[i]*v_dec[i];
    float mx = 0.999f/c;
    if (n2 > mx) { float s = sqrtf(mx/(n2+1e-10f)); for (int i = 0; i < dim; i++) v_dec[i] *= s; }
    float tv[32];
    if (n2 < 1e-15f) { memcpy(tv, v_dec, dim * sizeof(float)); }
    else { float vn = sqrtf(n2); float c_inv = 1.0f/sqrtf(c); float arg = c_inv*vn; if (arg > 0.999f) arg = 0.999f; float ah = atanhf(arg); float sc = ah/(c_inv*vn); for (int i = 0; i < dim; i++) tv[i] = v_dec[i]*sc; }
    float fv[32];
    for (int o = 0; o < dim; o++) { float s = m->levels[l].fb2[o]; for (int h = 0; h < fh; h++) s += m->levels[l].fw2[o*fh+h]*hba[h]; fv[o] = s; }
    float to[32]; for (int i = 0; i < dim; i++) to[i] = tv[i] - fv[i];
    float vn2 = 0; for (int i = 0; i < dim; i++) vn2 += to[i]*to[i];
    float vn = sqrtf(vn2);
    if (vn < 1e-8f) { memcpy(v_out, to, dim * sizeof(float)); }
    else { float c_inv = 1.0f/sqrtf(c); float th = tanhf(c_inv*vn); float sc = th/(c_inv*vn); for (int i = 0; i < dim; i++) v_out[i] = to[i]*sc; }
    n2 = 0; for (int i = 0; i < dim; i++) n2 += v_out[i]*v_out[i];
    if (n2 > mx) { float s = sqrtf(mx/(n2+1e-10f)); for (int i = 0; i < dim; i++) v_out[i] *= s; }
}

/* Generate a random training frame on-the-fly */
static void generate_random_frame(float* buf, int W, int H) {
    float t = rf();
    float phase = t * 2.0f * M_PI;
    int category = (int)(rf() * 50);
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float u = (float)x / (W - 1);
            float v = (float)y / (H - 1);
            int idx = (y * W + x) * 3;
            float r = 0.5f, g = 0.5f, b = 0.5f;
            
            switch (category) {
                case 0: r = 0.5f + 0.4f * sinf(phase + u * 3.0f); g = 0.5f + 0.4f * sinf(phase * 1.3f + v * 2.0f); b = 0.5f + 0.4f * cosf(phase * 0.7f + (u + v) * 2.5f); break;
                case 1: { float offset = fmodf(t * 2.0f, 1.0f); float cx = u - offset, cy = v - 0.5f; float dist = sqrtf(cx*cx + cy*cy); r = fmaxf(0.0f, 1.0f - dist * 2.0f); g = fmaxf(0.0f, 1.0f - dist * 2.5f) * 0.5f; b = fmaxf(0.0f, 1.0f - dist * 1.5f) * 0.8f; break; }
                case 2: { float cx = u - 0.5f, cy = v - 0.5f; float angle = atan2f(cy, cx) + phase; float dist = sqrtf(cx*cx + cy*cy); float ring = fmodf(angle * 3.0f + dist * 10.0f, 1.0f); r = ring * (1.0f - dist); g = ring * dist; b = (1.0f - ring) * (1.0f - dist * 0.5f); break; }
                case 3: { float zoom = 1.0f + 0.5f * sinf(phase * 2.0f); float cx = (u - 0.5f) * zoom + 0.5f; float cy = (v - 0.5f) * zoom + 0.5f; float dist = sqrtf((cx-0.5f)*(cx-0.5f) + (cy-0.5f)*(cy-0.5f)); r = fmaxf(0.0f, 1.0f - dist * 3.0f); g = fmaxf(0.0f, 1.0f - dist * 2.0f) * 0.7f; b = fmaxf(0.0f, 1.0f - dist * 4.0f) * 0.3f; break; }
                case 4: { r = g = b = 0.05f; for (int obj = 0; obj < 3; obj++) { float ox = 0.5f + 0.3f * sinf(phase * (1.0f + obj * 0.3f) + obj * 2.094f); float oy = 0.5f + 0.3f * cosf(phase * (0.7f + obj * 0.2f) + obj * 2.094f); float dist = sqrtf((u-ox)*(u-ox) + (v-oy)*(v-oy)); float intensity = fmaxf(0.0f, 1.0f - dist * 8.0f); r += intensity * (obj == 0 ? 1.0f : 0.0f); g += intensity * (obj == 1 ? 1.0f : 0.0f); b += intensity * (obj == 2 ? 1.0f : 0.0f); } break; }
                default: { float freq1 = 5.0f + rf() * 15.0f; float freq2 = 5.0f + rf() * 15.0f; float speed = 1.0f + rf() * 3.0f; r = sinf(u * freq1 + phase * speed) * 0.4f + 0.5f; g = sinf(v * freq2 + phase * speed * 0.8f) * 0.4f + 0.5f; b = sinf((u + v) * (freq1 + freq2) * 0.5f + phase * speed * 1.2f) * 0.4f + 0.5f; break; }
            }
            
            float noise = (rf() - 0.5f) * 0.05f;
            r += noise; g += noise; b += noise;
            
            float gray = 0.2989f * r + 0.5870f * g + 0.1140f * b;
            gray = fmaxf(0.0f, fminf(1.0f, gray));
            buf[idx + 0] = gray;
            buf[idx + 1] = gray;
            buf[idx + 2] = gray;
        }
    }
}

static float fwd_bwd(float* img, float* gpw, float*gpb, float*gdw, float*gdb,
                     float gfw1[4][32*97], float gfb1[4][32], float gfw2[4][32*32], float gfb2[4][32]){
    float loss=0;
    for(int p=0;p<NUM_PATCH;p++){
        int py=p/16, px=p%16;
        float pd[PATCH_PIX], gt[PATCH_PIX];
        int idx=0;
        for(int dy=0;dy<4;dy++)for(int dx=0;dx<4;dx++){
            int iy=py*4+dy, ix=px*4+dx, ii=(iy*64+ix)*3;
            gt[idx]=img[ii]; pd[idx]=gt[idx]; idx++;
        }
        float pre[DIM];
        for(int o=0;o<DIM;o++){float s=model.pb[o];for(int i=0;i<PATCH_PIX;i++)s+=model.pw[o*PATCH_PIX+i]*pd[i];pre[o]=s;}
        float mean=0;for(int i=0;i<DIM;i++)mean+=pre[i];mean/=DIM;
        float var=0;for(int i=0;i<DIM;i++){float d=pre[i]-mean;var+=d*d;}var/=DIM;
        float inv=1/sqrtf(var+1e-5f);
        float pl[DIM];for(int i=0;i<DIM;i++)pl[i]=(pre[i]-mean)*inv;
        float v[DIM];memcpy(v,pl,DIM*sizeof(float));
        float lvl_ctx[4][97], lvl_hb[4][32], lvl_hba[4][32];
        for(int l=0;l<4;l++){float vo[32];level_forward(&model, l, v, vo, lvl_ctx[l], lvl_hb[l], lvl_hba[l]);memcpy(v, vo, DIM*sizeof(float));}
        float lat[LDIM];for(int i=0;i<LDIM;i++)lat[i]=v[i];
        float dlat[LDIM];memset(dlat,0,sizeof(dlat));
        float v_dec[32]; memcpy(v_dec, lat, LDIM*sizeof(float));
        for(int l=3;l>=0;l--){float vo[32];level_inverse(&model, l, v_dec, vo, lvl_ctx[l], lvl_hba[l]);memcpy(v_dec, vo, DIM*sizeof(float));}
        float pl_dec[32]; for (int i = 0; i < DIM; i++) pl_dec[i] = v_dec[i];
        float pre_dec[32]; for (int i = 0; i < DIM; i++) pre_dec[i] = pl_dec[i] * sqrtf(var + 1e-5f) * inv + mean;
        for(int o=0;o<PATCH_PIX;o++){
            float pred=model.db[p*PATCH_PIX+o];
            for(int i=0;i<LDIM;i++)pred+=model.dw[p*PATCH_PIX*LDIM+o*LDIM+i]*lat[i];
            if(pred<0)pred=0; if(pred>1)pred=1;
            float d=pred-gt[o];loss+=d*d;
            float grad=2.0f*d/(float)(NUM_PATCH*PATCH_PIX);
            gdb[p*PATCH_PIX+o]+=grad;
            for(int i=0;i<LDIM;i++){gdw[p*PATCH_PIX*LDIM+o*LDIM+i]+=grad*lat[i];dlat[i]+=grad*model.dw[p*PATCH_PIX*LDIM+o*LDIM+i];}
        }
        float dv[DIM];for(int i=0;i<LDIM;i++)dv[i]=dlat[i];for(int i=LDIM;i<DIM;i++)dv[i]=0;
        for(int l=3;l>=0;l--){
            int fh=model.levels[l].fh;float c=model.levels[l].c;
            float dfv[DIM];for(int i=0;i<DIM;i++)dfv[i]=dv[i];
            float dhba[32];memset(dhba,0,sizeof(dhba));
            for(int o=0;o<DIM;o++){float g=dfv[o];gfb2[l][o]+=g;for(int h=0;h<fh;h++){gfw2[l][o*fh+h]+=g*lvl_hba[l][h];dhba[h]+=g*model.levels[l].fw2[o*fh+h];}}
            float dhb[32];for(int h=0;h<fh;h++)dhb[h]=dhba[h]*gb(lvl_hb[l][h]);
            float dctx[97];memset(dctx,0,sizeof(dctx));
            for(int h=0;h<fh;h++){float g=dhb[h];gfb1[l][h]+=g;for(int c2=0;c2<97;c2++){gfw1[l][h*97+c2]+=g*lvl_ctx[l][c2];dctx[c2]+=g*model.levels[l].fw1[h*97+c2];}}
            float dtv[DIM];for(int i=0;i<DIM;i++)dtv[i]=dctx[i];memcpy(dv,dtv,DIM*sizeof(float));
        }
        float dvar=0;for(int i=0;i<DIM;i++)dvar+=dv[i]*(pre[i]-mean);dvar/=(float)DIM;
        float dmean=0;for(int i=0;i<DIM;i++)dmean+=-dv[i]*inv;dmean/=(float)DIM;
        for(int i=0;i<DIM;i++){float dx=inv*(dv[i]-dmean-(pre[i]-mean)*dvar*inv*inv);gpb[i]+=dx;for(int j=0;j<PATCH_PIX;j++)gpw[i*PATCH_PIX+j]+=dx*pd[j];}
    }
    return loss/(float)(NUM_PATCH*PATCH_PIX);
}

int main(void){
    printf("GAAD WuBu Training - Online Data Generation\n");
    printf("============================================\n\n");
    init_model();
    printf("Model initialized\n");
    
    static float ggpw[1*DIM], ggpb[DIM];
    static float ggd[NUM_PATCH*PATCH_PIX*LDIM], ggdb[NUM_PATCH*PATCH_PIX];
    static float ggfw1[4][32*97], ggfb1[4][32], ggfw2[4][32*32], ggfb2[4][32];
    WubuSGDConfig cfg = {.learning_rate=1e-4f, .momentum_factor=0.9f, .weight_decay=1e-4f, .max_grad_norm=0.1f};
    WubuManifoldBinding euc_man = {.c=0.f, .manifold_enabled=0};
    WubuSGD opt_proj, opt_dec, opt_levels[4];
    wubu_sgd_init(&opt_proj, &cfg, euc_man, 1*DIM+DIM);
    wubu_sgd_init(&opt_dec, &cfg, euc_man, NUM_PATCH*PATCH_PIX*LDIM+NUM_PATCH*PATCH_PIX);
    for(int l=0;l<4;l++) wubu_sgd_init(&opt_levels[l], &cfg, euc_man, 32*97+32+32*32+32);
    FILE* log = fopen("output/train_log_online.csv", "w");
    fprintf(log, "epoch,loss,psnr\n");
    
    float* frame = (float*)malloc(IMG_W * IMG_H * 3 * sizeof(float));
    float eloss = 0;
    int total_steps = 0;
    
    for (int ep = 0; ep < EPOCHS; ep++) {
        if (ep < 10) {
            float lr = 1e-5f + (1e-4f - 1e-5f) * (ep / 10.0f);
            opt_proj.config.learning_rate = lr;
            opt_dec.config.learning_rate = lr;
            for(int l=0;l<4;l++) opt_levels[l].config.learning_rate = lr;
        }
        eloss = 0;
        for (int step = 0; step < STEPS_PER_EPOCH; step++) {
            memset(ggpw, 0, sizeof(ggpw)); memset(ggpb, 0, sizeof(ggpb));
            memset(ggd, 0, sizeof(ggd)); memset(ggdb, 0, sizeof(ggdb));
            for (int l = 0; l < 4; l++) { memset(ggfw1[l], 0, sizeof(ggfw1[l])); memset(ggfb1[l], 0, sizeof(ggfb1[l])); memset(ggfw2[l], 0, sizeof(ggfw2[l])); memset(ggfb2[l], 0, sizeof(ggfb2[l])); }
            float bl = 0;
            for (int b = 0; b < BATCH; b++) {
                generate_random_frame(frame, IMG_W, IMG_H);
                bl += fwd_bwd(frame, ggpw, ggpb, ggd, ggdb, ggfw1, ggfb1, ggfw2, ggfb2);
            }
            bl /= BATCH;
            eloss += bl;
            total_steps++;
            
            float proj_params[1*DIM+DIM], proj_grads[1*DIM+DIM];
            memcpy(proj_params, model.pw, 1*DIM*sizeof(float));
            memcpy(proj_params+1*DIM, model.pb, DIM*sizeof(float));
            memcpy(proj_grads, ggpw, 1*DIM*sizeof(float));
            memcpy(proj_grads+1*DIM, ggpb, DIM*sizeof(float));
            wubu_sgd_step_euclidean(&opt_proj, proj_params, proj_grads, 1*DIM+DIM);
            memcpy(model.pw, proj_params, 1*DIM*sizeof(float));
            memcpy(model.pb, proj_params+1*DIM, DIM*sizeof(float));
            wubu_sgd_step_euclidean(&opt_dec, model.db, ggdb, NUM_PATCH*PATCH_PIX);
            wubu_sgd_step_euclidean(&opt_dec, model.dw, ggd, NUM_PATCH*PATCH_PIX*LDIM);
            for (int l = 0; l < 4; l++) {
                float lp[4192], lg[4192];
                memcpy(lp, model.levels[l].fw1, 32*97*sizeof(float));
                memcpy(lp+32*97, model.levels[l].fb1, 32*sizeof(float));
                memcpy(lp+32*97+32, model.levels[l].fw2, 32*32*sizeof(float));
                memcpy(lp+32*97+32+32*32, model.levels[l].fb2, 32*sizeof(float));
                memcpy(lg, ggfw1[l], 32*97*sizeof(float));
                memcpy(lg+32*97, ggfb1[l], 32*sizeof(float));
                memcpy(lg+32*97+32, ggfw2[l], 32*32*sizeof(float));
                memcpy(lg+32*97+32+32*32, ggfb2[l], 32*sizeof(float));
                wubu_sgd_step_euclidean(&opt_levels[l], lp, lg, 4192);
                memcpy(model.levels[l].fw1, lp, 32*97*sizeof(float));
                memcpy(model.levels[l].fb1, lp+32*97, 32*sizeof(float));
                memcpy(model.levels[l].fw2, lp+32*97+32, 32*32*sizeof(float));
                memcpy(model.levels[l].fb2, lp+32*97+32+32*32, 32*sizeof(float));
            }
        }
        eloss /= STEPS_PER_EPOCH;
        float psnr = eloss < 1e-10 ? 100 : 10*log10f(1.0f/eloss);
        printf("Epoch %3d/%d | Loss: %.6f | PSNR: %.1f dB | Steps: %d\n", ep+1, EPOCHS, eloss, psnr, total_steps);
        fprintf(log, "%d,%.6f,%.2f\n", ep+1, eloss, psnr);
    }
    fclose(log);
    free(frame);
    FILE* fi = fopen("output/gaad_trained_online.bin", "wb");
    fwrite(&model, sizeof(Model), 1, fi);
    fclose(fi);
    printf("\nDone. Final PSNR: %.1f dB\n", 10*log10f(1.0f/(eloss > 1e-10f ? eloss : 1e-10f)));
    printf("Weights saved to output/gaad_trained_online.bin\n");
    wubu_sgd_free(&opt_proj); wubu_sgd_free(&opt_dec);
    for (int l = 0; l < 4; l++) wubu_sgd_free(&opt_levels[l]);
    return 0;
}
