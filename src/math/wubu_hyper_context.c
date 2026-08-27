/*
 * wubu_hyper_context.c — C21: Hyperbolic manifold priors for codec
 *
 * Embeds block statistics on the Poincare ball and uses hyperbolic
 * gyromidpoint for predictor fusion and context modeling.
 *
 * This is the WuBu signature: hyperbolic geometry meets video coding.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <math.h>
#include "wubu_hyperbolic.h"
#include "wubu_cabac.h"

/* Hyperbolic context state for a block */
typedef struct {
    float embedding[8];  /* 8-dim Poincare embedding of block stats */
    float curvature;     /* adaptive curvature parameter */
    int count;           /* number of blocks seen so far */
} HyperContext;

/* Compute hyperbolic gyromidpoint of two embeddings */
void hyper_gyromidpoint(float* out, const float* a, const float* b, int N, float c) {
    /* gyromidpoint = mobius_add(a, b) scaled by 0.5 in tangent space */
    float sum[8];
    for(int i=0;i<N;i++) sum[i] = a[i] + b[i];
    
    /* Project to Poincare ball */
    wubu_poincare_clip(out, sum, N, c);
    
    /* Scale by 0.5 (approximate gyromidpoint for nearby points) */
    float norm = 0;
    for(int i=0;i<N;i++) norm += out[i]*out[i];
    norm = sqrtf(norm + 1e-8f);
    
    if(norm > 1e-6f){
        float scale = 0.5f / norm;
        for(int i=0;i<N;i++) out[i] *= scale;
    }
    wubu_poincare_clip(out, out, N, c);
}

/* Update hyperbolic context with new block statistics */
void hyper_context_update(HyperContext* ctx, const int* coeffs, int n) {
    /* Compute block statistics */
    float mean = 0, var = 0, energy = 0;
    int significant = 0;
    for(int i=0;i<n;i++){
        mean += coeffs[i];
        energy += (float)(coeffs[i]*coeffs[i]);
        if(coeffs[i] != 0) significant++;
    }
    mean /= n;
    var = energy/n - mean*mean;
    if(var < 0) var = 0;
    
    /* Create new embedding from stats */
    float new_embed[8];
    new_embed[0] = mean / 100.0f;
    new_embed[1] = sqrtf(var) / 100.0f;
    new_embed[2] = (float)significant / (float)n;
    new_embed[3] = energy / 10000.0f;
    new_embed[4] = (coeffs[0] != 0) ? 1.0f : 0.0f;  /* DC coefficient present */
    new_embed[5] = (coeffs[1] != 0 || coeffs[8] != 0) ? 1.0f : 0.0f;  /* AC1/AC8 present */
    new_embed[6] = 0;
    new_embed[7] = 0;
    
    /* Update running embedding using gyromidpoint */
    if(ctx->count == 0){
        memcpy(ctx->embedding, new_embed, 8*sizeof(float));
    }else{
        /* Blend: new_embedding = gyromidpoint(old, new) */
        float blended[8];
        hyper_gyromidpoint(blended, ctx->embedding, new_embed, 8, ctx->curvature);
        memcpy(ctx->embedding, blended, 8*sizeof(float));
    }
    ctx->count++;
}

/* Use hyperbolic context to select CABAC contexts */
void hyper_context_select_cabac(HyperContext* ctx, CabacContext* cabac_ctx, int n_ctx) {
    /* Select context based on hyperbolic distance from origin */
    float dist = 0;
    for(int i=0;i<8;i++) dist += ctx->embedding[i]*ctx->embedding[i];
    dist = sqrtf(dist);
    
    /* Curvature-adaptive context selection */
    int ctx_idx = (int)(dist * 10.0f) % n_ctx;
    if(ctx_idx < 0) ctx_idx = 0;
    if(ctx_idx >= n_ctx) ctx_idx = n_ctx - 1;
    
    /* Initialize selected context */
    wubu_cabac_init_contexts(&cabac_ctx[ctx_idx], 1, 40, 10);
}
