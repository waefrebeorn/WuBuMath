/*
 * wubu_causal.c -- GAP-C035: Causal masking for autoregressive attention
 *
 * The upper-triangular mask that makes attention autoregressive: position
 * i may attend to positions 0..i only. Applied by zeroing masked weights
 * and renormalizing rows (equivalent to -inf pre-softmax).
 *
 * Gates: strictly upper triangle zeroed; diagonal + lower preserved
 * relative order; row sums still 1 after renormalization.
 */
#include "wubu_causal.h"
#include <string.h>

void wubu_causal_apply(float* attn,int N){
    /* zero strictly-upper, then renormalize each row */
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++)attn[(size_t)i*N+j]=0.0f;
        float s=0;
        for(int j=0;j<=i;j++)s+=attn[(size_t)i*N+j];
        if(s>1e-10f){
            for(int j=0;j<=i;j++)attn[(size_t)i*N+j]/=s;
        }else{
            /* all-zero allowed set: fall back to uniform causal */
            for(int j=0;j<=i;j++)attn[(size_t)i*N+j]=1.0f/(float)(i+1);
        }
    }
}

/* build the mask itself (1 = allowed, 0 = masked) */
void wubu_causal_mask(int N,uint8_t* mask){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            mask[(size_t)i*N+j]=(uint8_t)(j<=i);
}

/* check: does attn respect causality? (all j>i are zero) */
int wubu_causal_respected(const float* attn,int N,float eps){
    for(int i=0;i<N;i++)
        for(int j=i+1;j<N;j++)
            if(attn[(size_t)i*N+j]>eps)return 0;
    return 1;
}
