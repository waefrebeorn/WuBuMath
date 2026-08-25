/*
 * wubu_attnfrobenius.c -- GAP-B019: Attention Frobenius-norm monitor
 * (the ||P||_F gradient-explosion early-warning companion to B017)
 *
 * Research source: EMNLP 2025 variance-sensitivity paper, Proposition
 * 5.3 + Eq. 9: the attention probability matrix's Frobenius norm is
 * bounded in [1, sqrt(N)] — exactly 1 when every row is uniform
 * (max entropy), exactly sqrt(N) when every row is one-hot (collapsed).
 * And ||P||_F directly upper-bounds the attention layer's input-gradient
 * norm, so tracking it tracks gradient risk.
 *
 * Gate: uniform → ~1; one-hot → ~sqrt(N); and the two monitors agree:
 * high entropy ↔ low Frobenius.
 */
#include "wubu_attnfrobenius.h"
#include <math.h>

float wubu_af_frobenius(const float* attn,int N,int n_heads){
    /* attn: [n_heads, N, N] */
    double s2=0;
    long n_all=(long)n_heads*N*N;
    for(long i=0;i<n_all;i++)s2+=(double)attn[i]*attn[i];
    return (float)sqrt(s2);
}

/* normalized: 0 = collapsed (one-hot rows), 1 = uniform rows.
 * per-row p2 = sum_j p_ij^2 in [1/N, 1]; average then invert. */
float wubu_af_uniformity(const float* attn,int N,int n_heads){
    double acc=0;
    int rows=n_heads*N;
    for(int r=0;r<rows;r++){
        double p2=0;
        for(int j=0;j<N;j++)p2+=(double)attn[r*N+j]*attn[r*N+j];
        /* map p2 in [1/N,1] -> uniformity in [0,1] */
        double u=(p2-1.0/N)/(1.0-1.0/N);
        if(u<0)u=0;
        if(u>1)u=1;
        acc+=1.0-u;
    }
    return (float)(acc/rows);
}
