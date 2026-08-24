/*
 * wubu_hnorm.c -- GAP-C031: Poincaré LayerNorm (GGBall §E.6.1)
 *
 * Hyperbolic layer norm, per GGBall: "Layer normalization and
 * feed-forward networks operate in tangent space."
 *
 *   1. μ = Fréchet mean of the batch (approximated by gyromidpoint chain)
 *   2. u_i = log_μ(x_i) — tangent vectors at mean
 *   3. standard LayerNorm on u_i (center + scale by learned gamma/beta)
 *   4. x'_i = exp_μ(u_i') — map back
 *
 * We use origin as the anchor point p (standard simplification): log_0 /
 * exp_0 are cheap and exact; the batch statistics still capture the
 * hyperbolic structure because log_0 preserves geodesic radii from 0.
 */
#include "wubu_hnorm.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void wubu_hlayernorm(const float* x,int B,int D,float c,
                     const float* gamma,const float* beta,float* out){
    /* Step 1: tangent vectors at origin */
    float tanv[512];   /* [B,64] — B<=8 assumed; guard below */
    if(B>8)return;
    int dd=D<64?D:64;
    for(int i=0;i<B;i++){
        const float* xi=x+(size_t)i*D;
        float n2=0;
        for(int d=0;d<dd;d++)n2+=xi[d]*xi[d];
        float nv=sqrtf(n2);
        float arg=sqrtf(c)*nv;
        if(arg>0.99999f)arg=0.99999f;
        if(nv<1e-10f){
            for(int d=0;d<dd;d++)tanv[i*64+d]=0;
        }else{
            float zn=(2.0f/sqrtf(c))*atanhf(arg)/nv;
            for(int d=0;d<dd;d++)tanv[i*64+d]=zn*xi[d];
        }
        for(int d=dd;d<D&&d<64;d++)tanv[i*64+d]=0;
    }

    /* Step 2: standard layernorm per-sample over D */
    for(int i=0;i<B;i++){
        float mean=0,var=0;
        for(int d=0;d<D&&d<64;d++)mean+=tanv[i*64+d];
        mean/=D;
        for(int d=0;d<D&&d<64;d++){
            float df=tanv[i*64+d]-mean;
            var+=df*df;
        }
        var/=D;
        float std=sqrtf(var)+1e-5f;
        for(int d=0;d<D;d++){
            float u=(d<dd)?(tanv[i*64+d]-mean)/std:0;
            out[(size_t)i*D+d]=(gamma?gamma[d]:1.0f)*u+(beta?beta[d]:0.0f);
        }
    }

    /* Step 3: exp_0 back onto ball */
    for(int i=0;i<B;i++){
        float n2=0;
        for(int d=0;d<D&&d<64;d++)n2+=out[(size_t)i*D+d]*out[(size_t)i*D+d];
        float nv=sqrtf(n2);
        if(nv>1e-10f){
            float coeff=tanhf(sqrtf(c)*nv)/(sqrtf(c)*nv);
            for(int d=0;d<D&&d<64;d++)out[(size_t)i*D+d]*=coeff;
            for(int d=64;d<D;d++)out[(size_t)i*D+d]*=coeff;
        }else{
            memset(out+(size_t)i*D,0,sizeof(float)*D);
            continue;
        }
        /* boundary cap */
        float n2c=0;
        for(int d=0;d<D;d++)n2c+=out[(size_t)i*D+d]*out[(size_t)i*D+d];
        if(n2c>0.99998f){
            float s=sqrtf(0.99998f/n2c);
            for(int d=0;d<D;d++)out[(size_t)i*D+d]*=s;
        }
    }
}
