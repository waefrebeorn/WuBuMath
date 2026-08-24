/*
 * wubu_hattention.c -- Hyperbolic attention primitives (GAP-C016)
 *
 * Implements the two halves of hyperbolic attention (Gulcehre et al. 2018,
 * node: arXiv:1805.09786) on the Poincaré ball:
 *   matching    : attention weights = softmax(-d(q,k_i)/tau)
 *   aggregation : weighted Möbius gyromidpoint of values
 * The gyromidpoint follows geoopt's _weighted_midpoint:
 *   gamma_i = conformal factor at x_i
 *   num     = sum(gamma_i * w_i * x_i)
 *   den     = sum((gamma_i - 1) * |w_i|)
 *   m       = num/den, then mobius-scaled by 1/(1+sqrt(1+c|m|^2))
 */
#include "wubu_hattention.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

float wubu_hattn_distance(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1.0f-c*a2)*(1.0f-c*b2);
    if(den<1e-9f)den=1e-9f;
    float arg=1.0f+2.0f*c*ab2/den;
    return acoshf(arg>1.0f?arg:1.0f)/sqrtf(c);
}

void wubu_hattn_matching(const float* q,const float* keys,int K,int D,
                         float tau,float* weights){
    /* softmax(-d/tau) */
    float m=-1e30f,z=0;
    for(int k=0;k<K;k++){
        weights[k]=-wubu_hattn_distance(q,keys+(size_t)k*D,D,1.0f)/tau;
        if(weights[k]>m)m=weights[k];
    }
    for(int k=0;k<K;k++){weights[k]=expf(weights[k]-m);z+=weights[k];}
    for(int k=0;k<K;k++)weights[k]/=z;
}

void wubu_hattn_aggregate(const float* values,const float* weights,
                          int K,int D,float c,float* out){
    /* gyromidpoint: num=sum(gamma*w*x), den=sum((gamma-1)*w), then scale */
    float num[256];for(int d=0;d<D&&d<256;d++)num[d]=0;
    float den=1e-10f;
    for(int k=0;k<K;k++){
        const float* x=values+(size_t)k*D;
        float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
        float gamma=2.0f/(1.0f-c*n2);
        if(gamma<1.0f)gamma=1.0f;   /* numerical guard */
        for(int d=0;d<D&&d<256;d++)num[d]+=gamma*weights[k]*x[d];
        den+=(gamma-1.0f)*weights[k];
    }
    /* two_mean then Möbius scalar mul by 1/(1+sqrt(1 - c|tm|^2))
     * NOTE: geoopt uses k=-c internally (Poincare ball = k<0), so the
     * scalar-mul factor is 1/(1+sqrt(1-c|tm|^2)). For identical points
     * this reduces to the point itself (verified numerically). */
    float tm[256];float tm_n2=0;
    for(int d=0;d<D&&d<256;d++){tm[d]=num[d]/den;tm_n2+=tm[d]*tm[d];}
    float disc=1.0f-c*tm_n2;
    if(disc<1e-9f)disc=1e-9f;
    float s=1.0f/(1.0f+sqrtf(disc));
    for(int d=0;d<D&&d<256;d++)out[d]=tm[d]*s;
}
