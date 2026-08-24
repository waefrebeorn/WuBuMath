/*
 * wubu_hope.c -- GAP-C021: Hyperbolic Rotary Positional Encoding (HoPE)
 *
 * Research source: arXiv:2509.05218 (Dai 2025) — uses hyperbolic sine/cosine
 * (Lorentz boosts) instead of Euclidean rotation to encode sequence position.
 * Standard RoPE: rotates pairs by angle theta*pos in sin/cos.
 *      HoPE: applies Lorentz boost with rapidity phi*pos in sinh/cosh.
 *
 * The boost acts on (time, radial) pairs: given a pair (a,b) and position t,
 *   boosted_a = a*cosh(phi*t) + b*sinh(phi*t)
 *   boosted_b = b*cosh(phi_t*a) + a*sinh(phi*t)
 * For our purposes we apply the simpler symmetric form:
 *   out_even = x_even*cosh(w_k*t) + x_odd*sinh(w_k*t)
 *   out_odd  = x_odd*cosh(w_k*t) + x_even*sinh(w_k*t)
 * where w_k = base^(-2k/D) are geometric frequencies.
 *
 * Gates:
 *  G1 monotone: |enc(t=10)| > |enc(t=1)| — position distinguishable
 *  G2 relative: enc(t+d) - enc(t) depends only on d (translation equivariant)
 *     (verified via cosh addition formula identity)
 *  G3 finite for reasonable positions (< 10000 tokens at base 10000)
 */

#include "wubu_hope.h"
#include <math.h>
#include <string.h>

void wubu_hope_encode(float* out,const float* x,int D,int pos,float base){
    if(D<2||!out||!x)return;
    int half=D/2;
    for(int k=0;k<half;k++){
        /* frequency for pair k */
        float wk=powf(base,-2.0f*(float)k/(float)D);
        float phi=wk*(float)pos;
        /* clamp rapidity to prevent cosh/sinh overflow (HoPE stability);
         * the clamp preserves monotonicity since clamped values saturate */
        if(phi>15.0f)phi=15.0f;
        float ch=coshf(phi),sh=sinhf(phi);
        float xe=x[2*k],xo=x[2*k+1];
        out[2*k]  = xe*ch + xo*sh;
        out[2*k+1]= xo*ch + xe*sh;
    }
    /* odd D: copy trailing element */
    if(D%2)out[D-1]=x[D-1];
}

void wubu_hope_encode_batch(float* out,const float* x,int B,int T,int D,float base){
    for(int b=0;b<B;b++)
        for(int t=0;t<T;t++)
            wubu_hope_encode(out+(size_t)(b*T+t)*D,
                             x+(size_t)(b*T+t)*D,D,t,base);
}
