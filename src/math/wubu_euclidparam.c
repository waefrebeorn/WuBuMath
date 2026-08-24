/*
 * wubu_euclidparam.c -- GAP-C026: Euclidean parametrization of hyperbolic
 * space (the numerically-safest representation, Mishne et al. ICML 2023)
 *
 * Research finding: the Poincaré ball can only represent points up to
 * geodesic distance ~38 from origin in float64 (1-|x| underflows at ~2^-53);
 * the Lorentz model degrades softly past ~19 but has better optimization
 * (no gradient vanishing). The Euclidean parametrization gets BOTH:
 *   - store parameters z in R^D (unbounded — no representation limit)
 *   - obtain manifold points via exp_0(z) on demand
 *   - optimization behaves like Lorentz (no vanishing gradients)
 *
 * This module wraps that pattern: keep a Euclidean parameter vector,
 * produce on-ball Poincaré coordinates via exp_0 whenever needed.
 */
#include "wubu_euclidparam.h"
#include <math.h>
#include <string.h>

/* map unbounded R^D params to on-ball Poincaré coords:
 * x = tanh(|z|/2) * z/|z|  (unit curvature; general c: tanh(sqrt(c)|z|/2)/sqrt(c)) */
void wubu_ep_to_ball(const float* z,int D,float c,float* x){
    float n2=0;
    for(int d=0;d<D;d++)n2+=z[d]*z[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){for(int d=0;d<D;d++)x[d]=0;return;}
    float coeff=tanhf(sqrtf(c)*nv*0.5f)/(sqrtf(c)*nv);
    /* GAP-C026 fp32 safety: tanh saturates to exactly 1.0f for large nv,
     * putting x exactly ON the boundary (norm=1 in fp32). Cap at 0.99999
     * of rmax so downstream ops never see the boundary singularity. */
    float n2x=0;
    for(int d=0;d<D;d++){x[d]=coeff*z[d];n2x+=x[d]*x[d];}
    if(n2x>0.99998f){
        float s=sqrtf(0.99998f/n2x);
        for(int d=0;d<D;d++)x[d]*=s;
    }
}

/* inverse: on-ball point -> Euclidean parameter (log_0) */
void wubu_ep_from_ball(const float* x,int D,float c,float* z){
    float n2=0;
    for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){for(int d=0;d<D;d++)z[d]=0;return;}
    /* |z| = 2/sqrt(c) * atanh(sqrt(c)|x|); direction preserved */
    float arg=sqrtf(c)*nv;
    if(arg>0.999999f)arg=0.999999f;
    float zn=2.0f/sqrtf(c)*atanhf(arg);
    for(int d=0;d<D;d++)z[d]=zn*x[d]/nv;
}

/* SGD directly on the Euclidean parameters — no conformal factor, no
 * boundary explosion, exactly what Mishne et al. prove is stable. */
void wubu_ep_sgd_step(float* z,const float* grad,int D,float lr){
    for(int d=0;d<D;d++)z[d]-=lr*grad[d];
}

/* distance computed through the parametrization (for monitoring) */
float wubu_ep_distance(const float* z1,const float* z2,int D,float c){
    float x1[64],x2[64];
    int dd=D<64?D:64;
    wubu_ep_to_ball(z1,dd,c,x1);
    wubu_ep_to_ball(z2,dd,c,x2);
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<dd;d++){
        float df=x1[d]-x2[d];ab2+=df*df;
        a2+=x1[d]*x1[d];b2+=x2[d]*x2[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    float arg=1+2*c*ab2/den;
    return acoshf(arg>1?arg:1)/sqrtf(c);
}
