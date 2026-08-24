/*
 * wubu_hvae.c -- GAP-C024: Hyperbolic VAE sampling (wrapped normal)
 *
 * Research source: Nagano et al. ICML 2019 "A Wrapped Normal Distribution
 * on Hyperbolic Space for Gradient-Based Learning" (arXiv:1901.06033).
 *
 * The wrapped normal on the Poincaré ball:
 *   1. sample v ~ N(0, sigma²I) in R^D (Euclidean)
 *   2. lift to tangent space at origin: u = [v] (identity for Poincaré)
 *   3. exp_0(u) maps onto the ball — this is the sample z
 *
 * For the Lorentz model version: v=[0,ṽ], parallel transport to μ,
 * then sinh/cosh map onto hyperboloid. We implement the Poincaré
 * variant which is simpler and sufficient for our use.
 *
 * The log-density includes a Jacobian correction term that accounts for
 * the volume distortion of exp/log maps. This is essential for correct
 * KL divergence in the VAE loss.
 */
#include "wubu_hvae.h"
#include <math.h>
#include <stdlib.h>

static unsigned long hvae_rs=0xCAFEBABEL;
static float hvae_gauss(void){
    /* Box-Muller */
    float u1=(float)((hvae_rs=(hvae_rs*1103515245u+12345u)>>16)%10000)/10000.0f;
    if(u1<1e-7f)u1=1e-7f;
    float u2=(float)((hvae_rs=(hvae_rs*1103515245u+12345u)>>16)%10000)/10000.0f;
    return sqrtf(-2.0f*logf(u1))*cosf(2.0f*M_PI_F*u2);
}

void wubu_hvae_sample(float* z,const float* mu,const float* sigma,
                      int D,float c){
    /* Step 1: sample Euclidean noise */
    float v[64];int dd=D<64?D:64;
    for(int d=0;d<dd;d++)v[d]=hvae_gauss()*sigma[d];
    for(int d=dd;d<D;d++)v[d]=0;

    /* Step 2-3: exp_0(v) = tanh(sqrt(c)|v|/2)/sqrt(c) * v/|v|
     * This is the standard Poincaré exponential at origin. */
    float n2=0;
    for(int d=0;d<D;d++)n2+=v[d]*v[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){for(int d=0;d<D;d++)z[d]=mu[d];return;}
    float coeff=tanhf(sqrtf(c)*nv*0.5f)/(sqrtf(c)*nv);
    for(int d=0;d<D;d++)z[d]=mu[d]+coeff*v[d];

    /* project into ball */
    float zn2=0;for(int d=0;d<D;d++)zn2+=z[d]*z[d];
    float rmax=sqrtf(1.0f/c);
    if(zn2>rmax*rmax){
        float s=rmax/sqrtf(zn2);
        for(int d=0;d<D;d++)z[d]*=s;
    }
}

/* Jacobian-corrected log-density (simplified: uses conformal factor) */
float wubu_hvae_log_density(const float* z,const float* mu,
                             const float* sigma,int D,float c){
    /* Euclidean part: N(mu, sigma²) in tangent space */
    float eucl=0;
    for(int d=0;d<D;d++){
        float df=z[d]-mu[d];
        eucl+=df*df/(sigma[d]*sigma[d]);
    }
    eucl=-0.5f*eucl;
    /* Jacobian: log det(exp_x) ≈ D*log(λ_x) where λ_x is conformal factor */
    float n2=0;for(int d=0;d<D;d++)n2+=z[d]*z[d];
    float denom=1.0f-c*n2;if(denom<1e-4f)denom=1e-4f;
    float lam=2.0f/denom;
    float jac=D*logf(lam);
    return eucl+jac;
}
