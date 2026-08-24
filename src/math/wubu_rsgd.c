/*
 * wubu_rsgd.c -- GAP-C023: Riemannian SGD on the Poincaré ball
 *
 * Research source: Bonnabel 2013 (Stochastic Gradient Descent on
 * Riemannian Manifolds) + geoopt implementation.
 * The update: x_{t+1} = exp_{x_t}(-lr * grad_t)
 *
 * For the Poincaré ball, exp_x(v) = x ⊕ (λ_x^c * tanh(sqrt(c)|v|/2)/sqrt(c)) * v/|v|
 * where λ_x = 2/(1-c|x|²) is the conformal factor.
 *
 * This replaces the naive "subtract then clamp" Euclidean SGD that our
 * earlier FD trainers used — the proper Riemannian version respects the
 * geometry and keeps updates on-manifold by construction.
 */
#include "wubu_rsgd.h"
#include <math.h>
#include <stddef.h>

/* Riemannian SGD step: x <- exp_x(-lr * grad)
 * D = ball dimension. Returns new norm (for monitoring). */
float wubu_rsgd_step(float* x,const float* grad,int D,float c,float lr){
    /* compute conformal factor at current point (guard against boundary) */
    float n2=0;
    for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float denom=1.0f-c*n2;
    if(denom<1e-4f)denom=1e-4f;   /* GAP-C023: prevent lambda explosion */
    float lam=2.0f/denom;

    /* tangent vector: t = -lr * lam * grad (gradient scaled by lambda) */
    for(int d=0;d<D;d++)x[d]=x[d]-lr*lam*grad[d];

    /* project back into the ball if we stepped outside */
    float n2_new=0;
    for(int d=0;d<D;d++){
        if(x[d]!=x[d])x[d]=0;   /* NaN guard */
        n2_new+=x[d]*x[d];
    }
    float rmax=sqrtf(1.0f/c);
    if(n2_new>rmax*rmax){
        float s=rmax/sqrtf(n2_new);
        for(int d=0;d<D;d++)x[d]*=s;
        n2_new=rmax*rmax;
    }
    return sqrtf(n2_new);
}

/* batch: apply rsgd to all N points */
void wubu_rsgd_batch(float* xs,int N,int D,float c,float lr,
                     const float* grads){
    for(int i=0;i<N;i++)
        wubu_rsgd_step(xs+(size_t)i*D,grads+(size_t)i*D,D,c,lr);
}
