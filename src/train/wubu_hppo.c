/*
 * wubu_hppo.c -- GAP-H015: Hyperbolic PPO clipped surrogate
 * (manifold policy-ratio trust region)
 *
 * Research source: Schulman et al. 2017 (PPO-Clip) + the Riemannian
 * isometric correction insight (arXiv:2607.10169): on a manifold, the
 * policy ratio should be measured with GEODESIC distances, not raw
 * Euclidean parameter deltas, or the clip boundary is geometrically wrong.
 *
 * Our form: policy = Gaussian over tangent actions with mean mu on ball.
 *   ratio_i = exp(-d_c(mu_new, s_i)^2 / (2*sigma^2))
 *           / exp(-d_c(mu_old, s_i)^2 / (2*sigma^2))
 * i.e. a geodesic-Gaussian likelihood ratio.
 * Clipped surrogate:
 *   L = E[ min(ratio*A, clip(ratio, 1-eps, 1+eps)*A) ]
 */
#include "wubu_hppo.h"
#include <math.h>
#include <string.h>

static float hp_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

float wubu_hp_ratio(const float* mu_new,const float* mu_old,
                     const float* state,int D,float c,float sigma){
    float dn=hp_dist(mu_new,state,D,c);
    float dold=hp_dist(mu_old,state,D,c);
    /* log ratio of geodesic Gaussians */
    float logr=(-dn*dn/(2*sigma*sigma))-(-dold*dold/(2*sigma*sigma));
    if(logr>10.0f)logr=10.0f;
    if(logr<-10.0f)logr=-10.0f;
    return expf(logr);
}

float wubu_hp_surrogate(const float* mu_new,const float* mu_old,
                         const float* states,const float* advantages,
                         int N,int D,float c,float sigma,float eps){
    if(N<1)return 0;
    double total=0;
    for(int i=0;i<N;i++){
        const float* si=states+(size_t)i*D;
        float r=wubu_hp_ratio(mu_new,mu_old,si,D,c,sigma);
        float A=advantages[i];
        float rc=r;
        if(rc<1-eps)rc=1-eps;
        if(rc>1+eps)rc=1+eps;
        float unclipped=r*A;
        float clipped=rc*A;
        total+=(unclipped<clipped?unclipped:clipped);
    }
    return (float)(total/N);
}
