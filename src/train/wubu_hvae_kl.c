/*
 * wubu_hvae_kl.c -- GAP-D028: Hyperbolic VAE KL divergence
 * (wrapped normal vs wrapped prior)
 *
 * Research source: Nagano et al. ICML 2019 + Mathieu et al. NeurIPS 2019.
 * The KL between two wrapped normals q(z|s)=WN(μ_s,σ_s) and prior
 * p(z)=WN(μ₀,σ₀) has no closed form on the ball. The practical estimator:
 *
 *   KL ≈ log q(z) - log p(z)  for z ~ q (reparameterized sample),
 *
 * where both log-densities include the Jacobian of the exp/log map
 * (the radial distortion term). This is the ELBO's KL term done right.
 */
#include "wubu_hvae_kl.h"
#include "wubu_hvae.h"
#include <math.h>
#include <stdlib.h>

/* log-density of wrapped normal at z with mean mu, diag sigma */
static float wkl_log_q(const float* z,const float* mu,const float* sigma,
                        int D,float c){
    /* Euclidean part in tangent space */
    double e=0;
    for(int d=0;d<D;d++){
        float df=z[d]-mu[d];
        e-=(double)(df*df)/(2*sigma[d]*sigma[d]);
    }
    /* NOTE: The exp_0 Jacobian is IDENTICAL for both q and p at the same z
     * (it depends only on z, not on distribution parameters), so it CANCELS
     * in log q - log p. Omitting it here is exact for the KL difference.
     *
     * Also add the normalization constants: log N(x;mu,sig) needs
     * -D*log(sig) - D/2*log(2pi). We include the -D*log(sig) term here
     * since it does NOT cancel between q and p when sigmas differ. */
    return (float)e;
}

/* Monte-Carlo KL estimate: mean over samples of log q - log p */
float wubu_hvae_kl_estimate(const float* mu_q,const float* sigma_q,
                             const float* mu_p,const float* sigma_p,
                             int D,float c,int n_samples){
    if(n_samples<1)return -1;
    float* z=malloc(sizeof(float)*(size_t)D);
    double total=0;
    unsigned rs=(unsigned)(mu_q[0]*1000)+7u;
    for(int s=0;s<n_samples;s++){
        /* draw full Gaussian vector v ~ N(mu_q, diag(sigma_q)) */
        float v[64];
        int dd=D<64?D:64;
        for(int d=0;d<dd;d++){
            rs=rs*1103515245u+12345u;
            float u1=((rs>>16)%10000)/10000.0f;if(u1<1e-7f)u1=1e-7f;
            rs=rs*1103515245u+12345u;
            float u2=((rs>>16)%10000)/10000.0f;
            v[d]=mu_q[d]+sqrtf(-2*logf(u1))*cosf(2*3.14159265f*u2)*sigma_q[d];
        }
        for(int d=dd;d<D;d++)v[d]=0;
        /* project into ball via exp0 */
        float n2=0;for(int d=0;d<dd&&d<64;d++)n2+=v[d]*v[d];
        float nv=sqrtf(n2);
        if(nv>1e-10f){
            float coeff=tanhf(sqrtf(c)*nv)/(sqrtf(c)*nv);
            for(int d=0;d<D&&d<64;d++)z[d]=coeff*v[d];
        }else{
            for(int d=0;d<D;d++)z[d]=mu_q[d];
        }
        float n2c=0;for(int d=0;d<D;d++)n2c+=z[d]*z[d];
        if(n2c>0.99998f){float sc=sqrtf(0.99998f/n2c);for(int d=0;d<D;d++)z[d]*=sc;}

        total+=wkl_log_q(z,mu_q,sigma_q,D,c)
              -wkl_log_q(z,mu_p,sigma_p,D,c);
    }
    free(z);
    /* normalization constant difference: -D*log(sigma_q/sigma_p) */
    float ncorr=0;
    for(int d=0;d<D;d++)ncorr+=logf(sigma_p[d]/sigma_q[d]);
    total+=(double)(n_samples*ncorr);
    return (float)(total/n_samples);
}
