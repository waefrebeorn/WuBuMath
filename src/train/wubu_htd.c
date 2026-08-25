/*
 * wubu_htd.c -- GAP-H014: Hyperbolic TD learning with manifold values
 *
 * Alexander & Spiering 2010's "hyperbolically discounted TD" reimagined
 * for the Poincaré ball: value estimates live ON the ball (values are
 * ball points, not scalars — the codec's value = latent state target).
 *
 *   v_{t+1} = (1-a) ⊗ v_t ⊕ a ⊗ r_target
 *
 * where ⊕ is mobius_add and the step size decays with geodesic distance:
 * far targets move the estimate less per step (stability near boundary).
 * Bellman-style: V(s) ← V(s) + a_eff·(target - V(s)) along the geodesic,
 * with a_eff = lr / lambda_x (conformal damping).
 */
#include "wubu_htd.h"
#include <math.h>

static float htd_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

/* one TD step: move value toward target along the geodesic */
void wubu_htd_step(float* value,const float* target,int D,float c,
                   float lr,float gamma){
    /* conformal damping at current value */
    float n2=0;
    for(int d=0;d<D;d++)n2+=value[d]*value[d];
    float denom=1-c*n2;
    if(denom<1e-4f)denom=1e-4f;
    float lam=2.0f/denom;
    float a_eff=lr/(lam*(1+gamma));

    /* geodesic interpolation: v' = v ⊕ (a_eff ⊗ (target - v))
     * simplified to Euclidean-blend then project (valid near center) */
    float moved[64];
    int dd=D<64?D:64;
    for(int d=0;d<dd&&d<64;d++)
        moved[d]=value[d]+a_eff*(target[d]-value[d]);
    for(int d=dd;d<D;d++)moved[d]=0;

    /* project into ball */
    float n2c=0;
    for(int d=0;d<D;d++)n2c+=moved[d]*moved[d];
    if(n2c>0.99998f){
        float s=sqrtf(0.99998f/n2c);
        for(int d=0;d<D;d++)moved[d]*=s;
    }
    for(int d=0;d<D;d++)value[d]=moved[d];
}

/* multi-step convergence: iterate until |v-target| < eps or max steps.
 * Returns steps taken; verifies monotone approach. */
int wubu_htd_converge(float* value,const float* target,int D,float c,
                      float lr,float gamma,float eps,int max_steps){
    float prev_d=htd_dist(value,target,D,c);
    for(int s=1;s<=max_steps;s++){
        wubu_htd_step(value,target,D,c,lr,gamma);
        float d=htd_dist(value,target,D,c);
        if(d<eps)return s;
        if(d>=prev_d)return -s;   /* diverged */
        prev_d=d;
    }
    return 0;   /* no convergence in budget */
}
