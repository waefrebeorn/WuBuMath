/*
 * wubu_geodesic_path.c -- GAP-C025: Geodesic path sampling on the ball
 *
 * The geodesic from x0 to x1 on the Poincaré ball parameterized by t∈[0,1]:
 *   gamma(t) = x0 ⊕ (t ⊗ (⊖x0 ⊕ x1))
 *
 * This is the Möbius-geodesic: mobius_add(x0, scalar_mul(t, mobius_add(-x0, x1))).
 * Used for smooth latent interpolation between video frames (the P-frame
 * path), VAE latent morphing, and beam-canvas temporal sweeps.
 */
#include "wubu_geodesic_path.h"
#include <math.h>
#include <string.h>

static void gp_mobius_add(float* out,const float* u,const float* v,
                          int D,float c){
    float uu=0,vv=0,uv=0;
    for(int d=0;d<D;d++){
        uu+=u[d]*u[d];vv+=v[d]*v[d];uv+=u[d]*v[d];
    }
    float num1=1.0f+2.0f*c*uv+c*vv;
    float num2=1.0f-c*uu;
    float den=1.0f+2.0f*c*uv+c*c*uu*vv;
    if(den<1e-10f)den=1e-10f;
    for(int d=0;d<D;d++)
        out[d]=(num1*u[d]+num2*v[d])/den;
}

void wubu_gp_point(float* out,const float* x0,const float* x1,
                   int D,float c,float t){
    /* step 1: v = (-x0) ⊕ x1  — the tangent direction at x0 pointing to x1 */
    float neg_x0[64],v[64];
    int dd=D<64?D:64;
    for(int d=0;d<dd;d++)neg_x0[d]=-x0[d];
    for(int d=dd;d<D;d++)neg_x0[d]=0;
    gp_mobius_add(v,neg_x0,x1,D,c);

    /* step 2: scale by t (scalar multiplication in gyrovector space):
     * t⊗v = tanh(t*atanh(sqrt(c)|v|))/(sqrt(c)|v|) * v */
    float vn2=0;for(int d=0;d<D;d++)vn2+=v[d]*v[d];
    float nv=sqrtf(vn2);
    float tv[64];
    if(nv>1e-10f){
        float scaled=tanhf(t*atanhf(sqrtf(c)*nv))/(sqrtf(c)*nv);
        for(int d=0;d<D;d++)tv[d]=(d<dd)?scaled*v[d]:0;
    }else{
        memset(tv,0,sizeof(float)*D);
    }

    /* step 3: out = x0 ⊕ tv */
    gp_mobius_add(out,x0,tv,D,c);

    /* safety: project into ball */
    float n2=0;for(int d=0;d<D;d++)n2+=out[d]*out[d];
    if(n2>1.0f/c){float s=sqrtf(1.0f/(c*n2));for(int d=0;d<D;d++)out[d]*=s;}
}

void wubu_gp_path(const float* x0,const float* x1,int D,float c,
                  int num_steps,float* out /* [num_steps+1, D] */){
    for(int s=0;s<=num_steps;s++){
        float t=(float)s/(float)num_steps;
        wubu_gp_point(out+(size_t)s*D,x0,x1,D,c,t);
    }
}
