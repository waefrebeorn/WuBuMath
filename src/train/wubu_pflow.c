/*
 * wubu_pflow.c -- GAP-C030: Poincaré flow matching (GGBall formulation)
 *
 * Research source: GGBall arXiv:2506.07198 §3.3 + HypDiff (ICML 2024).
 * The deterministic geodesic path with time-warp κ(t):
 *   z_t = exp_{z1}( κ(t) · log_{z1}(z0) )
 * with κ(t)=1-t: t=0 → z0 (prior/noise), t=1 → z1 (data).
 *
 * This differs from C025's x0⊕(t⊗...) form by anchoring the log/exp at
 * z1 (the DATA endpoint), which is the correct direction for generation:
 * we integrate from noise TOWARD data, so the map must be exact at the
 * data end of the path.
 */
#include "wubu_pflow.h"
#include <math.h>
#include <string.h>

/* pointwise ops on ball (unit curvature helpers) */
static void pf_mobius_add(float* out,const float* u,const float* v,
                          int D,float c){
    float uu=0,vv=0,uv=0;
    for(int d=0;d<D;d++){uu+=u[d]*u[d];vv+=v[d]*v[d];uv+=u[d]*v[d];}
    float num1=1+2*c*uv+c*vv,num2=1-c*uu;
    float den=1+2*c*uv+c*c*uu*vv;
    if(den<1e-10f)den=1e-10f;
    for(int d=0;d<D;d++)out[d]=(num1*u[d]+num2*v[d])/den;
}
__attribute__((unused)) static void pf_exp0(const float* v,int D,float c,float* out){
    float n2=0;for(int d=0;d<D;d++)n2+=v[d]*v[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){for(int d=0;d<D;d++)out[d]=0;return;}
    float coeff=tanhf(sqrtf(c)*nv)/(sqrtf(c)*nv);
    for(int d=0;d<D;d++)out[d]=coeff*v[d];
    /* fp32 boundary cap */
    float n2c=0;for(int d=0;d<D;d++)n2c+=out[d]*out[d];
    if(n2c>0.99998f){float s=sqrtf(0.99998f/n2c);for(int d=0;d<D;d++)out[d]*=s;}
}
__attribute__((unused)) static void pf_log0(const float* x,int D,float c,float* out){
    float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){for(int d=0;d<D;d++)out[d]=0;return;}
    float arg=sqrtf(c)*nv;if(arg>0.99999f)arg=0.99999f;
    float zn=(2.0f/sqrtf(c))*atanhf(arg)/nv;
    for(int d=0;d<D;d++)out[d]=zn*x[d];
}

/* velocity field along the path: dz/dt at parameter t
 * For z_t = exp_{z1}(κ(t)·log_{z1}(z0)), the velocity in the tangent
 * space at z1 is dκ/dt · log_{z1}(z0) = -log_{z1}(z0).
 * We return the tangent vector AT z_t pointing toward z_1 (data). */
void wubu_pf_velocity(const float* z_data,const float* z_noise,
                      int D,float c,float* vel){
    /* v = (-z_data) ⊕ z_noise  — tangent at z_data toward noise,
     * then negate for the flow direction (noise → data) */
    float neg[64]={0},v[64];
    int dd=D<64?D:64;
    for(int d=0;d<dd;d++)neg[d]=-z_data[d];
    for(int d=dd;d<D;d++)neg[d]=0;
    pf_mobius_add(v,neg,z_noise,D,c);
    for(int d=0;d<D;d++){
        if(d<dd)vel[d]=-v[d];
        else vel[d]=0;
    }
}

/* single Euler step from z_t toward data with step size dt */
void wubu_pf_step(float* z,const float* z_data,int D,float c,float dt){
    float vel[64],shift[64];
    int dd=D<64?D:64;
    wubu_pf_velocity(z_data,z,D,c,vel);
    /* move by dt via mobius scalar + add: shift = dt ⊗ vel; z = z ⊕ shift */
    float vn2=0;for(int d=0;d<dd;d++)vn2+=vel[d]*vel[d];
    float nv=sqrtf(vn2);
    if(nv>1e-10f){
        float coeff=tanhf(dt*atanhf(sqrtf(c)*nv))/(sqrtf(c)*nv);
        for(int d=0;d<dd;d++)shift[d]=coeff*vel[d];
        for(int d=dd;d<D;d++)shift[d]=0;
        pf_mobius_add(z,z,shift,D,c);
    }
}

/* full trajectory: z_out[s] for s=0..steps (s=0 is noise, s=steps is near data) */
void wubu_pf_trajectory(const float* z_noise,const float* z_data,
                        int D,float c,int steps,float* z_out){
    float cur[64];
    int dd=D<64?D:64;
    memcpy(cur,z_noise,sizeof(float)*dd);
    for(int d=dd;d<D;d++)cur[d]=0;
    memcpy(z_out,cur,sizeof(float)*D);   /* slot 0 = noise */
    float dt=1.0f/(float)steps;
    for(int s=0;s<steps;s++){
        wubu_pf_step(cur,z_data,D,c,dt);   /* in-place advance */
        memcpy(z_out+(size_t)(s+1)*D,cur,sizeof(float)*D);
    }
}
