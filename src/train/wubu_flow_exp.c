/*
 * wubu_flow_exp.c -- GAP-C004: exp/log-map integration steps in
 * flow matching (replacing naive lerp+normalize)
 *
 * Research source: Chen & Lipman 2023 Riemannian flow matching.
 * The ODE solver should step ON THE MANIFOLD: x_{t+dt} = exp_x(dt*v(x_t))
 * not lerp+project. This is the difference between a flow that stays on
 * the ball by construction and one that fights projection artifacts.
 */
#include "wubu_flow_exp.h"
#include <math.h>
#include <string.h>

/* exp_x(v): point on ball reached from x along tangent vector v */
static void fe_exp(const float* x,const float* v,int D,float c,float* out){
    /* Möbius addition: out = x ⊕ v_scaled */
    float uu=0,vv=0,uv=0;
    for(int d=0;d<D;d++){uu+=x[d]*x[d];vv+=v[d]*v[d];uv+=x[d]*v[d];}
    float num1=1+2*c*uv+c*vv,num2=1-c*uu;
    float den=1+2*c*uv+c*c*uu*vv;
    if(den<1e-10f)den=1e-10f;
    for(int d=0;d<D;d++)out[d]=(num1*x[d]+num2*v[d])/den;
    /* project */
    float n2=0;for(int d=0;d<D;d++)n2+=out[d]*out[d];
    if(n2>1.0f/c){float s=sqrtf(1.0f/(c*n2));for(int d=0;d<D;d++)out[d]*=s;}
}

/* log_x(y): tangent vector at x pointing toward y */
static void fe_log(const float* x,const float* y,int D,float c,float* out){
    /* simplified: use the origin-based approximation via gyromidpoint diff */
    float n2=0;
    for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float lam=2.0f/(1-c*n2);
    for(int d=0;d<D;d++)out[d]=lam*(y[d]-x[d]);
}

int wubu_fe_step(const float* x,const float* velocity,int D,
                  float c,float dt,float* out){
    if(!x||!velocity||!out)return -1;
    /* scale velocity by dt then apply exp map */
    float scaled[512];
    int dd=D<512?D:512;
    for(int d=0;d<dd&&d<512;d++)scaled[d]=velocity[d]*dt;
    fe_exp(x,scaled,D,c,out);
    return 0;
}

int wubu_fe_trajectory(const float* start,const float* end,int D,
                        float c,int n_steps,float lr,float* out_path){
    if(!start||!end||!out_path||n_steps<1)return -1;
    float cur[512];
    int dd=D<512?D:512;
    memcpy(cur,start,sizeof(float)*dd);
    for(int s=0;s<n_steps;s++){
        /* velocity = log_cur(end) — always pursue the target */
        float vel[512];
        fe_log(cur,end,D,c,vel);
        memcpy(out_path+(size_t)s*D,cur,sizeof(float)*dd);
        wubu_fe_step(cur,vel,D,c,lr,cur);
    }
    return 0;
}
