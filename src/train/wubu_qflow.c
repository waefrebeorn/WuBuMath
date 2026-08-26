/*
 * wubu_qflow.c -- GAP-C001: Quaternion Hamilton-product-native flow field
 *
 * Research source: Chen & Lipman 2023 Riemannian Flow Matching.
 * Instead of treating latent vectors as flat and using Euclidean velocity,
 * this module defines a velocity field ON THE QUATERNION SPHERE:
 *   v(q,t) operates via Hamilton product and SLERP interpolation
 *   Integration steps use exp_q() / log_q() maps on S³
 *
 * The result: flows naturally follow geodesic paths on the rotation
 * manifold instead of cutting through the ambient space.
 */
#define M_PI 3.14159265358979f
#include "wubu_qflow.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Hamilton product for quaternions */
static void qf_hamilton(const float* a,const float* b,float* out){
    out[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
    out[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
    out[2]=a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
    out[3]=a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];
}

/* quaternion conjugate */
static void qf_conj(const float* a,float* out){
    out[0]=a[0];out[1]=-a[1];out[2]=-a[2];out[3]=-a[3];
}

/* log map: tangent vector at identity pointing toward q */
static void qf_log(const float* q,float* out){
    /* for unit quaternion: angle = 2*acos(|w|), axis = xyz/sin(angle/2) */
    float norm=sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if(norm<1e-10f){out[0]=out[1]=out[2]=out[3]=0;return;}

    float w=q[0]/norm;
    if(w>1.0f)w=1.0f;if(w<-1.0f)w=-1.0f;
    float theta=acosf(w);
    float sin_half=sinf(theta);

    if(fabsf(sin_half)<1e-8f){
        /* near identity: tangent is approximately xyz part */
        out[0]=0;out[1]=q[1]/norm;out[2]=q[2]/norm;out[3]=q[3]/norm;
    }else{
        out[0]=0;
        out[1]=2*theta*q[1]/(norm*sin_half);
        out[2]=2*theta*q[2]/(norm*sin_half);
        out[3]=2*theta*q[3]/(norm*sin_half);
    }
}

/* exp map: from identity, follow tangent vector to reach point on sphere */
static void qf_exp(const float* v,float* out){
    float theta=sqrtf(v[1]*v[1]+v[2]*v[2]+v[3]*v[3]);
    if(theta<1e-8f){
        /* near identity */
        out[0]=1.0f;out[1]=v[1];out[2]=v[2];out[3]=v[3];
    }else{
        float half=theta/2;
        float sin_half=sinf(half);
        out[0]=cosf(half);
        out[1]=sin_half*v[1]/theta;
        out[2]=sin_half*v[2]/theta;
        out[3]=sin_half*v[3]/theta;
    }
}

/* normalize quaternion to unit length */
static void qf_normalize(float* q){
    float n=sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if(n>1e-10f){q[0]/=n;q[1]/=n;q[2]/=n;q[3]/=n;}
}

/* compute velocity field: v = log(q_target ⊗ conj(q_current)) / dt
 * This gives the tangent vector at q_current pointing toward q_target */
void wubu_qf_velocity(const float* q_current,const float* q_target,
                       float dt,float* velocity){
    float conj_curr[4];
    qf_conj(q_current,conj_curr);
    /* delta = q_target ⊗ conj(q_current) — the relative rotation */
    float delta[4];
    qf_hamilton(q_target,conj_curr,delta);
    /* velocity in tangent space of q_current */
    qf_log(delta,velocity);
    /* velocity is the tangent vector; caller applies exp(v*dt)
     * NOTE: do NOT divide by dt here — the log already gives the correct
     * tangent vector for reaching the target in unit time */
}

/* integrate one step along the flow field using exponential map */
int wubu_qf_step(const float* q_current,const float* velocity,
                  float dt,float* q_next){
    /* tangent vector scaled by dt */
    float tangent[4]={0,velocity[1]*dt,velocity[2]*dt,velocity[3]*dt};
    /* exponential map takes us from current position along the geodesic */
    float delta[4];
    qf_exp(tangent,delta);
    /* compose: q_next = delta ⊗ q_current (left multiply for body frame) */
    qf_hamilton(q_current,delta,q_next);
    qf_normalize(q_next);
    return 0;
}

/* full trajectory: integrate from q_start to q_end over n_steps */
int wubu_qf_trajectory(const float* q_start,const float* q_end,
                        int n_steps,float* path){
    float cur[4];
    memcpy(cur,q_start,sizeof(float)*4);
    qf_normalize(cur);

    /* compute velocity ONCE for constant angular velocity */
    float vel[4];
    wubu_qf_velocity(cur,q_end,1.0f/n_steps,vel);
    
    float dt=1.0f/n_steps;
    memcpy(path,cur,sizeof(float)*4);  /* first point */

    for(int s=1;s<=n_steps;s++){
        wubu_qf_step(cur,vel,dt,cur);
        memcpy(path+(size_t)s*4,cur,sizeof(float)*4);
    }
    return 0;
}
