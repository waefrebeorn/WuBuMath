/*
 * wubu_qspline.c -- GAP-C064: Quaternion Catmull-Rom spline
 * (smooth camera trajectory through keyframe quaternions)
 *
 * Research source: arXiv:2602.16758 — B-spline + quaternion interpolation
 * for smooth trajectory generation. The codec stores KEY quaternion
 * keyframes and reconstructs intermediate rotations via spline, giving
 * smoother motion than linear SLERP between adjacent frames.
 *
 * Fewer keyframes needed = better compression at same quality.
 */
#define M_PI 3.14159265358979f
#include "wubu_qspline.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* quaternion dot product */
static float qs_dot(const float* a,const float* b){
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];
}

/* ensure shortest path (negate if dot<0) */
static void qs_align(const float* prev,float* q){
    if(qs_dot(prev,q)<0)
        for(int i=0;i<4;i++)q[i]=-q[i];
}

/* SLERP between two unit quaternions */
void wubu_qsp_slerp(const float* qa,const float* qb,float t,float* out){
    /* align for shortest path */
    float qb2[4];
    memcpy(qb2,qb,sizeof(float)*4);
    qs_align(qa,qb2);

    float cos_half=qs_dot(qa,qb2);
    if(cos_half>1.0f-1e-6f){
        /* nearly identical: lerp */
        for(int i=0;i<4;i++)out[i]=(1-t)*qa[i]+t*qb2[i];
    }else{
        float theta=acosf(cos_half);
        float sin_theta=sinf(theta);
        float wa=sinf((1-t)*theta)/sin_theta;
        float wb=sinf(t*theta)/sin_theta;
        for(int i=0;i<4;i++)out[i]=wa*qa[i]+wb*qb2[i];
    }
    /* normalize */
    float n=sqrtf(out[0]*out[0]+out[1]*out[1]+out[2]*out[2]+out[3]*out[3]);
    if(n>1e-10f)for(int i=0;i<4;i++)out[i]/=n;
}

/* Catmull-Rom spline through 4 control quaternions at parameter t∈[0,1] */
void wubu_qsp_catmull(const float* q0,const float* q1,
                       const float* q2,const float* q3,
                       float t,float* out){
    /* use the de Casteljau-like approach: triple SLERP with adjusted t */
    /* first compute intermediate points (approximating tangent control) */
    float t01,t12,t23;
    t01=t*(1+t*(0.5f*t));       /* quadratic ease-in toward midpoint */
    t12=2*t-1;                   /* centered */
    t23=t*t*(3-2*t);             /* smoothstep */

    float tmp1[4],tmp2[4],tmp3[4];
    wubu_qsp_slerp(q0,q1,t01,tmp1);
    wubu_qsp_slerp(q1,q2,fabsf(t12),tmp2);
    wubu_qsp_slerp(q2,q3,t23,tmp3);

    /* blend the three levels */
    float ab[4],bc[4];
    wubu_qsp_slerp(tmp1,tmp2,t,ab);
    wubu_qsp_slerp(tmp2,tmp3,t,bc);
    wubu_qsp_slerp(ab,bc,t,out);
}

/* fit a smooth trajectory through n keyframes and sample at arbitrary t */
void wubu_qsp_trajectory(const float* keys,int n_keys,
                          float t_global,float* out){
    if(n_keys<2){
        memcpy(out,keys,sizeof(float)*4);
        return;
    }
    /* find which segment we're in */
    float scaled=t_global*(n_keys-1);
    int seg=(int)scaled;
    if(seg>=n_keys-1)seg=n_keys-2;
    float local_t=scaled-seg;

    /* get 4 control points (clamped at boundaries) */
    const float* p0=keys+(size_t)(seg>0?seg-1:0)*4;
    const float* p1=keys+(size_t)seg*4;
    const float* p2=keys+(size_t)(seg+1<n_keys?seg+1:n_keys-1)*4;
    const float* p3=keys+(size_t)(seg+2<n_keys?seg+2:n_keys-1)*4;

    wubu_qsp_catmull(p0,p1,p2,p3,local_t,out);
}
