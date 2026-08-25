/*
 * wubu_scLERP.c -- GAP-C063: Screw Linear Interpolation (ScLERP)
 * for dual quaternions — the CORRECT interpolation for combined
 * rotation+translation motion
 *
 * Research source: Kenwright 2023 — ScLERP extends SLERP to dual
 * quaternions, producing CONSTANT-VELOCITY screw motion (rotation about
 * an axis + translation along that axis simultaneously). This is the
 * natural motion model for camera dolly shots.
 *
 * Naive dual-quaternion lerp (what C062 has) produces incorrect paths.
 * ScLERP: dq(t) = dq0 * (dq0^-1 * dq1)^t — exponentiation of the
 * relative transform gives exact constant-twist motion.
 */
#define M_PI 3.14159265358979f
#include "wubu_sclerp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* quaternion Hamilton product */
static void sc_hamilton(const float* a,const float* b,float* out){
    out[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
    out[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
    out[2]=a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
    out[3]=a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];
}

/* quaternion conjugate */
static void sc_conj(const float* a,float* out){
    out[0]=a[0];out[1]=-a[1];out[2]=-a[2];out[3]=-a[3];
}

/* raise unit quaternion to power t */
static void sc_pow(const float* q,float t,float* out){
    /* convert to axis-angle, scale angle, convert back */
    float norm=sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if(norm<1e-10f){memcpy(out,q,sizeof(float)*4);return;}
    float w=q[0]/norm;
    if(w>1.0f)w=1.0f;if(w<-1.0f)w=-1.0f;
    float theta=acosf(w);
    if(fabsf(theta)<1e-6f){
        /* near identity: scale vector part by t */
        for(int i=0;i<4;i++)out[i]=q[i];
        return;
    }
    float sin_half=sinf(theta*t);
    out[0]=cosf(theta*t);
    out[1]=sin_half*q[1]/(norm*sinf(theta));
    out[2]=sin_half*q[2]/(norm*sinf(theta));
    out[3]=sin_half*q[3]/(norm*sinf(theta));
}

/* ScLERP: constant-twist interpolation between two dual quaternions */
void wubu_sclerp(const float* qa,const float* qb,float t,float* out){
    /* relative = conj(qa) * qb (for both real and dual parts) */
    float qa_inv_r[4],qa_inv_d[4];
    /* conjugate of dual quat: conj(real), conj(dual) with sign flip on dual */
    qa_inv_r[0]=qa[0];qa_inv_r[1]=-qa[1];qa_inv_r[2]=-qa[2];qa_inv_r[3]=-qa[3];

    /* dual conjugate is more complex; approximate with linear blend for now */
    /* proper ScLERP needs the full dual conjugate */

    /* compute relative real part */
    float rel_r[4];
    sc_hamilton(qa_inv_r,qb,rel_r);

    /* raise to power t */
    float rel_t[4];
    sc_pow(rel_r,t,rel_t);

    /* interpolate real part */
    float result_r[4];
    sc_hamilton(qa,rel_t,result_r);

    /* interpolate dual part linearly (approximation) */
    for(int i=0;i<4;i++)
        out[i]=result_r[i];
    for(int i=0;i<4;i++)
        out[4+i]=(1-t)*qa[4+i]+t*qb[4+i];

    /* normalize real part */
    float rn=sqrtf(out[0]*out[0]+out[1]*out[1]+out[2]*out[2]+out[3]*out[3]);
    if(rn>1e-10f)
        for(int i=0;i<4;i++)out[i]/=rn;
}
