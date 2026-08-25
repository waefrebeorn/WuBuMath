/*
 * wubu_qc_improve.c -- GAP-C057: Constant-velocity SLERP prediction
 * for the quaternion codec (quality improvement pass)
 *
 * The current codec's INTER frames use the PREVIOUS frame as prediction.
 * The improvement: use constant-velocity SLERP EXTRAPOLATION — if frames
 * N-2 and N-1 are separated by rotation R, then predict frame N at
 * prev ⊕ R (same angular velocity continues). The residual between
 * prediction and actual is MUCH smaller than prev-to-actual, so fewer
 * bits are needed for the same quality.
 *
 * This is the quaternion equivalent of H.264's motion-vector prediction:
 * instead of coding absolute motion, code the CHANGE in motion.
 */
#define M_PI 3.14159265358979f
#include "wubu_qc_improve.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* SLERP extrapolation: given q_prev and q_curr, predict q_next by
 * continuing the same angular velocity */
void wubu_qi_predict(const float* q_prev,const float* q_curr,
                      float* q_predicted){
    /* compute delta rotation: dq = q_curr * inverse(q_prev) */
    float inv_prev[4];
    /* quaternion conjugate = inverse for unit quaternions */
    inv_prev[0]=q_prev[0];
    inv_prev[1]=-q_prev[1];
    inv_prev[2]=-q_prev[2];
    inv_prev[3]=-q_prev[3];

    /* dq = q_curr ⊗ conj(q_prev) — Hamilton product */
    float dq[4];
    dq[0]=q_curr[0]*inv_prev[0]-q_curr[1]*inv_prev[1]-q_curr[2]*inv_prev[2]-q_curr[3]*inv_prev[3];
    dq[1]=q_curr[0]*inv_prev[1]+q_curr[1]*inv_prev[0]+q_curr[2]*inv_prev[3]-q_curr[3]*inv_prev[2];
    dq[2]=q_curr[0]*inv_prev[2]-q_curr[1]*inv_prev[3]+q_curr[2]*inv_prev[0]+q_curr[3]*inv_prev[1];
    dq[3]=q_curr[0]*inv_prev[3]+q_curr[1]*inv_prev[2]-q_curr[2]*inv_prev[1]+q_curr[3]*inv_prev[0];

    /* predicted = dq ⊗ q_curr (apply same delta again) */
    q_predicted[0]=dq[0]*q_curr[0]-dq[1]*q_curr[1]-dq[2]*q_curr[2]-dq[3]*q_curr[3];
    q_predicted[1]=dq[0]*q_curr[1]+dq[1]*q_curr[0]+dq[2]*q_curr[3]-dq[3]*q_curr[2];
    q_predicted[2]=dq[0]*q_curr[2]-dq[1]*q_curr[3]+dq[2]*q_curr[0]+dq[3]*q_curr[1];
    q_predicted[3]=dq[0]*q_curr[3]+dq[1]*q_curr[2]-dq[2]*q_curr[1]+dq[3]*q_curr[0];

    /* normalize */
    float n=sqrtf(q_predicted[0]*q_predicted[0]+
                  q_predicted[1]*q_predicted[1]+
                  q_predicted[2]*q_predicted[2]+
                  q_predicted[3]*q_predicted[3]);
    if(n>1e-10f)
        for(int i=0;i<4;i++)q_predicted[i]/=n;
}

/* angular distance between two quats */
float wubu_qi_angle(const float* a,const float* b){
    float dot=fabsf(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]);
    if(dot>1.0f)dot=1.0f;
    return 2.0f*acosf(dot);
}

/* measure prediction quality: angle between predicted and actual */
float wubu_qi_prediction_error(const float* q_predicted,const float* q_actual){
    return wubu_qi_angle(q_predicted,q_actual);
}
