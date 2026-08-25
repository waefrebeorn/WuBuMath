/*
 * wubu_scene_cut.c -- GAP-C051: Scene-cut detection + adaptive KEY
 * frame insertion for the quaternion codec
 *
 * When the angular velocity between consecutive frames exceeds a
 * threshold, SLERP prediction breaks down (the rotation isn't smooth).
 * At those points we MUST insert a new KEY frame — otherwise error
 * accumulates and quality collapses. This is the same principle as
 * I-frame insertion in H.264, but triggered by ANGULAR discontinuity
 * rather than pixel-difference SAD.
 *
 * Also: SKIP frames. If angular velocity is below epsilon, the frame
 * is identical to the previous one → 0 bytes.
 */
#define M_PI 3.14159265358979f
#include "wubu_scene_cut.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* classify each inter-frame transition */
void wubu_sc_classify(const float* quat_frames,int n_frames,int D,
                       float cut_thresh,float skip_thresh,
                       uint8_t* out_type /* 0=SKIP 1=INTER 2=KEY */){
    if(n_frames<1)return;
    out_type[0]=2;   /* first frame always KEY */

    for(int i=1;i<n_frames;i++){
        const float* qp=quat_frames+(size_t)(i-1)*D;
        const float* qc=quat_frames+(size_t)i*D;

        /* dot product for unit quats gives cos(theta/2) */
        float dot=0;
        for(int d=0;d<D&&d<4;d++)dot+=qp[d]*qc[d];
        if(dot>1)dot=1;if(dot<-1)dot=-1;
        float angle=2*acosf(fabsf(dot));

        if(angle>=cut_thresh)
            out_type[i]=2;   /* scene cut: insert KEY */
        else if(angle<=skip_thresh)
            out_type[i]=0;   /* static: skip */
        else
            out_type[i]=1;   /* normal INTER */
    }
}

/* count frame types for rate estimation */
void wubu_sc_stats(const uint8_t* types,int n_frames,
                    int* n_key,int* n_inter,int* n_skip){
    *n_key=*n_inter=*n_skip=0;
    for(int i=0;i<n_frames;i++){
        switch(types[i]){
            case 0:(*n_skip)++;break;
            case 1:(*n_inter)++;break;
            case 2:(*n_key)++;break;
        }
    }
}

/* estimated total bytes given per-type costs */
long wubu_sc_estimate_bytes(int n_key,long key_bytes,
                              int n_inter,long inter_bytes,
                              int n_skip){
    return n_key*key_bytes+n_inter*inter_bytes;
}
