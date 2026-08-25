/*
 * wubu_keyselect.c -- GAP-C065: Rate-distortion-optimized keyframe
 * selection for quaternion trajectories
 *
 * The spline (C064) reconstructs smooth motion from sparse keyframes.
 * This module decides WHICH frames to keep as keys to minimize total
 * distortion at a given bit budget — the RD optimization for the
 * trajectory layer. Greedy farthest-point selection on angular distance.
 */
#define M_PI 3.14159265358979f
#include "wubu_keyselect.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* angular distance between two quaternions */
static float ks_angle(const float* a,const float* b){
    float dot=fabsf(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]);
    if(dot>1)dot=1;
    return 2*acosf(dot);
}

/* greedy farthest-point sampling: select k frames maximizing minimum
 * angular distance from already-selected keys */
int wubu_kf_select(const float* quats,int n_frames,int D,int n_keys,
                    int* out_indices){
    if(n_keys<1||n_frames<1)return 0;
    int dd=D<4?D:4;

    /* first key = frame 0 */
    out_indices[0]=0;
    int selected=1;

    /* min-distance array: for each frame, its distance to nearest key */
    float* min_dist=malloc(sizeof(float)*(size_t)n_frames);
    for(int i=0;i<n_frames;i++)
        min_dist[i]=ks_angle(quats+(size_t)i*D,quats);

    while(selected<n_keys&&selected<n_frames){
        /* pick frame with max min-dist */
        int best=-1;
        float best_d=-1;
        for(int i=0;i<n_frames;i++){
            if(min_dist[i]>best_d){best_d=min_dist[i];best=i;}
        }
        if(best<0||best_d<=0)break;
        out_indices[selected]=best;
        selected++;

        /* update min-dists */
        for(int i=0;i<n_frames;i++){
            float d=ks_angle(quats+(size_t)i*D,quats+(size_t)best*D);
            if(d<min_dist[i])min_dist[i]=d;
        }
    }
    free(min_dist);

    /* sort indices */
    for(int i=1;i<selected;i++){
        int key=out_indices[i];
        int j=i-1;
        while(j>=0&&out_indices[j]>key){out_indices[j+1]=out_indices[j];j--;}
        out_indices[j+1]=key;
    }
    return selected;
}

/* estimate reconstruction error at a given key set using SLERP interpolation
 * between consecutive keys */
float wubu_kf_estimate_error(const float* quats,int n_frames,int D,
                              const int* key_indices,int n_keys){
    float total_err=0;
    int count=0;
    int dd=D<4?D:4;
    for(int ki=0;ki<n_keys-1;ki++){
        int a=key_indices[ki],b=key_indices[ki+1];
        for(int f=a+1;f<b;f++){
            /* SLERP between keys a and b at parameter t */
            float t=(float)(f-a)/(b-a);
            const float* qa=quats+(size_t)a*D;
            const float* qb=quats+(size_t)b*D;
            /* simple lerp approximation of slerp */
            float interp[4];
            float cos_half=0;
            for(int d=0;d<dd;d++)cos_half+=qa[d]*qb[d];
            float theta=acosf(cos_half>1?1:(cos_half<-1?-1:cos_half));
            float sin_t=sinf(theta);
            if(sin_t<1e-6f){
                for(int d=0;d<dd;d++)interp[d]=(1-t)*qa[d]+t*qb[d];
            }else{
                float wa=sinf((1-t)*theta)/sin_t;
                float wb=sinf(t*theta)/sin_t;
                for(int d=0;d<dd;d++)interp[d]=wa*qa[d]+wb*qb[d];
            }
            float err=ks_angle(interp,quats+(size_t)f*D);
            total_err+=err;
            count++;
        }
    }
    return count>0?total_err/count:0;
}
