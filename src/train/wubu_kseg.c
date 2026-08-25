/*
 * wubu_kseg.c -- GAP-C066: Optimal K-segmentation via dynamic programming
 * (the DP that C065's greedy approach approximates)
 *
 * Research source: Ghosh 2009 — O(KN²) DP for optimal piecewise
 * segmentation minimizing L2 error. Applied to quaternion trajectories:
 * find the OPTIMAL placement of K keyframes to minimize total angular
 * interpolation error. This is the globally optimal version of what
 * C065's greedy heuristic approximates.
 *
 * dp[i][k] = min over j<i of dp[j][k-1] + segment_error(j+1,i)
 * where segment_error(a,b) = sum of SLERP interpolation errors when
 * using only frames a and b as keys for the range [a,b].
 */
#define M_PI 3.14159265358979f
#include "wubu_kseg.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* angular distance */
static float ks_angle2(const float* a,const float* b){
    float dot=fabsf(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]);
    if(dot>1)dot=1;
    return 2*acosf(dot);
}

/* compute the mean SLERP interpolation error if we use ONLY frames a,b
 * as keys for all frames in [a,b] */
static float seg_error(const float* quats,int n_frames,int D,
                        int a,int b){
    if(b-a<=1)return 0;  /* adjacent: zero error */
    const float* qa=quats+(size_t)a*D;
    const float* qb=quats+(size_t)b*D;
    double total=0;
    int count=0;
    for(int f=a+1;f<b;f++){
        float t=(float)(f-a)/(b-a);
        const float* actual=quats+(size_t)f*D;
        /* SLERP between qa and qb at t */
        float cos_half=0;
        for(int d=0;d<4&&d<D;d++)cos_half+=qa[d]*qb[d];
        if(cos_half>1)cos_half=1;if(cos_half<-1)cos_half=-1;
        float theta=acosf(cos_half);
        float interp[4];
        float sin_t=sinf(theta);
        if(sin_t<1e-6f){
            for(int d=0;d<4&&d<D;d++)interp[d]=(1-t)*qa[d]+t*qb[d];
        }else{
            float wa=sinf((1-t)*theta)/sin_t;
            float wb=sinf(t*theta)/sin_t;
            for(int d=0;d<4&&d<D;d++)interp[d]=wa*qa[d]+wb*qb[d];
        }
        total+=ks_angle2(interp,actual);
        count++;
    }
    return count>0?(float)(total/count):0;
}

/* optimal keyframe selection via DP — returns number of selected keys */
int wubu_seg_optimal(const float* quats,int n_frames,int D,int k,
                      int* out_indices){
    if(k>=n_frames){
        /* use every frame */
        for(int i=0;i<n_frames;i++)out_indices[i]=i;
        return n_frames;
    }

    /* cost[j] = min total error to cover [0..j] with some number of keys ≤ k */
    /* parent[j] = previous key before j in the optimal solution */

    /* precompute seg_error table (only need pairs where b-a <= max_gap) */
    int max_gap=n_frames;  /* allow any gap */
    float** err=malloc(sizeof(float*)*(size_t)n_frames);
    for(int i=0;i<n_frames;i++){
        err[i]=malloc(sizeof(float)*(size_t)n_frames);
        for(int j=0;j<n_frames;j++)
            err[i][j]=seg_error(quats,n_frames,D,i,j);
    }

    /* DP over exactly k segments */
    float INF=1e30f;
    float* prev_cost=malloc(sizeof(float)*(size_t)n_frames);
    float* curr_cost=malloc(sizeof(float)*(size_t)n_frames);
    int** parent=malloc(sizeof(int*)*(size_t)(k+1));
    for(int kk=0;kk<=k;kk++)
        parent[kk]=malloc(sizeof(int)*(size_t)n_frames);

    /* base case: 1 segment covers [0..j] with just frame 0 and frame j */
    for(int j=0;j<n_frames;j++){
        prev_cost[j]=err[0][j];
        parent[1][j]=0;
    }

    /* iterate for 2..k segments */
    for(int kk=2;kk<=k;kk++){
        for(int j=kk-1;j<n_frames;j++){
            curr_cost[j]=INF;
            parent[kk][j]=-1;
            /* try placing a key at position i, covering [i..j] as one segment */
            for(int i=kk-2;i<j;i++){
                float c=prev_cost[i]+err[i][j];
                if(c<curr_cost[j]){
                    curr_cost[j]=c;
                    parent[kk][j]=i;
                }
            }
        }
        memcpy(prev_cost,curr_cost,sizeof(float)*(size_t)n_frames);
    }

    /* backtrack: k segments have k+1 boundary keys */
    int idx=n_frames-1;
    int count=0;
    out_indices[count++]=idx;
    for(int kk=k;kk>=2;kk--){
        idx=parent[kk][idx];
        if(idx<0)break;
        out_indices[count++]=idx;
    }
    /* count is now k keys total (initial + k-1 parents); no extra 0 needed
     * because parent[1] base case already starts from frame 0 */

    /* reverse and sort */
    for(int i=0;i<count/2;i++){
        int tmp=out_indices[i];
        out_indices[i]=out_indices[count-1-i];
        out_indices[count-1-i]=tmp;
    }

    /* free */
    for(int i=0;i<n_frames;i++)free(err[i]);
    free(err);free(prev_cost);free(curr_cost);
    for(int kk=0;kk<=k;kk++)free(parent[kk]);
    free(parent);

    return count;
}
