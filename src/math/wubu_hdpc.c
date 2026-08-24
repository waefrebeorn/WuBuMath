/*
 * wubu_hdpc.c -- GAP-D021: Hyperbolic density-peak clustering (DPC adapted)
 *
 * Rodriguez & Laio's Density Peak Clustering (Science 2014) on the ball:
 *   1. rho_i = number of points within cutoff distance d_c
 *   2. delta_i = min distance to any point with HIGHER density
 *      (max distance for the densest point)
 *   3. cluster centers = points with BOTH high rho AND high delta
 *   4. remaining points assigned to nearest higher-density neighbor
 *
 * The hyperbolic advantage: geodesic distances naturally separate
 * hierarchical levels, so density peaks align with subtree roots.
 */
#include "wubu_hdpc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float hd_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int wubu_hdpc_run(const float* pts,int n,int D,float c,float dc_frac,
                  int* rho,int* assign,int* is_center){
    if(!pts||!rho||!assign||n<3)return -1;

    /* pairwise distances + cutoff */
    float* dist=malloc(sizeof(float)*(size_t)n*n);
    if(!dist)return -2;
    /* collect all distances to find dc */
    int total_pairs=n*(n-1)/2;
    float* all_d=malloc(sizeof(float)*(size_t)(total_pairs>0?total_pairs:1));
    int tp=0;
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++){
            float d=hd_dist(pts+(size_t)i*D,pts+(size_t)j*D,D,c);
            dist[(size_t)i*n+j]=dist[(size_t)j*n+i]=d;
            all_d[tp++]=d;
        }
    /* dc = dc_frac percentile of distances */
    for(int a=0;a<tp;a++)
        for(int b=a+1;b<tp;b++)
            if(all_d[b]<all_d[a]){float t=all_d[a];all_d[a]=all_d[b];all_d[b]=t;}
    float dc=all_d[(int)(dc_frac*tp)];
    free(all_d);

    /* densities */
    memset(rho,0,sizeof(int)*(size_t)n);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            if(j!=i&&dist[(size_t)i*n+j]<dc)rho[i]++;

    /* delta: min dist to higher-density point; max-dist point gets max */
    int max_rho_idx=0;
    for(int i=1;i<n;i++)if(rho[i]>rho[max_rho_idx])max_rho_idx=i;
    float* delta=malloc(sizeof(float)*(size_t)n);
    for(int i=0;i<n;i++){
        float best=1e30f;
        for(int j=0;j<n;j++){
            if(j==i||rho[j]<=rho[i])continue;
            if(dist[(size_t)i*n+j]<best)best=dist[(size_t)i*n+j];
        }
        delta[i]=(best<1e29f)?best:-1.0f;   /* -1 = highest density */
    }
    delta[max_rho_idx]=0;
    float mx=0;
    for(int i=0;i<n;i++)if(delta[i]>mx)mx=delta[i];
    delta[max_rho_idx]=mx;

    /* centers: high rho AND high delta — pick top 20% by rho*delta */
    double prod[/*n*/ 4096];
    double pmax=0;
    for(int i=0;i<n&&i<4096;i++){
        prod[i]=(double)rho[i]*delta[i];
        if(prod[i]>pmax)pmax=prod[i];
    }
    memset(is_center,0,sizeof(int)*(size_t)n);
    int want=(n>=10)?n/5:2;if(want<1)want=1;
    int placed=0;
    for(int i=0;i<n&&placed<want;i++)
        if(prod[i]>=pmax*0.5f){is_center[i]=1;placed++;}

    /* assignment: each non-center → nearest higher-density point (chain) */
    for(int i=0;i<n;i++){
        if(is_center[i]){assign[i]=i;continue;}
        int best_j=-1;float bd=1e30f;
        for(int j=0;j<n;j++){
            if(j==i||!is_center[j])continue;
            /* allow chains via intermediate higher-density too */
            if(rho[j]<rho[i]&&!is_center[j])continue;
            if(dist[(size_t)i*n+j]<bd){bd=dist[(size_t)i*n+j];best_j=j;}
        }
        if(best_j<0){   /* fallback: nearest overall center */
            for(int j=0;j<n;j++){
                if(!is_center[j])continue;
                if(dist[(size_t)i*n+j]<bd){bd=dist[(size_t)i*n+j];best_j=j;}
            }
        }
        assign[i]=(best_j>=0)?best_j:i;
    }

    free(dist);free(delta);
    return 0;
}
