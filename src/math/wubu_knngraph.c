/*
 * wubu_knngraph.c -- GAP-A007: k-NN graph construction on the ball
 *
 * Builds the adjacency structure that HGAT/GCN layers consume, directly
 * from raw on-ball embeddings (no manual graph needed): connect each
 * point to its k geodesically-nearest neighbors. Optional mutual filter
 * (keep only edges where BOTH endpoints chose each other) for cleaner
 * topology. Output in CSR format matching B015/C041's interface.
 */
#include "wubu_knngraph.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static float kg_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int wubu_knng_build(const float* pts,int n,int D,int k,float c,
                    int mutual,
                    int** out_idx,int** out_ptr,int* out_nnz){
    if(!pts||k<1||k>=n||!out_idx||!out_ptr||!out_nnz)return -1;

    /* full pairwise distance matrix */
    float* dm=malloc(sizeof(float)*(size_t)n*n);
    uint8_t* adj=malloc((size_t)n*n);
    if(!dm||!adj){free(dm);free(adj);return -2;}
    memset(adj,0,(size_t)n*n);

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            dm[(size_t)i*n+j]=(i==j)?1e30f:kg_dist(pts+(size_t)i*D,
                                                    pts+(size_t)j*D,D,c);

    /* k nearest per point */
    for(int i=0;i<n;i++){
        /* selection of k smallest */
        for(int a=0;a<k;a++){
            int mi=a;
            for(int b=a+1;b<n;b++)
                if(dm[(size_t)i*n+b]<dm[(size_t)i*n+mi])mi=b;
            float td=dm[(size_t)i*n+a];dm[(size_t)i*n+a]=dm[(size_t)i*n+mi];
            dm[(size_t)i*n+mi]=td;
            /* mark neighbor: store via a second pass using distances */
        }
    }
    /* re-select: mark adjacency by comparing against kth-smallest value */
    for(int i=0;i<n;i++){
        float vals[256];
        int cnt=n<256?n:256;
        for(int j=0;j<cnt;j++)vals[j]=dm[(size_t)i*n+j];
        float kth=vals[k-1];
        for(int j=0;j<n;j++)
            if(kg_dist(pts+(size_t)i*D,pts+(size_t)j*D,D,c)<=kth && j!=i)
                adj[(size_t)i*n+j]=1;
    }

    if(mutual){
        for(int i=0;i<n;i++)
            for(int j=i+1;j<n;j++)
                if(adj[(size_t)i*n+j]||adj[(size_t)j*n+i]){
                    uint8_t keep=adj[(size_t)i*n+j]&&adj[(size_t)j*n+i];
                    adj[(size_t)i*n+j]=adj[(size_t)j*n+i]=keep;
                }
    }

    /* CSR build */
    *out_nnz=0;
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            if(adj[(size_t)i*n+j])(*out_nnz)++;
    *out_idx=malloc(sizeof(int)*(size_t)(*out_nnz>0?*out_nnz:1));
    *out_ptr=malloc(sizeof(int)*(size_t)(n+1));
    if(!*out_idx||!*out_ptr){free(dm);free(adj);return -3;}
    int p=0;
    for(int i=0;i<n;i++){
        (*out_ptr)[i]=p;
        for(int j=0;j<n;j++)
            if(adj[(size_t)i*n+j])(*out_idx)[p++]=j;
    }
    (*out_ptr)[n]=p;
    free(dm);free(adj);
    return 0;
}
