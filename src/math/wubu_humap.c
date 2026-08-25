/*
 * wubu_humap.c -- GAP-D032: Hyperbolic UMAP-style layout
 * (fuzzy-graph cross entropy on the ball)
 *
 * Research source: McInnes et al. 2018 (UMAP). The pipeline:
 *   1. k-NN graph on input points (A007 provides this on-ball)
 *   2. fuzzy edge weights: w_h(i,j) = exp(-d_ij) for k-NN edges, 0 else
 *   3. low-dim weights: w_l(i,j) = 1/(1+d_c(y_i,y_j)) on the ball
 *   4. minimize fuzzy cross entropy:
 *      CE = Σ w_h·log(w_h/w_l) + (1-w_h)·log((1-w_h)/(1-w_l))
 *
 * SGD with negative sampling: attract positive edges, repel sampled
 * non-edges — UMAP's force interpretation, on the Poincaré ball.
 */
#include "wubu_humap.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float um_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

/* build fuzzy high-dim weights from a CSR k-NN graph */
int wubu_um_fuzzy_weights(const float* xs,const int* adj_idx,
                           const int* adj_ptr,int n,int D,float c,
                           float* W){
    for(int i=0;i<n;i++){
        int s=adj_ptr[i],e=adj_ptr[i+1];
        for(int p=s;p<e;p++){
            int j=adj_idx[p];
            float d=um_dist(xs+(size_t)i*D,xs+(size_t)j*D,D,c);
            W[(size_t)i*n+j]=expf(-d);
        }
    }
    /* symmetrize with max */
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++){
            float m=W[(size_t)i*n+j]>W[(size_t)j*n+i]?W[(size_t)i*n+j]:W[(size_t)j*n+i];
            W[(size_t)i*n+j]=W[(size_t)j*n+i]=m;
        }
    return 0;
}

float wubu_um_cross_entropy(const float* Wh,const float* ys,
                             int n,int D,float c){
    double ce=0;
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++){
            float wh=Wh[(size_t)i*n+j];
            if(wh<=0||wh>=1)continue;
            float wl=1.0f/(1.0f+um_dist(ys+(size_t)i*D,ys+(size_t)j*D,D,c));
            ce+=(double)wh*log(wh/(double)wl)
               +(double)(1-wh)*log((1-wh)/(double)(1-wl));
        }
    return (float)ce;
}
