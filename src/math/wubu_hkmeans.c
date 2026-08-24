/*
 * wubu_hkmeans.c -- GAP-C020: Hyperbolic k-means clustering on the Poincaré ball
 *
 * Research source: GGBall (arXiv:2506.07198) Algorithm 1 + PoincareKMeans.
 * The algorithm:
 *   1. init centroids (data-driven: first K points, or random on-ball)
 *   2. assign each point to nearest centroid by geodesic distance
 *   3. update each centroid as the Einstein midpoint of its assigned points
 *   4. repeat until max centroid displacement < epsilon or max_iter reached
 *
 * Gates:
 *  G1 well-separated clusters are recovered (adjusted rand >= 0.8)
 *  G2 all centroids stay inside the ball
 *  G3 assignments sum to N (every point assigned exactly once)
 */
#include "wubu_hkmeans.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

float wubu_hkmeans_distance(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1.0f-c*a2)*(1.0f-c*b2);
    if(den<1e-9f)den=1e-9f;
    float arg=1.0f+2.0f*c*ab2/den;
    return acoshf(arg>1.0f?arg:1.0f)/sqrtf(c);
}

/* weighted Möbius gyromidpoint of assigned points */
static void hkmeans_centroid(const float* pts,const int* assign,
                              int n,int D,int cluster,float c,float* out){
    float num[256];memset(num,0,sizeof(float)*(size_t)(D<256?D:256));
    float den=1e-10f;
    int count=0;
    for(int i=0;i<n;i++){
        if(assign[i]!=cluster)continue;
        const float* x=pts+(size_t)i*D;
        float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
        float gamma=2.0f/(1.0f-c*n2);
        if(gamma<1.0f)gamma=1.0f;
        for(int d=0;d<D&&d<256;d++)num[d]+=gamma*x[d];
        den+=(gamma-1.0f);
        count++;
    }
    if(count==0)return;   /* empty cluster: leave centroid unchanged */
    float tm[256],tm_n2=0;
    int dc=D<256?D:256;
    for(int d=0;d<dc;d++){tm[d]=num[d]/den;tm_n2+=tm[d]*tm[d];}
    float disc=1.0f-c*tm_n2;if(disc<1e-9f)disc=1e-9f;
    float sc=1.0f/(1.0f+sqrtf(disc));
    for(int d=0;d<D;d++)out[d]=tm[d]*sc;
}

int wubu_hkmeans(const float* pts,int n,int D,int K,float c,
                 int max_iter,float eps,
                 int* assignments,float* centroids){
    if(!pts||!assignments||!centroids||n<K||K<1)return -1;

    /* data-driven init: first K distinct points spread across the dataset */
    int stride=n/K;
    for(int k=0;k<K;k++)
        memcpy(centroids+(size_t)k*D,pts+(size_t)(k*stride)*D,sizeof(float)*D);

    int iter=0;
    for(iter=0;iter<max_iter;iter++){
        /* assignment */
        float max_shift=0;
        float old_centroids[256];
        memcpy(old_centroids,centroids,sizeof(float)*(size_t)K*D);
        for(int i=0;i<n;i++){
            float best=1e30f;int bk=0;
            for(int k=0;k<K;k++){
                float d=wubu_hkmeans_distance(pts+(size_t)i*D,centroids+(size_t)k*D,D,c);
                if(d<best){best=d;bk=k;}
            }
            assignments[i]=bk;
        }
        /* centroid update */
        for(int k=0;k<K;k++)
            hkmeans_centroid(pts,assignments,n,D,k,c,centroids+(size_t)k*D);
        /* convergence check */
        for(int k=0;k<K;k++){
            float shift=wubu_hkmeans_distance(old_centroids+(size_t)k*D,
                                               centroids+(size_t)k*D,D,c);
            if(shift>max_shift)max_shift=shift;
        }
        if(max_shift<eps)break;
    }
    return iter;
}
