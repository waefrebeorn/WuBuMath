/*
 * wubu_hkrum.c -- GAP-D018: Hyperbolic Krum outlier detection
 *
 * Krum (Blanchard et al. 2017) adapted to the Poincaré ball:
 * for each point i, score_i = sum of distances to its f nearest neighbors.
 * The point with the smallest score is the "most central" (least anomalous);
 * points with scores above threshold are outliers.
 *
 * Hyperbolic twist: distances are geodesic, so outliers that are merely
 * "far out in the hierarchy" (deep in the ball) are handled naturally —
 * the exponential volume growth means true anomalies stand out more than
 * in Euclidean space.
 */
#include "wubu_hkrum.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float hk_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int wubu_hkrum_scores(const float* pts,int n,int D,float c,int f,
                      float* scores){
    if(!pts||!scores||f<1||f>=n)return -1;
    /* for each point: sum distances to f NEAREST others */
    for(int i=0;i<n;i++){
        float dists[256];
        int dd=n-1<256?n-1:256;
        int cnt=0;
        for(int j=0;j<n&&cnt<dd;j++){
            if(j==i)continue;
            dists[cnt++]=hk_dist(pts+(size_t)i*D,pts+(size_t)j*D,D,c);
        }
        /* selection of f smallest */
        float sum=0;
        for(int k=0;k<f;k++){
            int mi=k;
            for(int m=k+1;m<cnt;m++)
                if(dists[m]<dists[mi])mi=m;
            float td=dists[k];dists[k]=dists[mi];dists[mi]=td;
            sum+=dists[k];
        }
        scores[i]=sum;
    }
    return 0;
}

/* mark outliers: return count; flags[i]=1 if outlier */
int wubu_hkrum_detect(const float* pts,int n,int D,float c,int f,
                       float multiplier,int* flags){
    float* scores=malloc(sizeof(float)*(size_t)n);
    if(!scores)return -1;
    wubu_hkrum_scores(pts,n,D,c,f,scores);
    /* median + MAD-based threshold (robust stats) */
    float* sorted=malloc(sizeof(float)*(size_t)n);
    memcpy(sorted,scores,sizeof(float)*(size_t)n);
    for(int a=0;a<n;a++)
        for(int b=a+1;b<n;b++)
            if(sorted[b]<sorted[a]){
                float t=sorted[a];sorted[a]=sorted[b];sorted[b]=t;
            }
    float med=(n%2)?sorted[n/2]:(sorted[n/2-1]+sorted[n/2])*0.5f;
    /* MAD */
    float* devs=malloc(sizeof(float)*(size_t)n);
    for(int i=0;i<n;i++)devs[i]=fabsf(scores[i]-med);
    for(int a=0;a<n;a++)
        for(int b=a+1;b<n;b++)
            if(devs[b]<devs[a]){
                float t=devs[a];devs[a]=devs[b];devs[b]=t;
            }
    float mad=(n%2)?devs[n/2]:(devs[n/2-1]+devs[n/2])*0.5f;
    float thresh=med+multiplier*(mad>1e-6f?mad:1e-6f);

    int count=0;
    for(int i=0;i<n;i++){
        flags[i]=(scores[i]>thresh);
        if(flags[i])count++;
    }
    free(scores);free(sorted);free(devs);
    return count;
}
