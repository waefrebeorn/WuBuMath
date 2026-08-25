/*
 * wubu_silhouette.c -- GAP-D034: Hyperbolic silhouette score
 * (geodesic cohesion/separation cluster validity)
 *
 * Rousseeuw 1987's silhouette, computed with geodesic distances:
 *   a(i) = mean d(x_i, x_j) for j in same cluster      (cohesion)
 *   b(i) = min over other clusters of mean d(x_i, x_j) (separation)
 *   s(i) = (b-a)/max(a,b)  in [-1, +1]
 *
 * This is THE evaluation primitive the clustering suite needed: C020,
 * D016, D021, D025 outputs can now be compared objectively.
 */
#include "wubu_silhouette.h"
#include <stdlib.h>
#include <math.h>

static float si_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

float wubu_sil_score(const float* pts,const int* assign,
                      int n,int D,int k,float c){
    if(n<3||k<2)return -2;   /* undefined */
    double total=0;
    int counted=0;
    for(int i=0;i<n;i++){
        int myc=assign[i];
        /* count cluster sizes */
        int own=0;
        for(int j=0;j<n;j++)if(assign[j]==myc&&j!=i)own++;
        if(own==0)continue;   /* singleton cluster: skip */

        /* a: mean distance to own cluster */
        double a_sum=0;
        for(int j=0;j<n;j++)
            if(j!=i&&assign[j]==myc)
                a_sum+=si_dist(pts+(size_t)i*D,pts+(size_t)j*D,D,c);
        float a=(float)(a_sum/own);

        /* b: min over OTHER clusters of mean distance */
        float b=1e30f;
        for(int kc=0;kc<k;kc++){
            if(kc==myc)continue;
            int cnt=0;
            double sum=0;
            for(int j=0;j<n;j++)
                if(assign[j]==kc&&j!=i){
                    sum+=si_dist(pts+(size_t)i*D,pts+(size_t)j*D,D,c);
                    cnt++;
                }
            if(cnt>0){
                float m=(float)(sum/cnt);
                if(m<b)b=m;
            }
        }
        if(b>=1e30f)continue;
        float mx=a>b?a:b;
        total+=(double)((b-a)/mx);
        counted++;
    }
    return counted>0?(float)(total/counted):-2;
}
