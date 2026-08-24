/*
 * wubu_hkmedoids.c -- GAP-D025: Hyperbolic k-medoids (PAM on the ball)
 *
 * Unlike k-means (C020) whose gyromidpoint centroids are synthetic points
 * sensitive to outliers, k-medoids picks ACTUAL data points as centers —
 * robust to outliers by construction. PAM algorithm:
 *   BUILD: greedily pick medoids maximizing coverage
 *   SWAP:  iteratively try swapping a medoid with a non-medoid; keep
 *          swaps that reduce total cost (sum of distances to nearest medoid)
 */
#include "wubu_hkmedoids.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float km_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

/* total cost given medoid set */
static float km_total_cost(const float* pts,int n,int D,float c,
                            const int* medoids,int k){
    float cost=0;
    for(int i=0;i<n;i++){
        float best=1e30f;
        for(int m=0;m<k;m++){
            float d=km_dist(pts+(size_t)i*D,pts+(size_t)medoids[m]*D,D,c);
            if(d<best)best=d;
        }
        cost+=best;
    }
    return cost;
}

int wubu_hkmedoids(const float* pts,int n,int D,int k,float c,
                   int iters,int* medoids,int* assign){
    if(!pts||!medoids||!assign||k<1||k>n)return -1;

    /* BUILD phase: first medoid = point minimizing sum of dists to all
     * (most central); subsequent = farthest from existing medoids. */
    int nm=0;
    {
        float best_cost=1e30f;int best_i=0;
        for(int i=0;i<n;i++){
            float s=0;
            for(int j=0;j<n;j++)s+=km_dist(pts+(size_t)i*D,pts+(size_t)j*D,D,c);
            if(s<best_cost){best_cost=s;best_i=i;}
        }
        medoids[nm++]=best_i;
    }
    while(nm<k){
        /* farthest-point selection */
        float best_d=-1;int best_i=-1;
        for(int i=0;i<n;i++){
            int is_med=0;
            for(int m=0;m<nm;m++)if(medoids[m]==i){is_med=1;break;}
            if(is_med)continue;
            float mind=1e30f;
            for(int m=0;m<nm;m++){
                float d=km_dist(pts+(size_t)i*D,pts+(size_t)medoids[m]*D,D,c);
                if(d<mind)mind=d;
            }
            if(mind>best_d){best_d=mind;best_i=i;}
        }
        medoids[nm++]=best_i;
    }

    /* SWAP phase */
    float cost=km_total_cost(pts,n,D,c,medoids,k);
    for(int it=0;it<iters;it++){
        int improved=0;
        for(int m=0;m<k;m++){
            for(int cand=0;cand<n;cand++){
                int is_med=0;
                for(int mm=0;mm<k;mm++)if(medoids[mm]==cand){is_med=1;break;}
                if(is_med)continue;
                int old=medoids[m];
                medoids[m]=cand;
                float nc=km_total_cost(pts,n,D,c,medoids,k);
                if(nc<cost-1e-6f){
                    cost=nc;improved=1;
                    break;   /* keep swap, restart this medoid scan */
                }else{
                    medoids[m]=old;
                }
            }
            if(improved)break;
        }
        if(!improved)break;
    }

    /* final assignment */
    for(int i=0;i<n;i++){
        float best=1e30f;int bm=0;
        for(int m=0;m<k;m++){
            float d=km_dist(pts+(size_t)i*D,pts+(size_t)medoids[m]*D,D,c);
            if(d<best){best=d;bm=m;}
        }
        assign[i]=bm;
    }
    return 0;
}
