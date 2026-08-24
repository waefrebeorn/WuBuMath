/*
 * wubu_hmds.c -- GAP-D019: Hyperbolic MDS (stress minimization)
 *
 * Given a target distance matrix D_ij, find on-ball embeddings minimizing
 * hyperbolic stress: Σ_ij w_ij (d(x_i,x_j) - D_ij)²
 * via Riemannian SGD with boundary-guarded steps (C023 recipe).
 *
 * Initialization: random small near origin. The gradient of pairwise
 * geodesic stress is approximated by the Euclidean direction between
 * points scaled by the residual — the standard practical MDS gradient.
 */
#include "wubu_hmds.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float hmds_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

float wubu_hmds_stress(const float* pts,int n,int D,float c,
                        const float* target){
    double s=0;
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++){
            float diff=hmds_dist(pts+(size_t)i*D,pts+(size_t)j*D,D,c)
                       -target[(size_t)i*n+j];
            s+=(double)(diff*diff)*4;
        }
    return (float)s;
}

int wubu_hmds_embed(const float* target,int n,int D,float c,
                    int iters,float lr,unsigned seed,
                    float* out /* [n,D] */){
    if(!target||!out||n<2)return -1;
    unsigned rs=seed*374761393u;
    for(int i=0;i<n*D;i++){
        rs=rs*1103515245u+12345u;
        out[i]=(float)((rs>>16)%2000)/40000.0f-0.025f;   /* tiny init */
    }

    for(int it=0;it<iters;it++){
        /* sample a random pair and take a step on its residual */
        int i=(int)((rs=(rs*1103515245u+12345u)>>16)%n);
        int j=(int)((rs=(rs*1103515245u+12345u)>>16)%n);
        if(i==j)continue;
        float d=hmds_dist(out+(size_t)i*D,out+(size_t)j*D,D,c);
        float res=d-target[(size_t)i*n+j];
        /* direction: move i away/toward j proportional to residual */
        float g[64];
        int dd=D<64?D:64;
        for(int k=0;k<dd;k++){
            float dir=out[(size_t)i*D+k]-out[(size_t)j*D+k];
            g[k]=res*dir/(d+1e-6f);
        }
        /* guarded step on point i */
        for(int k=0;k<dd;k++)out[(size_t)i*D+k]-=lr*g[k];
        /* opposite step on j */
        for(int k=0;k<dd;k++){
            float dir=out[(size_t)i*D+k]-out[(size_t)j*D+k];
            g[k]=res*(-dir)/(d+1e-6f);
        }
        for(int k=0;k<dd;k++)out[(size_t)j*D+k]-=lr*g[k];
        /* project both into ball */
        for(int idx=0;idx<n;idx++){
            float n2=0;
            for(int k=0;k<D;k++)n2+=out[(size_t)idx*D+k]*out[(size_t)idx*D+k];
            if(n2>0.99998f){float s=sqrtf(0.99998f/n2);
                for(int k=0;k<D;k++)out[(size_t)idx*D+k]*=s;}
        }
    }
    return 0;
}
