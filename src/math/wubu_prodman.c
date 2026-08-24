/*
 * wubu_prodman.c -- GAP-D014: Product manifold distance + embedding
 *
 * Research source: Gu, Sala, Goñi-Moreno, Ré NeurIPS 2019 "Learning
 * Mixed-Curvature Representations in Product Spaces".
 *
 * A product manifold M = H^{d1} × S^{d2} × R^{d3} assigns each point a
 * tuple of coordinates (one per factor). Distance is the L2 norm of the
 * per-factor distances:
 *    d_M(x,y) = sqrt( d_H(x1,y1)² + d_S(x2,y2)² + d_E(x3,y3)² )
 *
 * This lets us embed data with mixed local geometry — tree-like parts go
 * in the hyperbolic factor, cyclic parts in the spherical factor, and
 * flat parts in the Euclidean factor.
 *
 * We implement the 2-factor version H × E (most useful for WuBu):
 * hyperbolic captures hierarchy, Euclidean captures residual structure.
 */
#include "wubu_prodman.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

float wubu_pm_hyper_dist(const float* a,const float* b,int D,float c){
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

float wubu_pm_product_dist(const float* x,const float* y,
                            int D_hyp,int D_euc,float c){
    /* hyperbolic factor */
    float dh=wubu_pm_hyper_dist(x,y,D_hyp,c);
    /* euclidean factor */
    float de2=0;
    for(int d=D_hyp;d<D_hyp+D_euc;d++){
        float df=x[d]-y[d];
        de2+=df*df;
    }
    return sqrtf(dh*dh+de2);
}

/* project a full product point: hyperbolic part clamped to ball,
 * euclidean part untouched */
void wubu_pm_project(float* pt,int D_hyp,int D_euc,float c){
    float n2=0;
    for(int d=0;d<D_hyp;d++)n2+=pt[d]*pt[d];
    float rmax=sqrtf(1.0f/c);
    if(n2>rmax*rmax){
        float s=rmax/sqrtf(n2);
        for(int d=0;d<D_hyp;d++)pt[d]*=s;
    }
}

/* random product-manifold embedding init */
void wubu_pm_init_random(float* pts,int n,int D_hyp,int D_euc,
                          float c,unsigned* seed){
    for(int i=0;i<n;i++){
        float* p=pts+(size_t)i*(D_hyp+D_euc);
        /* hyperbolic: small random inside ball */
        for(int d=0;d<D_hyp;d++){
            *seed=(*seed)*1103515245u+12345u;
            p[d]=(float)((*seed>>16)%1000)/50000.0f-0.01f;
        }
        /* euclidean: standard normal-ish */
        for(int d=D_hyp;d<D_hyp+D_euc;d++){
            *seed=(*seed)*1103515245u+12345u;
            p[d]=((float)((*seed>>16)%2000)/2000.0f-0.5f)*0.5f;
        }
        wubu_pm_project(p,D_hyp,D_euc,c);
    }
}
