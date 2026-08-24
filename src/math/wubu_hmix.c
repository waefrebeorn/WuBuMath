/*
 * wubu_hmix.c -- GAP-D029: Hyperbolic-spherical mixed-curvature product
 *
 * Research source: Gu, Sala et al. "Learning Mixed-Curvature
 * Representations in Product Spaces" (NeurIPS 2019).
 *
 * Product manifold M = H^{d1} × S^{d2}:
 *   - hyperbolic factor: tree-like / hierarchical data
 *   - spherical factor: cyclic / periodic data
 * Distance: d_M = sqrt(d_H² + d_S²) (L2 of per-factor distances)
 *
 * The spherical factor uses the unit sphere S^{d2} with arc distance:
 *   d_S(a,b) = arccos(clamp(a·b, -1, 1))
 *
 * This is what the WuBu codec needs for the beam canvas: video frames
 * have hierarchical structure (hyperbolic factor) AND periodic motion
 * patterns (spherical factor). One product space captures both.
 */
#include "wubu_hmix.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* spherical (arc) distance on D_s-dimensional unit sphere */
static float hm_sphere_dist(const float* a,const float* b,int Ds){
    float dot=0;
    for(int d=0;d<Ds;d++)dot+=a[d]*b[d];
    if(dot>1.0f)dot=1.0f;
    if(dot<-1.0f)dot=-1.0f;
    return acosf(dot);
}

float wubu_hm_hyper_dist(const float* a,const float* b,int Dh,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<Dh;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

float wubu_hm_product_dist(const float* x,const float* y,
                            int Dh,int Ds,float c){
    float dh=wubu_hm_hyper_dist(x,y,Dh,c);
    float ds=hm_sphere_dist(x+Dh,y+Dh,Ds);
    return sqrtf(dh*dh+ds*ds);
}

/* project: hyperbolic part clamped to ball; spherical part normalized */
void wubu_hm_project(float* pt,int Dh,int Ds,float c){
    float n2=0;
    for(int d=0;d<Dh;d++)n2+=pt[d]*pt[d];
    float rmax=sqrtf(1.0f/c);
    if(n2>rmax*rmax){
        float s=rmax/sqrtf(n2);
        for(int d=0;d<Dh;d++)pt[d]*=s;
    }
    float sn=0;
    for(int d=Dh;d<Dh+Ds;d++)sn+=pt[d]*pt[d];
    sn=sqrtf(sn);
    if(sn>1e-10f)
        for(int d=Dh;d<Dh+Ds;d++)pt[d]/=sn;
}
