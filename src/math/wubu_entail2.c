/*
 * wubu_entail2.c -- GAP-D030: Entailment cone 2D cross-section check
 * (the Ganea ICML 2018 formula, explicit)
 *
 * Research source: Ganea, Bécigneul, Hofmann ICML 2018
 * (arXiv:1804.01882). The hyperbolic entailment cone at parent p has
 * apex angle k_p; a child c is entailed iff the angle between the
 * geodesic ray (p→c) and the radial direction at p is <= k_p/2.
 *
 * The paper's closed form for the optimal aperture at norm ||p||:
 *   sin(k_p) = 2*sqrt(c)*||p|| / (sqrt(1+c||p||²) + sqrt(c)*||p||)
 * (derived from the asymptotic angle analysis; children of deeper nodes
 * get wider cones).
 *
 * We implement the 2D angular test in the plane spanned by p and c:
 *   theta = angle between (c-p) and (-p) [pointing inward from p]
 *   entailment iff theta < k_p/2
 */
#include "wubu_entail2.h"
#include <math.h>

float wubu_ec_aperture(const float* p,int D,float c){
    /* optimal cone half-angle from the paper's closed form */
    float n2=0;
    for(int d=0;d<D;d++)n2+=p[d]*p[d];
    float np=sqrtf(n2);
    if(np<1e-8f)return 3.14159265f;   /* near origin: wide-open cone */
    float num=2.0f*np;
    float den=sqrtf(1.0f/c+n2)+np;
    if(den<1e-10f)den=1e-10f;
    float s=num/den;
    if(s>1.0f)s=1.0f;
    return asinf(s);
}

int wubu_ec_entailed(const float* parent,const float* child,
                      int D,float c){
    /* vector from parent toward origin (inward radial) */
    float inward[64];
    int dd=D<64?D:64;
    for(int d=0;d<dd;d++)inward[d]=-parent[d];

    /* vector from parent to child */
    float to_c[64],n2=0;
    for(int d=0;d<dd;d++){to_c[d]=child[d]-parent[d];n2+=to_c[d]*to_c[d];}
    float nt=sqrtf(n2);
    if(nt<1e-10f)return 1;   /* same point: trivially entailed */

    /* cos of angle between them */
    float dot=0,in2=0;
    for(int d=0;d<dd;d++){dot+=to_c[d]*inward[d];in2+=inward[d]*inward[d];}
    float ni=sqrtf(in2);
    if(ni<1e-10f)return 1;   /* parent at origin: everything inside */
    float cos_t=dot/(nt*ni);
    if(cos_t>1.0f)cos_t=1.0f;
    if(cos_t<-1.0f)cos_t=-1.0f;
    float theta=acosf(cos_t);

    float half=wubu_ec_aperture(parent,D,c)*0.5f;
    return theta<=half?1:0;
}
