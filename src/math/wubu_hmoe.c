/*
 * wubu_hmoe.c -- GAP-C044: Hyperbolic mixture-of-experts routing
 * (geodesic-distance gate over expert prototypes)
 *
 * Research source: Shazeer 2017 sparse MoE + Switch Transformer
 * top-k routing. Hyperbolic version: each expert has a prototype on the
 * ball; the gate logits are NEGATIVE geodesic distances (closer expert
 * = higher affinity), top-k selected, gyromidpoint-weighted output.
 *
 * The hyperbolic advantage: distance-based routing naturally partitions
 * the manifold into Voronoi-like expert regions that follow the data's
 * hierarchical shape — experts own cones of the ball, not half-spaces.
 */
#include "wubu_hmoe.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float moe_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int wubu_hmoe_route(const float* x,const float* protos,int E,int D,
                     float c,int topk,float tau,
                     int* sel_idx,float* sel_w){
    if(topk<1||topk>E)return -1;
    /* logits = -d/tau */
    float logits[64];
    int Ee=E<64?E:64;
    float mx=-1e30f;
    for(int e=0;e<Ee&&e<E;e++){
        logits[e]=-moe_dist(x,protos+(size_t)e*D,D,c)/tau;
        if(logits[e]>mx)mx=logits[e];
    }
    /* softmax then top-k by brute force (E small) */
    float p[64],z=0;
    for(int e=0;e<Ee&&e<E;e++){p[e]=expf(logits[e]-mx);z+=p[e];}
    for(int e=0;e<Ee&&e<E;e++)p[e]/=(z>0?z:1);

    int used[64];memset(used,0,sizeof(used));
    for(int k2=0;k2<topk;k2++){
        int bi=-1;float bv=-1;
        for(int e=0;e<Ee&&e<E;e++)
            if(!used[e]&&p[e]>bv){bv=p[e];bi=e;}
        used[bi]=1;
        sel_idx[k2]=bi;sel_w[k2]=bv;
    }
    /* renormalize selected weights to sum 1 */
    float s=0;
    for(int k2=0;k2<topk;k2++)s+=sel_w[k2];
    if(s>0)for(int k2=0;k2<topk;k2++)sel_w[k2]/=s;
    return 0;
}

/* weighted gyromidpoint of selected expert outputs */
void wubu_hmoe_combine(const float* outs,const int* sel_idx,
                        const float* sel_w,int topk,int D,float c,
                        float* out){
    float num[64],den=1e-10f,tm[64];
    int dd=D<64?D:64;
    memset(num,0,sizeof(float)*(size_t)dd);
    for(int k2=0;k2<topk;k2++){
        const float* o=outs+(size_t)sel_idx[k2]*D;
        float n2=0;for(int d=0;d<dd;d++)n2+=o[d]*o[d];
        float gamma=2/(1-c*n2);if(gamma<1)gamma=1;
        for(int d=0;d<dd;d++)num[d]+=gamma*sel_w[k2]*o[d];
        den+=(gamma-1)*sel_w[k2];
    }
    float tn2=0;
    for(int d=0;d<dd;d++){tm[d]=num[d]/den;tn2+=tm[d]*tm[d];}
    float disc=1-c*tn2;if(disc<1e-9f)disc=1e-9f;
    float sc=1/(1+sqrtf(disc));
    for(int d=0;d<D;d++)out[d]=(d<dd)?tm[d]*sc:0;
    /* project */
    float n2c=0;for(int d=0;d<D;d++)n2c+=out[d]*out[d];
    if(n2c>0.99998f){float s=sqrtf(0.99998f/n2c);for(int d=0;d<D;d++)out[d]*=s;}
}

float wubu_hmoe_load_balance(const int* assignments,int N,int E){
    /* coefficient of variation of expert usage (lower = balanced) */
    if(N<=0||E<=0)return 0;
    int counts[64];memset(counts,0,sizeof(counts));
    int Ee=E<64?E:E;
    for(int i=0;i<N;i++){
        int e=assignments[i];
        if(e>=0&&e<Ee)counts[e]++;
    }
    double mean=(double)N/E,var=0;
    for(int e=0;e<Ee&&e<E;e++){
        double df=counts[e]-mean;var+=df*df;
    }
    var/=E;
    return mean>0?(float)(sqrtf((float)(var))/mean):0;
}
