/*
 * wubu_kvcache.c -- GAP-C036: KV cache for hyperbolic autoregressive decoding
 *
 * Incremental generation: cache past K/V (on-ball Poincaré coordinates),
 * append new tokens without recomputing attention over the full history.
 * Single-token decode path: new query attends to [cached..., new] keys.
 */
#include "wubu_kvcache.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int wubu_kvc_init(WubuKVCache* kc,int max_T,int D,float c){
    if(max_T<1||D<1)return -1;
    kc->max_T=max_T;kc->len=0;kc->D=D;kc->c=c;
    kc->keys=malloc(sizeof(float)*(size_t)max_T*D);
    kc->vals=malloc(sizeof(float)*(size_t)max_T*D);
    if(!kc->keys||!kc->vals)return -2;
    memset(kc->keys,0,sizeof(float)*(size_t)max_T*D);
    memset(kc->vals,0,sizeof(float)*(size_t)max_T*D);
    return 0;
}
void wubu_kvc_free(WubuKVCache* kc){
    free(kc->keys);free(kc->vals);
    kc->keys=NULL;kc->vals=NULL;
}
void wubu_kvc_reset(WubuKVCache* kc){kc->len=0;}

int wubu_kvc_append(WubuKVCache* kc,const float* k_new,const float* v_new){
    if(kc->len>=kc->max_T)return -1;
    memcpy(kc->keys+(size_t)kc->len*kc->D,k_new,sizeof(float)*kc->D);
    memcpy(kc->vals+(size_t)kc->len*kc->D,v_new,sizeof(float)*kc->D);
    kc->len++;
    return 0;
}

/* single-token decode: q attends over all cached entries.
 * attn weights = softmax(-geodesic(q,k)/tau); out = midpoint of vals. */
void wubu_kvc_decode(const WubuKVCache* kc,const float* q,
                     float tau,float* out){
    int D=kc->D,c_int=1;
    float c=kc->c;
    float* w=malloc(sizeof(float)*(size_t)(kc->len>0?kc->len:1));
    float mx=-1e30f,z=0;
    for(int i=0;i<kc->len;i++){
        const float* k=kc->keys+(size_t)i*D;
        float ab2=0,a2=0,b2=0;
        for(int d=0;d<D;d++){
            float df=q[d]-k[d];ab2+=df*df;
            a2+=q[d]*q[d];b2+=k[d]*k[d];
        }
        float den=(1-c*a2)*(1-c*b2);
        if(den<1e-9f)den=1e-9f;
        float arg=1+2*c*ab2/den;
        float dist=acoshf(arg>1?arg:1)/sqrtf(c);
        w[i]=-dist/tau;
        if(w[i]>mx)mx=w[i];
    }
    for(int i=0;i<kc->len;i++){w[i]=expf(w[i]-mx);z+=w[i];}
    if(z>0)for(int i=0;i<kc->len;i++)w[i]/=z;

    /* weighted gyromidpoint of cached values */
    float num[64],den=1e-10f,tm[64];
    int dd=D<64?D:64;
    memset(num,0,sizeof(float)*(size_t)dd);
    for(int i=0;i<kc->len;i++){
        const float* v=kc->vals+(size_t)i*D;
        float n2=0;for(int d=0;d<dd;d++)n2+=v[d]*v[d];
        float gamma=2/(1-c*n2);
        if(gamma<1)gamma=1;
        for(int d=0;d<dd;d++)num[d]+=gamma*w[i]*v[d];
        den+=(gamma-1)*w[i];
    }
    float tn2=0;
    for(int d=0;d<dd;d++){tm[d]=num[d]/den;tn2+=tm[d]*tm[d];}
    float disc=1-c*tn2;if(disc<1e-9f)disc=1e-9f;
    float sc=1/(1+sqrtf(disc));
    for(int d=0;d<D;d++)out[d]=(d<dd)?tm[d]*sc:0;
    free(w);
    (void)c_int;
}
