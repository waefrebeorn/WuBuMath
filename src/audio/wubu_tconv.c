/*
 * wubu_tconv.c -- GAP-C048: Hyperbolic 1D temporal convolution
 * (tangent-space sliding window + gyromidpoint pooling)
 *
 * The temporal feature extractor for the codec's frame sequences:
 * a window of T consecutive on-ball frames is mapped to the LOCAL
 * tangent space of the window's gyromidpoint, convolved with a small
 * kernel, and re-projected. Local tangent = the paper-proven best
 * approximation point (HGCN Table 2 finding, same as B018).
 *
 * Output: one on-ball embedding per window position (stride 1),
 * sequence length preserved via edge padding.
 */
#include "wubu_tconv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void tc_log0_at(const float* x,const float* mu,int D,float c,
                        float* out){
    /* log_mu(x) simplified to log_0 of the difference direction:
     * for nearby points this approximates the local log map */
    float v[512];
    int dd=D<512?D:512;
    for(int d=0;d<dd&&d<512;d++)v[d]=x[d]-mu[d];
    float n2=0;
    for(int d=0;d<dd&&d<512;d++)n2+=v[d]*v[d];
    float nv=sqrtf(n2);
    if(nv>1e-10f){
        float arg=sqrtf(c)*nv;
        if(arg>0.999999f)arg=0.999999f;
        float zn=atanhf(arg)/(sqrtf(c)*nv);
        for(int d=0;d<dd&&d<512;d++)out[d]=zn*v[d];
    }else{
        memset(out,0,sizeof(float)*(size_t)dd);
    }
    for(int d=512;d<D;d++)out[d]=0;
}

int wubu_tc_conv1d(const float* seq,int T,int D,int K,float c,
                    const float* kernel,float bias,
                    unsigned seed,float* out){
    if(!seq||!out||T<1||K<1)return -1;
    if(!kernel){
        /* default kernel: uniform */
        float* kdef=malloc(sizeof(float)*(size_t)K);
        if(!kdef)return -2;
        for(int i=0;i<K;i++)kdef[i]=1.0f/K;
        int r=wubu_tc_conv1d(seq,T,D,K,c,kdef,bias,seed,out);
        free(kdef);
        return r;
    }

    /* pad by replicating edges */
    float* padded=malloc(sizeof(float)*(size_t)(T+2*(K-1))*D);
    if(!padded)return -3;
    for(int t=-K+1;t<T+K-1;t++){
        int ti=t<0?0:(t>=T?T-1:t);
        memcpy(padded+(size_t)(t+K-1)*D,seq+(size_t)ti*D,sizeof(float)*D);
    }

    for(int t=0;t<T;t++){
        /* window midpoint (gyro-average approximation: Euclidean mean
         * then project — valid for tight windows) */
        float mid[512];
        int dd=D<512?D:512;
        memset(mid,0,sizeof(float)*(size_t)dd);
        for(int k2=0;k2<K;k2++)
            for(int d=0;d<dd&&d<512;d++)
                mid[d]+=padded[(size_t)(t+k2)*D+d];
        for(int d=0;d<dd&&d<512;d++)mid[d]/=K;
        float n2=0;
        for(int d=0;d<dd&&d<512;d++)n2+=mid[d]*mid[d];
        if(n2>0.999f/c){
            float s=sqrtf(0.999f/(c*n2));
            for(int d=0;d<dd&&d<512;d++)mid[d]*=s;
        }

        /* tangent-space kernel mix at local origin */
        float acc[512];
        memset(acc,0,sizeof(float)*(size_t)dd);
        float wsum=0;
        for(int k2=0;k2<K;k2++){
            float tang[512];
            tc_log0_at(padded+(size_t)(t+k2)*D,mid,D,c,tang);
            for(int d=0;d<dd&&d<512;d++)acc[d]+=kernel[k2]*tang[d];
            wsum+=kernel[k2];
        }
        if(wsum<=0)wsum=1;
        /* exp from local origin back to ball: exp_mid(v) ≈ project(mid+v) */
        float moved[512];
        for(int d=0;d<dd&&d<512;d++)moved[d]=mid[d]+acc[d]/wsum+bias;
        float n2c=0;
        for(int d=0;d<D&&d<512;d++)n2c+=moved[d]*moved[d];
        float rmax=sqrtf(1.0f/c)*0.999f;
        if(n2c>rmax*rmax){
            float s=rmax/sqrtf(n2c);
            for(int d=0;d<D&&d<512;d++)moved[d]*=s;
        }
        for(int d=0;d<D&&d<512;d++)out[(size_t)t*D+d]=moved[d];
        for(int d=512;d<D;d++)out[(size_t)t*D+d]=0;
    }
    free(padded);
    return 0;
}
