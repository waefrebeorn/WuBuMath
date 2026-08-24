/*
 * wubu_attn_rollout.c -- GAP-B016: Attention rollout on the ball
 *
 * Attention rollout (Abnar & Zuidema 2020) tracks how information flows
 * through stacked attention layers by multiplying normalized attention
 * matrices. In hyperbolic form: each layer's aggregation pulls nodes
 * toward their attended neighbors' midpoint, so the rollout measures
 * cumulative geodesic contraction from input to output.
 *
 * Gates:
 *  G1 rollout weights are nonneg and sum to ~1 per row
 *  G2 identity-attention rollout = identity
 *  G3 two-layer rollout = product of single layers (associativity)
 */
#include "wubu_attn_rollout.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* single-layer rollout: A_norm + I, renormalized rows */
void wubu_ar_layer(const float* attn,int N,float* out){
    for(int i=0;i<N;i++){
        float row_sum=0;
        for(int j=0;j<N;j++){
            /* add identity residual connection */
            out[i*N+j]=attn[i*N+j]+((i==j)?1.0f:0.0f);
            row_sum+=out[i*N+j];
        }
        if(row_sum>1e-10f)
            for(int j=0;j<N;j++)out[i*N+j]/=row_sum;
    }
}

/* multi-layer rollout: iteratively multiply */
void wubu_ar_rollout(const float** attns,const int* num_layers_ptr,
                      int N,float* out){
    int L=*num_layers_ptr;
    float* tmp=malloc(sizeof(float)*(size_t)N*N);
    float* acc=malloc(sizeof(float)*(size_t)N*N);

    wubu_ar_layer(attns[0],N,acc);
    for(int l=1;l<L;l++){
        wubu_ar_layer(attns[l],N,tmp);
        /* matrix multiply acc = tmp @ acc */
        float* prod=malloc(sizeof(float)*(size_t)N*N);
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++){
                float s=0;
                for(int k=0;k<N;k++)s+=tmp[i*N+k]*acc[k*N+j];
                prod[i*N+j]=s;
            }
        memcpy(acc,prod,sizeof(float)*(size_t)N*N);
        free(prod);
    }
    memcpy(out,acc,sizeof(float)*(size_t)N*N);
    free(tmp);free(acc);
}

/* hyperbolic distance-weighted attention matrix from embeddings */
void wubu_ar_distance_attention(const float* embs,int D,int N,
                                 float c,float tau,float* attn){
    for(int i=0;i<N;i++){
        float mx=-1e30f,z=0;
        for(int j=0;j<N;j++){
            float ab2=0,a2=0,b2=0;
            for(int d=0;d<D;d++){
                float df=embs[(size_t)i*D+d]-embs[(size_t)j*D+d];
                ab2+=df*df;
                a2+=embs[(size_t)i*D+d]*embs[(size_t)i*D+d];
                b2+=embs[(size_t)j*D+d]*embs[(size_t)j*D+d];
            }
            float den=(1-c*a2)*(1-c*b2);
            if(den<1e-9f)den=1e-9f;
            float arg=1+2*c*ab2/den;
            float d=acoshf(arg>1?arg:1)/sqrtf(c);
            attn[i*N+j]=-d/tau;
            if(attn[i*N+j]>mx)mx=attn[i*N+j];
        }
        for(int j=0;j<N;j++){attn[i*N+j]=expf(attn[i*N+j]-mx);z+=attn[i*N+j];}
        for(int j=0;j<N;j++)attn[i*N+j]/=z;
    }
}
