/*
 * wubu_hmha.c -- GAP-C033: Hyperbolic multi-head attention
 *
 * GGBall §E.6.2 pattern: split D into H heads of size dh=D/H; per head:
 *   Q,K,V = gyrolinear projections of x (tangent space)
 *   attn  = softmax(-d(q,k)/tau) over keys
 *   head_out = gyromidpoint of values weighted by attn
 * Concatenate heads, one output gyrolinear.
 *
 * All on-ball: projections via log0->matmul->exp0; attention distance is
 * the geodesic; aggregation is the Einstein midpoint.
 */
#include "wubu_hmha.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void mha_log0(const float* x,int D,float c,float* v){
    float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){for(int d=0;d<D;d++)v[d]=0;return;}
    float arg=sqrtf(c)*nv;if(arg>0.99999f)arg=0.99999f;
    float zn=(2.0f/sqrtf(c))*atanhf(arg)/nv;
    for(int d=0;d<D;d++)v[d]=zn*x[d];
}
static void mha_exp0(const float* v,int D,float c,float* x){
    float n2=0;for(int d=0;d<D;d++)n2+=v[d]*v[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){memset(x,0,sizeof(float)*D);return;}
    float coeff=tanhf(sqrtf(c)*nv)/(sqrtf(c)*nv);
    for(int d=0;d<D;d++)x[d]=coeff*v[d];
    float n2c=0;for(int d=0;d<D;d++)n2c+=x[d]*x[d];
    if(n2c>0.99998f){float s=sqrtf(0.99998f/n2c);for(int d=0;d<D;d++)x[d]*=s;}
}
/* project x [D] -> y [Dh] on-ball via W [Dh,D] */
static void mha_project(const float* W,const float* x,int D,int Dh,
                         float c,float* y){
    float zv[64],tv[64];
    int dd=D<64?D:64;
    mha_log0(x,D,c,zv);
    for(int j=0;j<Dh&&j<64;j++){
        float acc=0;
        for(int k=0;k<dd;k++)acc+=W[(size_t)j*D+k]*zv[k];
        tv[j]=acc;
    }
    mha_exp0(tv,Dh,c,y);
}
static float mha_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}
static void mha_midpoint(const float* vals,const float* w,int K,int D,
                          float c,float* out){
    float num[64],den=1e-10f,tm[64];
    int dd=D<64?D:64;
    memset(num,0,sizeof(float)*(size_t)dd);
    for(int k=0;k<K;k++){
        const float* x=vals+(size_t)k*D;
        float n2=0;for(int d=0;d<dd;d++)n2+=x[d]*x[d];
        float gamma=2/(1-c*n2);
        if(gamma<1)gamma=1;
        for(int d=0;d<dd;d++)num[d]+=gamma*w[k]*x[d];
        den+=(gamma-1)*w[k];
    }
    float tn2=0;
    for(int d=0;d<dd;d++){tm[d]=num[d]/den;tn2+=tm[d]*tm[d];}
    float disc=1-c*tn2;if(disc<1e-9f)disc=1e-9f;
    float sc=1/(1+sqrtf(disc));
    for(int d=0;d<D;d++)out[d]=(d<dd)?tm[d]*sc:0;
}

int wubu_hmha_forward(const float* Wq,const float* Wk,const float* Wv,
                      const float* Wo,
                      const float* x,int N,int D,int heads,
                      float tau,float c,float* out){
    if(!Wq||!Wk||!Wv||!Wo||!x||!out||heads<1||D%heads)return -1;
    int Dh=D/heads;

    /* project Q,K,V per position */
    float* Q=malloc(sizeof(float)*(size_t)N*D);
    float* K=malloc(sizeof(float)*(size_t)N*D);
    float* V=malloc(sizeof(float)*(size_t)N*D);
    if(!Q||!K||!V){free(Q);free(K);free(V);return -2;}
    for(int i=0;i<N;i++){
        mha_project(Wq,x+(size_t)i*D,D,D,c,Q+(size_t)i*D);
        mha_project(Wk,x+(size_t)i*D,D,D,c,K+(size_t)i*D);
        mha_project(Wv,x+(size_t)i*D,D,D,c,V+(size_t)i*D);
    }

    /* per head, per query: attend + aggregate */
    float* head_out=malloc(sizeof(float)*(size_t)D);
    float* kv_vals=malloc(sizeof(float)*(size_t)N*(size_t)Dh);
    float* w=malloc(sizeof(float)*(size_t)N);
    for(int h=0;h<heads;h++){
        int off=h*Dh;
        for(int i=0;i<N;i++){
            const float* q=Q+(size_t)i*D+off;
            /* attention weights over all N keys in this head's slice */
            float mx=-1e30f,z=0;
            for(int j=0;j<N;j++){
                w[j]=-mha_dist(q,K+(size_t)j*D+off,Dh,c)/tau;
                if(w[j]>mx)mx=w[j];
            }
            for(int j=0;j<N;j++){w[j]=expf(w[j]-mx);z+=w[j];}
            for(int j=0;j<N;j++)w[j]/=z;
            /* gather this head's value slice and midpoint */
            for(int j=0;j<N;j++)
                memcpy(kv_vals+(size_t)j*Dh,V+(size_t)j*D+off,sizeof(float)*Dh);
            mha_midpoint(kv_vals,w,N,Dh,c,head_out);
            /* write head slice into a temp at i */
            memcpy(out+(size_t)i*D+off,head_out,sizeof(float)*Dh);
        }
    }

    /* output projection Wo applied to concatenated heads (in-place ok via tmp) */
    float* tmp=malloc(sizeof(float)*(size_t)N*D);
    if(!tmp){free(Q);free(K);free(V);free(head_out);free(kv_vals);free(w);return -3;}
    for(int i=0;i<N;i++)
        mha_project(Wo,out+(size_t)i*D,D,D,c,tmp+(size_t)i*D);
    memcpy(out,tmp,sizeof(float)*(size_t)N*D);

    free(Q);free(K);free(V);free(tmp);free(head_out);free(kv_vals);free(w);
    return 0;
}
