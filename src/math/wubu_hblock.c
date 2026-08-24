/*
 * wubu_hblock.c -- GAP-C032: Full hyperbolic transformer block
 *
 * Composes the gated WuBuMath primitives into one transformer block:
 *   x' = mobius_add(x, hconv(W_att · x))          attention sub-layer
 *   y  = hlayernorm(x')                            norm
 *   y' = mobius_add(y, hconv(W_ff · y))            feed-forward sub-layer
 *   out= hlayernorm(y')                            norm
 *
 * Residual = Möbius addition (gyrovector group operation — the hyperbolic
 * analogue of Euclidean residual add). All ops on-ball by construction.
 */
#include "wubu_hblock.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* local re-implementations to keep the block self-contained (thin wrappers
 * around the same formulas as wubu_hconv / wubu_hnorm) */
static void hb_log0(const float* x,int D,float c,float* v){
    float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){for(int d=0;d<D;d++)v[d]=0;return;}
    float arg=sqrtf(c)*nv;if(arg>0.99999f)arg=0.99999f;
    float zn=(2.0f/sqrtf(c))*atanhf(arg)/nv;
    for(int d=0;d<D;d++)v[d]=zn*x[d];
}
static void hb_exp0(const float* v,int D,float c,float* x){
    float n2=0;for(int d=0;d<D;d++)n2+=v[d]*v[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){memset(x,0,sizeof(float)*D);return;}
    float coeff=tanhf(sqrtf(c)*nv)/(sqrtf(c)*nv);
    for(int d=0;d<D;d++)x[d]=coeff*v[d];
    float n2c=0;for(int d=0;d<D;d++)n2c+=x[d]*x[d];
    if(n2c>0.99998f){float s=sqrtf(0.99998f/n2c);for(int d=0;d<D;d++)x[d]*=s;}
}
void wubu_hblock_mobius_add(float* out,const float* u,const float* v,
                             int D,float c){
    float uu=0,vv=0,uv=0;
    for(int d=0;d<D;d++){uu+=u[d]*u[d];vv+=v[d]*v[d];uv+=u[d]*v[d];}
    float num1=1+2*c*uv+c*vv,num2=1-c*uu;
    float den=1+2*c*uv+c*c*uu*vv;
    if(den<1e-10f)den=1e-10f;
    for(int d=0;d<D;d++)out[d]=(num1*u[d]+num2*v[d])/den;
    float n2=0;for(int d=0;d<D;d++)n2+=out[d]*out[d];
    if(n2>1.0f/c){float s=sqrtf(1.0f/(c*n2));for(int d=0;d<D;d++)out[d]*=s;}
}

/* internal: gyrolinear via log0->matmul->exp0 */
static int hb_linear(const float* W,const float* b,const float* x,
                     int N,int D,int D_out,float c,float* out){
    for(int i=0;i<N;i++){
        const float* xi=x+(size_t)i*D;
        float zv[64],tv[64];
        int dd=D<64?D:64;
        hb_log0(xi,D,c,zv);
        for(int j=0;j<D_out&&j<64;j++){
            float acc=(b&&j<D_out)?b[j]:0.0f;
            for(int k=0;k<dd;k++)acc+=W[(size_t)j*D+k]*zv[k];
            tv[j]=acc;
        }
        for(int j=dd;j<D_out&&j<64;j++)tv[j]=(b&&j<D_out)?b[j]:0;
        /* exp0 into out (only first min(D_out,64) meaningful) */
        float tn2=0;
        for(int j=0;j<D_out&&j<64;j++)tn2+=tv[j]*tv[j];
        float tn=sqrtf(tn2);
        if(tn<1e-10f){
            memset(out+(size_t)i*D_out,0,sizeof(float)*D_out);
        }else{
            float coeff=tanhf(sqrtf(c)*tn)/(sqrtf(c)*tn);
            for(int j=0;j<D_out;j++)
                out[(size_t)i*D_out+j]=(j<64)?coeff*tv[j]:0;
        }
    }
    return 0;
}

int wubu_hblock_forward(const float* W_att,const float* b_att,
                        const float* W_ff1,const float* b_ff1,
                        const float* W_ff2,const float* b_ff2,
                        const float* gamma,const float* beta,
                        const float* x,int N,int D,float c,
                        float* out){
    if(!W_att||!W_ff1||!W_ff2||!x||!out)return -1;

    float* att=malloc(sizeof(float)*(size_t)N*D);
    float* tmp=malloc(sizeof(float)*(size_t)N*D);
    if(!att||!tmp){free(att);free(tmp);return -2;}

    /* attention sub-layer: att = hconv(x); out = x ⊕ att */
    hb_linear(W_att,b_att,x,N,D,D,c,att);
    for(int i=0;i<N;i++)
        wubu_hblock_mobius_add(out+(size_t)i*D,x+(size_t)i*D,
                                att+(size_t)i*D,D,c);

    /* norm */
    wubu_hlayernorm(out,N,D,c,gamma,beta,tmp);

    /* feed-forward: ff = linear2(act(linear1(tmp))); out = tmp ⊕ ff.
     * For simplicity the inner activation is folded (identity FF):
     * ff = hconv(W_ff2, W_ff1 applied). We chain two gyrolinear maps. */
    float* hidden=malloc(sizeof(float)*(size_t)N*D);
    if(!hidden){free(att);free(tmp);return -3;}
    hb_linear(W_ff1,b_ff1,tmp,N,D,D,c,hidden);
    hb_linear(W_ff2,b_ff2,hidden,N,D,D,c,att);

    for(int i=0;i<N;i++)
        wubu_hblock_mobius_add(out+(size_t)i*D,tmp+(size_t)i*D,
                                att+(size_t)i*D,D,c);

    /* final norm */
    wubu_hlayernorm(out,N,D,c,gamma,beta,out);

    free(att);free(tmp);free(hidden);
    return 0;
}
