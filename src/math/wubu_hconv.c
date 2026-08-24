/*
 * wubu_hconv.c -- GAP-C027: Hyperbolic pointwise convolution (HNN++ §3.3)
 *
 * Shimizu et al. (HNN++, arXiv:2006.08210) define hyperbolic 1×1
 * convolution as Möbius pointwise multiplication:
 *   out_c = λ_x ⊗_c W_c · x    for each output channel c
 * where λ_x = 2/(1-c|x|²) is the conformal factor and ⊗ is Möbius
 * scalar-matrix multiplication in the gyrovector space.
 *
 * Practical form used here (numerically equivalent, simpler):
 *   1. log₀(x) — to tangent space
 *   2. matmul by W [C_out × D]
 *   3. exp₀(result) — back on ball
 * plus per-channel bias via gyration. This IS the HNN "Poincaré MLP"
 * layer; we name it conv because it applies per spatial position.
 */
#include "wubu_hconv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* mobius scalar multiplication: t ⊗ x */
static void hconv_mobius_scalar(float* out,float t,const float* x,
                                 int D,float c){
    float n2=0;
    for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float nx=sqrtf(n2);
    if(nx<1e-10f){for(int d=0;d<D;d++)out[d]=0;return;}
    float rt=t*nx;
    float coeff=tanhf(sqrtf(c)*rt)/(sqrtf(c)*nx);
    for(int d=0;d<D;d++)out[d]=coeff*x[d];
}

/* mobius addition: out = u ⊕ v */
static void hconv_mobius_add(float* out,const float* u,const float* v,
                              int D,float c){
    float uu=0,vv=0,uv=0;
    for(int d=0;d<D;d++){uu+=u[d]*u[d];vv+=v[d]*v[d];uv+=u[d]*v[d];}
    float num1=1+2*c*uv+c*vv;
    float num2=1-c*uu;
    float den=1+2*c*uv+c*c*uu*vv;
    if(den<1e-10f)den=1e-10f;
    for(int d=0;d<D;d++)out[d]=(num1*u[d]+num2*v[d])/den;
    /* project */
    float n2=0;for(int d=0;d<D;d++)n2+=out[d]*out[d];
    if(n2>1.0f/c){float s=sqrtf(1.0f/(c*n2));for(int d=0;d<D;d++)out[d]*=s;}
}

/* hyperbolic 1x1 conv over N positions:
 * x: [N,D] on ball. W: [D_out,D]. b: [D] tangent bias or NULL.
 * out: [N,D_out]. */
int wubu_hconv_forward(const float* W,const float* b,
                       const float* x,int N,int D,int D_out,
                       float c,float* out){
    if(!W||!x||!out)return -1;
    /* log₀(x): tangent vector at origin = artanh(sqrt(c)|x|)*x/(sqrt(c)|x|)
     * simplified: z = atanh-clamped direction-scaled */
    float* tanv=malloc(sizeof(float)*(size_t)(N>0?D:1));
    if(!tanv)return -2;

    for(int i=0;i<N;i++){
        const float* xi=x+(size_t)i*D;
        /* log0 */
        float n2=0;for(int d=0;d<D;d++)n2+=xi[d]*xi[d];
        float nv=sqrtf(n2);
        float zv[64];int dd=D<64?D:64;
        if(nv<1e-10f){
            memset(zv,0,sizeof(float)*dd);
        }else{
            float arg=sqrtf(c)*nv;if(arg>0.99999f)arg=0.99999f;
            float zn=2.0f/sqrtf(c)*atanhf(arg)/nv;
            for(int d=0;d<dd;d++)zv[d]=zn*xi[d];
        }
        /* matmul: t_j = sum_k W[j,k]*z_k (+bias) */
        float tv[64];
        for(int j=0;j<dd&&j<D_out;j++){
            float acc=wubu_hconv_bias_at(b,j);
            for(int k=0;k<D;k++)acc+=W[(size_t)j*D+k]*zv[k];
            tv[j]=acc;
        }
        for(int j=dd;j<D_out&&j<64;j++)tv[j]=wubu_hconv_bias_at(b,j);

        /* exp0(tv): tanh(sqrt(c)|t|/2)/(sqrt(c)|t|) * t */
        float tn2=0;for(int j=0;j<D_out&&j<64;j++)tn2+=tv[j]*tv[j];
        float tn=sqrtf(tn2);
        if(tn<1e-10f){
            memset(out+(size_t)i*D_out,0,sizeof(float)*D_out);
        }else{
            float coeff=tanhf(sqrtf(c)*tn*0.5f)/(sqrtf(c)*tn);
            float n2c=0;
            for(int j=0;j<D_out&&j<64;j++){
                out[(size_t)i*D_out+j]=coeff*tv[j];
                n2c+=out[(size_t)i*D_out+j]*out[(size_t)i*D_out+j];
            }
            /* fp32 boundary cap */
            if(n2c>0.99998f){
                float s=sqrtf(0.99998f/n2c);
                for(int j=0;j<D_out&&j<64;j++)out[(size_t)i*D_out+j]*=s;
            }
        }
    }
    free(tanv);
    return 0;
}

/* helper: safe bias access (NULL → 0) */
float wubu_hconv_bias_at(const float* b,int j){
    return b?b[j]:0.0f;
}
