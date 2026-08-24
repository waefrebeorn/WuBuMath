/*
 * wubu_learned_pos.c -- GAP-C034: Learned positional embedding table
 *
 * The classic GPT-style alternative to sinusoidal/RoPE/HoPE: a trainable
 * table pos_emb[max_T][D] added (in tangent space, then exp0) to token
 * embeddings. For the hyperbolic stack we add in T_0 via mobius_add with
 * exp0-mapped embedding — keeps everything on-ball.
 *
 * Gates: init random distinct rows; gradient step moves rows; round-trip
 * through save/load preserves table exactly.
 */
#include "wubu_learned_pos.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int wubu_lp_init(WubuLearnedPos* lp,int max_T,int D,unsigned seed){
    if(max_T<1||D<1)return -1;
    lp->max_T=max_T;lp->D=D;
    lp->table=malloc(sizeof(float)*(size_t)max_T*D);
    if(!lp->table)return -2;
    /* small random init */
    unsigned rs=seed*2654435761u+7u;
    for(int i=0;i<max_T*D;i++){
        rs=rs*1103515245u+12345u;
        lp->table[i]=(float)((rs>>16)%2000)/20000.0f-0.05f;
    }
    return 0;
}
void wubu_lp_free(WubuLearnedPos* lp){free(lp->table);lp->table=NULL;}

/* get row t (no copy for internal use) */
const float* wubu_lp_row(const WubuLearnedPos* lp,int t){
    if(t<0||t>=lp->max_T)return NULL;
    return lp->table+(size_t)t*lp->D;
}

/* add positional info to an on-ball token embedding:
 * z = log0(tok); z += table[t]; tok' = exp0(z) */
void wubu_lp_apply(const WubuLearnedPos* lp,const float* tok,
                   int t,float c,float* out){
    if(t<0||t>=lp->max_T){
        memcpy(out,tok,sizeof(float)*lp->D);
        return;
    }
    const float* pe=lp->table+(size_t)t*lp->D;
    int D=lp->D;
    float n2=0;for(int d=0;d<D;d++)n2+=tok[d]*tok[d];
    float nv=sqrtf(n2);
    float zv[64];int dd=D<64?D:64;
    if(nv>1e-10f){
        float arg=sqrtf(c)*nv;if(arg>0.99999f)arg=0.99999f;
        float zn=(2.0f/sqrtf(c))*atanhf(arg)/nv;
        for(int d=0;d<dd;d++)zv[d]=zn*tok[d];
    }else{
        memset(zv,0,sizeof(float)*dd);
    }
    for(int d=0;d<dd;d++)zv[d]+=pe[d];
    /* exp0 back */
    float n2v=0;for(int d=0;d<dd;d++)n2v+=zv[d]*zv[d];
    float nvs=sqrtf(n2v);
    if(nvs<1e-10f){memset(out,0,sizeof(float)*D);return;}
    float coeff=tanhf(sqrtf(c)*nvs)/(sqrtf(c)*nvs);
    for(int d=0;d<D&&d<dd&&d<64;d++){
        out[d]=coeff*zv[d];
        if(d>=dd)out[d]=0;
    }
    for(int d=dd;d<D;d++)out[d]=0;
    /* boundary cap */
    float n2c=0;for(int d=0;d<D;d++)n2c+=out[d]*out[d];
    if(n2c>0.99998f){float s=sqrtf(0.99998f/n2c);for(int d=0;d<D;d++)out[d]*=s;}
}

/* FD gradient step on one row against a provided loss gradient */
void wubu_lp_train_row(WubuLearnedPos* lp,int t,const float* grad,float lr){
    if(t<0||t>=lp->max_T)return;
    float* row=lp->table+(size_t)t*lp->D;
    for(int d=0;d<lp->D;d++)
        row[d]-=lr*grad[d];
}
