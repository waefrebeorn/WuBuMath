/*
 * wubu_kgtrans.c -- GAP-D036: Hyperbolic TransE-style knowledge graph
 * embedding (translation via Möbius addition)
 *
 * Research source: Bordes 2013 (TransE) + hyperbolic KG line
 * (arXiv:2305.13015, 3H-TH). The scoring: head + relation ≈ tail
 * becomes, on the ball:
 *
 *   score(h,r,t) = -d_c( mobius_add(h, r_vec), t )
 *
 * where r_vec lives in the tangent space at origin (relations as
 * tangent translations). Training: margin loss over corrupted triples.
 *
 * The hyperbolic payoff: hierarchically-structured KGs (subclass-of
 * chains) get exponentially more expressive embeddings per dimension.
 */
#include "wubu_kgtrans.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float kt_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int wubu_kg_init(WubuKG* kg,int n_ent,int n_rel,int D,float c,unsigned seed){
    if(n_ent<2||n_rel<1||D<1)return -1;
    kg->n_ent=n_ent;kg->n_rel=n_rel;kg->D=D;kg->c=c;
    kg->ent=malloc(sizeof(float)*(size_t)n_ent*D);
    kg->rel=malloc(sizeof(float)*(size_t)n_rel*D);
    if(!kg->ent||!kg->rel)return -2;
    unsigned rs=seed*374761393u+29u;
    for(int i=0;i<n_ent*D;i++){
        rs=rs*1103515245u+12345u;
        kg->ent[i]=((float)((rs>>16)%2000))/40000.0f-0.025f;
    }
    for(int i=0;i<n_rel*D;i++){
        rs=rs*1103515245u+12345u;
        kg->rel[i]=((float)((rs>>16)%2000))/20000.0f-0.05f;
    }
    return 0;
}
void wubu_kg_free(WubuKG* kg){free(kg->ent);free(kg->rel);}

float wubu_kg_score(const WubuKG* kg,int h,int r,int t){
    int D=kg->D;float c=kg->c;
    const float* he=kg->ent+(size_t)h*D;
    const float* rv=kg->rel+(size_t)r*D;
    const float* te=kg->ent+(size_t)t*D;
    /* translated = h ⊕ r_vec */
    float uu=0,vv=0,uv=0;
    for(int d=0;d<D;d++){uu+=he[d]*he[d];vv+=rv[d]*rv[d];uv+=he[d]*rv[d];}
    float cc=kg->c;
            float num1=1+2*cc*uv+cc*vv,num2=1-cc*uu;
    float den=1+2*c*uv+c*c*uu*vv;
    if(den<1e-10f)den=1e-10f;
    float moved[512];
    int dd=D<512?D:512;
    for(int d=0;d<dd&&d<512;d++)moved[d]=(num1*he[d]+num2*rv[d])/den;
    /* project into ball */
    float n2=0;
    for(int d=0;d<dd&&d<512;d++)n2+=moved[d]*moved[d];
    if(n2>1.0f/c){float s=sqrtf(1.0f/(c*n2));for(int d=0;d<dd&&d<512;d++)moved[d]*=s;}
    return -kt_dist(moved,te,D,c);
}

/* one margin-loss training pass over a triple list */
float wubu_kg_train_epoch(WubuKG* kg,const int* heads,const int* rels,
                           const int* tails,int n,float margin,float lr,
                           unsigned* seed){
    int D=kg->D;
    double total=0;
    for(int i=0;i<n;i++){
        int h=heads[i],r=rels[i],t=tails[i];
        /* corrupt tail */
        *seed=*seed*1103515245u+12345u;
        int tc=(*seed>>8)%kg->n_ent;
        if(tc==t)tc=(tc+1)%kg->n_ent;
        float sp=wubu_kg_score(kg,h,r,t);      /* positive */
        float sn=wubu_kg_score(kg,h,r,tc);     /* negative */
        float loss=margin+sn-sp;
        total+=loss>0?loss:0;
        if(loss>0){
            /* geodesic pull: move true tail toward translated head,
             * push corrupt tail away from it */
            float he2[512],rv[512],moved[512];
            int dd=D<512?D:512;
            memcpy(he2,kg->ent+(size_t)h*D,sizeof(float)*dd);
            memcpy(rv,kg->rel+(size_t)r*D,sizeof(float)*dd);
            float uu=0,vv=0,uv=0;
            for(int d=0;d<dd&&d<512;d++){uu+=he2[d]*he2[d];vv+=rv[d]*rv[d];uv+=he2[d]*rv[d];}
            float cc=kg->c;
            float num1=1+2*cc*uv+cc*vv,num2=1-cc*uu;
            float mden=1+2*cc*uv+cc*cc*uu*vv;
            if(mden<1e-10f)mden=1e-10f;
            for(int d=0;d<dd&&d<512;d++)moved[d]=(num1*he2[d]+num2*rv[d])/mden;
            float f=lr*0.1f;
            for(int d=0;d<dd&&d<512;d++){
                kg->ent[(size_t)t*D+d]+=f*(moved[d]-kg->ent[(size_t)t*D+d]);
                kg->ent[(size_t)tc*D+d]-=f*(moved[d]-kg->ent[(size_t)tc*D+d]);
            }
            /* reproject all touched entities into ball */
            for(int e=0;e<kg->n_ent;e++){
                float n2=0;
                for(int d=0;d<D&&d<512;d++)n2+=kg->ent[(size_t)e*D+d]*kg->ent[(size_t)e*D+d];
                if(n2>1.0f/kg->c){
                    float s=sqrtf(1.0f/(kg->c*n2));
                    for(int d=0;d<D&&d<512;d++)kg->ent[(size_t)e*D+d]*=s;
                }
            }
        }
    }
    return (float)(total/n);
}

int wubu_kg_rank(const WubuKG* kg,int h,int r,int t){
    /* rank of true tail among all entities by score (lower = better) */
    float sp=wubu_kg_score(kg,h,r,t);
    int rank=1;
    for(int e=0;e<kg->n_ent;e++){
        if(e==t)continue;
        if(wubu_kg_score(kg,h,r,e)>sp)rank++;
    }
    return rank;
}
