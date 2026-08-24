/*
 * wubu_poincare_emb.c -- GAP-D017: Poincaré embedding layer with
 * negative-sampling loss (Nickel & Kiela NeurIPS 2017)
 *
 * The original hyperbolic embedding training recipe:
 *   L = -log( exp(-d(u,v))/Σ_{n∈N(u)} exp(-d(u,n)) )
 * where u=positive target, N(u)=negative samples, d = geodesic distance.
 * Riemannian-projected SGD (we use our C023 boundary-guarded step).
 */
#include "wubu_poincare_emb.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>

static float pe_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

/* boundary-safe rSGD step (from C023 recipe) */
static void pe_rsgd(float* x,const float* grad,int D,float lr){
    for(int d=0;d<D;d++)x[d]-=lr*grad[d];
    float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
    if(n2>0.99998f){float s=sqrtf(0.99998f/n2);for(int d=0;d<D;d++)x[d]*=s;}
}

int wubu_pe_init(WubuPEmb* pe,int n_items,int D,float c,float lr,unsigned seed){
    if(n_items<2||D<1)return -1;
    pe->n=n_items;pe->D=D;pe->c=c;pe->lr=lr;
    pe->emb=malloc(sizeof(float)*(size_t)n_items*D);
    if(!pe->emb)return -2;
    unsigned rs=seed*2246822519u;
    /* tiny init near origin */
    for(int i=0;i<n_items*D;i++){
        rs=rs*1103515245u+12345u;
        pe->emb[i]=(float)((rs>>16)%2000)/200000.0f-0.005f;
    }
    return 0;
}
void wubu_pe_free(WubuPEmb* pe){free(pe->emb);pe->emb=NULL;}

float wubu_pe_train_edge(WubuPEmb* pe,int u,int v,
                          const int* negatives,int n_neg){
    int D=pe->D;
    const float* pu=pe->emb+(size_t)u*D;
    const float* pv=pe->emb+(size_t)v*D;
    float d_pos=pe_dist(pu,pv,D,pe->c);

    /* denominator: exp(-d_pos) + sum exp(-d_neg) */
    float z=expf(-d_pos);
    float neg_scores[64];
    int nn=n_neg<64?n_neg:64;
    for(int k=0;k<nn;k++){
        neg_scores[k]=pe_dist(pu,pe->emb+(size_t)negatives[k]*D,D,pe->c);
        z+=expf(-neg_scores[k]);
    }
    /* loss = -log( exp(-d_pos)/z ) = log(z) + d_pos */
    float loss=logf(z)+d_pos;
    int dd=D<64?D:64;

    /* Nickel-Kiela practical gradient: Euclidean proxy direction
     * (x-v)/d scaled by sigmoid weight, Riemannian-projected step. */
    {
        float sig=1/(1+expf(d_pos));
        float g[D>64?64:64];
        int dd=D<64?D:64;
        for(int d=0;d<dd;d++)g[d]=-sig*(pv[d]-pu[d])/ (d_pos+1e-6f);
        pe_rsgd((float*)pu,g,dd,pe->lr);

        float gv[64];
        for(int d=0;d<dd;d++)gv[d]=sig*(pv[d]-pu[d])/(d_pos+1e-6f);
        pe_rsgd((float*)pv,gv,dd,pe->lr);
    }
    /* push negatives away */
    for(int k=0;k<nn;k++){
        int ni=negatives[k];
        if(ni==v||ni==u)continue;
        float dn=neg_scores[k];
        if(dn<=0)dn=1e-3f;
        float sig_neg=1/(1+expf(-dn));   /* want dn large → loss small when far */
        float* pn=pe->emb+(size_t)ni*D;
        float gn[64];
        for(int d=0;d<dd&&d<D;d++)
            gn[d]=sig_neg*(pn[d]-pu[d])/(dn+1e-6f);
        pe_rsgd(pn,gn,dd,pe->lr);
    }
    return loss;
}
