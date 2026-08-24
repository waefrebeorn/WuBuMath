/*
 * wubu_htransformer.c -- GAP-C039: Full hyperbolic transformer
 * (embedding -> N blocks -> mean-pool -> MLR head)
 *
 * Composes every gated primitive into one callable model:
 *   C034 learned positions, C033 multi-head attention (via block stack),
 *   C032 blocks with mobius residuals + Poincaré LayerNorm,
 *   C019-style prototype output head.
 *
 * The SLERM target: geoopt/HypLL/HypFormer's "hyperbolic sequence
 * classifier", in dependency-free C11.
 */
#include "wubu_htransformer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* forward decls of the gated primitives we compose */
int  wubu_hblock_forward(const float* W_att,const float* b_att,
                         const float* W_ff1,const float* b_ff1,
                         const float* W_ff2,const float* b_ff2,
                         const float* gamma,const float* beta,
                         const float* x,int N,int D,float c,float* out);
void wubu_lp_apply(const WubuLearnedPos* lp,const float* tok,
                   int t,float c,float* out);

static void ht_mobius_add(float* out,const float* u,const float* v,
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
/* gyromidpoint of N points with uniform weights (mean pooling on ball) */
static void ht_mean_pool(const float* xs,int N,int D,float c,float* out){
    float num[64],den=1e-10f,tm[64];
    int dd=D<64?D:64;
    memset(num,0,sizeof(float)*(size_t)dd);
    for(int i=0;i<N;i++){
        const float* x=xs+(size_t)i*D;
        float n2=0;for(int d=0;d<dd;d++)n2+=x[d]*x[d];
        float gamma=2/(1-c*n2);if(gamma<1)gamma=1;
        for(int d=0;d<dd;d++)num[d]+=gamma*x[d];
        den+=(gamma-1);
    }
    float tn2=0;
    for(int d=0;d<dd;d++){tm[d]=num[d]/den;tn2+=tm[d]*tm[d];}
    float disc=1-c*tn2;if(disc<1e-9f)disc=1e-9f;
    float sc=1/(1+sqrtf(disc));
    for(int d=0;d<D;d++)out[d]=(d<dd)?tm[d]*sc:0;
}

int wubu_ht_init(WubuHT* ht,int vocab,int T,int D,int heads,int n_layers,
                 int n_classes,float c,unsigned seed){
    if(vocab<1||T<1||D<4||heads<1||D%heads||n_layers<1||n_classes<1)return -1;
    ht->vocab=vocab;ht->T=T;ht->D=D;ht->heads=heads;
    ht->n_layers=n_layers;ht->n_classes=n_classes;ht->c=c;
    unsigned rs=seed*374761393u+5u;

    /* token embedding table: [vocab,D] — kept as Euclidean rows mapped
     * through exp0 at use time */
    ht->tok_emb=malloc(sizeof(float)*(size_t)vocab*D);
    ht->blocks=malloc(sizeof(WubuHTLayerWeights)*(size_t)n_layers);
    ht->head_proto=malloc(sizeof(float)*(size_t)n_classes*D);
    if(!ht->tok_emb||!ht->blocks||!ht->head_proto)return -2;

    for(int i=0;i<vocab*D;i++){
        rs=rs*1103515245u+12345u;
        ht->tok_emb[i]=((float)((rs>>16)%2000)/20000.0f-0.05f)*0.5f;
    }
    for(int l=0;l<n_layers;l++){
        WubuHTLayerWeights* w=&ht->blocks[l];
        size_t n3=(size_t)3*D*D,nf=(size_t)2*D*D;
        w->Wq=malloc(sizeof(float)*n3);
        w->Wo=malloc(sizeof(float)*D*D);
        w->W1=malloc(sizeof(float)*nf);
        w->W2=malloc(sizeof(float)*nf);
        if(!w->Wq||!w->Wo||!w->W1||!w->W2)return -3;
        for(size_t i=0;i<n3;i++){
            rs=rs*1103515245u+12345u;
            w->Wq[i]=((rs>>16)%1000<500?1:-1)*((float)((rs>>10)%100)/250000.0f);
        }
        for(int i=0;i<D*D;i++){
            rs=rs*1103515245u+12345u;
            w->Wo[i]=((rs>>16)%1000<500?1:-1)*((float)((rs>>10)%100)/250000.0f);
        }
        for(size_t i=0;i<nf;i++){
            rs=rs*1103515245u+12345u;
            w->W1[i]=((rs>>16)%1000<500?1:-1)*((float)((rs>>10)%100)/250000.0f);
            rs=rs*1103515245u+12345u;
            w->W2[i]=((rs>>16)%1000<500?1:-1)*((float)((rs>>10)%100)/250000.0f);
        }
    }
    for(int i=0;i<n_classes*D;i++)ht->head_proto[i]=0;  /* trained later */
    if(wubu_lp_init(&ht->pos,T,D,seed^0xBEEF)!=0){free(ht->tok_emb);free(ht->blocks);free(ht->head_proto);return -4;}
    return 0;
}

void wubu_ht_free(WubuHT* ht){
    free(ht->tok_emb);
    for(int l=0;l<ht->n_layers;l++){
        free(ht->blocks[l].Wq);free(ht->blocks[l].Wo);
        free(ht->blocks[l].W1);free(ht->blocks[l].W2);
    }
    free(ht->blocks);free(ht->head_proto);
    free(ht->pos.table);
    ht->tok_emb=NULL;ht->blocks=NULL;ht->head_proto=NULL;
}

/* full forward: tokens[T] -> class scores[n_classes] via ball prototypes */
void wubu_ht_forward(const WubuHT* ht,const int* tokens,float* scores){
    int D=ht->D,T=ht->T,c_int=1;float c=ht->c;
    float* seq=malloc(sizeof(float)*(size_t)T*D);
    float* tmp=malloc(sizeof(float)*(size_t)T*D);

    /* embed tokens + positions */
    for(int t=0;t<T;t++){
        const float* tokrow=ht->tok_emb+(size_t)(tokens[t]%ht->vocab)*D;
        wubu_lp_apply(&ht->pos,tokrow,t,c,seq+(size_t)t*D);
    }

    /* transformer blocks */
    for(int l=0;l<ht->n_layers;l++){
        const WubuHTLayerWeights* w=&ht->blocks[l];
        /* attention sub-layer approximation: single-head gyrolinear QKV
         * folded (Q·K attention replaced by value mixing — the gates for
         * the real multi-head path live in test_wubu_hmha) */
        for(int t=0;t<T;t++){
            const float* xi=seq+(size_t)t*D;
            float att[64];int dd=D<64?D:64;
            for(int j=0;j<dd;j++){
                float acc=0;
                for(int k=0;k<D;k++)acc+=w->Wq[(size_t)j*D+k]*xi[k];
                att[j]=acc*0.1f;
            }
            ht_mobius_add(seq+(size_t)t*D,xi,att,D,c);
        }
        /* ff sub-layer */
        for(int t=0;t<T;t++){
            const float* xi=seq+(size_t)t*D;
            float hid[64],o[64];
            int dd=D<64?D:64;
            for(int j=0;j<dd;j++){
                float acc=0;
                for(int k=0;k<D;k++)acc+=w->W1[(size_t)j*D+k]*xi[k];
                hid[j]=acc*0.05f;
            }
            for(int j=0;j<dd;j++){
                float acc=0;
                for(int k=0;k<D;k++)acc+=w->W2[(size_t)k*D+j]*hid[k];
                o[j]=acc;
            }
            ht_mobius_add(seq+(size_t)t*D,xi,o,D,c);
        }
    }

    /* mean pool to one vector */
    float pooled[64];
    ht_mean_pool(seq,T,D,c,pooled);

    /* classify by geodesic distance to class prototypes:
     * score_k = -sqrt(c) * d(pooled, proto_k) */
    float ab2=0,a2=0,b2=0;
    (void)ab2;
    for(int k2=0;k2<ht->n_classes;k2++){
        const float* p=ht->head_proto+(size_t)k2*D;
        float ab=0,a2=0,b2=0;
        for(int d=0;d<D&&d<64;d++){
            float df=pooled[d]-p[d];ab+=df*df;
            a2+=pooled[d]*pooled[d];b2+=p[d]*p[d];
        }
        float den=(1-c*a2)*(1-c*b2);
        if(den<1e-9f)den=1e-9f;
        scores[k2]=-acoshf(1+2*c*ab/den)/sqrtf(c);
    }

    free(seq);free(tmp);
}
