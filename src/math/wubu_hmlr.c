/*
 * wubu_hmlr.c -- GAP-C019: Hyperbolic multinomial logistic regression
 *
 * Prototype-based Poincare-ball MLR (the practical simplification of
 * Ganea et al. 2018's hyperplane-distance MLR):
 *   logit_k = -(sqrt(c) * d_c(x, p_k)) * z_k
 * where p_k = class prototype (on-ball), z_k = learnable scale per class.
 * prob_k = softmax(logit). Distance-based logits preserve the hierarchical
 * geometry; prototype form is stable and sufficient for our gates.
 */
#include "wubu_hmlr.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

float wubu_hmlr_distance(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1.0f-c*a2)*(1.0f-c*b2);
    if(den<1e-9f)den=1e-9f;
    float arg=1.0f+2.0f*c*ab2/den;
    return acoshf(arg>1.0f?arg:1.0f)/sqrtf(c);
}

int wubu_hmlr_init(WubuHMLR* m,int num_classes,int D,float c){
    if(num_classes<2||D<1)return -1;
    m->K=num_classes;m->D=D;m->c=c;
    m->proto=malloc(sizeof(float)*(size_t)num_classes*D);
    m->z=malloc(sizeof(float)*(size_t)num_classes);
    if(!m->proto||!m->z)return -2;
    for(int k=0;k<num_classes;k++){
        /* prototypes spread inside the ball at increasing radius */
        float r=0.1f+0.5f*(float)k/num_classes;
        for(int d=0;d<D;d++){
            unsigned s=2166136261u^(unsigned)k^(unsigned)(d*2654435761u);
            s=s*16777619u+(unsigned)d;
            m->proto[(size_t)k*D+d]=((float)(s%1000)/500.0f-1.0f)*0.3f*r;
        }
        /* renormalize to radius r */
        float n2=0;for(int d=0;d<D;d++)n2+=m->proto[(size_t)k*D+d]*m->proto[(size_t)k*D+d];
        float n=sqrtf(n2);if(n>1e-9f)for(int d=0;d<D;d++)m->proto[(size_t)k*D+d]*=r/n;
        m->z[k]=1.0f;
    }
    return 0;
}
void wubu_hmlr_free(WubuHMLR* m){free(m->proto);free(m->z);m->proto=NULL;m->z=NULL;}
static int step_hint_dummy;

void wubu_hmlr_logits(const WubuHMLR* m,const float* x,float* logits){
    for(int k=0;k<m->K;k++)
        logits[k]=-m->c*m->z[k]*wubu_hmlr_distance(x,m->proto+(size_t)k*m->D,m->D,m->c);
}

void wubu_hmlr_softmax(const float* logits,int K,float* probs){
    float mx=-1e30f;
    for(int k=0;k<K;k++)if(logits[k]>mx)mx=logits[k];
    float z=0;
    for(int k=0;k<K;k++){probs[k]=expf(logits[k]-mx);z+=probs[k];}
    for(int k=0;k<K;k++)probs[k]/=z;
}

/* FD training step: minimize CE loss over prototypes + scales */
float wubu_hmlr_train_step(WubuHMLR* m,const float* xs,const int* labels,
                           int N,int lr){
    int K=m->K,D=m->D;
    /* data-driven init: on the FIRST call only, seed each prototype as the
     * mean of its labeled samples — gives FD a meaningful starting basin. */
    if(m->step_hint==0 && N>0){
        memset(m->proto,0,sizeof(float)*(size_t)m->K*m->D);
        int* cnt=calloc((size_t)m->K,sizeof(int));
        for(int i=0;i<N;i++){int cl=labels[i];cnt[cl]++;
            for(int d=0;d<D;d++)m->proto[(size_t)cl*D+d]+=xs[(size_t)i*D+d];}
        for(int k=0;k<K;k++)if(cnt[k]>0)
            for(int d=0;d<D;d++)m->proto[(size_t)k*D+d]/=cnt[k];
        free(cnt);
        m->step_hint=1;
    }
    /* current loss */
    #define HMLR_LOSS() do{ \
        double L=0; \
        for(int i=0;i<N;i++){ \
            float lg[64]; wubu_hmlr_logits(m,xs+(size_t)i*D,lg); \
            float mn=-1e30f; for(int k=0;k<K;k++)if(lg[k]>mn)mn=lg[k]; \
            float zz=0; for(int k=0;k<K;k++)zz+=expf(lg[k]-mn); \
            L-=log((expf(lg[labels[i]]-mn)+1e-12)/(zz+1e-12)); \
        } \
        cur=(float)L/N; \
    }while(0)
    float cur;
    HMLR_LOSS();
    float eps=1e-3f;
    size_t np=(size_t)K*D;
    for(size_t k=0;k<np;k++){
        float old=m->proto[k];
        m->proto[k]=old+eps; HMLR_LOSS(); float lp=cur;
        m->proto[k]=old-eps; HMLR_LOSS(); float lm=cur;
        m->proto[k]=old-lr*(lp-lm)/(2*eps);
    }
    for(int k2=0;k2<K;k2++){
        float old=m->z[k2];
        m->z[k2]=old+eps; HMLR_LOSS(); float lp=cur;
        m->z[k2]=old-eps; HMLR_LOSS(); float lm=cur;
        float nw=old-lr*(lp-lm)/(2*eps);
        m->z[k2]=(nw>0.05f)?nw:0.05f;
    }
    HMLR_LOSS();
    #undef HMLR_LOSS
    return cur;
}
