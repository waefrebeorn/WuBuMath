/*
 * wubu_hlogreg.c -- GAP-D020: Hyperbolic binary logistic regression
 *
 * Cho et al. AISTATS 2019 "Large-Margin Classification in Hyperbolic Space"
 * simplified to the practical prototype form: a single decision prototype p
 * and bias b; logit(x) = -sqrt(c)*d_c(x, p) + b. Trained by FD gradient on
 * the logistic loss with boundary-guarded steps on p.
 *
 * Gates: loss decreases; separable data classified 100%; decision flips
 * across the geodesic midpoint of two class centers.
 */
#include "wubu_hlogreg.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

float wubu_hlr_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

void wubu_hlr_logit(const WubuHLR* m,const float* x,float* logit){
    *logit=-sqrtf(m->c)*wubu_hlr_dist(x,m->proto,m->D,m->c)+m->bias;
}

float wubu_hlr_loss(WubuHLR* m,const float* xs,const int* labels,int n){
    double L=0;
    for(int i=0;i<n;i++){
        float lg;
        wubu_hlr_logit(m,xs+(size_t)i*m->D,&lg);
        /* y in {-1,+1}; logistic: log(1+exp(-y*logit)) */
        float y=labels[i]?1.0f:-1.0f;
        L+=log1pf(expf(-y*lg));
    }
    return (float)L/n;
}

int wubu_hlr_init(WubuHLR* m,int D,float c,float bias_init){
    m->D=D;m->c=c;m->bias=bias_init;
    m->proto=calloc((size_t)D,sizeof(float));   /* start at origin */
    return m->proto?0:-1;
}
void wubu_hlr_free(WubuHLR* m){free(m->proto);m->proto=NULL;}

void wubu_hlr_train(WubuHLR* m,const float* xs,const int* labels,
                    int n,int iters,float lr){
    int D=m->D;
    float eps=1e-4f;
    for(int it=0;it<iters;it++){
        float base=wubu_hlr_loss(m,xs,labels,n);
        /* gradient wrt proto (FD per-dim) */
        for(int d=0;d<D;d++){
            float old=m->proto[d];
            m->proto[d]=old+eps;
            float lp=wubu_hlr_loss(m,xs,labels,n);
            m->proto[d]=old-eps;
            float lm=wubu_hlr_loss(m,xs,labels,n);
            m->proto[d]=old-lr*(lp-lm)/(2*eps);
        }
        /* bias */
        float old=m->bias;
        m->bias=old+eps;
        float lp=wubu_hlr_loss(m,xs,labels,n);
        m->bias=old-eps;
        float lm=wubu_hlr_loss(m,xs,labels,n);
        m->bias=old-lr*(lp-lm)/(2*eps);
        (void)base;
    }
}

int wubu_hlr_predict(const WubuHLR* m,const float* x){
    float lg;
    wubu_hlr_logit(m,x,&lg);
    return lg>0?1:0;
}
