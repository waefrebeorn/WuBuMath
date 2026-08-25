/*
 * wubu_temp_learn.c -- GAP-D007: Learned temperature for contrastive loss
 *
 * CLIP's tau is a fixed hyperparameter. Learning it (as MERU does)
 * lets the model adapt its sharpness: early training wants soft targets
 * (high tau), later wants hard discrimination (low tau).
 */
#include "wubu_temp_learn.h"
#include <stdlib.h>
#include <math.h>

float wubu_tl_logit(const float* a,const float* b,int D,float c,float log_tau){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    float dist=acoshf(1+2*c*ab2/den)/sqrtf(c);
    return -dist/expf(log_tau);
}

/* InfoNCE with learnable log_tau; returns loss and updated log_tau */
float wubu_tl_infonce(const float* anchor,const float* positive,
                       const float* negatives,int n_neg,
                       int D,float c,float* log_tau,float lr){
    /* positive logit */
    float lp=wubu_tl_logit(anchor,positive,D,c,*log_tau);
    /* denominator: exp(lp) + sum of negatives */
    double denom=exp((double)lp);
    float max_neg=-1e30f;
    float neg_logits[64];
    int nn=n_neg<64?n_neg:64;
    for(int k=0;k<nn;k++){
        neg_logits[k]=wubu_tl_logit(anchor,negatives+(size_t)k*D,D,c,*log_tau);
        if(neg_logits[k]>max_neg)max_neg=neg_logits[k];
    }
    for(int k=0;k<nn;k++)denom+=exp((double)neg_logits[k]);
    double loss=-log(exp((double)lp)/denom);

    /* proper gradient: dL/dlogTau = sum_j softmax_j * dlogit_j/dlogTau
     * where dlogit/dlogTau = |logit| (since logit=-d/exp(lt), d/tau=-logit)
     * dL/dlt = sum_j p_j*(-logit_j) + logit_p  (only positive contributes to numerator) */
    double sm_pos=exp((double)lp)/denom;
    float grad=0;
    grad+=(float)(-sm_pos*lp);   /* numerator term */
    grad+=(float)(sm_pos*lp);    /* cancels! */
    /* negatives in denominator */
    for(int k=0;k<nn;k++){
        double sm_neg=exp((double)neg_logits[k])/denom;
        grad+=(float)(sm_neg*neg_logits[k]);
    }
    grad+=lp;   /* from positive's own denominator contribution */
    /* actually the full derivation gives:
     * dL/dlt = -(1-p_pos)*|lp| + sum_k p_k*|lneg_k| */
    float p_pos=(float)(exp((double)lp)/denom);
    grad=-((1.0f-p_pos)*(float)fabs((double)lp))+(float)(p_pos*0);/* neg terms */
    for(int k=0;k<nn;k++){
        double smk=exp((double)neg_logits[k])/denom;
        grad+=(float)(smk*(double)fabs(neg_logits[k]));
    }
    *log_tau-=grad*lr*0.001f;    return (float)loss;
}
