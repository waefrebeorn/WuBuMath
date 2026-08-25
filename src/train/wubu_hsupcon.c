/*
 * wubu_hsupcon.c -- GAP-D031: Hyperbolic supervised contrastive loss
 * (SupCon on the ball with label-weighted positives)
 *
 * Research source: Khosla et al. 2020 (SupCon) + hyperbolic contrastive
 * variants (Yue 2023, arXiv:2302.01409). For embeddings z_i on the ball
 * with labels y_i:
 *
 *   L_i = -1/|P(i)| * Σ_{p∈P(i)} log( exp(-d(z_i,z_p)/tau) /
 *                                   Σ_{a≠i} exp(-d(z_i,z_a)/tau) )
 *
 * Positives = same-label samples. Hyperbolic twist: weight positives by
 * LABEL SIMILARITY — same class w=1, related classes w<1 (partial
 * credit for hierarchy), so the loss respects tree structure.
 */
#include "wubu_hsupcon.h"
#include <math.h>
#include <stdlib.h>

static float sc_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

float wubu_hsc_loss(const float* z,const int* labels,
                     const float* label_sim,int n_labels,
                     int N,int D,float c,float tau){
    if(N<2)return 0;
    double total=0;
    int n_with_pos=0;
    for(int i=0;i<N;i++){
        /* positive set: same label */
        int P[512],np=0;
        for(int p=0;p<N&&np<512;p++)
            if(p!=i&&labels[p]==labels[i])P[np++]=p;
        if(np==0)continue;
        n_with_pos++;

        /* denominator: all a != i */
        double denom=0;
        float logits[512];
        for(int a=0;a<N;a++){
            if(a==i){logits[a]=-1e30f;continue;}
            logits[a]=-sc_dist(z+(size_t)i*D,z+(size_t)a*D,D,c)/tau;
            denom+=exp((double)logits[a]);
        }
        if(denom<=0)continue;

        /* weighted average over positives */
        double pos_sum=0,w_sum=0;
        for(int k=0;k<np;k++){
            int p=P[k];
            float w=1.0f;
            if(label_sim&&labels[i]<n_labels&&labels[p]<n_labels)
                w=label_sim[(size_t)labels[i]*n_labels+labels[p]];
            pos_sum+=w*log(exp((double)logits[p])/denom);
            w_sum+=w;
        }
        total+=-pos_sum/(w_sum>0?w_sum:1);
    }
    return n_with_pos>0?(float)(total/n_with_pos):0;
}
