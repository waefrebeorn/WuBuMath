/*
 * wubu_hbandit.c -- GAP-H012: Hyperbolic contextual bandit
 * (Thompson sampling on the Poincaré ball)
 *
 * The BearRL connection: contexts embedded on the ball; each action has
 * a prototype on the ball. Reward model per action: Beta(alpha,beta)
 * where the posterior is updated by geodesic-proximity-weighted counts:
 *
 *   weight = exp(-d(context, proto_action)/tau)
 *
 * Actions with prototypes near the current context get their Beta
 * posterior updated strongly; far ones barely move. Thompson sampling:
 * draw theta_a ~ Beta(alpha_a, beta_a), pick argmax.
 */
#include "wubu_hbandit.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float hb_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int wubu_hb_init(WubuHB* b,int n_actions,int D,float tau,unsigned seed){
    if(n_actions<1||D<1)return -1;
    b->n=n_actions;b->D=D;b->tau=tau;b->seed=seed;
    b->proto=malloc(sizeof(float)*(size_t)n_actions*D);
    b->alpha=malloc(sizeof(float)*(size_t)n_actions);
    b->beta=malloc(sizeof(float)*(size_t)n_actions);
    if(!b->proto||!b->alpha||!b->beta)return -2;
    unsigned rs=seed*374761393u+13u;
    for(int i=0;i<n_actions*D;i++){
        rs=rs*1103515245u+12345u;
        b->proto[i]=(float)((rs>>16)%2000)/20000.0f-0.05f;
    }
    for(int a=0;a<n_actions;a++){b->alpha[a]=1.0f;b->beta[a]=1.0f;}
    return 0;
}
void wubu_hb_free(WubuHB* b){free(b->proto);free(b->alpha);free(b->beta);}

int wubu_hb_select(WubuHB* b,const float* context,float c){
    int best=-1;float best_theta=-1;
    for(int a=0;a<b->n;a++){
        /* Beta sample via two uniforms (Cheng's approximation simplified) */
        b->seed=b->seed*1103515245u+12345u;
        float u1=((b->seed>>16)%10000)/10000.0f;if(u1<1e-6f)u1=1e-6f;
        b->seed=b->seed*1103515245u+12345u;
        float u2=((b->seed>>16)%10000)/10000.0f;if(u2<1e-6f)u2=1e-6f;
        /* gamma-ish approximation: use mean + noise scaled by uncertainty */
        float mean=b->alpha[a]/(b->alpha[a]+b->beta[a]);
        float var=b->alpha[a]*b->beta[a]/
                  ((b->alpha[a]+b->beta[a])*(b->alpha[a]+b->beta[a])*(b->alpha[a]+b->beta[a]+1));
        float noise=sqrtf(-2*logf(u1))*cosf(2*3.14159265f*u2)*sqrtf(var);
        float theta=mean+noise;
        if(theta>best_theta){best_theta=theta;best=a;}
    }
    return best;
}

void wubu_hb_update(WubuHB* b,int action,const float* context,
                    float reward,float c){
    if(action<0||action>=b->n)return;
    /* proximity-weighted update strength */
    float d=hb_dist(context,b->proto+(size_t)action*b->D,b->D,c);
    float w=expf(-d/b->tau);
    if(w<0.01f)w=0.01f;   /* floor: always learn a little */
    if(reward>0.5f)b->alpha[action]+=w;
    else           b->beta[action]+=w;
}

float wubu_hb_mean_reward(const WubuHB* b,int action){
    if(action<0||action>=b->n)return -1;
    return b->alpha[action]/(b->alpha[action]+b->beta[action]);
}
