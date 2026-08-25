/*
 * wubu_hreplay.c -- GAP-H013: Hyperbolic prioritized experience replay
 *
 * PER (Schaul et al. 2016) hyperbolized: transitions' states live on the
 * ball; priority blends TD-error with a geodesic DIVERSITY bonus —
 * samples far from everything already in the buffer carry more
 * information (novel region of the manifold). Sampling is
 * priority-proportional.
 *
 *   prio_i = |td_error_i| + eps + diversity_bonus_i
 *   diversity_bonus_i = min_j d(state_i, state_j) over stored states
 */
#include "wubu_hreplay.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float rp_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int wubu_rp_init(WubuRP* r,int cap,int D,float c,float div_weight){
    r->cap=cap;r->len=0;r->head=0;r->D=D;r->c=c;
    r->div_w=div_weight;
    r->states=malloc(sizeof(float)*(size_t)cap*D);
    r->prio=calloc((size_t)cap,sizeof(float));
    if(!r->states||!r->prio)return -1;
    return 0;
}
void wubu_rp_free(WubuRP* r){free(r->states);free(r->prio);}

void wubu_rp_add(WubuRP* r,const float* state,float td_error){
    int slot=r->head;
    memcpy(r->states+(size_t)slot*r->D,state,sizeof(float)*r->D);

    /* diversity: distance to nearest stored state (excluding self-slot) */
    float mind=1e30f;
    for(int i=0;i<r->len;i++){
        if(i==slot)continue;
        float d=rp_dist(state,r->states+(size_t)i*r->D,r->D,r->c);
        if(d<mind)mind=d;
    }
    if(r->len==0)mind=1.0f;   /* first sample: max novelty */

    r->prio[slot]=fabsf(td_error)+1e-4f+r->div_w*mind;
    r->head=(r->head+1)%r->cap;
    if(r->len<r->cap)r->len++;
}

int wubu_rp_sample(const WubuRP* r,unsigned* seed){
    if(r->len==0)return -1;
    /* total priority */
    double total=0;
    for(int i=0;i<r->len;i++)total+=r->prio[i];
    if(total<=0)return -1;
    *seed=*seed*1103515245u+12345u;
    float u=((float)((*seed>>16)%10000))/10000.0f;
    double target=u*total;
    double acc=0;
    for(int i=0;i<r->len;i++){
        acc+=r->prio[i];
        if(acc>=target)return i;
    }
    return r->len-1;
}

/* update priority after replay (new TD error) */
void wubu_rp_update_prio(WubuRP* r,int idx,float td_error){
    if(idx<0||idx>=r->len)return;
    /* recompute diversity vs current contents */
    float mind=1e30f;
    for(int i=0;i<r->len;i++){
        if(i==idx)continue;
        float d=rp_dist(r->states+(size_t)idx*r->D,
                        r->states+(size_t)i*r->D,r->D,r->c);
        if(d<mind)mind=d;
    }
    if(r->len<=1)mind=1.0f;
    r->prio[idx]=fabsf(td_error)+1e-4f+r->div_w*mind;
}
