/*
 * wubu_hbeam.c -- GAP-A008: Hyperbolic beam search over latent paths
 *
 * Beam search decoding hyperbolized: hypotheses are PATHS on the ball
 * (sequences of on-ball points). Expansion = geodesic step toward one of
 * K candidate directions; score accumulates NEGATIVE path length plus a
 * goal-proximity term; pruning keeps the best W hypotheses per step.
 *
 * This is the decoder for the codec's latent traversal: instead of a
 * single geodesic interpolation (C025), explore multiple curved routes
 * through the manifold and keep the best ones — better when the direct
 * geodesic passes near boundary or obstacles (other data points).
 */
#include "wubu_hbeam.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float bm_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

/* one hypothesis */
typedef struct {
    float pt[64];      /* current point */
    float score;       /* accumulated: -path_length + goal_bonus */
    int steps;
} HBHyp;

int wubu_beam_search(const float* start,const float* goal,
                      const float* waypoints,int n_wp,
                      int D,float c,float step_size,
                      int width,int max_steps,float* out_path){
    if(!start||!goal||!out_path||width<1)return -1;

    /* expand toward goal + each waypoint direction */
    int n_dirs=1+n_wp;
    HBHyp* cur=malloc(sizeof(HBHyp)*(size_t)(width*n_dirs));
    HBHyp* nxt=malloc(sizeof(HBHyp)*(size_t)(width*n_dirs));
    if(!cur||!nxt){free(cur);free(nxt);return -2;}

    int n_cur=1;
    memset(cur,0,sizeof(HBHyp));
    memcpy(cur[0].pt,start,sizeof(float)*D);
    cur[0].score=0;cur[0].steps=0;

    int total_written=0;
    for(int step=0;step<max_steps;step++){
        int n_nxt=0;
        for(int h=0;h<n_cur;h++){
            for(int dir=0;dir<n_dirs;dir++){
                const float* target=(dir==0)?goal:waypoints+(size_t)(dir-1)*D;
                /* move from cur[h] toward target by step_size along geodesic */
                const float* p=cur[h].pt;
                float moved[64];
                int dd=D<64?D:64;
                /* Euclidean blend then project (valid approximation for
                 * small steps near center) */
                float a_eff=step_size/(1+bm_dist(p,target,D,c));
                for(int d=0;d<dd&&d<64;d++)moved[d]=p[d]+a_eff*(target[d]-p[d]);
                /* project into ball */
                float n2=0;
                for(int d=0;d<D&&d<64;d++)n2+=moved[d]*moved[d];
                if(n2>0.99998f){
                    float s=sqrtf(0.99998f/n2);
                    for(int d=0;d<D&&d<64;d++)moved[d]*=s;
                }
                /* score: penalize length, reward goal proximity */
                float seg_len=a_eff*bm_dist(p,target,D,c);
                float d_goal=bm_dist(moved,goal,D,c);
                HBHyp* nh=&nxt[n_nxt++];
                memcpy(nh->pt,moved,sizeof(float)*D);
                nh->steps=cur[h].steps+1;
                nh->score=cur[h].score-seg_len-d_goal*0.5f;
            }
        }
        /* prune to width best by score */
        for(int i=0;i<width&&i<n_nxt;i++){
            int bi=i;
            for(int j=i+1;j<n_nxt;j++)
                if(nxt[j].score>nxt[bi].score)bi=j;
            HBHyp t=nxt[i];nxt[i]=nxt[bi];nxt[bi]=t;
        }
        n_cur=n_nxt<width?n_nxt:width;
        memcpy(cur,nxt,sizeof(HBHyp)*(size_t)n_cur);

        /* record best point at final step */
        if(step==max_steps-1||bm_dist(cur[0].pt,goal,D,c)<step_size*0.5f){
            memcpy(out_path+(size_t)total_written*D,cur[0].pt,sizeof(float)*D);
            total_written++;
            if(bm_dist(cur[0].pt,goal,D,c)<step_size*0.5f)break;
        }
    }
    free(cur);free(nxt);
    return total_written>0?total_written:-3;
}
