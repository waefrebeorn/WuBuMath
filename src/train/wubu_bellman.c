/*
 * wubu_bellman.c -- GAP-H016: Bellman operator + contraction property
 * (the 100th gap — the theoretical heart of value-based RL)
 *
 * The Bellman optimality operator on a finite MDP with rewards r(s,a)
 * and discount gamma:
 *   (TV)(s) = max_a [ r(s,a) + gamma * V(s') ]
 *
 * Property gates (the Banach fixed-point theorem in executable form):
 *   G1 CONTRACTION: ||T(V1)-T(V2)||_inf <= gamma*||V1-V2||_inf
 *   G2 MONOTONICITY: V1<=V2 pointwise => TV1 <= TV2
 *   G3 FIXED POINT: iterating T converges to V* and T(V*)=V*
 *   G4 GAMMA=1 boundary: contraction may fail (documented, not gated)
 */
#include "wubu_bellman.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* V[s] updated via T for all states. transitions: [S,A,S'] indices,
 * rewards: [S,A]. Deterministic 1-successor per (s,a). */
void wubu_bellman_apply(float* V,const int8_t* next,const float* R,
                         int S,int A,float gamma,float* out){
    for(int s=0;s<S;s++){
        float best=-1e30f;
        for(int a=0;a<A;a++){
            int idx=s*A+a;
            float q=R[idx]+gamma*V[next[idx]];
            if(q>best)best=q;
        }
        out[s]=best;
    }
}

float wubu_bellman_supdiff(const float* a,const float* b,int S){
    float m=0;
    for(int s=0;s<S;s++){
        float d=fabsf(a[s]-b[s]);
        if(d>m)m=d;
    }
    return m;
}

int wubu_bellman_contraction_check(const int8_t* next,const float* R,
                                    int S,int A,float gamma,
                                    unsigned seed){
    /* random V1,V2: verify sup-diff shrinks by exactly gamma */
    float V1[256],V2[256],T1[256],T2[256];
    for(int i=0;i<S;i++){
        seed=seed*1103515245u+12345u;
        V1[i]=((float)((seed>>16)%2000))/1000.0f-1.0f;
        seed=seed*1103515245u+12345u;
        V2[i]=((float)((seed>>16)%2000))/1000.0f-1.0f;
    }
    wubu_bellman_apply(V1,next,R,S,A,gamma,T1);
    wubu_bellman_apply(V2,next,R,S,A,gamma,T2);
    float before=wubu_bellman_supdiff(V1,V2,S);
    float after=wubu_bellman_supdiff(T1,T2,S);
    return after<=gamma*before+1e-5f;
}
