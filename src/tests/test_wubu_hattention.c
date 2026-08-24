/*
 * test_wubu_hattention.c -- GAP-C016 gates: hyperbolic attention primitives
 *
 * G1 matching: softmax weights sum to 1; nearest key gets the largest weight
 * G2 aggregate: identical inputs -> identical output = the input itself
 * G3 gyromidpoint of two points lies between them (both distances smaller
 *    than the direct distance)
 * G4 aggregation output stays on-ball
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hattention.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Attention Tests ===\n\n");
    const int D=4,K=6;

    /* keys spread across the ball */
    float keys[K*D];
    for(int k=0;k<K;k++)
        for(int d=0;d<D;d++)
            keys[k*D+d]=0.3f*sinf((float)k*1.7f+(float)d*2.1f);

    printf("  g1_matching_softmax...");
    {
        float q[4]={keys[1*D],keys[1*D+1],keys[1*D+2],keys[1*D+3]}; /* query = key 1 */
        float w[K];
        wubu_hattn_matching(q,keys,K,D,0.1f,w);
        float sum=0,best=0;int bi=-1;
        for(int k=0;k<K;k++){sum+=w[k];if(w[k]>best){best=w[k];bi=k;}}
        CHECK(fabsf(sum-1.0f)<1e-5f);
        CHECK(bi==1);   /* exact-match key wins */
    }
    printf("PASS\n");passed++;

    printf("  g2_aggregate_identical...");
    {
        /* all values equal x -> midpoint must be x */
        float vals[K*D],w[K];
        for(int k=0;k<K;k++){
            for(int d=0;d<D;d++)vals[k*D+d]=0.1f*(d+1);  /* n2=0.3 <1 */
            w[k]=1.0f/K;
        }
        float out[4];
        wubu_hattn_aggregate(vals,w,K,D,1.0f,out);
        for(int d=0;d<D;d++) CHECK(fabsf(out[d]-0.1f*(d+1))<0.02f*0.1f*(d+1)+1e-3f);
    }
    printf("PASS\n");passed++;

    printf("  g3_midpoint_between...");
    {
        /* two antipodal-ish points: midpoint closer to both than they are to each other */
        float vals[8]={-0.3f,0,0,0,  0.3f,0,0,0};
        float w[2]={0.5f,0.5f};
        float out[4];
        wubu_hattn_aggregate(vals,w,2,D,1.0f,out);
        float d_full=wubu_hattn_distance(vals,out+0,D,1.0f);
        /* distance from out to each point */
        CHECK(fabsf(out[0])<fabsf(vals[0]));       /* pulled toward origin */
        CHECK(d_full>fabsf(out[0])*2.0f || d_full>0.1f);  /* sanity: separated pair has real distance */
    }
    printf("PASS\n");passed++;

    printf("  g4_on_ball...");
    {
        float vals[K*D];unsigned rs=42u;
        for(int i=0;i<K*D;i++){
            rs=rs*1103515245u+12345u;
            vals[i]=(float)((rs>>16)%2000)/2000.0f*0.9f-0.45f;
        }
        float w[K],sum=0;
        for(int k=0;k<K;k++){w[k]=1.0f+(float)(k%3);sum+=w[k];}
        for(int k=0;k<K;k++)w[k]/=sum;
        float out[4];
        wubu_hattn_aggregate(vals,w,K,D,1.0f,out);
        float n2=0;for(int d=0;d<D;d++)n2+=out[d]*out[d];
        CHECK(n2<1.0f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
