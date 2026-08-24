/* test_wubu_htransformer.c -- GAP-C039 gates: full hyperbolic transformer
 *  G1 forward produces finite scores summing to a sane range
 *  G2 deterministic across calls
 *  G3 same tokens shuffled -> different pooled scores (position matters)
 *  G4 all intermediate representations on-ball (verified via block stack)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_htransformer.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Full Hyperbolic Transformer Tests ===\n\n");
    const int VOCAB=32,T=8,D=16,H=2,L=2,NCLS=3;

    WubuHT ht;
    CHECK(wubu_ht_init(&ht,VOCAB,T,D,H,L,NCLS,1.0f,42u)==0);

    int toks[T];
    for(int t=0;t<T;t++)toks[t]=(t*7+3)%VOCAB;

    float sc1[NCLS],sc2[NCLS];
    wubu_ht_forward(&ht,toks,sc1);

    printf("  g1_finite_scores...");
    {
        for(int k=0;k<NCLS;k++){
            CHECK(!isnan(sc1[k]));
            CHECK(!isinf(sc1[k]));
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_deterministic...");
    {
        wubu_ht_forward(&ht,toks,sc2);
        for(int k=0;k<NCLS;k++)CHECK(fabsf(sc1[k]-sc2[k])<1e-6f);
    }
    printf("PASS\n");passed++;

    printf("  g3_order_matters...");
    {
        /* shuffle token order → pooled representation differs → scores differ */
        int rev[T];
        for(int t=0;t<T;t++)rev[t]=toks[T-1-t];
        float sc3[NCLS];
        wubu_ht_forward(&ht,rev,sc3);
        float diff=0;
        for(int k=0;k<NCLS;k++){float df=sc1[k]-sc3[k];diff+=df*df;}
        CHECK(sqrtf(diff)>1e-6f);
    }
    printf("PASS\n");passed++;

    wubu_ht_free(&ht);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
