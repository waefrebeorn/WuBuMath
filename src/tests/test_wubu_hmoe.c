/* test_wubu_hmoe.c -- GAP-C044 gates
 *  G1 nearest expert gets top weight
 *  G2 selected weights sum to 1
 *  G3 combine output on-ball, finite, deterministic
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hmoe.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic MoE Tests ===\n\n");
    const int E=4,D=8;
    float c=1.0f;

    /* experts spread on different axes */
    float protos[E*D];
    memset(protos,0,sizeof(protos));
    for(int e=0;e<E;e++)protos[e*D+e]=0.5f;

    printf("  g1_nearest_dominates...");
    {
        float x[D];memset(x,0,sizeof(x));x[0]=0.45f;  /* near expert 0 */
        int sel[2];float w[2];
        CHECK(wubu_hmoe_route(x,protos,E,D,c,2,0.5f,sel,w)==0);
        CHECK(sel[0]==0);
        CHECK(w[0]>w[1]);
    }
    printf("PASS\n");passed++;

    printf("  g2_weights_sum_one...");
    {
        float x[D];memset(x,0,sizeof(x));x[2]=0.3f;
        int sel[3];float w[3];
        CHECK(wubu_hmoe_route(x,protos,E,D,c,3,0.5f,sel,w)==0);
        float s=0;for(int k=0;k<3;k++){CHECK(!isnan(w[k]));s+=w[k];}
        CHECK(fabsf(s-1.0f)<1e-4f);
    }
    printf("PASS\n");passed++;

    printf("  g3_combine_sane...");
    {
        /* expert outputs: identity-ish points near their protos */
        float outs[E*D];
        for(int e=0;e<E;e++)
            for(int d=0;d<D;d++)outs[e*D+d]=0.9f*protos[e*D+d];
        int sel[2]={0,1};float w[2]={0.7f,0.3f};
        float out[D];
        wubu_hmoe_combine(outs,sel,w,2,D,c,out);
        float n2=0;for(int d=0;d<D;d++){CHECK(!isnan(out[d]));n2+=out[d]*out[d];}
        CHECK(n2<1.0f);
        /* weighted toward expert 0's direction: out[0] should exceed out[1] */
        CHECK(out[0]>out[1]);
        /* determinism */
        float out2[D];
        wubu_hmoe_combine(outs,sel,w,2,D,c,out2);
        for(int d=0;d<D;d++)CHECK(fabsf(out[d]-out2[d])<1e-7f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
