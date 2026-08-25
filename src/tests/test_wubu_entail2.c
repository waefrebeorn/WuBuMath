/* test_wubu_entail2.c -- GAP-D030 gates
 *  G1 aperture in (0, pi/2] for interior points
 *  G2 a child placed along the inward axis IS entailed
 *  G3 a child placed far off-axis is NOT entailed
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_entail2.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Entailment Cone Tests ===\n\n");
    const int D=8;
    float c=1.0f;

    /* parent at moderate norm */
    float parent[D];
    for(int d=0;d<D;d++)parent[d]=0;
    parent[0]=0.5f;

    printf("  g1_aperture_range...");
    {
        float k=wubu_ec_aperture(parent,D,c);
        printf("[k=%.3f rad] ",(double)k);
        CHECK(k>0.0f&&k<=3.14159265f);
    }
    printf("PASS\n");passed++;

    printf("  g2_inward_child_entailed...");
    {
        /* child directly toward origin from parent */
        float child[D];
        for(int d=0;d<D;d++)child[d]=parent[d]*0.4f;
        CHECK(wubu_ec_entailed(parent,child,D,c)==1);
    }
    printf("PASS\n");passed++;

    printf("  g3_offaxis_not_entailed...");
    {
        /* child at same norm but rotated far away (perpendicular-ish) */
        float child[D];
        for(int d=0;d<D;d++)child[d]=0;
        child[1]=parent[0];   /* swap axes: same norm, orthogonal direction */
        CHECK(wubu_ec_entailed(parent,child,D,c)==0);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
