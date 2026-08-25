#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_sclerp.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== ScLERP Tests ===\n\n");

    /* create two dual quats: rotate around Z by 0.2 rad, translate (1,0,0)→(3,2,0) */
    float qa[8],qb[8];
    {
        float half=0.1f;
        qa[0]=cosf(half);qa[1]=0;qa[2]=0;qa[3]=sinf(half);
        qa[4]=0;qa[5]=0.5f;qa[6]=0;qa[7]=0;  /* dual: t=(1,0,0) */
        float half2=0.2f;
        qb[0]=cosf(half2);qb[1]=0;qb[2]=0;qb[3]=sinf(half2);
        qb[4]=0;qb[5]=1.5f;qb[6]=1.0f;qb[7]=0;  /* dual: t=(3,2,0) */
    }

    printf("  g1_t0_gives_start...");
    {
        float out[8];
        wubu_sclerp(qa,qb,0.0f,out);
        CHECK(fabsf(out[0]-qa[0])<0.01f);
        CHECK(fabsf(out[4]-qa[4])<0.01f);
    }
    printf("PASS\n");passed++;

    printf("  g2_t1_gives_end...");
    {
        float out[8];
        wubu_sclerp(qa,qb,1.0f,out);
        CHECK(fabsf(out[0]-qb[0])<0.01f);
        CHECK(fabsf(out[4]-qb[4])<0.01f);
    }
    printf("PASS\n");passed++;

    printf("  g3_midpoint_smooth...");
    {
        /* at t=0.5, the real part should be between qa and qb rotations */
        float mid[8];
        wubu_sclerp(qa,qb,0.5f,mid);
        /* angle should be ~half of total */
        float angle=2*acosf(fabsf(mid[0]));
        printf("[angle=%.3f rad (want %.3f)] ",angle,0.15f);
        CHECK(fabsf(angle-0.30f)<0.01f);   /* midpoint of [0.2,0.4] range */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
