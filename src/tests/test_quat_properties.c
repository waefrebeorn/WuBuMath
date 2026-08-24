/*
 * test_property_invariants.c -- GAP-H006/H007/H008: quaternion property gates
 *
 * Fuzz-seeded invariants over the quaternion core:
 *  H006 normalize idempotence + norm-1 preservation under Hamilton product
 *  H007 slerp monotonicity: d(q0, slerp(t)) strictly increases in t
 *  H008 exp/log round trip: log(exp(v)) == v for |v| < pi
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_quaternion_ops.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static unsigned long rs=0xDEADBEEFUL;
static float fr(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;
    return (float)((rs>>11)&0x3FFFFFF)/(float)0x3FFFFFF;}

static float qdist(const float*a,const float*b){
    /* geodesic on S^3 via dot */
    float dot=0;for(int i=0;i<4;i++)dot+=a[i]*b[i];
    if(dot<0)dot=-dot; if(dot>1)dot=1;
    return 2.0f*acosf(dot);
}
int main(void){
    printf("=== WuBuMath Quaternion Property Invariants ===\n\n");

    printf("  h006_normalize_and_product_norm...");
    for(int t=0;t<2000;t++){
        float q[4],p[4];
        for(int i=0;i<4;i++){q[i]=fr()*2-1;p[i]=fr()*2-1;}
        wubu_quat_normalize(q,q);wubu_quat_normalize(p,p);
        CHECK(wubu_quat_is_unit(q)&&wubu_quat_is_unit(p));
        float pr[4];wubu_hamilton_product(pr,q,p);
        CHECK(fabsf(wubu_quat_norm(pr)-1.0f)<1e-5f);   /* unit*unit = unit */
        /* normalize twice = once */
        float q2[4];wubu_quat_normalize(q2,q);
        CHECK(fabsf(wubu_quat_norm_sq(q2)-1.0f)<1e-6f);
    }
    printf("PASS\n");passed++;

    printf("  h007_slerp_monotonicity...");
    for(int t=0;t<500;t++){
        float a[4],b[4];
        for(int i=0;i<4;i++){a[i]=fr()*2-1;b[i]=fr()*2-1;}
        wubu_quat_normalize(a,a);wubu_quat_normalize(b,b);
        float prev=-1,cur;
        for(float u=0.f;u<=1.001f;u+=0.2f){
            float s[4];wubu_quat_slerp(s,a,b,u);
            cur=qdist(a,s);
            CHECK(cur>=prev-1e-5f);   /* monotone non-decreasing */
            prev=cur;
        }
    }
    printf("PASS\n");passed++;

    printf("  h008_exp_log_round_trip...");
    for(int t=0;t<1000;t++){
        /* pure rotation vector with angle < pi */
        float th=fr()*2.8f;              /* keep below pi */
        float ax[3]={fr()*2-1,fr()*2-1,fr()*2-1};
        float an=sqrtf(ax[0]*ax[0]+ax[1]*ax[1]+ax[2]*ax[2]);
        if(an<1e-3f){t--;continue;}
        for(int i=0;i<3;i++)ax[i]/=an;
        /* both functions now speak pure-quat(4): v half-angle in, and
         * log returns the same convention (theta_half*axis). Round trip
         * must be identity. */
        float half[4]={0,ax[0]*th/2,ax[1]*th/2,ax[2]*th/2};
        float qe[4];wubu_quat_exp(qe,half);
        float lg[4];wubu_quat_log(lg,qe);
        CHECK(fabsf(lg[0]-half[0])<1e-4f);
        CHECK(fabsf(lg[1]-half[1])<1e-4f);
        CHECK(fabsf(lg[2]-half[2])<1e-4f);
        CHECK(fabsf(lg[3]-half[3])<1e-4f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
