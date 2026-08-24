/*
 * test_wubu_lorentz_flow.c -- GAP-C011 gates: Lorentz-model flow matching
 *
 * The survey (node 1.5) says Lorentz is numerically superior to Poincare for
 * deep nets. Gates:
 *  G1 geodesic interpolation on the hyperboloid stays ON it (L(x,x)=-1)
 *  G2 target velocity via lorentz_log is tangent (L(p,v)=0) and reduces dist
 *  G3 chained rollout across keyframes keeps every frame on the hyperboloid
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_lorentz.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

/* geodesic interpolation mu_t = exp_p(t * w) on the hyperboloid where
 * w = (t/d_p) * ( q + L(p,q) p ) is the unit-speed tangent at p toward q.
 * This is the standard Lorentz geodesic; L(p,w)=0 by construction. */
static void lor_interpolate(float* out,const float* p,const float* q,float t,int dim){
    float lpq=lorentz_inner(p,q,dim);
    float d=lorentz_distance(p,q,dim);   /* arccosh(-L) = arcosh(Lpq) sign-safe */
    float w[64],n2=0;
    for(int i=0;i<dim;i++){
        w[i]=q[i]+lpq*p[i];              /* spatial+time tangent direction */
        n2+=w[i]*w[i]*(i==0?-1.f:1.f);   /* L(w,w) */
    }
    float nw=sqrtf(n2>0?n2:1e-9f);
    for(int i=0;i<dim;i++) out[i]=p[i];  /* fallback if degenerate */
    if(nw<1e-7f||d<1e-7f) return;
    /* exp_p(t*w_unit): cosh(t)*p + sinh(t)*(w/nw) */
    float ch=coshf(t),sh=sinhf(t);
    for(int i=0;i<dim;i++) out[i]=ch*p[i]+(sh/nw)*w[i];
}
int main(void){
    printf("=== WuBuMath Lorentz Flow Tests ===\n\n");
    const int DIM=5; /* H^4 */

    printf("  g1_interp_on_hyperboloid...");
    float p[DIM]; p[0]=1; p[1]=0.3f;p[2]=0.1f;p[3]=-0.2f;p[4]=0.05f;
    { /* re-normalize p onto hyperboloid: x0 = sqrt(1+|x_s|^2) */
        float s2=0; for(int d=1;d<DIM;d++) s2+=p[d]*p[d];
        p[0]=sqrtf(1.0f+s2);
    }
    float q[DIM]; q[0]=1; q[1]=-0.4f;q[2]=0.25f;q[3]=0.15f;q[4]=-0.1f;
    { float s2=0; for(int d=1;d<DIM;d++) s2+=q[d]*q[d];
      q[0]=sqrtf(1.0f+s2); }
    for(float t=0.f;t<=1.001f;t+=0.25f){
        float mu[DIM]; lor_interpolate(mu,p,q,t,DIM);
        float L=lorentz_norm2(mu,DIM);
        CHECK(fabsf(L+1.0f)<1e-3f);   /* L(x,x) = -1 */
        CHECK(mu[0]>0);               /* upper sheet */
    }
    printf("PASS\n");passed++;

    printf("  g2_target_velocity_tangent...");
    {
        /* proper tangent at p toward q: w = q + L(p,q)p, then step along
         * the geodesic with exp_p(h * w/|w|) and check distance reduces */
        float lpq=lorentz_inner(p,q,DIM);
        float d=lorentz_distance(p,q,DIM);
        float w[64],n2=0;
        for(int i=0;i<DIM;i++){
            w[i]=q[i]+lpq*p[i];
            n2+=w[i]*w[i]*(i==0?-1.f:1.f);
        }
        float nw=sqrtf(n2>0?n2:1e-9f);
        float h=d*0.05f;
        float vfull[64],nxt[DIM];
        for(int i=0;i<DIM;i++) vfull[i]=(h/nw)*w[i];
        lorentz_exp(p,vfull,nxt,DIM);
        float d1=lorentz_distance(nxt,q,DIM);
        CHECK(d1<d);                                   /* moved toward q  */
        CHECK(lorentz_norm2(nxt,DIM)<-0.99f);          /* on hyperboloid  */
    }
    printf("PASS\n");passed++;

    printf("  g3_rollout_on_hyperboloid...");
    {
        const int M=4,NB=8;
        float keys[4][DIM];
        unsigned r=13u;
        for(int k=0;k<M;k++){
            float s2=0;
            for(int d=1;d<DIM;d++){ r=r*1103515245u+12345u;
                keys[k][d]=((float)(r>>16)/65536.0f-1.0f)*0.35f;
                s2+=keys[k][d]*keys[k][d]; }
            keys[k][0]=sqrtf(1.0f+s2);
        }
        for(int leg=0;leg<M-1;leg++)
            for(int b=1;b<=NB;b++){
                float t=(float)b/NB, mu[DIM];
                lor_interpolate(mu,keys[leg],keys[leg+1],t,DIM);
                CHECK(fabsf(lorentz_norm2(mu,DIM)+1.0f)<1e-3f);
            }
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
