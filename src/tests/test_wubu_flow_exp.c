#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_flow_exp.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
static float fd(const float*a,const float*b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){float df=a[d]-b[d];ab2+=df*df;a2+=a[d]*a[d];b2+=b[d]*b[d];}
    float den=(1-c*a2)*(1-c*b2);if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}
int main(void){
    printf("=== Flow Exp Integration Tests ===\n\n");
    const int D=8;
    float c=1.0f;

    printf("  g1_step_on_ball...");
    {
        float x[D],v[D],out[D];
        unsigned rs=42u;
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            x[d]=((float)((rs>>16)%2000))/20000.0f-0.05f;
            rs=rs*1103515245u+12345u;
            v[d]=((float)((rs>>16)%2000))/1000.0f-0.5f;
        }
        CHECK(wubu_fe_step(x,v,D,c,0.01f,out)==0);
        float n2=0;for(int d=0;d<D;d++)n2+=out[d]*out[d];
        CHECK(n2<1.0f);
    }
    printf("PASS\n");passed++;

    printf("  g2_trajectory_monotone...");
    {
        float start[D],end[D],path[30*D];
        for(int d=0;d<D;d++){
            start[d]=((float)(d%3)-1)*0.15f;
            end[d]=((float)((d+2)%5)-2)*0.12f;
        }
        int n=20;
        CHECK(wubu_fe_trajectory(start,end,D,c,n,0.1f,path)==0);
        float prev=fd(start,end,D,c);
        int mono=1;
        for(int s=0;s<n;s++){
            float cur_d=fd(path+(size_t)s*D,end,D,c);
            if(cur_d>prev+0.001f){mono=0;break;}
            prev=cur_d;
        }
        CHECK(mono);
    }
    printf("PASS\n");passed++;
    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
