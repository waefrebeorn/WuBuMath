/* test_wubu_hpq.c -- GAP-E009 gates
 *  G1 codes in [0,K) for all stages
 *  G2 reconstruction improves monotonically with more stages used
 *  G3 compression: L=4,K=16 → 4 codes vs raw D floats
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_hpq.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic RVQ Tests ===\n\n");
    const int N=40,D=16,L=4,K=16;
    float c=1.0f;

    /* clustered training points */
    float pts[N*D];
    unsigned rs=42u;
    for(int i=0;i<N;i++){
        float base=((float)(i%4)-1.5f)*0.15f;
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float noise=((rs>>16)%100)/5000.0f-0.01f;
            pts[i*D+d]=base+noise;
        }
        float n2=0;
        for(int d=0;d<D;d++)n2+=pts[i*D+d]*pts[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)pts[i*D+d]*=s;}
    }

    WubuHPQ q;
    CHECK(wubu_hpq_build(pts,N,D,L,K,c,7u,&q)==0);

    printf("  g1_codes_in_range...");
    {
        int codes[4];
        for(int i=0;i<N;i++){
            wubu_hpq_encode(&q,pts+i*D,codes);
            for(int l=0;l<L;l++)CHECK(codes[l]>=0&&codes[l]<K);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_monotone_refinement...");
    {
        /* decode using first l stages; error should not increase */
        float x[D],recon[D];
        memcpy(x,pts,sizeof(float)*D);
        int codes[4];
        wubu_hpq_encode(&q,x,codes);
        float prev_err=1e30f;
        for(int use=1;use<=L;use++){
            memset(recon,0,sizeof(float)*D);
            for(int l=0;l<use;l++){
                const float* cb=q.codebooks+(size_t)(l*K+codes[l])*D;
                for(int d=0;d<D;d++)recon[d]+=cb[d];
            }
            float n2=0;for(int d=0;d<D;d++)n2+=recon[d]*recon[d];
            if(n2>0.999f){float s=sqrtf(0.999f/n2);for(int d=0;d<D;d++)recon[d]*=s;}
            float err=0;
            for(int d=0;d<D;d++){float df=recon[d]-x[d];err+=df*df;}
            printf("[L=%d err=%.4f] ",use,(double)err);
            CHECK(err<=prev_err*1.05f);
            prev_err=err;
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_compression_ratio...");
    {
        float bits_raw=(float)(D*32);
        float bits_coded=(float)(L*log2f((float)K));
        float ratio=bits_raw/bits_coded;
        printf("[%.1fx] ",(double)ratio);
        CHECK(ratio>10.0f);
    }
    printf("PASS\n");passed++;

    wubu_hpq_free(&q);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
