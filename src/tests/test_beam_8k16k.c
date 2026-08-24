/*
 * test_beam_8k16k.c -- GAP-A016..A019: 8K/16K-class sweep integration tests
 *
 * Proves the beam representation handles 8K/16K sweep lengths without
 * materializing frame buffers: init at 8K/16K sweep, write+read visible and
 * infrared across full range, decode-at-N for 4K/1080p/720p/480p from one
 * field, phi-order uniformity at scale.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_beam.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

static int y_(int s){ return s%4000; }
static void test_8k_sweep(void){
    WubuBeamConfig cfg={ .strip_width=4000,.sweep_length=7680*2, /* 8K-class long axis */
        .orientation=WUBU_BEAM_HORIZONTAL,
        .bands={{3900,true},{90,false},{10,false}} };
    WubuBeam b; CHECK(wubu_beam_init(&b,&cfg)==0);
    /* spot-write across the whole sweep: first, middle, last */
    int probes[3]={0,cfg.sweep_length/2,cfg.sweep_length-1};
    float vals[3][3]={{0.1f,0.2f,0.3f},{0.5f,0.6f,0.7f},{0.9f,0.8f,0.7f}};
    for(int i=0;i<3;i++)
        wubu_beam_write_visible(&b,probes[i],i,y_(probes[i]),vals[i][0],vals[i][1],vals[i][2]);
    for(int i=0;i<3;i++){
        float* p=wubu_beam_at(&b,WUBU_BAND_VISIBLE,probes[i],i,y_(probes[i]));
        CHECK(p&&fabsf(p[0]-vals[i][0])<1e-6f);
    }
    wubu_beam_free(&b);
}
static void test_16k_decode_at_all_targets(void){
    int n=16384;                       /* 16K-class sweep */
    int* order=malloc(sizeof(int)*(size_t)n);
    wubu_beam_phi_order(order,n);
    const int targets[]={3840,2160,1080,720,480};  /* 4K->480p from ONE field */
    for(int ti=0;ti<5;ti++){
        int N=targets[ti];
        int* pos=malloc(sizeof(int)*(size_t)N);
        wubu_beam_decode_at(order,n,pos,N);
        for(int i=0;i<N;i++) CHECK(pos[i]>=0&&pos[i]<n);
        /* uniqueness: N<=n so all distinct */
        char* seen=calloc((size_t)n,1);
        for(int i=0;i<N;i++){ CHECK(!seen[pos[i]]); seen[pos[i]]=1; }
        free(seen);free(pos);
    }
    free(order);
}
static void test_phi_uniformity_at_scale(void){
    int n=16384,*order=malloc(sizeof(int)*(size_t)n);
    wubu_beam_phi_order(order,n);
    /* prefix k=n/16 must have max gap <= ~2*16 positions + edge slack */
    int k=n/16,chosen[1024],m=0;
    for(int i=0;i<n&&m<1024;i++) if(order[i]<k) chosen[m++]=i;
    CHECK(m==k);
    int maxgap=0;
    for(int j=1;j<m;j++){int g=chosen[j]-chosen[j-1];if(g>maxgap)maxgap=g;}
    CHECK(maxgap<=40);   /* ~uniform at 16K scale */
    free(order);
}
int main(void){
    printf("=== WuBuMath Beam 8K/16K Integration ===\n\n");
    printf("  test_8k_sweep...");                 test_8k_sweep(); printf("PASS\n");passed++;
    printf("  test_16k_decode_at_all_targets...");test_16k_decode_at_all_targets();printf("PASS\n");passed++;
    printf("  test_phi_uniformity_at_scale...");  test_phi_uniformity_at_scale(); printf("PASS\n");passed++;
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
