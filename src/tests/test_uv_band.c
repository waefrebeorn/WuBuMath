/*
 * test_uv_band.c -- GAP-B013 gate: UV band carries provenance payload.
 *
 * Gates:
 *  G1 UV payload survives write/read exactly (watermark = lossless)
 *  G2 UV is invisible to the renderer
 *  G3 all three bands coexist without cross-contamination
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_beam.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== UV Band Provenance Tests ===\n\n");
    WubuBeamConfig cfg={ .strip_width=40,.sweep_length=8,
        .orientation=WUBU_BEAM_HORIZONTAL,
        .bands={{30,true},{7,false},{3,false}} };
    WubuBeam b; CHECK(wubu_beam_init(&b,&cfg)==0);

    /* G1: 21-float provenance signature into UV band (3 wide x 40 rows).
     * Write signature row f at (bx=f, y=f) for f=0..6. */
    float sig[21];
    for(int i=0;i<21;i++) sig[i]=(float)(i*7919%997)/997.0f;
    int cell=0;
    for(int f=0;f<7;f++)
        for(int c=0;c<3;c++){
            int bx=cell%3,y=cell/3;
            float* p=wubu_beam_at(&b,WUBU_BAND_UV,1,bx,y);
            CHECK(p);
            *p=sig[cell];
            cell++;
        }
    CHECK(cell==21);
    printf("  g1_uv_lossless...");
    cell=0;
    for(int f=0;f<7;f++)
        for(int c=0;c<3;c++){
            int bx=cell%3,y=cell/3;
            const float* p=wubu_beam_at(&b,WUBU_BAND_UV,1,bx,y);
            CHECK(p);
            CHECK(fabsf(*p-sig[cell])<1e-6f);
            cell++;
        }
    printf("PASS\n");passed++;

    /* G2 + G3 combined: visible marker, IR marker, UV untouched */
    printf("  g2g3_coexistence...");
    wubu_beam_write_visible(&b,2,0,5, 1.0f,1.0f,1.0f);
    wubu_beam_write_infrared(&b,2,0,5,-1.0f,-1.0f,-1.0f);
    {
        float* uv=wubu_beam_at(&b,WUBU_BAND_UV,2,0,5);
        CHECK(uv&&uv[0]!=1.0f&&uv[0]!=-1.0f);   /* UV untouched by others */

        b.sweep_pos=2;
        float vis[30*40*3];
        wubu_beam_render_strip(&b,vis);
        size_t idx=(size_t)(5*30+0)*3;
        CHECK(vis[idx]==1.0f && vis[idx+1]==1.0f);  /* visible intact */

        float inv[10*40*3];
        int n=wubu_beam_read_invisible(&b,inv,sizeof(inv)/sizeof(float));
        CHECK(n==10*40*3);
        /* reader indexes ONLY invisible bands: IR first (bx 0..6) */
        size_t ir_idx=((size_t)0*40+5)*3;
        CHECK(inv[ir_idx]==-1.0f);
        /* UV sig starts after all IR cells in the reader's layout */
        size_t uv_off=(size_t)7*40;
        CHECK(fabsf(inv[(uv_off+0)*3]-sig[0])<1e-6f);
    }
    printf("PASS\n");passed++;

    wubu_beam_free(&b);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
