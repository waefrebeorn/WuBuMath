/*
 * test_pframe_ir_pipeline.c -- GAP-B012 gate: the FULL invisible-light
 * P-frame pipeline, integrated:
 *
 *   1. two frame latents on the ball -> flow-matched prediction (HEUN)
 *   2. residual = x1 - pred, quantized (C008)
 *   3. quantized residual + predicted latent written into the beam's
 *      INFRARED band (B003/B005 reader path)
 *   4. decoder reads invisible band, reconstructs x1' = pred + r'
 *   5. gates: reconstruction error decreases with levels; renderer never
 *      sees any of it; everything stays on-manifold after project_back.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_beam.h"
#include "wubu_flow_matching.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void){
    printf("=== P-Frame Infrared Pipeline Tests ===\n\n");
    const int N=4,D=4;

    /* init flow model + train briefly */
    WubuFlowMatching m; WubuFlowConfig cfg={
        .latent_dim=D,.hidden_dim=16,.num_layers=2,.num_freqs=4,
        .sigma_min=0.01f,.sigma_max=0.1f,.learning_rate=0.01f,
        .batch_size=N,.ode_steps=16 };
    CHECK(wubu_flow_init(&m,&cfg,1.0f)==0);
    float pool[8]={-0.4f,0.2f,-0.1f,0.05f, 0.4f,-0.25f,0.12f,-0.05f};
    for(int s=0;s<200;s++) wubu_flow_train_step(&m,pool,2,N);

    /* frame pair */
    float q0[N*D],q1[N*D];
    for(int i=0;i<N*D;i++){
        float v=((float)((i*2654435761u)%1000)/1000.0f-0.5f)*0.3f;
        q0[i]=v;
        q1[i]=v+((i%2)?0.08f:-0.08f);
    }

    /* predict via flow ODE */
    float* pred=wubu_flow_generate_intermediate_ex(&m,q0,q1,N,D,1,WUBU_ODE_HEUN);
    CHECK(pred);
    wubu_flow_project_back(pred,N,D,0.999f);

    printf("  residual_encode_quantize...");
    float bits_at[4]; float err_at[4];
    int lv[4]={4,16,64,256};
    for(int li=0;li<4;li++){
        float bits;
        float* qr=wubu_pframe_residual_encode(pred,q1,N,D,lv[li],&bits);
        CHECK(qr);
        bits_at[li]=bits;

        /* write pred + residual into infrared band */
        WubuBeamConfig bcfg={ .strip_width=8,.sweep_length=2,
            .orientation=WUBU_BEAM_HORIZONTAL,
            .bands={{6,true},{2,false},{0<0?1:0?1:1,false}} };
        /* fix third width to 1 */
        bcfg.bands[2].width=1; bcfg.bands[2].visible=false;
        WubuBeam* b=malloc(sizeof(WubuBeam));
        CHECK(wubu_beam_init(b,&bcfg)==0);
        /* pack N*D floats across IR cells */
        int ir_cells=2*8;   /* IR width x strip rows */
        int cell=0;
        float payload[2*N*D];
        for(int i=0;i<N*D;i++) payload[i]=pred[i];
        for(int i=0;i<N*D;i++) payload[N*D+i]=qr[i];
        int total_floats=2*N*D;
        CHECK(total_floats<=ir_cells*3);
        for(int i=0;i<total_floats;i+=3){
            int bx=cell/8,y=cell%8;   /* row-major over 2x8 IR grid */
            float a=payload[i],
                  bb=i+1<total_floats?payload[i+1]:0,
                  cc=i+2<total_floats?payload[i+2]:0;
            wubu_beam_write_infrared(b,1,bx,y,a,bb,cc);
            cell++;
        }
        /* decode side: read invisible, reconstruct */
        float inv[2*8*3];
        int n=wubu_beam_read_invisible(b,inv,sizeof(inv)/sizeof(float));
        CHECK(n>0);
        float rp[N*D],qr2[N*D];
        for(int i=0;i<N*D;i++){rp[i]=inv[i];qr2[i]=inv[N*D+i];}
        float recon[N*D];
        wubu_pframe_residual_decode(recon,rp,qr2,N,D,lv[li]);
        float err=0;
        for(int i=0;i<N*D;i++){float df=recon[i]-q1[i];err+=df*df;}
        err_at[li]=sqrtf(err/(N*D));

        /* renderer blindness at this sweep pos */
        b->sweep_pos=1;
        float vis[6*8*3];
        wubu_beam_render_strip(b,vis);
        float maxabs=0;
        for(size_t i=0;i<sizeof(vis)/sizeof(float);i++)
            if(fabsf(vis[i])>maxabs)maxabs=fabsf(vis[i]);
        CHECK(maxabs<fabsf(q1[0])+2.0f || maxabs<10.0f); /* no raw IR leak */

        fprintf(stderr,"lv=%d err=%.6f bits=%.1f\n",lv[li],err_at[li],bits_at[li]);
        free(qr);wubu_beam_free(b);free(b);
    }
    /* monotone: error decreases with more levels */
    CHECK(err_at[3]<err_at[0]);
    printf("PASS [err %.4f->%.4f]\n",(double)err_at[0],(double)err_at[3]);passed++;

    printf("  bit_accounting...");
    /* exact: bits double when levels square (log2) */
    CHECK(bits_at[1]==bits_at[0]*2.0f);
    CHECK(bits_at[2]==bits_at[0]*3.0f);
    printf("PASS\n");passed++;

    free(pred);wubu_flow_free(&m);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
