/*
 * test_pair_retrieval.c -- GAP-D006 + D010 gates
 *
 * D006: corpus structure exists (caption similarity tracks scene identity)
 * D010: full recall@k eval harness — train on train split, evaluate
 *       recall@1/5/10 on the HELD-OUT val split. Generalization gate.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_pairdata.h"
#include "../train/wubu_manifold_clip.c"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

/* recall@k: for each val image, its pair must rank in top-k text neighbors */
static float recall_at_k(WubuManifoldClip* m,const float* fi,const float* ft,
                         int B,int F,int k,int D){
    float* ea=malloc(sizeof(float)*B*D);
    float* eb=malloc(sizeof(float)*B*D);
    wubu_mclip_embed(m,fi,F,m->proj_a,D,ea,B);
    wubu_mclip_embed(m,ft,F,m->proj_b,D,eb,B);
    int hits=0;
    for(int i=0;i<B;i++){
        /* collect distances to all texts */
        float dist[512];int order[512];
        for(int j=0;j<B&&j<512;j++){
            dist[j]=wubu_mclip_geodesic(ea+i*D,eb+j*D,D,expf(m->log_c));
            order[j]=j;
        }
        /* selection sort partial: find k smallest */
        for(int a=0;a<k;a++)
            for(int b2=a+1;b2<B;b2++)
                if(dist[order[b2]]<dist[order[a]]){int t=order[a];order[a]=order[b2];order[b2]=t;}
        for(int a=0;a<k;a++) if(order[a]==i){hits++;break;}
    }
    free(ea);free(eb);
    return (float)hits/B;
}
int main(void){
    printf("=== Pair Data + Retrieval Harness Tests ===\n\n");

    WubuPairCorpus corp;
    WubuPairDataConfig pcfg={ .num_pairs=400,.feat_dim=16,
        .caption_noise=0.08f,.val_split=0.25f,.seed=2026u };
    CHECK(wubu_pairdata_init(&corp,&pcfg)==0);
    CHECK(corp.train_n==300);
    printf("  d006_corpus_structure...PASS\n");passed++;

    /* D006 structure gate: same-scene pairs (identical index) closer than
     * random cross pairs even BEFORE training */
    float sim_same=0,sim_rand=0;
    for(int i=0;i<100;i++){
        float d=wubu_mclip_geodesic(corp.image_feat+(size_t)i*16,
                                    corp.text_feat+(size_t)i*16,16,1.0f);
        sim_same+=d;
        int j=(i*7919+13)%400;
        sim_rand+=wubu_mclip_geodesic(corp.image_feat+(size_t)i*16,
                                      corp.text_feat+(size_t)j*16,16,1.0f);
    }
    CHECK(sim_same<sim_rand);
    printf("  d006_caption_tracks_scene...PASS\n");passed++;

    /* D010: train on train split only; evaluate on val split */
    WubuManifoldClip m;
    WubuMclipConfig cfg={ .embed_dim=8,.feat_dim=16,.lr=0.15f };
    CHECK(wubu_mclip_init(&m,&cfg)==0);
    for(int s=0;s<500;s++)
        wubu_mclip_train_step(&m,corp.image_feat,corp.text_feat,
                              corp.train_n>64?64:corp.train_n,16);
    float r1=recall_at_k(&m,
        corp.image_feat+(size_t)corp.train_n*16,
        corp.text_feat+(size_t)corp.train_n*16,
        corp.cfg.num_pairs-corp.train_n,16,1,wubu_mclip_curvature(&m));
    float r5=recall_at_k(&m,
        corp.image_feat+(size_t)corp.train_n*16,
        corp.text_feat+(size_t)corp.train_n*16,
        corp.cfg.num_pairs-corp.train_n,16,5,wubu_mclip_curvature(&m));
    printf("  d010_val_recall@1=%.2f @5=%.2f  ",(double)r1,(double)r5);
    CHECK(r1>1.0f/100.0f);   /* above chance on unseen scenes */
    CHECK(r5>r1);            /* monotone in k */
    printf("PASS\n");passed++;

    wubu_mclip_free(&m);wubu_pairdata_free(&corp);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
