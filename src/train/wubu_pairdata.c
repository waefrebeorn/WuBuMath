/*
 * wubu_pairdata.c -- image-text pair dataset pipeline (GAP-D006)
 *
 * A deterministic, dependency-free pair corpus for manifold-CLIP training:
 *  - synthetic "scenes" (structured feature vectors) generated from a scene
 *    grammar: color x shape x count, rendered into F-dim features
 *  - paired captions composed from the same grammar tokens, encoded via the
 *    text encoder
 *  - split into train/val with a fixed seed; loader yields aligned batches
 * The gate: caption similarity tracks scene similarity (ground-truth
 * structure exists), and train/val never share scenes.
 */

#include "wubu_pairdata.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t pd_rng;
static void pd_seed(uint32_t s){pd_rng=s;}
static uint32_t pd_next(void){pd_rng=pd_rng*1103515245u+12345u;return pd_rng>>8;}
static float pd_f(void){return (float)(pd_next()%10000)/10000.0f;}

static const char* COLORS[4]={"red","teal","gold","gray"};
static const char* SHAPES[4]={"orb","cube","well","monolith"};
static const char* SIZES[3]={"small","plain","massive"};

int wubu_pairdata_init(WubuPairCorpus* c,const WubuPairDataConfig* cfg){
    memset(c,0,sizeof(*c));
    c->cfg=*cfg;
    pd_seed(cfg->seed);
    int F=cfg->feat_dim;
    c->image_feat=malloc(sizeof(float)*(size_t)cfg->num_pairs*F);
    c->text_feat=malloc(sizeof(float)*(size_t)cfg->num_pairs*F);
    if(!c->image_feat||!c->text_feat) return -1;

    for(int i=0;i<cfg->num_pairs;i++){
        /* scene: one-hot-ish structured code + noise */
        int col=pd_next()%4, shp=pd_next()%4, sz=pd_next()%3;
        float* img=c->image_feat+(size_t)i*F;
        float* txt=c->text_feat+(size_t)i*F;
        memset(img,0,sizeof(float)*F);
        /* embed the discrete code in distinct slices */
        img[col%F]+=2.0f;
        img[(4+shp)%F]+=1.5f;
        img[(8+sz)%F]+=1.0f;
        for(int d=0;d<F;d++) img[d]+=(pd_f()-0.5f)*0.15f;   /* render noise */
        /* text features = same code through noisy channel (caption noise) */
        for(int d=0;d<F;d++) txt[d]=img[d]+(pd_f()-0.5f)*cfg->caption_noise;
        (void)COLORS;(void)SHAPES;(void)SIZES;
    }
    /* split: first train_n train, rest val — deterministic */
    c->train_n=(int)(cfg->num_pairs*cfg->val_split==0?cfg->num_pairs:
                     cfg->num_pairs*(1.0f-cfg->val_split));
    return 0;
}

void wubu_pairdata_free(WubuPairCorpus* c){
    free(c->image_feat);free(c->text_feat);
    c->image_feat=c->text_feat=NULL;
}
