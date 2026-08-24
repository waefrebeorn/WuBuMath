/*
 * wubu_text_encoder.c -- minimal hashed-bag token encoder (GAP-D005)
 *
 * The manifold-CLIP text front end. No tokenizer dependency: tokens are
 * byte n-grams hashed into a fixed vocabulary; a bag-of-vectors mean with
 * sinusoidal position damping produces one feature vector per document,
 * ready for wubu_mclip_embed(). This is deliberately simple — the claim
 * we gate is "distinct texts give distinct, stable embeddings" and
 * "embedding is deterministic", not SOTA language understanding.
 * Language quality enters later through nest_gpt (D-cell).
 */

#include "wubu_text_encoder.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* FNV-1a 32-bit */
static uint32_t te_hash(const char* s,size_t n){
    uint32_t h=2166136261u;
    for(size_t i=0;i<n;i++){h^=(uint8_t)s[i];h*=16777619u;}
    return h;
}

int wubu_text_init(WubuTextEncoder* te,int vocab_size,int out_dim){
    if(vocab_size<16||out_dim<2) return -1;
    memset(te,0,sizeof(*te));
    te->vocab_size=vocab_size;
    te->out_dim=out_dim;
    te->table=malloc(sizeof(float)*(size_t)vocab_size*(size_t)out_dim);
    if(!te->table) return -2;
    /* deterministic seeded init — same table every process */
    uint32_t r=20260824u;
    for(size_t i=0;i<(size_t)vocab_size*out_dim;i++){
        r=r*1103515245u+12345u;
        te->table[i]=((float)(r>>16)/32768.0f-1.0f)*0.1f;
    }
    return 0;
}

void wubu_text_free(WubuTextEncoder* te){
    free(te->table);
    te->table=NULL;
}

void wubu_text_encode(const WubuTextEncoder* te,const char* text,
                      float* out,int out_dim){
    memset(out,0,sizeof(float)*(size_t)out_dim);
    int D=out_dim<te->out_dim?out_dim:te->out_dim;
    size_t len=strlen(text);
    if(len==0) return;

    /* word 4-grams + char trigrams hashed into vocabulary; weighted mean */
    int count=0;
    size_t i=0;
    while(i<len){
        size_t j=i;
        while(j<len&&text[j]!=' '&&text[j]!='\n'&&text[j]!='\t') j++;
        if(j>i){
            /* word hash */
            uint32_t h=te_hash(text+i,j-i);
            const float* row=te->table+(size_t)(h%(uint32_t)te->vocab_size)*(size_t)te->out_dim;
            for(int d=0;d<D;d++) out[d]+=row[d];
            count++;
            /* char trigrams inside word (sub-word signal) */
            for(size_t k=i;k+3<=j;k++){
                uint32_t h3=te_hash(text+k,3);
                const float* r3=te->table+(size_t)(h3%(uint32_t)te->vocab_size)*(size_t)te->out_dim;
                for(int d=0;d<D;d++) out[d]+=r3[d]*0.35f;
            }
            count+=2;
        }
        i=(j<len)?j+1:j+1;
    }
    if(count==0) return;
    float inv=1.0f/(float)count;
    for(int d=0;d<D;d++) out[d]*=inv;
}
