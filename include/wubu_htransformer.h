/* GAP-C039: full hyperbolic transformer */
#ifndef WUBU_HTRANSFORMER_H
#define WUBU_HTRANSFORMER_H
#include "wubu_learned_pos.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    float* Wq;  /* [D,D] folded QKV-attention weight */
    float* Wo;  /* [D,D] output projection */
    float* W1;  /* [D,D] FF up */
    float* W2;  /* [D,D] FF down */
} WubuHTLayerWeights;

typedef struct {
    int vocab,T,D,heads,n_layers,n_classes;
    float c;
    float* tok_emb;            /* [vocab,D] */
    WubuHTLayerWeights* blocks;/* [n_layers] */
    float* head_proto;         /* [n_classes,D] ball prototypes */
    WubuLearnedPos pos;
} WubuHT;

int  wubu_ht_init(WubuHT* ht,int vocab,int T,int D,int heads,int n_layers,
                  int n_classes,float c,unsigned seed);
void wubu_ht_free(WubuHT* ht);
void wubu_ht_forward(const WubuHT* ht,const int* tokens,float* scores);
#ifdef __cplusplus
}
#endif
#endif
