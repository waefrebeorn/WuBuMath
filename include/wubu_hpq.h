/* GAP-E009: hyperbolic residual vector quantization */
#ifndef WUBU_HPQ_H
#define WUBU_HPQ_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int L,K,D;
    float c;
    float* codebooks;  /* [L, K, D] */
} WubuHPQ;

int  wubu_hpq_build(const float* pts,int n,int D,int L,int K,float c,
                    unsigned seed,WubuHPQ* q);
void wubu_hpq_free(WubuHPQ* q);
int  wubu_hpq_encode(const WubuHPQ* q,const float* x,int* codes);
void wubu_hpq_decode(const WubuHPQ* q,const int* codes,float* out);
#ifdef __cplusplus
}
#endif
#endif
