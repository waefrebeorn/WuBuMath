/* GAP-C044: hyperbolic mixture-of-experts routing */
#ifndef WUBU_HMOE_H
#define WUBU_HMOE_H
#ifdef __cplusplus
extern "C" {
#endif
/* route x to top-k of E expert prototypes; sel_idx/sel_w: [topk] */
int  wubu_hmoe_route(const float* x,const float* protos,int E,int D,
                      float c,int topk,float tau,
                      int* sel_idx,float* sel_w);
void wubu_hmoe_combine(const float* outs,const int* sel_idx,
                        const float* sel_w,int topk,int D,float c,
                        float* out);
float wubu_hmoe_load_balance(const int* assignments,int N,int E);
#ifdef __cplusplus
}
#endif
#endif
