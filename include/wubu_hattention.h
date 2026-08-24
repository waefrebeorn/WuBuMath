/* GAP-C016: hyperbolic attention primitives on the Poincare ball */
#ifndef WUBU_HATTENTION_H
#define WUBU_HATTENTION_H
#ifdef __cplusplus
extern "C" {
#endif
/* geodesic distance on the ball (curvature c) */
float wubu_hattn_distance(const float* a,const float* b,int D,float c);
/* softmax(-d(q,k)/tau): weights[K] */
void wubu_hattn_matching(const float* q,const float* keys,int K,int D,
                         float tau,float* weights);
/* weighted Mobius gyromidpoint of values with given weights -> out[D] */
void wubu_hattn_aggregate(const float* values,const float* weights,
                          int K,int D,float c,float* out);
#ifdef __cplusplus
}
#endif
#endif
