/* wubu_moe_router.h */
#ifndef WUBU_MOE_ROUTER_H
#define WUBU_MOE_ROUTER_H
#include <stddef.h>
/* Hyperbolic MoE router: score_i = -dist_H(x, proto_i); softmax; top-k.
   protos: nexperts x dim. scores: nexperts. topk: k indices out. */
void wubu_moe_route(const double *x, const double *protos, int nexperts, size_t dim, int k, int *topk, double *scores);
#endif