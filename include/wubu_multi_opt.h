/* wubu_multi_opt.h */
#ifndef WUBU_MULTI_OPT_H
#define WUBU_MULTI_OPT_H
#include <stddef.h>
/* Mixed optimizer: Euclidean params use Adam, hyperbolic params use RAdam.
   Simple: step both with same lr. */
void wubu_multi_opt_step(double *euclid_params, const double *euclid_grad, size_t ne,
                         double *hyp_params, const double *hyp_grad, size_t nh, double lr);
#endif