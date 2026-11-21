/* wubu_hyp_residual.h */
#ifndef WUBU_HYP_RESIDUAL_H
#define WUBU_HYP_RESIDUAL_H
#include <stddef.h>
/* Hyperbolic residual: out = x ⊕ (Mobius transform of f(x)), projected to ball */
void wubu_hyp_residual(const double *x, const double *fx, double *out, size_t dim);
#endif