/* wubu_hyperbolic_conv.h */
#ifndef WUBU_HYPERBOLIC_CONV_H
#define WUBU_HYPERBOLIC_CONV_H
#include <stddef.h>
/* Mobius matrix-vector product on Poincare ball (Ganea et al.) */
void wubu_mobius_matvec(const double *M, const double *x, double *out, size_t in, size_t outd, double c);
/* Pointwise nonlinearity in tangent space then exp back */
void wubu_hyp_tanh(const double *x, double *out, size_t dim);
#endif