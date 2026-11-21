/* wubu_expmap.h */
#ifndef WUBU_EXPMAP_H
#define WUBU_EXPMAP_H
#include <stddef.h>
/* Riemannian exp/log on Poincare ball */
void wubu_poincare_exp(const double *x, const double *v, double c, double *out, size_t dim);
void wubu_poincare_log(const double *x, const double *y, double c, double *out, size_t dim);
#endif