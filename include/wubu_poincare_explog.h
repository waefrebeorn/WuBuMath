/* wubu_poincare_explog.h */
#ifndef WUBU_POINCARE_EXPLOG_H
#define WUBU_POINCARE_EXPLOG_H
#include <stddef.h>
/* Exponential map at x in Poincare ball: exp_x(v) */
void wubu_poincare_exp(const double *x, const double *v, double *out, size_t dim, double c);
/* Log map: log_x(y) */
void wubu_poincare_log(const double *x, const double *y, double *out, size_t dim, double c);
#endif