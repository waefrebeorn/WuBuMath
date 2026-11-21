/* wubu_lorentz_explog.h */
#ifndef WUBU_LORENTZ_EXPLOG_H
#define WUBU_LORENTZ_EXPLOG_H
#include <stddef.h>
/* Exponential map on Lorentz model: x in L^{n}, v tangent at x */
void wubu_lorentz_exp(const double *x, const double *v, double *out, size_t dim);
void wubu_lorentz_log(const double *x, const double *y, double *out, size_t dim);
#endif