/* wubu_hyp_dist.h */
#ifndef WUBU_HYP_DIST_H
#define WUBU_HYP_DIST_H
#include <stddef.h>
double wubu_poincare_dist(const double *u, const double *v, size_t dim, double c);
double wubu_lorentz_dist(const double *u, const double *v, size_t dim);
#endif