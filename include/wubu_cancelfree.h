/* wubu_cancelfree.h */
#ifndef WUBU_CANCELFREE_H
#define WUBU_CANCELFREE_H
#include <stddef.h>
/* Cancellation-free hyperbolic distance (hyperbolix-style).
   Computes arcosh without subtracting large nearly-equal terms. */
double wubu_cf_poincare_dist(const double *u, const double *v, size_t dim, double c);
double wubu_cf_lorentz_dist(const double *u, const double *v, size_t dim);
#endif