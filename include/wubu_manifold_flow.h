/* wubu_manifold_flow.h -- Manifold flow matching (geodesic interpolants) */
#ifndef WUBU_MANIFOLD_FLOW_H
#define WUBU_MANIFOLD_FLOW_H
#include <stddef.h>
/* Geodesic interpolation on Poincare ball: x_t = exp_x(t * log_x(y)) */
void wubu_poincare_interp(const double *x, const double *y, double t, double *out, size_t dim);
/* Conditional flow matching velocity: v = (log_x(y))/sigma */
void wubu_flow_velocity(const double *x, const double *y, double sigma, double *v, size_t dim);
#endif