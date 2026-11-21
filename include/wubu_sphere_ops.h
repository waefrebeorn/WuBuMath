/* wubu_sphere_ops.h */
#ifndef WUBU_SPHERE_OPS_H
#define WUBU_SPHERE_OPS_H
#include <stddef.h>
/* Project vector v to tangent space of unit sphere at x */
void wubu_sphere_tangent_proj(const double *x, const double *v, double *out, size_t dim);
/* Exponential map on unit sphere: exp_x(v) */
void wubu_sphere_exp(const double *x, const double *v, double *out, size_t dim);
#endif