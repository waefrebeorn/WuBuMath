/* wubu_riem_grad.h */
#ifndef WUBU_RIEM_GRAD_H
#define WUBU_RIEM_GRAD_H
#include <stddef.h>
/* Project Euclidean gradient to tangent space of Lorentz hyperboloid at x.
   g_euclid (dim+1), x (dim+1), out (dim+1): out = g + <g,x>_L x (with metric). */
void wubu_lorentz_grad_proj(const double *x, const double *g, double *out, size_t dim);
#endif