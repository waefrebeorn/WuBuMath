/* wubu_klein_model.h */
#ifndef WUBU_KLEIN_MODEL_H
#define WUBU_KLEIN_MODEL_H
#include <stddef.h>
/* Klein model: hyperbolic point in unit ball, geodesics are straight lines.
   Distance: arcosh((1 - u·v)/sqrt((1-|u|^2)(1-|v|^2))) */
double wubu_klein_dist(const double *u, const double *v, size_t dim);
#endif