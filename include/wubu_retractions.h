/* wubu_retractions.h */
#ifndef WUBU_RETRACTIONS_H
#define WUBU_RETRACTIONS_H
#include <stddef.h>
/* Retract x += v back onto unit sphere (normalize) */
void wubu_retract_sphere(double *x, const double *v, size_t dim);
/* Retract onto Stiefel manifold via QR (Gram-Schmidt) for a single vector */
void wubu_retract_stiefel(double *x, size_t dim);
#endif