/* wubu_hyp_mlr.h */
#ifndef WUBU_HYP_MLR_H
#define WUBU_HYP_MLR_H
#include <stddef.h>
/* Hyperbolic multinomial logistic regression (Ganea et al.).
   x: point in ball (dim). a_k: hyperplane normal per class (ncls x dim).
   p_k: class prob output (ncls). c: curvature. */
void wubu_hyp_mlr(const double *x, const double *a, const double *b, int ncls, size_t dim, double c, double *probs);
#endif