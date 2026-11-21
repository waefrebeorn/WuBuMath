/* wubu_fisher.h */
#ifndef WUBU_FISHER_H
#define WUBU_FISHER_H
#include <stddef.h>
/* Fisher information matrix (diagonal approximation): F_ii = E[(d log p/dtheta_i)^2] */
void wubu_fisher_diag(const double *grads, int n, size_t dim, double *F);
/* Natural gradient: theta -= lr * F^{-1} * grad */
void wubu_natural_grad_step(double *theta, const double *grad, const double *F, size_t dim, double lr);
#endif