/* wubu_riemannian_adam.h -- Riemannian Adam (adaptive SGD on manifolds) */
#ifndef WUBU_RIEMANNIAN_ADAM_H
#define WUBU_RIEMANNIAN_ADAM_H
#include <stddef.h>

typedef struct {
    double *m;   /* first moment (tangent) */
    double *v;   /* second moment */
    double beta1, beta2, eps;
    double lr;
    int t;
    size_t dim;
} wubu_radam_state_t;

wubu_radam_state_t *wubu_radam_create(size_t dim, double lr, double b1, double b2, double eps);
void wubu_radam_free(wubu_radam_state_t *s);
/* step: g is Riemannian gradient in tangent space at x; updates x in place via retraction */
void wubu_radam_step(wubu_radam_state_t *s, double *x, const double *g,
                     void (*retract)(double *x, const double *dx, size_t dim));
#endif