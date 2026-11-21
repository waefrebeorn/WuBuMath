#include "wubu_riemannian_adam.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

wubu_radam_state_t *wubu_radam_create(size_t dim, double lr, double b1, double b2, double eps) {
    wubu_radam_state_t *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->m = calloc(dim, sizeof(double));
    s->v = calloc(dim, sizeof(double));
    if (!s->m || !s->v) { free(s->m); free(s->v); free(s); return NULL; }
    s->dim = dim; s->lr = lr; s->beta1 = b1; s->beta2 = b2; s->eps = eps; s->t = 0;
    return s;
}

void wubu_radam_free(wubu_radam_state_t *s) {
    if (!s) return;
    free(s->m); free(s->v); free(s);
}

void wubu_radam_step(wubu_radam_state_t *s, double *x, const double *g,
                     void (*retract)(double *x, const double *dx, size_t dim)) {
    s->t++;
    double bc1 = 1.0 - pow(s->beta1, s->t);
    double bc2 = 1.0 - pow(s->beta2, s->t);
    double *dxv = malloc(s->dim * sizeof(double));
    if (!dxv) return;
    for (size_t i = 0; i < s->dim; i++) {
        s->m[i] = s->beta1 * s->m[i] + (1 - s->beta1) * g[i];
        s->v[i] = s->beta2 * s->v[i] + (1 - s->beta2) * g[i] * g[i];
        double mh = s->m[i] / bc1;
        double vh = s->v[i] / bc2;
        dxv[i] = -s->lr * mh / (sqrt(vh) + s->eps);
    }
    if (retract) retract(x, dxv, s->dim);
    free(dxv);
}