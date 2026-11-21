#include "wubu_adam.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
void wubu_adam_init(wubu_adam_state *s, size_t dim) {
    s->m = calloc(dim, sizeof(double));
    s->v = calloc(dim, sizeof(double));
    s->t = 0;
}
void wubu_adam_step(wubu_adam_state *s, double *theta, const double *grad, size_t dim, double lr, double b1, double b2, double eps) {
    s->t++;
    for (size_t i = 0; i < dim; i++) {
        s->m[i] = b1*s->m[i] + (1-b1)*grad[i];
        s->v[i] = b2*s->v[i] + (1-b2)*grad[i]*grad[i];
        double mh = s->m[i]/(1-pow(b1,s->t));
        double vh = s->v[i]/(1-pow(b2,s->t));
        theta[i] -= lr*mh/(sqrt(vh)+eps);
    }
}
void wubu_adam_free(wubu_adam_state *s) {
    free(s->m); free(s->v);
}