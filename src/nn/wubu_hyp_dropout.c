#include "wubu_hyp_dropout.h"
#include <stdlib.h>
#include <math.h>

void wubu_hyp_dropout(double *x, size_t dim, double p, unsigned int *rng) {
    double scale = 1.0/(1.0-p);
    for (size_t i = 0; i < dim; i++) {
        *rng = *rng * 1103515245u + 12345u;
        double r = (double)(*rng) / (double)0xFFFFFFFFu;
        if (r < p) x[i] = 0;
        else x[i] *= scale;
    }
    /* project to ball */
    double rr = 0; for (size_t i = 0; i < dim; i++) rr += x[i]*x[i];
    rr = sqrt(rr);
    if (rr >= 0.99) { double s = 0.99/rr; for (size_t i = 0; i < dim; i++) x[i] *= s; }
}