#include "wubu_hyp_posenc.h"
#include <math.h>

void wubu_hyp_posenc(double *x, size_t dim, int pos) {
    for (size_t i = 0; i < dim; i++) {
        double freq = 1.0 / pow(10000.0, (double)(2*(i/2)) / (double)dim);
        double val = (i % 2 == 0) ? sin(pos * freq) : cos(pos * freq);
        x[i] += 0.1 * val;
    }
    /* project to ball */
    double r = 0; for (size_t i = 0; i < dim; i++) r += x[i]*x[i];
    r = sqrt(r);
    if (r >= 0.99) { double s = 0.99/r; for (size_t i = 0; i < dim; i++) x[i] *= s; }
}