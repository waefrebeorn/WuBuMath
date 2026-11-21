#include "wubu_ptc.h"
#include <math.h>
#include <string.h>

void wubu_ptc_average(const double *vecs, int nvec, size_t dim, double *out) {
    for (size_t d = 0; d < dim; d++) {
        double s = 0;
        for (int i = 0; i < nvec; i++) s += vecs[i*dim + d];
        out[d] = s / (nvec > 0 ? nvec : 1);
    }
}