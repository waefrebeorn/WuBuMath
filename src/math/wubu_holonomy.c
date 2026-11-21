#include "wubu_holonomy.h"
void wubu_holonomy_apply(const double *R, const double *v, double *out, size_t dim) {
    for (size_t i = 0; i < dim; i++) {
        double s = 0;
        for (size_t j = 0; j < dim; j++) s += R[i*dim+j]*v[j];
        out[i] = s;
    }
}