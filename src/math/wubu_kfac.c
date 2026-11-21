#include "wubu_kfac.h"
double wubu_kfac_trace(const double *A, const double *G, size_t dim) {
    double tA = 0, tG = 0;
    for (size_t i = 0; i < dim; i++) { tA += A[i*dim+i]; tG += G[i*dim+i]; }
    return tA * tG;
}