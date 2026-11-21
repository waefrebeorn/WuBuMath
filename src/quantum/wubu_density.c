#include "wubu_density.h"

double wubu_density_trace(const double *rho, int n) {
    double t = 0;
    for (int i = 0; i < n; i++) t += rho[(i*n+i)*2];
    return t;
}

double wubu_density_purity(const double *rho, int n) {
    /* Tr(rho^2) = sum_{i,j} |rho_{ij}|^2 for Hermitian rho */
    double p = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double re = rho[(i*n+j)*2], im = rho[(i*n+j)*2+1];
            p += re*re + im*im;
        }
    return p;
}