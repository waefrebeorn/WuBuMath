/* wubu_density.h */
#ifndef WUBU_DENSITY_H
#define WUBU_DENSITY_H
#include <stddef.h>
/* Trace of density matrix rho (n x n complex stored as 2n^2 doubles): sum of diagonal reals */
double wubu_density_trace(const double *rho, int n);
/* Purity Tr(rho^2) for pure/mixed state diagnostic (real part) */
double wubu_density_purity(const double *rho, int n);
#endif