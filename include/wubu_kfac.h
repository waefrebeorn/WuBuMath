/* wubu_kfac.h */
#ifndef WUBU_KFAC_H
#define WUBU_KFAC_H
#include <stddef.h>
/* KFAC: approximate Fisher with Kronecker factors A (input) and G (gradient).
   Returns curvature approximation: F ≈ A ⊗ G. Simplified: trace. */
double wubu_kfac_trace(const double *A, const double *G, size_t dim);
#endif