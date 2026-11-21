/* wubu_horopca.h */
#ifndef WUBU_HOROPCA_H
#define WUBU_HOROPCA_H
#include <stddef.h>
/* HoroPCA: project points to a horosphere (tangent space at a base point).
   Simplified: Euclidean PCA on tangent vectors at origin. */
void wubu_horopca(const double *pts, int n, size_t dim, double *components, int ncomp);
#endif