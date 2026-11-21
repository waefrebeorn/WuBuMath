/* wubu_gromov_delta.h */
#ifndef WUBU_GROMOV_DELTA_H
#define WUBU_GROMOV_DELTA_H
#include <stddef.h>
/* Gromov delta-hyperbolicity of a distance matrix (n x n).
   Returns delta = max over triples of (d(x,y)+d(x,z)-d(y,z))/2 etc. */
double wubu_gromov_delta(const double *D, int n);
#endif