/* wubu_nbody.h */
#ifndef WUBU_NBODY_H
#define WUBU_NBODY_H
#include <stddef.h>
/* N-body on Poincare ball: gravitational attraction.
   positions: [N*dim], velocities: [N*dim], masses: [N]
   Returns total energy. */
double wubu_nbody_step(double *pos, double *vel, const double *mass, int N, size_t dim,
    double G, double dt, double softening);
#endif