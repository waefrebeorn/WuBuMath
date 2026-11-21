/* wubu_leapfrog.h */
#ifndef WUBU_LEAPFROG_H
#define WUBU_LEAPFROG_H
#include <stddef.h>
/* Velocity Verlet / leapfrog on Poincare ball.
   Returns final energy. */
double wubu_leapfrog(const double *q0, const double *p0,
    double (*gradV)(const double*, size_t, int, void*),
    void *ctx, size_t dim, double dt, int nsteps,
    double *q_final, double *p_final);
#endif