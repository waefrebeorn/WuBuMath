#include "wubu_leapfrog.h"
#include "wubu_symplectic.h"
#include "wubu_expmap.h"
#include <math.h>
#include <string.h>
double wubu_leapfrog(const double *q0, const double *p0,
    double (*gradV)(const double*, size_t, int, void*),
    void *ctx, size_t dim, double dt, int nsteps,
    double *q_final, double *p_final) {
    double q[64], p[64];
    memcpy(q, q0, dim*sizeof(double));
    memcpy(p, p0, dim*sizeof(double));
    for (int step = 0; step < nsteps; step++) {
        /* half-kick */
        double ph[64];
        for (size_t i = 0; i < dim; i++) ph[i] = p[i] - 0.5*dt*gradV(q, dim, (int)i, ctx);
        /* drift */
        double v[64];
        for (size_t i = 0; i < dim; i++) v[i] = dt * ph[i];
        double qn[64];
        wubu_poincare_exp(q, v, 1.0, qn, dim);
        /* half-kick */
        for (size_t i = 0; i < dim; i++) p_final[i] = ph[i] - 0.5*dt*gradV(qn, dim, (int)i, ctx);
        memcpy(q, qn, dim*sizeof(double));
        memcpy(p, p_final, dim*sizeof(double));
    }
    memcpy(q_final, q, dim*sizeof(double));
    memcpy(p_final, p, dim*sizeof(double));
    /* energy */
    double ke = 0, ve = 0;
    for (size_t i = 0; i < dim; i++) { ke += 0.5*p_final[i]*p_final[i]; ve += 0.5*q_final[i]*q_final[i]; }
    return ke + ve;
}