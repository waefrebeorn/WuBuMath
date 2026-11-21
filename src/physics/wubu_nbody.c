#include "wubu_nbody.h"
#include "wubu_expmap.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void compute_forces(const double *pos, const double *mass, int N, size_t dim,
    double G, double softening, double *force, double *pe) {
    memset(force, 0, (size_t)N*dim*sizeof(double));
    *pe = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            double r2 = softening*softening;
            for (size_t d = 0; d < dim; d++) {
                double diff = pos[j*dim+d]-pos[i*dim+d];
                r2 += diff*diff;
            }
            double r = sqrt(r2);
            double fmag = G * mass[i] * mass[j] / (r2 * r);
            *pe -= G * mass[i] * mass[j] / r;
            for (size_t d = 0; d < dim; d++) {
                double diff = pos[j*dim+d]-pos[i*dim+d];
                force[i*dim+d] += fmag * diff;
                force[j*dim+d] -= fmag * diff;
            }
        }
    }
}

double wubu_nbody_step(double *pos, double *vel, const double *mass, int N, size_t dim,
    double G, double dt, double softening) {
    double *force = calloc((size_t)N*dim, sizeof(double));
    double pe;
    compute_forces(pos, mass, N, dim, G, softening, force, &pe);
    for (int i = 0; i < N; i++)
        for (size_t d = 0; d < dim; d++)
            vel[i*dim+d] += (force[i*dim+d]/mass[i]) * dt;
    for (int i = 0; i < N; i++) {
        double v[8];
        for (size_t d = 0; d < dim && d < 8; d++) v[d] = vel[i*dim+d]*dt;
        wubu_poincare_exp(&pos[i*dim], v, 1.0, &pos[i*dim], dim);
    }
    free(force);
    double ke = 0;
    for (int i = 0; i < N; i++) for (size_t d = 0; d < dim; d++) ke += 0.5*mass[i]*vel[i*dim+d]*vel[i*dim+d];
    return ke + pe;
}
