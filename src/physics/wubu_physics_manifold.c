#include "wubu_physics_manifold.h"
#include <math.h>
double wubu_hamiltonian(const double *x, const double *p, size_t dim) {
    double kin = 0, pot = 0;
    for (size_t i = 0; i < dim; i++) { kin += 0.5*p[i]*p[i]; pot += 0.5*x[i]*x[i]; }
    return kin + pot;
}