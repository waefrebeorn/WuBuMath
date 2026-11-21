/* wubu_physics_manifold.h */
#ifndef WUBU_PHYSICS_MANIFOLD_H
#define WUBU_PHYSICS_MANIFOLD_H
#include <stddef.h>
/* Physics-informed: Hamiltonian on hyperbolic space.
   H = 0.5 * g^{ij} p_i p_j + V(x). Simplified: kinetic = 0.5*sum(p^2), potential = sum(x^2). */
double wubu_hamiltonian(const double *x, const double *p, size_t dim);
#endif