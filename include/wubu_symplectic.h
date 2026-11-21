/* wubu_symplectic.h */
#ifndef WUBU_SYMPLECTIC_H
#define WUBU_SYMPLECTIC_H
#include <stddef.h>
/* Symplectic Euler on Poincare ball:
   p_{n+1} = p_n - dt * grad_V(q_n)
   q_{n+1} = exp_q(q_n, dt * p_{n+1})  (Riemannian exp map)
   Returns energy (Hamiltonian) at final state. */
double wubu_symplectic_step(const double *q, const double *p,
    double (*gradV)(const double*, size_t, int, void*),
    void *ctx, size_t dim, double dt, double *q_out, double *p_out);
#endif