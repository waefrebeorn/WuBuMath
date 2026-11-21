#include "wubu_fib_fmove.h"
#include <math.h>

void wubu_fib_fmatrix(double *F) {
    /* Fibonacci anyon F-move matrix (recoupling):
       φ = (1+√5)/2 (golden ratio)
       F = [[1/φ, 1/√φ],
            [1/√φ, -1/φ]]
       This is a unitary 2x2 matrix for Fibonacci anyon fusion. */
    double phi = (1.0 + sqrt(5.0)) / 2.0;
    double inv_phi = 1.0 / phi;
    double inv_sqrt_phi = 1.0 / sqrt(phi);
    F[0] = inv_phi;
    F[1] = inv_sqrt_phi;
    F[2] = inv_sqrt_phi;
    F[3] = -inv_phi;
}
