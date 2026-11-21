#include "wubu_fibonacci_braid.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void wubu_fib_braid_phase(int outcome, double *re, double *im) {
    /* R_tau^tau (outcome vacuum/tau): e^{-4πi/5} for tau, e^{?} for vacuum.
       Use standard: R_{ττ}^{1} = e^{-4π i /5}, R_{ττ}^{τ} = e^{3π i /5} (up to convention). */
    if (outcome == 0) {
        /* vacuum channel: phase e^{4π i/5}? use e^{-4πi/5} */
        *re = cos(-4*M_PI/5.0); *im = sin(-4*M_PI/5.0);
    } else {
        *re = cos(3*M_PI/5.0); *im = sin(3*M_PI/5.0);
    }
}

double wubu_fib_quantum_dim(void) {
    return (1.0 + sqrt(5.0)) / 2.0; /* golden ratio */
}