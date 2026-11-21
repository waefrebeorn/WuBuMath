#include "wubu_cnot.h"

void wubu_cnot(double *state, int control) {
    /* state layout: [re00,im00, re01,im01, re10,im10, re11,im11] */
    /* CNOT: if control qubit=0 (first), flip target on |1> control */
    /* control=0 means first qubit controls */
    if (control == 0) {
        /* swap |10> <-> |11> */
        for (int k = 0; k < 2; k++) {
            double t = state[4+k]; state[4+k] = state[6+k]; state[6+k] = t;
        }
    } else {
        /* second qubit controls: swap |01> <-> |11> */
        for (int k = 0; k < 2; k++) {
            double t = state[2+k]; state[2+k] = state[6+k]; state[6+k] = t;
        }
    }
}

void wubu_cphase(double *state, double phase_re, double phase_im) {
    /* apply phase to |11> component (indices 6,7) */
    double re = state[6], im = state[7];
    state[6] = re*phase_re - im*phase_im;
    state[7] = re*phase_im + im*phase_re;
}