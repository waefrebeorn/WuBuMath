/* wubu_cnot.h */
#ifndef WUBU_CNOT_H
#define WUBU_CNOT_H
#include <stddef.h>
/* Apply CNOT to 2-qubit state (4 amplitudes, complex: 8 doubles). control=0/1. */
void wubu_cnot(double *state, int control);
/* Apply controlled-phase (phase on |11>) */
void wubu_cphase(double *state, double phase_re, double phase_im);
#endif