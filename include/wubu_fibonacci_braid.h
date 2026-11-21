/* wubu_fibonacci_braid.h */
#ifndef WUBU_FIBONACCI_BRAID_H
#define WUBU_FIBONACCI_BRAID_H
/* Returns the braiding phase (complex) for two Fibonacci anyons fusing to outcome.
   outcome: 0 (vacuum) or 1 (tau). out_re, out_im filled. */
void wubu_fib_braid_phase(int outcome, double *re, double *im);
/* Quantum dimension of tau anyon */
double wubu_fib_quantum_dim(void);
#endif