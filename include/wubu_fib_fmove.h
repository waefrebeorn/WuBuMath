/* wubu_fib_fmove.h */
#ifndef WUBU_FIB_FMOVE_H
#define WUBU_FIB_FMOVE_H
/* Fibonacci F-matrix (2x2) for fusion channel tau x tau -> vacuum/tau.
   F = 1/sqrt(phi) * [[1, 1/phi?]] ... simplified returns golden ratio matrix. */
void wubu_fib_fmatrix(double *F);
#endif