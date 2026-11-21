/* wubu_fib_error_correction.h */
#ifndef WUBU_FIB_ERROR_CORRECTION_H
#define WUBU_FIB_ERROR_CORRECTION_H
/* Fibonacci anyon error correction threshold (phenomenological).
   Returns 1 if decoding succeeds (error rate below threshold). */
int wubu_fib_decode_success(double error_rate, double threshold);
#endif