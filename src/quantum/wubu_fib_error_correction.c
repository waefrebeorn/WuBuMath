#include "wubu_fib_error_correction.h"
#include <math.h>

/* Fibonacci anyon error correction:
   The Fibonacci code has a threshold around p_th ~ 0.109 for independent
   errors on a triangular lattice. Below threshold, logical error rate
   decays exponentially with code distance. */
int wubu_fib_decode_success(double error_rate, double threshold) {
    /* threshold should be ~0.109 for Fibonacci anyons */
    if (threshold <= 0) threshold = 0.109;
    if (error_rate < 0) error_rate = 0;
    if (error_rate > 1) error_rate = 1;
    /* Success if error rate is below threshold */
    if (error_rate >= threshold) return 0;
    /* Additional check: even below threshold, very high error rates near
       threshold have high logical failure probability */
    double margin = threshold - error_rate;
    if (margin < 0.01) return (error_rate < threshold * 0.95) ? 1 : 0;
    return 1;
}
