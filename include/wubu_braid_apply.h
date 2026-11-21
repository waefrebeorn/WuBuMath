/* wubu_braid_apply.h */
#ifndef WUBU_BRAID_APPLY_H
#define WUBU_BRAID_APPLY_H
#include <stddef.h>
/* Apply a braid word (sequence of 2x2 complex-ish gates stored as 4 doubles
   re,im per entry? simplified: real 2x2 matrices) to a 2-vector state.
   gates: nwords x 4 (a,b,c,d). word: indices. n: length. state: 2 doubles. */
void wubu_braid_apply(const double *gates, const int *word, int n, double *state);
#endif