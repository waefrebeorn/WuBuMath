/* wubu_solovay_kitaev.h */
#ifndef WUBU_SOLOVAY_KITAEV_H
#define WUBU_SOLOVAY_KITAEV_H
#include <stddef.h>
/* Approximate target unit quaternion by a word over a gate set.
   gates: array of n_gates quaternions (4 each). out_word: indices (max len).
   Returns word length used. */
int wubu_sk_approx(const double *target, const double *gates, int ngates,
                   int max_len, int *out_word);
#endif