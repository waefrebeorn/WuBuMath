/* wubu_mitm_search.h */
#ifndef WUBU_MITM_SEARCH_H
#define WUBU_MITM_SEARCH_H
#include <stddef.h>
/* Meet-in-the-middle: given target (complex 4x4 = 32 doubles), find best braid word.
   Simplified: returns best distance found. */
double wubu_mitm_search(const double *target, const double *gate_set, int ngates, int half_len);
#endif