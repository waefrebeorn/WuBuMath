/* wubu_anyon_fusion.h */
#ifndef WUBU_ANYON_FUSION_H
#define WUBU_ANYON_FUSION_H
#include <stddef.h>
/* Fibonacci anyon fusion: type 0=vacuum, 1=tau.
   Returns number of possible outcomes (1 or 2), fills outcomes[]. */
int wubu_fibonacci_fuse(int a, int b, int *outcomes);
#endif