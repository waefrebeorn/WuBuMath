/* wubu_jones.h */
#ifndef WUBU_JONES_H
#define WUBU_JONES_H
#include <stddef.h>
/* Jones polynomial evaluation at t (simplified: returns a scalar) */
double wubu_jones_eval(const int *braid_word, int n, double t);
#endif