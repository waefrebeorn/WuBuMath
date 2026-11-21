/* wubu_init.h */
#ifndef WUBU_INIT_H
#define WUBU_INIT_H
#include <stddef.h>
/* Weight initialization schemes */
void wubu_init_xavier(double *w, size_t rows, size_t cols, unsigned seed);
void wubu_init_he(double *w, size_t rows, size_t cols, unsigned seed);
void wubu_init_uniform(double *w, size_t n, double lo, double hi, unsigned seed);
#endif