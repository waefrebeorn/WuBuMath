#include "wubu_init.h"
#include <stdlib.h>
#include <math.h>
void wubu_init_xavier(double *w, size_t rows, size_t cols, unsigned seed) {
    srand(seed);
    double scale = sqrt(2.0/(rows+cols));
    for (size_t i = 0; i < rows*cols; i++) w[i] = ((double)rand()/RAND_MAX*2-1)*scale;
}
void wubu_init_he(double *w, size_t rows, size_t cols, unsigned seed) {
    srand(seed);
    double scale = sqrt(2.0/rows);
    for (size_t i = 0; i < rows*cols; i++) w[i] = ((double)rand()/RAND_MAX*2-1)*scale;
}
void wubu_init_uniform(double *w, size_t n, double lo, double hi, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) w[i] = lo + (hi-lo)*(double)rand()/RAND_MAX;
}